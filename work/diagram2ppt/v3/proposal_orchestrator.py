"""Multi-agent proposal orchestration for v3 reconstruction.

This module moves reconstruction away from a strict one-defect serial loop.
For each semantic region task, agents work in sandboxed candidate IRs.  The
planner renders and verifies candidates, then commits only proposals that beat
the current blackboard according to real rendered evidence.
"""
from __future__ import annotations

import copy
import json
import os
import signal
import threading
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat

from work.diagram2ppt.v2 import render as v2_render
from . import (
    builder,
    caption_recovery,
    ir as IR,
    renderer,
    rendering_methods,
    procedural_surface,
    strategy,
    typography,
    verifier,
)
from .agents.base import Agent


def run(
    ir: dict,
    original: Image.Image,
    image_path: str | Path,
    out_dir: str | Path,
    agents: dict[str, Agent],
    task_graph: dict,
    log: Callable[[str], None] = print,
    max_tasks: int = 4,
) -> dict:
    """Run region-level multi-agent proposal selection.

    The function mutates ``ir`` only when a proposal candidate is accepted.
    Every agent operates on a copy during proposal generation.
    """
    out_root = Path(out_dir) / "proposal_phase"
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "proposal-phase-v3-representation-aware",
        "start_metrics": copy.deepcopy(ir.get("metrics", {})),
        "representation_plan": (ir.get("strategy_plan") or {}).get("representation_plan", {}),
        "tasks": [],
        "accepted": 0,
    }
    scheduled_tasks = _scheduled_tasks(task_graph, max_tasks)
    has_visual_tasks = any(
        t.get("kind") == "visual_region_defect"
        for t in (task_graph.get("tasks", []) or [])
    )
    report["scheduler"] = {
        "input_tasks": len(task_graph.get("tasks", []) or []),
        "scheduled_tasks": len(scheduled_tasks),
        "max_tasks": max_tasks,
        "visual_tasks_only": (
            has_visual_tasks
            and not _env_bool("I2E_PROPOSAL_INCLUDE_REGION_TASKS", default=False)
        ),
        "include_defect_clusters": _env_bool(
            "I2E_PROPOSAL_INCLUDE_DEFECT_CLUSTERS", default=False),
        "task_timeout_sec": _task_timeout_sec(),
    }

    for task_index, task in enumerate(scheduled_tasks):
        available_roles = [r for r in task.get("agent_roles", []) if r in agents]
        if not available_roles:
            continue
        task_dir = out_root / f"{task_index:02d}_{_safe_name(task.get('id', 'task'))}"
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            task_report = _run_task_with_timeout(
                ir=ir,
                original=original,
                image_path=image_path,
                task=task,
                task_dir=task_dir,
                agents=agents,
                roles=available_roles,
                log=log,
            )
        except TimeoutError as exc:
            task_report = {
                "task_id": task.get("id"),
                "kind": task.get("kind"),
                "roles": available_roles,
                "decision": "timeout",
                "reason": str(exc),
                "candidates": [],
            }
        report["tasks"].append(task_report)
        if task_report.get("decision") == "accept":
            IR.restore(ir, task_report["_accepted_ir"])
            task_report.pop("_accepted_ir", None)
            report["accepted"] += 1
            log("[ProposalPlanner] accepted "
                f"{task.get('id')} via {task_report.get('accepted_candidate')}: "
                f"{task_report.get('reason')}")
        else:
            log("[ProposalPlanner] rejected "
                f"{task.get('id')}: {task_report.get('reason')}")

    report["end_metrics"] = copy.deepcopy(ir.get("metrics", {}))
    (out_root / "proposal_report.json").write_text(
        json.dumps(_strip_ir_payloads(report), indent=2, ensure_ascii=False, default=str))
    ir.setdefault("proposal_history", []).append(_strip_ir_payloads(report))
    return report


def _scheduled_tasks(task_graph: dict, max_tasks: int) -> list[dict[str, Any]]:
    """Select proposal tasks for the current bounded agent round.

    Region-level visual tasks are the audit system's primary work unit.  Raw
    defect clusters are fallback work; when visual tasks exist, single-defect
    repair can handle those later without blocking the whole proposal phase.
    """
    tasks = list(task_graph.get("tasks", []) or [])
    has_visual_tasks = any(t.get("kind") == "visual_region_defect" for t in tasks)
    if has_visual_tasks:
        if not _env_bool("I2E_PROPOSAL_INCLUDE_REGION_TASKS", default=False):
            tasks = [t for t in tasks if t.get("kind") == "visual_region_defect"]
        elif not _env_bool("I2E_PROPOSAL_INCLUDE_DEFECT_CLUSTERS", default=False):
            tasks = [t for t in tasks if t.get("kind") != "defect_cluster"]
    if max_tasks <= 0:
        return tasks
    return tasks[:max_tasks]


def _run_task_with_timeout(**kwargs: Any) -> dict:
    timeout = _task_timeout_sec()
    if timeout <= 0 or threading.current_thread() is not threading.main_thread():
        return _run_task(**kwargs)
    old_handler = signal.getsignal(signal.SIGALRM)
    task = kwargs.get("task") or {}
    try:
        signal.signal(signal.SIGALRM, _raise_task_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        return _run_task(**kwargs)
    except TimeoutError as exc:
        raise TimeoutError(
            f"proposal task {task.get('id') or '<unknown>'} exceeded "
            f"{timeout}s budget"
        ) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _raise_task_timeout(signum, frame) -> None:
    raise TimeoutError("proposal task timed out")


def _task_timeout_sec() -> int:
    try:
        return int(os.environ.get("I2E_PROPOSAL_TASK_TIMEOUT", "120") or 0)
    except ValueError:
        return 120


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _run_task(
    ir: dict,
    original: Image.Image,
    image_path: str | Path,
    task: dict,
    task_dir: Path,
    agents: dict[str, Agent],
    roles: list[str],
    log: Callable[[str], None],
) -> dict:
    before_metrics = copy.deepcopy(ir.get("metrics", {}))
    base_rank = _rank(before_metrics)
    before_region_delta = None
    before_region_evidence: dict[str, float] | None = None
    target_bbox = _task_bbox(task)
    if target_bbox is not None:
        try:
            baseline_png = _render_candidate_image(
                ir,
                original,
                image_path,
                task_dir / "baseline",
                stem="baseline",
            )
            before_region_evidence = _region_visual_evidence(
                original, baseline_png, target_bbox)
            before_region_delta = before_region_evidence["composite_delta"]
            _write_region_triptych(
                original,
                baseline_png,
                target_bbox,
                task_dir / "baseline_region_triptych.png",
                label="baseline",
            )
        except Exception as exc:
            log(f"[ProposalPlanner] baseline region scoring failed for "
                f"{task.get('id')}: {type(exc).__name__}: {exc}")
    candidates = []

    for role in roles:
        if not rendering_methods.candidate_allowed(task, [role]):
            continue
        cand = _candidate_from_roles(
            base_ir=ir,
            original=original,
            task=task,
            roles=[role],
            agents=agents,
            log=log,
        )
        if cand["changed"]:
            candidates.append(cand)

    forbidden = set(task.get("forbid_agents") or [])
    required = set(task.get("required_agents") or [])
    has_required_single = any(
        required.issubset(set(c.get("roles") or []))
        for c in candidates
    ) if required else False
    merged_roles = [role for role in roles if role not in forbidden]
    should_try_merged = (
        len(merged_roles) > 1
        and rendering_methods.candidate_allowed(task, merged_roles)
        and not _prefer_single_method_candidate(task)
    )
    if should_try_merged:
        merged = _candidate_from_roles(
            base_ir=ir,
            original=original,
            task=task,
            roles=merged_roles,
            agents=agents,
            log=log,
        )
        if merged["changed"]:
            candidates.append(merged)

    evaluated = []
    for idx, cand in enumerate(candidates):
        cand_dir = task_dir / f"{idx:02d}_{_safe_name(cand['name'])}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        try:
            eval_result = _render_verify_candidate(
                cand["ir"], original, image_path, cand_dir)
            cand["metrics"] = eval_result.get("metrics", {})
            cand["defect_count"] = len(eval_result.get("defects", []))
            if target_bbox is not None and eval_result.get("rendered_png"):
                cand["before_region_evidence"] = before_region_evidence
                cand["target_region_evidence"] = _region_visual_evidence(
                    original, eval_result["rendered_png"], target_bbox)
                cand["target_region_delta"] = cand["target_region_evidence"]["composite_delta"]
                cand["target_region_gain"] = (
                    None if before_region_delta is None
                    else before_region_delta - cand["target_region_delta"]
                )
                cand["target_region_evidence_gain"] = _region_evidence_gain(
                    before_region_evidence,
                    cand.get("target_region_evidence"),
                )
                _write_region_triptych(
                    original,
                    eval_result["rendered_png"],
                    target_bbox,
                    cand_dir / "target_region_triptych.png",
                    label=cand["name"],
                )
            cand["structure_scores"] = _candidate_structure_scores(cand, task)
            cand["representation_satisfied"] = _representation_satisfied(cand, task)
            cand["rank"] = _rank(cand["metrics"])
            cand["accepted_by_rank"] = cand["rank"] < base_rank
            (cand_dir / "candidate_ir.json").write_text(
                json.dumps(cand["ir"], indent=2, ensure_ascii=False, default=str))
            evaluated.append(cand)
        except Exception as exc:
            evaluated.append({
                "name": cand["name"],
                "roles": cand["roles"],
                "changed": cand["changed"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    viable = [
        c for c in evaluated
        if (
            c.get("rank") is not None
            and _passes_target_region_gate(c, task)
            and _is_real_gain(before_metrics, c.get("metrics", {}), task)
        )
    ]
    task_report = {
        "task_id": task.get("id"),
        "kind": task.get("kind"),
        "roles": list(roles),
        "task": {k: v for k, v in task.items() if k != "primary_defect"},
        "representation": _representation_contract_summary(task),
        "before_metrics": before_metrics,
        "target_bbox": target_bbox,
        "before_region_delta": before_region_delta,
        "before_region_evidence": before_region_evidence,
        "candidates": [_candidate_summary(c) for c in evaluated],
        "decision": "reject",
        "accepted_candidate": "",
        "reason": "no candidate improved rendered verification",
    }
    method_locked = _preferred_method_locked_candidate(task, evaluated, before_metrics)
    if method_locked is not None:
        task_report.update(_prepare_acceptance(
            base_ir=ir,
            selected=method_locked,
            task=task,
            original=original,
            image_path=image_path,
            task_dir=task_dir,
            reason=(
                "method-locked visual transaction; planner requires procedural "
                "native reconstruction over metric-favored residual fragments"
            ),
        ))
        return task_report
    chart_component = _preferred_chart_component_candidate(task, evaluated, before_metrics)
    if chart_component is not None:
        task_report.update(_prepare_acceptance(
            base_ir=ir,
            selected=chart_component,
            task=task,
            original=original,
            image_path=image_path,
            task_dir=task_dir,
            reason=(
                "semantic chart override; native chart panel reconstruction "
                "preferred over residual trace metrics"
            ),
        ))
        return task_report
    preferred = _preferred_visual_component_candidate(task, evaluated, before_metrics)
    if preferred is not None:
        task_report.update(_prepare_acceptance(
            base_ir=ir,
            selected=preferred,
            task=task,
            original=original,
            image_path=image_path,
            task_dir=task_dir,
            reason=(
                "visual component override; native region rebuild preferred "
                "over metric-stable no-op"
            ),
        ))
        return task_report
    if not viable:
        return task_report

    best = min(viable, key=lambda c: c["rank"])
    task_report.update(_prepare_acceptance(
        base_ir=ir,
        selected=best,
        task=task,
        original=original,
        image_path=image_path,
        task_dir=task_dir,
        reason=_gain_reason(before_metrics, best.get("metrics", {})),
    ))
    return task_report


def _prepare_acceptance(
    base_ir: dict,
    selected: dict,
    task: dict,
    original: Image.Image,
    image_path: str | Path,
    task_dir: Path,
    reason: str,
) -> dict:
    """Build the actual blackboard transaction committed for a selected agent.

    Agent candidates are allowed to have global side effects while exploring:
    caption recovery, typography normalization, or helper generation may touch
    unrelated regions.  The planner commits only the task-owned region and then
    re-renders that merged blackboard so acceptance is based on the real state
    that will survive.
    """
    accepted_ir, transaction = _regional_transaction_ir(
        base_ir=base_ir,
        candidate_ir=selected.get("ir") or {},
        task=task,
        changed=selected.get("changed") or [],
    )
    accepted_dir = task_dir / "accepted_transaction"
    try:
        result = _render_verify_candidate(
            accepted_ir,
            original,
            image_path,
            accepted_dir,
        )
        after_metrics = result.get("metrics", {})
        target_bbox = _task_bbox(task)
        if target_bbox is not None and result.get("rendered_png"):
            baseline = selected.get("before_region_evidence")
            region_evidence = _region_visual_evidence(
                original,
                result["rendered_png"],
                target_bbox,
            )
            transaction["target_region_evidence"] = region_evidence
            transaction["target_region_evidence_gain"] = _region_evidence_gain(
                baseline,
                region_evidence,
            )
            _write_region_triptych(
                original,
                result["rendered_png"],
                target_bbox,
                accepted_dir / "accepted_region_triptych.png",
                label=f"accepted:{selected.get('name')}",
            )
        transaction["metrics"] = after_metrics
    except Exception as exc:
        after_metrics = selected.get("metrics", {})
        transaction["verification_error"] = f"{type(exc).__name__}: {exc}"

    accepted_ir.setdefault("proposal_transactions", []).append({
        "task_id": task.get("id"),
        "candidate": selected.get("name"),
        "reason": reason,
        **transaction,
    })
    return {
        "decision": "accept",
        "accepted_candidate": selected.get("name"),
        "after_metrics": after_metrics,
        "reason": reason,
        "transaction": transaction,
        "_accepted_ir": accepted_ir,
    }


def _regional_transaction_ir(
    base_ir: dict,
    candidate_ir: dict,
    task: dict,
    changed: list[str],
) -> tuple[dict, dict]:
    """Merge only task-owned candidate changes into a fresh blackboard copy."""
    if not candidate_ir:
        return IR.snapshot(base_ir), {
            "mode": "empty_candidate",
            "committed": [],
            "removed": [],
            "ignored": list(changed),
        }
    commit_bbox = _commit_bbox(task)
    if commit_bbox is None or task.get("kind") == "defect_cluster":
        return IR.snapshot(candidate_ir), {
            "mode": "full_candidate",
            "bbox": commit_bbox,
            "committed": list(changed),
            "removed": [],
            "ignored": [],
        }

    changed_ids = {str(x) for x in changed if x}
    base = IR.snapshot(base_ir)
    candidate_by_id = {
        str(el.get("id")): el
        for el in candidate_ir.get("elements", [])
        if el.get("id") is not None
    }
    base_by_id = {
        str(el.get("id")): el
        for el in base.get("elements", [])
        if el.get("id") is not None
    }
    replace_ids: set[str] = set()
    remove_ids: set[str] = set()
    ignored: list[str] = []

    for eid in changed_ids:
        cand_el = candidate_by_id.get(eid)
        base_el = base_by_id.get(eid)
        if cand_el is not None:
            if _element_owned_by_task(cand_el, task, commit_bbox, changed_ids):
                replace_ids.add(eid)
            else:
                ignored.append(eid)
        elif base_el is not None:
            if _element_owned_by_task(base_el, task, commit_bbox, changed_ids):
                remove_ids.add(eid)
            else:
                ignored.append(eid)

    committed: list[str] = []
    kept = [
        el for el in base.get("elements", [])
        if str(el.get("id")) not in replace_ids
        and str(el.get("id")) not in remove_ids
    ]
    for el in candidate_ir.get("elements", []):
        eid = str(el.get("id"))
        if eid not in replace_ids:
            continue
        kept.append(copy.deepcopy(el))
        committed.append(eid)
    base["elements"] = kept
    base.setdefault("history", []).append({
        "agent": "ProposalPlanner",
        "action": "regional_transaction_commit",
        "round": base.get("round", 0),
        "task_id": task.get("id"),
        "candidate_changed": list(changed_ids),
        "committed": committed,
        "removed": sorted(remove_ids),
        "ignored": sorted(set(ignored)),
        "bbox": commit_bbox,
    })
    return base, {
        "mode": "regional",
        "bbox": commit_bbox,
        "committed": committed,
        "removed": sorted(remove_ids),
        "ignored": sorted(set(ignored)),
    }


def _commit_bbox(task: dict) -> list[float] | None:
    boxes = []
    for key in ("bbox", "strategy_bbox"):
        bbox = task.get(key)
        if bbox and len(bbox) == 4:
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            if x1 > x0 and y1 > y0:
                boxes.append([x0, y0, x1, y1])
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def _element_owned_by_task(
    el: dict,
    task: dict,
    bbox: list[float],
    changed_ids: set[str],
) -> bool:
    eid = str(el.get("id") or "")
    if not eid:
        return False
    if eid in set(str(x) for x in task.get("element_ids", []) if x):
        return True
    eb = el.get("bbox")
    if _method_owned_id(eid, task):
        if not eb or len(eb) != 4:
            return True
        return _bbox_overlap_fraction(eb, bbox) > 0.0 or _bbox_center_inside(eb, bbox)
    if not eb or len(eb) != 4:
        return False
    return _bbox_overlap_fraction(eb, bbox) >= 0.10 or _bbox_center_inside(eb, bbox)


def _method_owned_id(eid: str, task: dict) -> bool:
    method = str(task.get("locked_method") or ((task.get("representation") or {}).get("method") or ""))
    prefixes = {
        "procedural_surface": (
            "proc_",
            "surface_seed_",
            "bottom_mini_",
            "bottom_check",
            "bottom_nuisance_",
        ),
        "pipeline_context_layout": ("pipeline_context_", "generic_flow_"),
        "chart_parser": ("chart_", "generic_chart_"),
        "auditor_card_layout": ("auditor_",),
        "component_layout": ("action_card_",),
        "failure_summary_layout": ("failure_summary_",),
        "mini_surface_checklist": ("bottom_mini_", "bottom_check", "bottom_nuisance_"),
        "cross_panel_bridge": ("cross_panel_bridge_",),
    }.get(method, ())
    return bool(prefixes and eid.startswith(prefixes))


def _candidate_from_roles(
    base_ir: dict,
    original: Image.Image,
    task: dict,
    roles: list[str],
    agents: dict[str, Agent],
    log: Callable[[str], None],
) -> dict:
    cand_ir = IR.snapshot(base_ir)
    before_ids = _element_signature(cand_ir)
    before_fingerprints = _element_fingerprints(cand_ir)
    changed: list[str] = []
    defect = task.get("primary_defect")

    for role in roles:
        agent = agents.get(role)
        if not agent:
            continue
        try:
            out = agent.run(
                cand_ir,
                original,
                defect=defect,
                task=task,
                proposal_mode=True,
                expected_fixes=task.get("defect_ids", []),
            )
            if isinstance(out, list):
                changed.extend(str(x) for x in out)
        except Exception as exc:
            log(f"[ProposalPlanner] {role} proposal failed for "
                f"{task.get('id')}: {type(exc).__name__}: {exc}")

    caption_stats = caption_recovery.apply(cand_ir)
    if caption_stats.get("captions_added") or caption_stats.get("captions_normalized"):
        changed.append("global_bottom_caption")
    if (
        caption_stats.get("solution_subtitles_added")
        or caption_stats.get("solution_subtitles_normalized")
    ):
        changed.append("solution_subtitle")
    typo_stats = typography.apply(cand_ir)
    if typo_stats.get("styled"):
        after_fingerprints = _element_fingerprints(cand_ir)
        changed.extend([
            eid for eid, fp in after_fingerprints.items()
            if before_fingerprints.get(eid) != fp
        ])
    after_ids = _element_signature(cand_ir)
    changed.extend(sorted(after_ids.symmetric_difference(before_ids)))
    changed = _dedupe(changed)
    return {
        "name": "+".join(roles),
        "roles": roles,
        "changed": changed,
        "ir": cand_ir,
    }


def _render_verify_candidate(
    cand_ir: dict,
    original: Image.Image,
    image_path: str | Path,
    cand_dir: Path,
) -> dict:
    rendered_png = _render_candidate_image(
        cand_ir,
        original,
        image_path,
        cand_dir,
        stem="candidate",
    )
    result = verifier.verify(cand_ir, str(image_path), rendered_png)
    strategy.apply_defect_strategy(cand_ir)
    result["defects"] = cand_ir.get("defects", [])
    result["metrics"] = cand_ir.get("metrics", result.get("metrics", {}))
    result["rendered_png"] = rendered_png
    return result


def _render_candidate_image(
    cand_ir: dict,
    original: Image.Image,
    image_path: str | Path,
    cand_dir: Path,
    stem: str,
) -> str:
    cand_dir.mkdir(parents=True, exist_ok=True)
    pptx_path = cand_dir / f"{stem}.pptx"
    builder.build_pptx(cand_ir, str(pptx_path))
    if renderer.is_available():
        try:
            return renderer.render_isolated(str(pptx_path), str(cand_dir / f"{stem}.true.png"))
        except Exception:
            pass
    rendered_png = str(cand_dir / f"{stem}.proxy.png")
    _ensure_proxy_image(cand_ir, image_path, original)
    v2_render.render(cand_ir, original).save(rendered_png)
    return rendered_png


def _prefer_single_method_candidate(task: dict) -> bool:
    """Avoid expensive mixed-agent candidates when the method owner spoke.

    Component and semantic-chart tasks are transactions: the planner is asking
    for a method-specific native rebuild, then measuring it.  Adding Icon/Text/
    Style/Connector agents into the same candidate often triggers extra remote
    calls and makes the causal attribution unclear.  Follow-up agents should run
    as later tasks after the owning method candidate is accepted.
    """
    return str(task.get("acceptance_policy") or "") in {
        "method_locked_visual",
        "semantic_chart",
        "component_visual",
    }


def _rank(metrics: dict) -> tuple:
    return (
        round(float(metrics.get("visual_delta", 1.0)), 4),
        int(metrics.get("critical_defect_count", 9999)),
        float(metrics.get("text_layout_error", 1.0)),
        int(metrics.get("text_layout_mismatch_count", 9999)),
        float(metrics.get("text_template_error", 1.0)),
        int(metrics.get("text_template_mismatch_count", 9999)),
        int(metrics.get("defect_count", 9999)),
        -float(metrics.get("text_accuracy", 0.0)),
        -float(metrics.get("coverage_explained", 0.0)),
    )


def _is_real_gain(before: dict, after: dict, task: dict) -> bool:
    if not after:
        return False
    if after.get("native_fraction_count", 1.0) < 1.0:
        return False
    visual_gain = float(before.get("visual_delta", 1.0)) - float(after.get("visual_delta", 1.0))
    critical_gain = int(before.get("critical_defect_count", 9999)) - int(after.get("critical_defect_count", 9999))
    defect_gain = int(before.get("defect_count", 9999)) - int(after.get("defect_count", 9999))
    text_gain = float(after.get("text_accuracy", 0.0)) - float(before.get("text_accuracy", 0.0))
    layout_gain = (
        float(before.get("text_layout_error", 1.0))
        - float(after.get("text_layout_error", 1.0))
    )
    layout_count_gain = (
        int(before.get("text_layout_mismatch_count", 9999))
        - int(after.get("text_layout_mismatch_count", 9999))
    )
    template_gain = (
        float(before.get("text_template_error", 1.0))
        - float(after.get("text_template_error", 1.0))
    )
    template_count_gain = (
        int(before.get("text_template_mismatch_count", 9999))
        - int(after.get("text_template_mismatch_count", 9999))
    )
    coverage_gain = float(after.get("coverage_explained", 0.0)) - float(before.get("coverage_explained", 0.0))
    overflow_delta = (
        int(after.get("typography_overflow_count", 0))
        - int(before.get("typography_overflow_count", 0))
    )

    policy = str(task.get("acceptance_policy") or "")
    if policy not in {"method_locked_visual", "semantic_chart", "component_visual"}:
        if overflow_delta > 2 and visual_gain < 0.030:
            return False
        if critical_gain < 0 and text_gain < -0.02:
            return False
        if text_gain < -0.10:
            return False
        if text_gain < -0.08 and visual_gain < 0.040:
            return False
        if critical_gain < -2 and visual_gain < 0.030:
            return False
    if text_gain < -0.005 and visual_gain < 0.01:
        return False
    roles = set(task.get("agent_roles") or [])
    if "TextLayoutAgent" in roles and task.get("kind") == "defect_cluster":
        return (
            (layout_gain >= 0.010 or layout_count_gain > 0)
            and visual_gain > -0.010
            and text_gain > -0.020
        )
    if "TemplateSlotAgent" in roles and task.get("kind") == "defect_cluster":
        return (
            (template_gain >= 0.010 or template_count_gain > 0)
            and visual_gain > -0.012
            and text_gain > -0.025
        )
    if "ConnectorAgent" in roles and task.get("kind") == "defect_cluster":
        return visual_gain >= 0.001 or (defect_gain >= 3 and visual_gain >= 0.0)
    if policy == "method_locked_visual":
        return (
            visual_gain >= 0.001
            or (coverage_gain >= 0.010 and visual_gain > -0.004)
            or (critical_gain > 0 and visual_gain > -0.004)
        )
    if task.get("kind") == "visual_region_defect":
        # Most visual-review tasks still need rendered evidence.  Narrow
        # method-selection overrides live in _preferred_visual_component_candidate
        # below; do not accept metric-stable no-ops here.
        return (
            visual_gain >= 0.002
            or (critical_gain > 0 and visual_gain > -0.004)
            or (defect_gain >= 2 and visual_gain > -0.004)
        )
    if visual_gain >= 0.002:
        return True
    if critical_gain > 0 and visual_gain > -0.006:
        return True
    if defect_gain >= 2 and visual_gain > -0.004:
        return True
    if layout_gain >= 0.018 and visual_gain > -0.006 and text_gain > -0.02:
        return True
    if task.get("kind") in {"component_card_row", "pipeline_context_row"}:
        return text_gain >= 0.015 and visual_gain > -0.006
    if task.get("kind") == "procedural_3d_surface":
        return coverage_gain >= 0.015 and visual_gain > -0.006
    return False


def _preferred_method_locked_candidate(
    task: dict,
    evaluated: list[dict],
    before: dict,
) -> dict | None:
    if task.get("acceptance_policy") != "method_locked_visual":
        return None
    before_visual = float(before.get("visual_delta", 1.0))
    before_critical = int(before.get("critical_defect_count", 9999))
    required = set(task.get("required_agents") or [])
    candidates = []
    for cand in evaluated:
        metrics = cand.get("metrics") or {}
        roles = set(cand.get("roles") or [])
        if not metrics or metrics.get("native_fraction_count", 1.0) < 1.0:
            continue
        if required and not required.issubset(roles):
            continue
        if cand.get("name") != "ProceduralSurfaceAgent":
            continue
        if len(cand.get("changed") or []) < 1:
            continue
        if not cand.get("representation_satisfied"):
            continue
        structure = (cand.get("structure_scores") or {}).get("procedural_surface", 0.0)
        debt = _region_visual_debt(cand)
        contract_pass = (
            structure >= 0.95
            and cand.get("representation_satisfied") is True
            and _passes_target_region_gate(cand, task)
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.090
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 40
        )
        structural_pass = (
            structure >= 0.82
            and debt <= 0.015
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.095
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 28
        )
        if not _visual_acceptance_guard(
            cand,
            before,
            task,
            max_global_debt=0.095,
            max_critical_debt=40,
        ):
            continue
        if not _passes_target_region_gate(cand, task) and not structural_pass and not contract_pass:
            continue
        allowed_visual_debt = 0.090 if contract_pass else 0.095 if structural_pass else 0.055
        if float(metrics.get("visual_delta", 1.0)) > before_visual + allowed_visual_debt:
            continue
        candidates.append(cand)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (
        float((c.get("metrics") or {}).get("visual_delta", 1.0)),
        -len(c.get("changed") or []),
    ))
    return candidates[0]


def _preferred_chart_component_candidate(
    task: dict,
    evaluated: list[dict],
    before: dict,
) -> dict | None:
    region_id = str(task.get("region_id") or "")
    task_id = str(task.get("id") or "")
    is_q0 = region_id == "q0_coverage_charts" or "q0_coverage" in task_id
    if task.get("acceptance_policy") != "semantic_chart":
        return None
    before_visual = float(before.get("visual_delta", 1.0))
    before_critical = int(before.get("critical_defect_count", 9999))
    candidates = []
    for cand in evaluated:
        metrics = cand.get("metrics") or {}
        roles = set(cand.get("roles") or [])
        if not metrics or "ChartAgent" not in roles:
            continue
        if metrics.get("native_fraction_count", 1.0) < 1.0:
            continue
        if len(cand.get("changed") or []) < 3:
            continue
        scores = cand.get("structure_scores") or {}
        typo_ok = _typography_contract_pass(cand, min_score=0.72)
        q0_score = float(scores.get("q0_coverage", 0.0))
        generic_score = float(scores.get("generic_chart", 0.0))
        q0_method_pass = (
            is_q0
            and
            q0_score >= 0.92
            and cand.get("representation_satisfied") is True
            and typo_ok
            and len(cand.get("changed") or []) >= 30
            and _region_visual_debt(cand) <= 0.055
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.090
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 38
        )
        generic_method_pass = (
            not is_q0
            and generic_score >= 0.72
            and cand.get("representation_satisfied") is True
            and len(cand.get("changed") or []) >= 3
            and _region_visual_debt(cand) <= 0.020
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.040
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 16
        )
        structural_pass = _component_structural_pass(
            cand, roles, metrics, before_visual, before_critical)
        if not _visual_acceptance_guard(
            cand,
            before,
            task,
            max_global_debt=0.040,
            min_region_gain=-0.014,
            max_critical_debt=24,
            max_text_drop=0.20,
        ):
            continue
        if (
            not _passes_target_region_gate(cand, task)
            and not structural_pass
            and not q0_method_pass
            and not generic_method_pass
        ):
            continue
        if float(metrics.get("visual_delta", 1.0)) > before_visual + 0.055:
            if not structural_pass and not q0_method_pass and not generic_method_pass:
                continue
        candidates.append(cand)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (
        0 if c.get("name") == "ChartAgent" else 1,
        _region_visual_debt(c),
        float((c.get("metrics") or {}).get("visual_delta", 1.0)),
        -len(c.get("changed") or []),
    ))
    return candidates[0]


def _preferred_visual_component_candidate(
    task: dict,
    evaluated: list[dict],
    before: dict,
) -> dict | None:
    """Pick component reconstruction when scalar metrics punish method fixes.

    Some regions are visibly wrong because the method is wrong, not because a
    single residual box is large.  Replacing noisy OCR fragments with a clean
    native component can temporarily hurt text/residual metrics while making
    the slide structurally closer.  Keep this override narrow and native-only.
    """
    region_id = str(task.get("region_id") or "")
    is_auditor = (
        task.get("kind") == "auditor_method_cards"
        or region_id in {"auditor_cards", "region_auditor_cards"}
        or task.get("acceptance_policy") == "component_visual"
    )
    if not is_auditor:
        return None
    before_visual = float(before.get("visual_delta", 1.0))
    before_critical = int(before.get("critical_defect_count", 9999))
    candidates = []
    component_agents = {
        "LayoutAgent",
        "PipelineContextAgent",
        "ActionCardAgent",
        "AuditorCardAgent",
        "FailureSummaryAgent",
        "BottomMiniSurfaceAgent",
    }
    for cand in evaluated:
        metrics = cand.get("metrics") or {}
        roles = set(cand.get("roles") or [])
        if not metrics or not roles.intersection(component_agents):
            continue
        if metrics.get("native_fraction_count", 1.0) < 1.0:
            continue
        scores = cand.get("structure_scores") or {}
        typo_ok = _typography_contract_pass(cand, min_score=0.70)
        auditor_method_pass = (
            "AuditorCardAgent" in roles
            and float(scores.get("auditor_cards", 0.0)) >= 0.92
            and typo_ok
            and len(cand.get("changed") or []) >= 50
            and _region_visual_debt(cand) <= 0.015
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.095
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 42
        )
        action_method_pass = (
            "ActionCardAgent" in roles
            and float(scores.get("action_cards", 0.0)) >= 0.84
            and typo_ok
            and len(cand.get("changed") or []) >= 30
            and _region_visual_debt(cand) <= 0.014
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.085
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 36
        )
        pipeline_method_pass = (
            "PipelineContextAgent" in roles
            and float(scores.get("pipeline_context", 0.0)) >= 0.92
            and typo_ok
            and len(cand.get("changed") or []) >= 40
            and _region_visual_debt(cand) <= 0.006
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.040
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 24
        )
        representation_pass = (
            cand.get("representation_satisfied") is True
            and task.get("acceptance_policy") == "component_visual"
            and roles.intersection(component_agents)
            and typo_ok
            and len(cand.get("changed") or []) >= 10
            and metrics.get("native_fraction_count", 1.0) >= 1.0
            and _region_visual_debt(cand) <= 0.009
            and _region_ink_closer(cand)
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.075
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 30
        )
        failure_method_pass = (
            "FailureSummaryAgent" in roles
            and float(scores.get("failure_summary", 0.0)) >= 0.92
            and typo_ok
            and len(cand.get("changed") or []) >= 15
            and _region_visual_debt(cand) <= 0.026
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.085
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 36
        )
        bottom_method_pass = (
            "BottomMiniSurfaceAgent" in roles
            and float(scores.get("bottom_mini_surface", 0.0)) >= 0.90
            and typo_ok
            and len(cand.get("changed") or []) >= 12
            and _region_visual_debt(cand) <= 0.012
            and float(metrics.get("visual_delta", 1.0)) <= before_visual + 0.085
            and int(metrics.get("critical_defect_count", 9999)) <= before_critical + 40
        )
        component_method_pass = (
            auditor_method_pass or action_method_pass or pipeline_method_pass
            or failure_method_pass or bottom_method_pass or representation_pass
        )
        is_auditor_candidate = "AuditorCardAgent" in roles
        is_failure_candidate = "FailureSummaryAgent" in roles
        is_bottom_candidate = "BottomMiniSurfaceAgent" in roles
        if not _visual_acceptance_guard(
            cand,
            before,
            task,
            max_global_debt=(
                0.050 if is_auditor_candidate else
                0.035 if is_failure_candidate else
                0.018 if is_bottom_candidate else
                0.030 if "ActionCardAgent" in roles else
                0.012
            ),
            min_region_gain=(
                -0.012 if is_auditor_candidate else
                -0.020 if is_failure_candidate else
                -0.008 if is_bottom_candidate and _region_ink_closer(cand) else
                -0.014 if "ActionCardAgent" in roles else
                0.001
            ),
            max_critical_debt=(
                32 if is_auditor_candidate else
                16 if is_failure_candidate else
                12 if is_bottom_candidate else
                18 if "ActionCardAgent" in roles else
                8
            ),
        ):
            continue
        if float(metrics.get("visual_delta", 1.0)) > before_visual + 0.012:
            if not component_method_pass and not _component_structural_pass(cand, roles, metrics, before_visual, before_critical):
                continue
        if int(metrics.get("critical_defect_count", 9999)) > before_critical + 2:
            if not component_method_pass and not _component_structural_pass(cand, roles, metrics, before_visual, before_critical):
                continue
        if len(cand.get("changed") or []) < 10:
            continue
        if not component_method_pass and not _passes_target_region_gate(cand, task, min_gain=0.002):
            continue
        if not component_method_pass and not _component_structural_pass(cand, roles, metrics, before_visual, before_critical, relaxed=True):
            continue
        candidates.append(cand)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (
        0 if set(c.get("roles") or []).intersection({
            "PipelineContextAgent",
            "ActionCardAgent",
            "AuditorCardAgent",
            "FailureSummaryAgent",
            "BottomMiniSurfaceAgent",
        }) else 1,
        -float(c.get("target_region_evidence_gain")
               if c.get("target_region_evidence_gain") is not None
               else c.get("target_region_gain") or 0.0),
        float((c.get("metrics") or {}).get("visual_delta", 1.0)),
    ))
    return candidates[0]


def _region_ink_closer(candidate: dict) -> bool:
    evidence = candidate.get("target_region_evidence") or {}
    before = candidate.get("before_region_evidence") or {}
    if not evidence or not before:
        return True
    original = float(evidence.get("original_ink", before.get("original_ink", 0.0)) or 0.0)
    current = float(evidence.get("rendered_ink", 0.0) or 0.0)
    previous = float(before.get("rendered_ink", 0.0) or 0.0)
    return abs(original - current) <= abs(original - previous) + 0.015


def _typography_contract_pass(candidate: dict, min_score: float = 0.70) -> bool:
    score = candidate.get("typography_contract_score") or {}
    if not score:
        return True
    if int(score.get("texts") or 0) <= 0:
        return True
    return float(score.get("score") or 0.0) >= float(min_score)


def _gain_reason(before: dict, after: dict) -> str:
    parts = []
    for key in ("visual_delta", "critical_defect_count", "defect_count",
                "text_accuracy", "coverage_explained"):
        if key in before or key in after:
            parts.append(f"{key}: {before.get(key)} -> {after.get(key)}")
    return "; ".join(parts)


def _component_structural_pass(
    candidate: dict,
    roles: set[str],
    metrics: dict,
    before_visual: float,
    before_critical: int,
    relaxed: bool = False,
) -> bool:
    scores = candidate.get("structure_scores") or {}
    visual = float(metrics.get("visual_delta", 1.0))
    critical = int(metrics.get("critical_defect_count", 9999))
    debt = _region_visual_debt(candidate)
    if "AuditorCardAgent" in roles and float(scores.get("auditor_cards", 0.0)) >= 0.82:
        if relaxed:
            return debt <= 0.012
        return debt <= 0.012 and visual <= before_visual + 0.070 and critical <= before_critical + 30
    if "PipelineContextAgent" in roles and float(scores.get("pipeline_context", 0.0)) >= 0.82:
        if relaxed:
            return debt <= 0.004
        return debt <= 0.004 and visual <= before_visual + 0.032 and critical <= before_critical + 14
    if "FailureSummaryAgent" in roles and float(scores.get("failure_summary", 0.0)) >= 0.82:
        if relaxed:
            return debt <= 0.004
        return debt <= 0.004 and visual <= before_visual + 0.040 and critical <= before_critical + 12
    if "ChartAgent" in roles and float(scores.get("q0_coverage", 0.0)) >= 0.82:
        if relaxed:
            return debt <= 0.006
        return debt <= 0.006 and visual <= before_visual + 0.060 and critical <= before_critical + 16
    if "ActionCardAgent" in roles and float(scores.get("action_cards", 0.0)) >= 0.82:
        if relaxed:
            return debt <= 0.014
        return debt <= 0.014 and visual <= before_visual + 0.055 and critical <= before_critical + 18
    if "BottomMiniSurfaceAgent" in roles and float(scores.get("bottom_mini_surface", 0.0)) >= 0.82:
        if relaxed:
            return debt <= 0.006
        return debt <= 0.006 and visual <= before_visual + 0.050 and critical <= before_critical + 14
    return False


def _passes_target_region_gate(candidate: dict, task: dict, min_gain: float | None = None) -> bool:
    """Require visual evidence inside the task bbox when that evidence exists.

    Global metrics can improve while the assigned region gets worse, especially
    when residual fragments elsewhere dominate the score.  Region tasks are
    planner assignments, so their candidates must be accountable to the region
    they were asked to fix.
    """
    region_gain = candidate.get("target_region_evidence_gain")
    if region_gain is None:
        region_gain = candidate.get("target_region_gain")
    if region_gain is None:
        return True
    policy = str(task.get("acceptance_policy") or "")
    kind = str(task.get("kind") or "")
    if min_gain is None:
        if policy in {"component_visual", "method_locked_visual", "semantic_chart"}:
            min_gain = 0.0
        elif kind in {"visual_region_defect", "procedural_3d_surface"}:
            min_gain = 0.0
        else:
            min_gain = -0.003
    return float(region_gain) >= float(min_gain)


def _region_visual_debt(candidate: dict) -> float:
    gain = candidate.get("target_region_evidence_gain")
    if gain is None:
        gain = candidate.get("target_region_gain")
    if gain is None:
        return 0.0
    return max(0.0, -float(gain))


def _visual_acceptance_guard(
    candidate: dict,
    before: dict,
    task: dict,
    max_global_debt: float,
    min_region_gain: float = 0.001,
    max_critical_debt: int | None = None,
    max_text_drop: float = 0.080,
) -> bool:
    """Final safety check for representation-aware overrides.

    A component can be structurally complete and still be visually wrong.  The
    planner may prefer native reconstruction, but it cannot merge a candidate
    that makes its assigned crop worse or materially damages the full slide.
    """
    metrics = candidate.get("metrics") or {}
    if not metrics or metrics.get("native_fraction_count", 1.0) < 1.0:
        return False
    visual_debt = (
        float(metrics.get("visual_delta", 1.0))
        - float(before.get("visual_delta", 1.0))
    )
    region_gain = candidate.get("target_region_evidence_gain")
    if region_gain is None:
        region_gain = candidate.get("target_region_gain")
    if region_gain is not None and float(region_gain) < min_region_gain:
        if not (visual_debt <= -0.010 and float(region_gain) >= -0.006):
            return False
    if visual_debt > max_global_debt:
        return False
    critical_debt = (
        int(metrics.get("critical_defect_count", 9999))
        - int(before.get("critical_defect_count", 9999))
    )
    if max_critical_debt is None:
        max_critical_debt = 40 if str(task.get("acceptance_policy") or "") == "method_locked_visual" else 8
    if critical_debt > max_critical_debt and visual_debt >= 0.0:
        return False
    text_drop = (
        float(before.get("text_accuracy", 0.0))
        - float(metrics.get("text_accuracy", 0.0))
    )
    if text_drop > max_text_drop and visual_debt > -0.010:
        return False
    return True


def _candidate_summary(candidate: dict) -> dict:
    return {
        "name": candidate.get("name"),
        "roles": candidate.get("roles", []),
        "changed": candidate.get("changed", []),
        "metrics": candidate.get("metrics"),
        "target_region_delta": candidate.get("target_region_delta"),
        "target_region_gain": candidate.get("target_region_gain"),
        "target_region_evidence": candidate.get("target_region_evidence"),
        "target_region_evidence_gain": candidate.get("target_region_evidence_gain"),
        "structure_scores": candidate.get("structure_scores"),
        "typography_contract_score": candidate.get("typography_contract_score"),
        "representation_satisfied": candidate.get("representation_satisfied"),
        "rank": list(candidate["rank"]) if candidate.get("rank") else None,
        "error": candidate.get("error"),
    }


def _representation_contract_summary(task: dict) -> dict:
    rep = task.get("representation") or {}
    if not rep:
        return {}
    return {
        "method": rep.get("method") or task.get("locked_method"),
        "family": rep.get("family"),
        "owner_agent": rep.get("owner_agent"),
        "required_agents": rep.get("required_agents") or task.get("required_agents", []),
        "forbid_agents": rep.get("forbid_agents") or task.get("forbid_agents", []),
        "acceptance_policy": rep.get("acceptance_policy") or task.get("acceptance_policy"),
        "native_expression": rep.get("native_expression") or task.get("expected_native_expression", ""),
        "visual_evidence": rep.get("visual_evidence", []),
        "typography_contract": rep.get("typography_contract") or task.get("typography_contract", {}),
    }


def _strip_ir_payloads(report: dict) -> dict:
    clean = copy.deepcopy(report)
    for task in clean.get("tasks", []):
        task.pop("_accepted_ir", None)
    return clean


def _element_signature(ir: dict) -> set[str]:
    return {str(e.get("id")) for e in ir.get("elements", [])}


def _element_fingerprints(ir: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for el in ir.get("elements", []):
        eid = str(el.get("id"))
        out[eid] = json.dumps(el, sort_keys=True, ensure_ascii=False, default=str)
    return out


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_+" else "_" for ch in str(name))[:64]


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _ensure_proxy_image(ir: dict, image_path: str | Path, original: Image.Image) -> None:
    ir.setdefault("image", {
        "path": str(image_path),
        "width": original.width,
        "height": original.height,
    })


def _candidate_structure_scores(candidate: dict, task: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    typo_score = typography.score_contract(candidate.get("ir") or {}, task)
    if typo_score.get("texts", 0) > 0:
        scores["typography_contract"] = float(typo_score.get("score", 0.0))
        candidate["typography_contract_score"] = typo_score
    region_id = str(task.get("region_id") or "")
    task_id = str(task.get("id") or "")
    if task.get("kind") == "procedural_3d_surface" or region_id == "left_surface":
        scores["procedural_surface"] = _procedural_surface_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if (
        task.get("kind") == "auditor_method_cards"
        or region_id in {"auditor_cards", "region_auditor_cards"}
    ):
        scores["auditor_cards"] = _auditor_cards_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if (
        task.get("kind") == "pipeline_context_row"
        or region_id == "pipeline_context"
        or str(task.get("locked_method") or "") == "pipeline_context_layout"
    ):
        scores["pipeline_context"] = _pipeline_context_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if (
        task.get("kind") == "failure_summary_panel"
        or str(task.get("region_id") or "") == "failure_summary"
    ):
        scores["failure_summary"] = _failure_summary_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if region_id == "q0_coverage_charts" or "q0_coverage" in task_id:
        scores["q0_coverage"] = _q0_coverage_structure_score(
            candidate.get("ir") or {},
            task,
        )
    elif str(task.get("locked_method") or "") == "chart_parser" or task.get("kind") == "chart":
        scores["generic_chart"] = _generic_chart_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if task.get("kind") == "component_card_row" or region_id == "action_cards":
        scores["action_cards"] = _action_cards_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if task.get("kind") == "bottom_mini_surface" or region_id == "bottom_mini_surface":
        scores["bottom_mini_surface"] = _bottom_mini_surface_structure_score(
            candidate.get("ir") or {},
            task,
        )
    if task.get("kind") == "cross_panel_bridge" or region_id == "region_cross_panel_bridge":
        scores["cross_panel_bridge"] = _cross_panel_bridge_structure_score(
            candidate.get("ir") or {},
            task,
        )
    return scores


def _representation_satisfied(candidate: dict, task: dict) -> bool:
    method = str(task.get("locked_method") or ((task.get("representation") or {}).get("method") or ""))
    scores = candidate.get("structure_scores") or {}
    roles = set(candidate.get("roles") or [])
    if method == "procedural_surface":
        return "ProceduralSurfaceAgent" in roles and float(scores.get("procedural_surface", 0.0)) >= 0.88
    if method == "chart_parser":
        return (
            "ChartAgent" in roles
            and (
                float(scores.get("q0_coverage", 0.0)) >= 0.90
                or float(scores.get("generic_chart", 0.0)) >= 0.72
            )
        )
    if method == "pipeline_context_layout":
        return "PipelineContextAgent" in roles and float(scores.get("pipeline_context", 0.0)) >= 0.90
    if method == "auditor_card_layout":
        return "AuditorCardAgent" in roles and float(scores.get("auditor_cards", 0.0)) >= 0.90
    if method == "component_layout":
        return "ActionCardAgent" in roles and float(scores.get("action_cards", 0.0)) >= 0.90
    if method == "failure_summary_layout":
        return "FailureSummaryAgent" in roles and float(scores.get("failure_summary", 0.0)) >= 0.90
    if method == "mini_surface_checklist":
        return "BottomMiniSurfaceAgent" in roles and float(scores.get("bottom_mini_surface", 0.0)) >= 0.90
    if method == "cross_panel_bridge":
        return "CrossPanelBridgeAgent" in roles and float(scores.get("cross_panel_bridge", 0.0)) >= 0.90
    return True


def _cross_panel_bridge_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    score = 0.0
    for el in ir.get("elements", []):
        if str(el.get("id", "")) != "cross_panel_bridge_problem_to_solution":
            continue
        icon = el.get("icon") or {}
        is_arrow_shape = (
            el.get("type") == "arrow"
            or (el.get("type") == "icon" and icon.get("kind") == "arrow")
        )
        if not is_arrow_shape or not el.get("bbox"):
            continue
        if _bbox_overlap_fraction(el.get("bbox"), bbox) < 0.20:
            continue
        points = el.get("points") or []
        thickness = float(el.get("thickness") or el.get("line_width") or 0)
        ext = el.get("ext") or {}
        score += 0.34
        score += 0.24 if len(points) == 4 and float(points[2]) > float(points[0]) else 0.0
        score += 0.22 if thickness >= 28 else 0.08 if thickness >= 12 else 0.0
        score += 0.12 if ext.get("component") == "cross_panel_bridge" else 0.0
        score += 0.08 if str(el.get("color") or "").lower() in {"#2b7fb6", "#2a78a8", "#2f83b7"} else 0.0
        return min(1.0, score)
    return 0.0


def _pipeline_context_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    elements = ir.get("elements", [])
    cards = [
        e for e in elements
        if str(e.get("id", "")).startswith(("pipeline_context_card_", "generic_flow_card_"))
    ]
    texts = [
        e for e in elements
        if str(e.get("id", "")).startswith(("pipeline_context_text_", "generic_flow_text_"))
    ]
    arrows = [
        e for e in elements
        if str(e.get("id", "")).startswith(("pipeline_context_arrow_", "generic_flow_arrow_"))
    ]
    icons = [
        e for e in elements
        if str(e.get("id", "")).startswith("pipeline_context_icon_")
        and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.01
    ]
    plot_icons = [e for e in icons if e.get("type") == "dotcloud" and (e.get("paths") or [])]
    primitive_icons = [e for e in icons if e.get("type") in {"rounded_rect", "oval", "line", "dotcloud"}]
    title = any(str(e.get("id", "")) in {"pipeline_context_title", "generic_flow_title"} for e in elements)
    separator = any(str(e.get("id", "")) == "pipeline_context_separator" for e in elements)
    arrows_with_points = sum(1 for e in arrows if len(e.get("points") or []) == 4)
    parts = [
        0.20 if len(cards) >= 4 else 0.0,
        0.16 if len(texts) >= 4 else 0.0,
        0.16 if len(arrows) >= 3 and arrows_with_points >= 3 else 0.0,
        0.16 if len(primitive_icons) >= 12 else 0.08 if len(primitive_icons) >= 6 else 0.0,
        0.10 if len(plot_icons) >= 2 else 0.0,
        0.08 if title else 0.0,
        0.04 if separator else 0.0,
        0.10 if _pipeline_spread_ok(cards, bbox) else 0.0,
    ]
    return sum(parts)


def _pipeline_spread_ok(cards: list[dict], bbox: list[float]) -> bool:
    if len(cards) < 4:
        return False
    centers = sorted((float(e["bbox"][0]) + float(e["bbox"][2])) / 2 for e in cards if e.get("bbox"))
    if len(centers) < 4:
        return False
    span = centers[-1] - centers[0]
    bw = float(bbox[2]) - float(bbox[0])
    return span >= bw * 0.62


def _failure_summary_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    elements = ir.get("elements", [])
    panel = any(str(e.get("id", "")) == "failure_summary_panel" for e in elements)
    title = any(str(e.get("id", "")) == "failure_summary_title" for e in elements)
    icons = [
        e for e in elements
        if str(e.get("id", "")).startswith("failure_summary_icon_")
        and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.01
    ]
    rows = [
        e for e in elements
        if str(e.get("id", "")).startswith("failure_summary_text_")
        and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.01
    ]
    readable = sum(
        1 for e in rows
        if len(str(e.get("text") or "")) >= 12
        and float(e.get("font_size") or 0) >= 10
    )
    parts = [
        0.22 if panel else 0.0,
        0.16 if title else 0.0,
        0.22 if len(icons) >= 3 else 0.08 if len(icons) >= 1 else 0.0,
        0.28 if readable >= 3 else 0.12 if readable >= 2 else 0.0,
        0.12 if _vertical_stack_ok(rows, bbox, minimum=3) else 0.0,
    ]
    return sum(parts)


def _q0_coverage_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    elements = ir.get("elements", [])
    panel = any(str(e.get("id", "")) == "chart_q0_panel" for e in elements)
    title_ids = {
        str(e.get("id", ""))
        for e in elements
        if str(e.get("id", "")).startswith("chart_q0_title")
    }
    title = (
        "chart_q0_title" in title_ids
        or {"chart_q0_title_q", "chart_q0_title_sub", "chart_q0_title_rest"}.issubset(title_ids)
    )
    axes = [
        e for e in elements
        if str(e.get("id", "")).startswith("chart_q0_")
        and e.get("type") == "line"
    ]
    lines = [
        e for e in elements
        if str(e.get("id", "")).startswith("chart_q0_line_")
        and (e.get("paths") or [])
    ]
    bars = [
        e for e in elements
        if str(e.get("id", "")).startswith("chart_q0_bar_")
        and e.get("type") == "rect"
    ]
    labels = [
        e for e in elements
        if str(e.get("id", "")).startswith("chart_q0_")
        and e.get("type") == "text"
        and len(str(e.get("text") or "")) >= 2
    ]
    parts = [
        0.16 if panel else 0.0,
        0.12 if title else 0.0,
        0.18 if len(axes) >= 4 else 0.08 if len(axes) >= 2 else 0.0,
        0.18 if len(lines) >= 2 else 0.08 if len(lines) >= 1 else 0.0,
        0.14 if len(bars) >= 2 else 0.0,
        0.14 if len(labels) >= 10 else 0.07 if len(labels) >= 6 else 0.0,
        0.08 if _chart_panel_spread_ok(elements, bbox) else 0.0,
    ]
    return sum(parts)


def _generic_chart_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    region = str(task.get("region_id") or task.get("id") or "")
    elements = ir.get("elements", [])
    owned = [
        e for e in elements
        if str(e.get("id", "")).startswith("generic_chart_")
        and (not region or region in str(e.get("id", "")))
        and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.0
    ]
    axes = [e for e in owned if e.get("type") == "line"]
    series = [e for e in owned if e.get("type") == "freeform" and (e.get("paths") or [])]
    path_points = 0
    for e in series:
        for p in e.get("paths") or []:
            path_points += len(p.get("points") or [])
    parts = [
        0.28 if len(axes) >= 2 else 0.12 if len(axes) >= 1 else 0.0,
        0.36 if len(series) >= 1 and path_points >= 8 else 0.18 if len(series) >= 1 else 0.0,
        0.18 if _generic_chart_spread_ok(owned, bbox) else 0.0,
        0.10 if all((e.get("ext") or {}).get("component") == "generic_chart" for e in owned) and owned else 0.0,
        0.08 if len(owned) >= 3 else 0.0,
    ]
    return sum(parts)


def _generic_chart_spread_ok(elements: list[dict], bbox: list[float]) -> bool:
    if len(elements) < 3:
        return False
    x0 = min(float(e["bbox"][0]) for e in elements if e.get("bbox"))
    y0 = min(float(e["bbox"][1]) for e in elements if e.get("bbox"))
    x1 = max(float(e["bbox"][2]) for e in elements if e.get("bbox"))
    y1 = max(float(e["bbox"][3]) for e in elements if e.get("bbox"))
    bw = max(1.0, float(bbox[2]) - float(bbox[0]))
    bh = max(1.0, float(bbox[3]) - float(bbox[1]))
    return (x1 - x0) >= bw * 0.35 and (y1 - y0) >= bh * 0.35


def _action_cards_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    elements = ir.get("elements", [])
    cards = [
        e for e in elements
        if str(e.get("id", "")).startswith("action_card_")
        and e.get("type") == "rounded_rect"
    ]
    icons = [e for e in elements if str(e.get("id", "")).startswith("action_card_icon_")]
    titles = [e for e in elements if str(e.get("id", "")).startswith("action_card_title_")]
    bodies = [e for e in elements if str(e.get("id", "")).startswith("action_card_body_")]
    connectors = [e for e in elements if str(e.get("id", "")).startswith("action_card_connector_")]
    parts = [
        0.22 if len(cards) >= 4 else 0.0,
        0.16 if len(icons) >= 4 else 0.0,
        0.18 if len(titles) >= 4 else 0.0,
        0.18 if len(bodies) >= 4 else 0.0,
        0.10 if len(connectors) >= 3 else 0.0,
        0.16 if _horizontal_spread_ok(cards, bbox, minimum=4) else 0.0,
    ]
    return sum(parts)


def _bottom_mini_surface_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    elements = ir.get("elements", [])
    owned = [
        e for e in elements
        if str(e.get("id", "")).startswith(("bottom_mini_", "bottom_check", "bottom_nuisance_"))
        and e.get("bbox")
    ]
    in_slot = (
        bbox is not None
        and owned
        and sum(1 for e in owned if _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.35)
        >= max(1, int(len(owned) * 0.82))
    )
    surface = any(
        str(e.get("id", "")) == "bottom_mini_surface_surface"
        and e.get("type") == "surface"
        and (bbox is None or _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.55)
        for e in elements
    )
    axes = [e for e in elements if str(e.get("id", "")).startswith("bottom_mini_axis_")]
    vectors = [e for e in elements if str(e.get("id", "")).startswith("bottom_mini_vec_")]
    panel = any(str(e.get("id", "")) == "bottom_checklist_panel" for e in elements)
    checks = [
        e for e in elements
        if str(e.get("id", "")).startswith("bottom_check_")
        and e.get("type") == "icon"
    ]
    texts = [e for e in elements if str(e.get("id", "")).startswith("bottom_check_text_")]
    label = any(
        str(e.get("id", "")) in {"bottom_nuisance_label", "auditor_cheap_nuisance_label"}
        for e in elements
    )
    parts = [
        0.22 if surface else 0.0,
        0.12 if len(axes) >= 2 else 0.0,
        0.14 if len(vectors) >= 2 else 0.0,
        0.16 if panel else 0.0,
        0.16 if len(checks) >= 3 else 0.0,
        0.14 if len(texts) >= 3 else 0.0,
        0.06 if label else 0.0,
    ]
    score = sum(parts)
    if bbox is not None:
        score = min(score, 0.84) if not in_slot else score
    return score


def _vertical_stack_ok(elements: list[dict], bbox: list[float], minimum: int) -> bool:
    centers = sorted(
        (float(e["bbox"][1]) + float(e["bbox"][3])) / 2
        for e in elements
        if e.get("bbox")
    )
    if len(centers) < minimum:
        return False
    span = centers[-1] - centers[0]
    bh = float(bbox[3]) - float(bbox[1])
    return span >= bh * 0.22


def _horizontal_spread_ok(elements: list[dict], bbox: list[float], minimum: int) -> bool:
    centers = sorted(
        (float(e["bbox"][0]) + float(e["bbox"][2])) / 2
        for e in elements
        if e.get("bbox")
    )
    if len(centers) < minimum:
        return False
    span = centers[-1] - centers[0]
    bw = float(bbox[2]) - float(bbox[0])
    return span >= bw * 0.55


def _chart_panel_spread_ok(elements: list[dict], bbox: list[float]) -> bool:
    q0 = [
        e for e in elements
        if str(e.get("id", "")).startswith("chart_q0_") and e.get("bbox")
    ]
    if len(q0) < 10:
        return False
    x0 = min(float(e["bbox"][0]) for e in q0)
    x1 = max(float(e["bbox"][2]) for e in q0)
    bw = float(bbox[2]) - float(bbox[0])
    return (x1 - x0) >= bw * 0.65


def _auditor_cards_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    elements = ir.get("elements", [])
    cards = [e for e in elements if str(e.get("id", "")).startswith("auditor_card_")]
    nums = [e for e in elements if str(e.get("id", "")).startswith("auditor_num_")]
    titles = [e for e in elements if str(e.get("id", "")).startswith("auditor_title_")]
    formulas = [e for e in elements if str(e.get("id", "")).startswith("auditor_formula_")]
    visuals = [
        e for e in elements
        if str(e.get("id", "")).startswith("auditor_visual_")
        and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.01
    ]
    bridges = [e for e in elements if str(e.get("id", "")).startswith("auditor_bridge_")]
    group_bracket = [
        e for e in elements
        if str(e.get("id", "")).startswith("auditor_cheap_nuisance_bracket_")
    ]
    dot_or_surface = sum(1 for e in visuals if e.get("type") in {"dotcloud", "surface"})
    line_or_arrow = sum(1 for e in visuals if e.get("type") in {"line", "arrow"})
    has_paths = any((e.get("paths") or []) for e in visuals)
    has_surface = any(e.get("type") == "surface" for e in visuals)
    has_segment = any("segment" in str(e.get("id", "")) for e in visuals)
    parts = [
        0.18 if len(cards) >= 5 else 0.0,
        0.12 if len(nums) >= 5 else 0.0,
        0.14 if len(titles) >= 5 else 0.0,
        0.10 if len(formulas) >= 4 else 0.0,
        0.14 if dot_or_surface >= 4 else 0.06 if dot_or_surface >= 2 else 0.0,
        0.10 if line_or_arrow >= 8 else 0.04 if line_or_arrow >= 3 else 0.0,
        0.06 if has_paths else 0.0,
        0.06 if has_surface else 0.0,
        0.05 if has_segment else 0.0,
        0.03 if len(bridges) >= 3 else 0.0,
        0.02 if len(group_bracket) >= 3 else 0.0,
    ]
    return sum(parts)


def _procedural_surface_structure_score(ir: dict, task: dict) -> float:
    bbox = _task_bbox(task)
    if bbox is None:
        return 0.0
    candidates = []
    for el in ir.get("elements", []):
        if el.get("type") != "surface" or not el.get("bbox"):
            continue
        if _bbox_overlap_fraction(el.get("bbox"), bbox) < 0.20:
            continue
        ext = el.get("ext") or {}
        curves = (el.get("wave_bands") or {}).get("curves") or []
        streamlines = el.get("streamlines") or []
        dots = el.get("dots") or []
        heat = el.get("heat_regions") or []
        blue = sum(1 for d in dots if _is_blueish(d.get("color")))
        red = sum(1 for d in dots if _is_reddish(d.get("color")))
        x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
        bw, bh = x1 - x0, y1 - y0
        canvas = ir.get("canvas") or {}
        width = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 1)
        elements = ir.get("elements", [])
        axes = [
            e for e in elements
            if str(e.get("id", "")).startswith("proc_axis_")
            and e.get("bbox")
            and _bbox_overlap_fraction(e.get("bbox"), [x0 - 80.0, y0 - 80.0, x1, y1]) > 0.0
        ]
        vectors = [e for e in elements if str(e.get("id", "")).startswith("proc_vec_")]
        risk = [e for e in elements if str(e.get("id", "")).startswith("proc_risk_")]
        ci = [e for e in elements if str(e.get("id", "")).startswith("proc_ci_")]
        formula = [
            e for e in elements
            if str(e.get("id", "")).startswith("proc_formula_")
            or (
                e.get("type") in {"formula", "text"}
                and e.get("bbox")
                and _bbox_overlap_fraction(e.get("bbox"), [x0 + bw * 0.35, y0 - 90.0, x1, y0 + bh * 0.18]) > 0.05
                and ("β" in str(e.get("text") or e.get("latex") or e.get("ext", {}).get("latex") or "")
                     or "gamma" in str(e.get("text") or e.get("latex") or "").lower()
                     or "γ" in str(e.get("text") or e.get("latex") or e.get("ext", {}).get("latex") or ""))
            )
        ]
        parts = [
            0.12 if ext.get("procedural_surface") else 0.0,
            0.14 if len(curves) >= 7 else 0.08 if len(curves) >= 4 else 0.0,
            0.11 if len(streamlines) >= 9 else 0.06 if len(streamlines) >= 5 else 0.0,
            0.11 if blue >= 90 else 0.06 if blue >= 45 else 0.0,
            0.08 if red >= 18 else 0.04 if red >= 8 else 0.0,
            0.06 if len(heat) >= 2 else 0.0,
            0.07 if x0 <= width * 0.04 and x1 <= width * 0.52 else 0.02,
            0.05 if bw >= width * 0.36 and bh >= 330 else 0.0,
            0.08 if len(axes) >= 3 else 0.03 if len(axes) >= 2 else 0.0,
            0.08 if len(vectors) >= 4 else 0.03 if len(vectors) >= 2 else 0.0,
            0.07 if len(risk) >= 3 else 0.02 if len(risk) >= 1 else 0.0,
            0.06 if len(ci) >= 7 else 0.03 if len(ci) >= 3 else 0.0,
            0.07 if len(formula) >= 1 else 0.0,
        ]
        candidates.append(sum(parts))
    return max(candidates or [0.0])


def _is_blueish(color: Any) -> bool:
    rgb = _hex_rgb(color)
    return bool(rgb and rgb[2] >= rgb[0] and rgb[1] >= rgb[0])


def _is_reddish(color: Any) -> bool:
    rgb = _hex_rgb(color)
    return bool(rgb and rgb[0] > rgb[1] * 1.25 and rgb[0] > rgb[2] * 1.25)


def _hex_rgb(color: Any) -> tuple[int, int, int] | None:
    if not color:
        return None
    s = str(color).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _task_bbox(task: dict) -> list[float] | None:
    bbox = task.get("bbox") or (task.get("primary_defect") or {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox_overlap_fraction(a: list | tuple | None, b: list | tuple | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / area


def _bbox_center_inside(a: list | tuple | None, b: list | tuple | None) -> bool:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return False
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    cx = (ax0 + ax1) / 2.0
    cy = (ay0 + ay1) / 2.0
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _region_visual_delta(original: Image.Image, rendered_png: str | Path, bbox: list[float]) -> float:
    rendered = Image.open(rendered_png).convert("RGB")
    orig = original.convert("RGB")
    if rendered.size != orig.size:
        rendered = rendered.resize(orig.size)
    x0, y0, x1, y1 = _clamped_bbox(bbox, orig.size)
    if x1 <= x0 or y1 <= y0:
        return 1.0
    lhs = orig.crop((x0, y0, x1, y1))
    rhs = rendered.crop((x0, y0, x1, y1))
    diff = ImageChops.difference(lhs, rhs)
    stat = ImageStat.Stat(diff)
    return float(sum(stat.mean) / (len(stat.mean) * 255.0))


def _region_visual_evidence(
    original: Image.Image,
    rendered_png: str | Path,
    bbox: list[float],
) -> dict[str, float]:
    """Return crop-level evidence for planner acceptance.

    RGB residual alone rewards noisy residual traces and punishes clean native
    reconstructions unevenly.  The planner therefore looks at three signals:
    color difference, edge/geometry difference, and foreground ink density.
    """
    rendered = Image.open(rendered_png).convert("RGB")
    orig = original.convert("RGB")
    if rendered.size != orig.size:
        rendered = rendered.resize(orig.size)
    x0, y0, x1, y1 = _clamped_bbox(bbox, orig.size)
    if x1 <= x0 or y1 <= y0:
        return {
            "color_delta": 1.0,
            "edge_delta": 1.0,
            "ink_delta": 1.0,
            "original_ink": 0.0,
            "rendered_ink": 0.0,
            "composite_delta": 1.0,
        }

    lhs = orig.crop((x0, y0, x1, y1))
    rhs = rendered.crop((x0, y0, x1, y1))
    color_delta = _mean_abs_delta(lhs, rhs)

    lhs_edges = ImageOps.grayscale(lhs).filter(ImageFilter.FIND_EDGES)
    rhs_edges = ImageOps.grayscale(rhs).filter(ImageFilter.FIND_EDGES)
    edge_delta = _mean_abs_delta(lhs_edges.convert("RGB"), rhs_edges.convert("RGB"))

    original_ink = _foreground_ink_fraction(lhs)
    rendered_ink = _foreground_ink_fraction(rhs)
    ink_delta = abs(original_ink - rendered_ink)
    composite = 0.55 * color_delta + 0.30 * edge_delta + 0.15 * ink_delta
    return {
        "color_delta": round(float(color_delta), 6),
        "edge_delta": round(float(edge_delta), 6),
        "ink_delta": round(float(ink_delta), 6),
        "original_ink": round(float(original_ink), 6),
        "rendered_ink": round(float(rendered_ink), 6),
        "composite_delta": round(float(composite), 6),
    }


def _region_evidence_gain(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> float | None:
    if not before or not after:
        return None
    return float(before.get("composite_delta", 1.0)) - float(after.get("composite_delta", 1.0))


def _mean_abs_delta(lhs: Image.Image, rhs: Image.Image) -> float:
    diff = ImageChops.difference(lhs, rhs)
    stat = ImageStat.Stat(diff)
    return float(sum(stat.mean) / (len(stat.mean) * 255.0))


def _foreground_ink_fraction(img: Image.Image) -> float:
    gray = ImageOps.grayscale(img)
    hist = gray.histogram()
    total = max(1, sum(hist))
    # Count non-white-ish pixels.  The threshold intentionally keeps faint
    # contours and small gray labels because those are method-critical here.
    ink = sum(hist[:246])
    return float(ink) / float(total)


def _write_region_triptych(
    original: Image.Image,
    rendered_png: str | Path,
    bbox: list[float],
    out_path: str | Path,
    label: str,
) -> None:
    rendered = Image.open(rendered_png).convert("RGB")
    orig = original.convert("RGB")
    if rendered.size != orig.size:
        rendered = rendered.resize(orig.size)
    x0, y0, x1, y1 = _clamped_bbox(bbox, orig.size)
    if x1 <= x0 or y1 <= y0:
        return
    lhs = orig.crop((x0, y0, x1, y1))
    rhs = rendered.crop((x0, y0, x1, y1))
    diff = ImageOps.autocontrast(ImageChops.difference(lhs, rhs))
    label_h = 34
    w, h = lhs.size
    out = Image.new("RGB", (w * 3, h + label_h), "white")
    for idx, img in enumerate((lhs, rhs, diff)):
        out.paste(img, (idx * w, label_h))
    draw = ImageDraw.Draw(out)
    for idx, text in enumerate(("original", label[:28], "diff")):
        draw.text((idx * w + 10, 10), text, fill=(0, 0, 0))
    out.save(out_path)


def _clamped_bbox(bbox: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = bbox
    pad_x = max(4.0, (x1 - x0) * 0.02)
    pad_y = max(4.0, (y1 - y0) * 0.02)
    return (
        max(0, min(width, int(x0 - pad_x))),
        max(0, min(height, int(y0 - pad_y))),
        max(0, min(width, int(x1 + pad_x))),
        max(0, min(height, int(y1 + pad_y))),
    )
