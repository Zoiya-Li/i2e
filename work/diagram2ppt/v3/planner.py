"""Central Planner / Orchestrator Agent for diagram2ppt v3.

The Planner owns the Global Native IR Blackboard, schedules specialist agents,
renders through real PowerPoint, runs the Verifier, and accepts or rolls back
patches until the reconstruction passes or a stopping condition is met.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from work.diagram2ppt.v2 import decompose as v2_decompose
from work.diagram2ppt.v2 import postprocess as v2_postprocess
from work.diagram2ppt.v2 import render as v2_render
from work.diagram2ppt.v2.handlers import process_all as v2_process_all
from work.diagram2ppt.v2.vlm import VLMClient
from . import (
    builder,
    content_orchestrator,
    component_cleanup,
    ir as IR,
    migrate,
    quality_gate,
    proposal_orchestrator,
    perception_orchestrator,
    renderer,
    residual_completion,
    semantic,
    strategy,
    task_graph,
    verifier,
    visual_review,
)
from .agents.base import Agent


DEFAULT_MAX_ROUNDS = 15


class Planner:
    """Top-level orchestrator for Image → native PPTX reconstruction."""

    def __init__(
        self,
        image_path: str,
        out_dir: str = "work/diagram2ppt/v3_out",
        vlm: Any | None = None,
        log: Callable[[str], None] = print,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self.image_path = Path(image_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vlm = vlm or VLMClient()
        self.log = log
        self.max_rounds = max_rounds

        self.original = Image.open(image_path).convert("RGB")
        self.ir: dict | None = None
        self._snapshots: list[tuple[int, dict]] = []
        self._patch_preimages: dict[str, dict] = {}
        self.agents: dict[str, Agent] = {}
        self._agent_attempts: dict[str, int] = {}
        self._failed_routes: dict[str, int] = {}
        self._visual_review_cache: dict[str, dict] = {}
        self.strategy_plan: dict | None = None

    def register_agent(self, agent: Agent) -> None:
        self.agents[agent.name] = agent

    def plan(self) -> dict:
        """Initialize the IR from the input source."""
        self.log(f"[Planner] planning reconstruction for {self.image_path}")
        w, h = self.original.size

        # Stage 1: planner-controlled perception blackboard.  CV/OCR/VLM
        # perception agents submit evidence independently; the planner fuses
        # that evidence into candidate entities.  The old v2 decompose path is
        # now only a legacy recovery path.
        decompose_recovered = False
        try:
            if os.environ.get("I2E_LEGACY_DECOMPOSE", "0") == "1":
                entities = v2_decompose.decompose(str(self.image_path), self.vlm, log=self.log)
            else:
                entities, perception_report = perception_orchestrator.run(
                    str(self.image_path),
                    self.original,
                    self.vlm,
                    self.out_dir,
                    log=self.log,
                )
                self.log("[Planner] perception blackboard: "
                         f"evidence={perception_report['summary']['evidence']} "
                         f"entities={perception_report['summary']['entities']} "
                         f"sources={perception_report['summary']['sources']}")
                if not entities:
                    raise RuntimeError("perception blackboard produced no entities")
        except Exception as exc:
            cached = self._best_cached_decompose()
            if cached is None:
                self.log("[Planner] perception blackboard failed; falling back "
                         f"to legacy decompose ({type(exc).__name__}: {exc})")
                try:
                    entities = v2_decompose.decompose(str(self.image_path), self.vlm, log=self.log)
                except Exception:
                    raise
            else:
                decompose_recovered = True
                self.log("[Planner] perception failed; recovered from "
                         f"same-source memory ({type(exc).__name__}: {exc})")
                entities = copy.deepcopy(cached)
        entities = self._merge_structure_memory(entities)
        self._normalize_entities(entities)
        self.strategy_plan = strategy.plan_from_entities(entities, w, h)
        (self.out_dir / "strategy_plan.json").write_text(
            json.dumps(self.strategy_plan, indent=2, ensure_ascii=False))
        self.log("[Planner] strategy: "
                 f"regions={len(self.strategy_plan.get('regions', []))} "
                 f"policy={self.strategy_plan.get('candidate_policy', {})}")
        (self.out_dir / "decompose.json").write_text(
            json.dumps(entities, indent=2, ensure_ascii=False))

        # Stage 2: planner-visible content task orchestration.  Concrete
        # extractors can still reuse v2 handlers, but scheduling/failures are
        # now task-level blackboard events instead of a hidden hard flow.
        processed_from_memory = False
        if (
            decompose_recovered
            and os.environ.get("I2E_PROCESS_AFTER_DECOMPOSE_FALLBACK", "0") != "1"
        ):
            cached = self._best_cached_processed()
            if cached is not None:
                self.log("[Planner] skipped live content handlers after "
                         "decompose fallback; recovered same-source processed memory")
                entities = copy.deepcopy(cached)
                self._normalize_entities(entities)
                processed_from_memory = True
        if not processed_from_memory:
            try:
                max_workers = int(os.environ.get("I2E_HANDLER_MAX_WORKERS", "3"))
                if os.environ.get("I2E_LEGACY_PROCESS", "0") == "1":
                    v2_process_all(entities, self.original, self.vlm,
                                   max_workers=max_workers, log=self.log)
                else:
                    entities, content_report = content_orchestrator.run(
                        entities,
                        self.original,
                        self.vlm,
                        self.out_dir,
                        max_workers=max_workers,
                        log=self.log,
                    )
                    self.log("[Planner] content tasks: "
                             f"tasks={content_report['summary']['tasks']} "
                             f"ok={content_report['summary']['ok']} "
                             f"failed={content_report['summary']['failed']}")
            except Exception as exc:
                cached = self._best_cached_processed()
                if cached is None:
                    raise
                self.log("[Planner] content provider failed; recovered from "
                         f"same-source processed memory ({type(exc).__name__}: {exc})")
                entities = copy.deepcopy(cached)
                self._normalize_entities(entities)
        self.strategy_plan = strategy.plan_from_entities(entities, w, h)
        (self.out_dir / "strategy_plan_processed.json").write_text(
            json.dumps(self.strategy_plan, indent=2, ensure_ascii=False))
        self.log("[Planner] strategy refined: "
                 f"regions={len(self.strategy_plan.get('regions', []))} "
                 f"policy={self.strategy_plan.get('candidate_policy', {})}")

        # Stage 3: select the round-0 IR by real rendered evidence, not by
        # proxy counts.  This catches cases where a "more complete" processed
        # layer is visually worse after native rendering.
        self.ir, selected_entities = self._select_initial_ir_by_render(entities, w, h)
        (self.out_dir / "processed.json").write_text(
            json.dumps(selected_entities, indent=2, ensure_ascii=False, default=_jsonable))
        self.ir["status"] = "extracting"
        self._push_snapshot()
        self._save_ir("ir_00_plan.json")
        return self.ir

    def _select_initial_ir_by_render(self, current_entities: list[dict],
                                     w: int, h: int) -> tuple[dict, list[dict]]:
        """Choose the initial blackboard by true rendered verification.

        Proxy quality scores are useful for pruning, but the objective is
        visual/native reconstruction.  Candidate processed layers must compete
        after postprocess → migrate → QualityGate → build → render → verify.
        """
        candidates = self._initial_processed_candidates(current_entities)
        if not candidates:
            ir, _ = self._ir_from_processed_entities(current_entities, w, h,
                                                     log=self.log)
            return ir, current_entities

        cand_root = self.out_dir / "candidates"
        cand_root.mkdir(parents=True, exist_ok=True)
        reports: list[dict] = []
        best: tuple[tuple, dict, list[dict]] | None = None

        expanded = strategy.candidate_specs(self.strategy_plan, candidates)

        for idx, spec in enumerate(expanded):
            name = spec["name"]
            raw_entities = spec["entities"]
            cdir = cand_root / f"{idx:02d}_{_safe_name(name)}"
            cdir.mkdir(parents=True, exist_ok=True)
            try:
                cand_ir, postprocessed = self._ir_from_processed_entities(
                    raw_entities, w, h,
                    log=lambda msg, n=name: self.log(f"[candidate:{n}] {msg}"),
                    enable_component_motifs=bool(spec.get("component_motifs")),
                    enable_procedural_surfaces=bool(spec.get("procedural_surfaces")),
                )
                pptx_path = cdir / "candidate.pptx"
                build_stats = builder.build_pptx(cand_ir, str(pptx_path))
                if renderer.is_available():
                    rendered_png = renderer.render_isolated(
                        str(pptx_path), str(cdir / "candidate.true.png"))
                else:
                    rendered_png = str(cdir / "candidate.proxy.png")
                    _ensure_proxy_image(cand_ir, self.image_path, self.original)
                    v2_render.render(cand_ir, self.original).save(rendered_png)
                result = verifier.verify(cand_ir, str(self.image_path), rendered_png)
                strategy.apply_defect_strategy(cand_ir)
                result["defects"] = cand_ir.get("defects", [])
                metrics = result["metrics"]
                rep_score = _initial_representation_score(cand_ir, self.strategy_plan)
                rank = _candidate_rank(metrics, rep_score)
                report = {
                    "name": name,
                    "rank": list(rank),
                    "representation_score": rep_score,
                    "metrics": metrics,
                    "build": build_stats,
                    "raw_score": round(_processed_quality_score(raw_entities), 4),
                }
                (cdir / "ir_verified.json").write_text(
                    json.dumps(cand_ir, indent=2, ensure_ascii=False, default=_jsonable))
                reports.append(report)
                self.log("[Planner] candidate "
                         f"{name}: visual_delta={metrics.get('visual_delta')} "
                         f"critical={metrics.get('critical_defect_count')} "
                         f"text={metrics.get('text_accuracy')}")
                if best is None or rank < best[0]:
                    best = (rank, cand_ir, copy.deepcopy(raw_entities))
                if (strategy.residual_allowed(self.strategy_plan, spec, idx)
                        and _should_try_residual_completion(metrics)):
                    resid_name = f"{name}_residual"
                    resid_dir = cand_root / f"{idx:02d}_{_safe_name(resid_name)}"
                    resid_dir.mkdir(parents=True, exist_ok=True)
                    resid_ir = copy.deepcopy(cand_ir)
                    added = residual_completion.add_residual_freeforms(
                        resid_ir,
                        self.original,
                        rendered_png,
                        defects=result.get("defects", []),
                        replace_unreliable=True,
                        log=lambda msg, n=resid_name: self.log(f"[candidate:{n}] {msg}"),
                    )
                    if added:
                        resid_pptx = resid_dir / "candidate.pptx"
                        resid_build = builder.build_pptx(resid_ir, str(resid_pptx))
                        if renderer.is_available():
                            resid_png = renderer.render_isolated(
                                str(resid_pptx), str(resid_dir / "candidate.true.png"))
                        else:
                            resid_png = str(resid_dir / "candidate.proxy.png")
                            _ensure_proxy_image(resid_ir, self.image_path, self.original)
                            v2_render.render(resid_ir, self.original).save(resid_png)
                        resid_result = verifier.verify(
                            resid_ir, str(self.image_path), resid_png)
                        strategy.apply_defect_strategy(resid_ir)
                        resid_result["defects"] = resid_ir.get("defects", [])
                        resid_metrics = resid_result["metrics"]
                        resid_rep_score = _initial_representation_score(resid_ir, self.strategy_plan)
                        resid_rank = _candidate_rank(resid_metrics, resid_rep_score)
                        reports.append({
                            "name": resid_name,
                            "rank": list(resid_rank),
                            "representation_score": resid_rep_score,
                            "metrics": resid_metrics,
                            "build": resid_build,
                            "raw_score": round(_processed_quality_score(raw_entities), 4),
                            "residual_freeforms": added,
                        })
                        (resid_dir / "ir_verified.json").write_text(
                            json.dumps(resid_ir, indent=2, ensure_ascii=False,
                                       default=_jsonable))
                        self.log("[Planner] candidate "
                                 f"{resid_name}: visual_delta="
                                 f"{resid_metrics.get('visual_delta')} "
                                 f"critical={resid_metrics.get('critical_defect_count')} "
                                 f"text={resid_metrics.get('text_accuracy')} "
                                 f"freeforms={added}")
                        if best is None or resid_rank < best[0]:
                            best = (resid_rank, resid_ir, copy.deepcopy(raw_entities))
            except Exception as exc:
                reports.append({
                    "name": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_score": round(_processed_quality_score(raw_entities), 4),
                })
                self.log(f"[Planner] candidate {name} failed: {type(exc).__name__}: {exc}")

        (cand_root / "report.json").write_text(
            json.dumps(reports, indent=2, ensure_ascii=False, default=_jsonable))

        if best is None:
            self.log("[Planner] all rendered candidates failed; falling back to proxy memory")
            selected = self._select_best_processed_entities(current_entities)
            ir, _ = self._ir_from_processed_entities(selected, w, h, log=self.log)
            return ir, selected

        selected_report = min(
            (r for r in reports if "rank" in r),
            key=lambda r: tuple(r["rank"]),
        )
        self.log("[Planner] selected initial candidate "
                 f"{selected_report['name']} by real render/verify")
        return best[1], best[2]

    def _ir_from_processed_entities(self, entities: list[dict], w: int, h: int,
                                    log: Callable[[str], None],
                                    enable_component_motifs: bool = False,
                                    enable_procedural_surfaces: bool = False) -> tuple[dict, list[dict]]:
        v2_ir = {"image": {"width": w, "height": h},
                 "elements": copy.deepcopy(entities)}
        v2_postprocess.dedup_overlapping(v2_ir, log=log)
        v2_postprocess.drop_junk_text(v2_ir, log=log)
        v2_postprocess.drop_hollow_group_shapes(v2_ir, log=log)
        v2_postprocess.repair_icons_by_context(v2_ir, log=log)
        v2_postprocess.separate_overlapping_panels(v2_ir, log=log)
        out_ir = migrate.from_v2_entities(
            v2_ir["elements"], str(self.image_path), w, h, round=0)
        strategy.apply_ir_strategy(out_ir, self.strategy_plan)
        quality_gate.apply(out_ir, self.original, log=log,
                           enable_component_motifs=enable_component_motifs,
                           enable_procedural_surfaces=enable_procedural_surfaces)
        strategy.apply_ir_strategy(out_ir, self.strategy_plan)
        self._apply_planned_initializers(
            out_ir,
            enable_component_motifs=enable_component_motifs,
            enable_procedural_surfaces=enable_procedural_surfaces,
            log=log,
        )
        strategy.apply_ir_strategy(out_ir, self.strategy_plan)
        quality_gate.apply(out_ir, self.original, log=log,
                           enable_component_motifs=False,
                           enable_procedural_surfaces=False)
        return out_ir, v2_ir["elements"]

    def _apply_planned_initializers(
        self,
        out_ir: dict,
        *,
        enable_component_motifs: bool,
        enable_procedural_surfaces: bool,
        log: Callable[[str], None],
    ) -> None:
        """Run owner agents during candidate construction.

        Initial selection must compare complete native-method candidates, not a
        weak base IR plus later repair hopes.  This turns the strategy plan into
        actual candidate blackboards before render/verify scoring.
        """
        if not (enable_component_motifs or enable_procedural_surfaces):
            return
        for region in (self.strategy_plan or {}).get("regions", []):
            method = str(region.get("primary_method") or "")
            rep = region.get("representation") or {}
            owner = str(rep.get("owner_agent") or region.get("preferred_agent") or "")
            if method == "procedural_surface":
                if not enable_procedural_surfaces:
                    continue
            elif method in {
                "chart_parser",
                "pipeline_context_layout",
                "auditor_card_layout",
                "component_layout",
                "failure_summary_layout",
                "mini_surface_checklist",
                "cross_panel_bridge",
            }:
                if not enable_component_motifs:
                    continue
            else:
                continue
            agent = self.agents.get(owner)
            if agent is None:
                continue
            task = {
                "id": f"initial_{region.get('id') or method}",
                "kind": region.get("kind"),
                "region_id": region.get("id"),
                "bbox": region.get("bbox"),
                "locked_method": method,
                "representation": rep,
                "expected_native_expression": rep.get("native_expression"),
                "objective": region.get("reason") or f"initialize {method}",
            }
            try:
                changed = agent.run(out_ir, self.original, task=task)
            except Exception as exc:
                log(f"[Initializers] {owner} failed for {region.get('id')}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            if changed:
                log(f"[Initializers] {owner} initialized {region.get('id')} "
                    f"changed={len(changed)}")

    def _initial_processed_candidates(self, current_entities: list[dict],
                                      max_candidates: int = 4) -> list[tuple[str, list[dict]]]:
        raw: list[tuple[str, list[dict]]] = [("current", copy.deepcopy(current_entities))]
        cached = [
            (f"cache_{i:02d}", copy.deepcopy(c))
            for i, c in enumerate(self._cached_processed_candidates())
        ]
        cached.sort(key=lambda item: _processed_quality_score(item[1]), reverse=True)
        raw.extend(cached)

        out: list[tuple[str, list[dict]]] = []
        seen: set[str] = set()
        for name, ents in raw:
            sig = hashlib.sha256(json.dumps(
                ents, sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8")).hexdigest()
            if sig in seen:
                continue
            seen.add(sig)
            out.append((name, ents))
            if len(out) >= max_candidates:
                break
        return out

    def _merge_structure_memory(self, entities: list[dict]) -> list[dict]:
        """Merge prior high-structure detections for the same input image.

        The structure VLM is non-deterministic: on the same framework image it
        sometimes returns 3 structural boxes and sometimes 70+.  The Planner is
        supposed to be a blackboard orchestrator, so it should not throw away a
        better prior decomposition for the same source.  We only import
        non-text structure boxes; current OCR/formula results remain current.
        """
        current_struct = [e for e in entities if _is_structure_type(e)]
        best = self._best_cached_structure()
        if not best:
            return entities
        cached_struct = [e for e in best if _is_structure_type(e)]
        if len(cached_struct) <= len(current_struct) + 8:
            return entities

        merged = list(entities)
        added = 0
        for cand in cached_struct:
            if any(cand.get("type") == e.get("type") and
                   _iou(cand.get("bbox", []), e.get("bbox", [])) > 0.55
                   for e in merged if "bbox" in e):
                continue
            c = copy.deepcopy(cand)
            c.pop("z", None)
            if not c.get("id"):
                c["id"] = _stable_entity_id(c, len(merged))
            c["content"] = None
            merged.append(c)
            added += 1
        if added:
            self.log(f"[Planner] structure memory: +{added} cached non-text boxes "
                     f"({len(current_struct)} → {len(current_struct) + added})")
        return merged

    def _best_cached_decompose(self) -> list[dict] | None:
        """Find the richest prior decompose.json for this same image hash."""
        root = self.out_dir.parent
        source_hash = _file_hash(self.image_path)
        best: tuple[int, list[dict]] | None = None
        for p in root.glob("v3_out*/decompose.json"):
            if p.parent.resolve() == self.out_dir.resolve():
                continue
            try:
                plan = json.loads((p.parent / "ir_00_plan.json").read_text())
                src = plan.get("source", {}).get("path") or ""
                src_path = Path(src)
                if not src_path.is_absolute():
                    src_path = Path.cwd() / src_path
                if not src_path.exists() or _file_hash(src_path) != source_hash:
                    continue
                data = json.loads(p.read_text())
            except Exception:
                continue
            n = sum(1 for e in data if _is_structure_type(e))
            if best is None or n > best[0]:
                best = (n, data)
        return best[1] if best else None

    def _best_cached_structure(self) -> list[dict] | None:
        return self._best_cached_decompose()

    def _select_best_processed_entities(self, entities: list[dict]) -> list[dict]:
        """Use the best same-source processed layer when current OCR is worse.

        The VLM/OCR source stage is non-deterministic.  On framework.png the
        same image alternates between a usable 96-text layer and a noisy
        105-text layer that tanks text accuracy.  The planner should treat
        prior same-hash processed outputs as blackboard memory, not keep
        rebuilding from a worse perception sample.
        """
        current_score = _processed_quality_score(entities)
        best_entities = entities
        best_score = current_score
        for cand in self._cached_processed_candidates():
            score = _processed_quality_score(cand)
            if score > best_score + 0.08:
                best_entities = copy.deepcopy(cand)
                best_score = score
        if best_entities is not entities:
            self._normalize_entities(best_entities)
            self.log("[Planner] processed memory: reused better same-source "
                     f"layer (score {current_score:.3f} → {best_score:.3f})")
        return best_entities

    def _cached_processed_candidates(self) -> list[list[dict]]:
        root = self.out_dir.parent
        source_hash = _file_hash(self.image_path)
        out: list[list[dict]] = []
        for p in root.glob("v3_out*/processed.json"):
            if p.parent.resolve() == self.out_dir.resolve():
                continue
            try:
                plan = json.loads((p.parent / "ir_00_plan.json").read_text())
                src = plan.get("source", {}).get("path") or ""
                src_path = Path(src)
                if not src_path.is_absolute():
                    src_path = Path.cwd() / src_path
                if not src_path.exists() or _file_hash(src_path) != source_hash:
                    continue
                out.append(json.loads(p.read_text()))
            except Exception:
                continue
        return out

    def _best_cached_processed(self) -> list[dict] | None:
        best: tuple[float, list[dict]] | None = None
        for cand in self._cached_processed_candidates():
            score = _processed_quality_score(cand)
            if best is None or score > best[0]:
                best = (score, cand)
        return copy.deepcopy(best[1]) if best else None

    @staticmethod
    def _normalize_entities(entities: list[dict]) -> None:
        for idx, e in enumerate(entities):
            if not e.get("id"):
                e["id"] = _stable_entity_id(e, idx)

    def run_round(self, agent_name: str, **kwargs: Any) -> dict:
        """Execute one specialist agent and record the patch."""
        if self.ir is None:
            raise RuntimeError("call plan() before run_round()")
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ValueError(f"unknown agent: {agent_name}")

        ir_before = IR.snapshot(self.ir)
        metrics_before = copy.deepcopy(self.ir.get("metrics") or IR.metrics(self.ir))
        round_num = self.ir["round"] + 1
        self.ir["round"] = round_num
        self.ir["status"] = "extracting"

        self.log(f"[Planner] round {round_num}: running {agent_name}")
        changed = agent.run(self.ir, self.original, **kwargs)

        patch_id = f"patch_{round_num:03d}_{uuid.uuid4().hex[:6]}"
        metrics_after = copy.deepcopy(self.ir.get("metrics") or IR.metrics(self.ir))
        IR.record_patch(
            self.ir,
            patch_id=patch_id,
            agent=agent_name,
            changed=changed if isinstance(changed, list) else [],
            expected_fixes=kwargs.get("expected_fixes", []),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            decision="pending",
            reason="",
        )
        self._patch_preimages[patch_id] = ir_before
        self._push_snapshot()
        self._save_ir(f"ir_{round_num:02d}_{agent_name}.json")
        return self.ir["patches"][-1]

    def render_and_verify(self) -> dict:
        """Build PPTX, render through real PowerPoint, run Verifier."""
        if self.ir is None:
            raise RuntimeError("call plan() before render_and_verify()")

        self.ir["status"] = "building"
        pptx_path = self.out_dir / "diagram_v3.pptx"

        try:
            build_stats = builder.build_pptx(self.ir, str(pptx_path))
            self.log(f"[Planner] built {pptx_path}: {build_stats}")
        except builder.BuildBlockedError as exc:
            self.ir["status"] = "failed"
            self.ir["defects"].extend({
                "id": f"defect_build_{i}",
                "type": "build_blocked",
                "element_id": r.get("element_id", ""),
                "bbox": [0, 0, self.original.width, self.original.height],
                "severity": 1.0,
                "reason": r["reason"],
                "suggested_agent": _suggest_agent_for_blocker(r),
            } for i, r in enumerate(exc.reasons))
            self.ir["metrics"] = IR.metrics(self.ir)
            self._save_ir("ir_failed_build.json")
            return {
                "built": False,
                "blockers": exc.reasons,
                "metrics": self.ir["metrics"],
            }

        self.ir["status"] = "verifying"

        if not renderer.is_available():
            # Fallback to PIL proxy render for environments without PowerPoint.
            self.log("[Planner] true PowerPoint renderer unavailable; using PIL proxy")
            proxy_png = self.out_dir / "diagram_v3.proxy.png"
            _ensure_proxy_image(self.ir, self.image_path, self.original)
            v2_render.render(self.ir, self.original).save(proxy_png)
            rendered_png = str(proxy_png)
            compare_png = str(self.out_dir / "diagram_v3.compare.png")
            _write_proxy_compare(self.original, proxy_png, compare_png)
        else:
            try:
                rendered_png = renderer.render_isolated(
                    str(pptx_path), str(self.out_dir / "diagram_v3.true.png"))
                # Also produce the comparison image for human inspection.
                renderer.compare_isolated(
                    str(pptx_path),
                    str(self.image_path),
                    str(self.out_dir / "diagram_v3.compare.png"),
                )
                compare_png = str(self.out_dir / "diagram_v3.compare.png")
            except Exception as exc:
                self.log("[Planner] true PowerPoint render failed; using PIL proxy: "
                         f"{type(exc).__name__}: {exc}")
                proxy_png = self.out_dir / "diagram_v3.proxy.png"
                _ensure_proxy_image(self.ir, self.image_path, self.original)
                v2_render.render(self.ir, self.original).save(proxy_png)
                rendered_png = str(proxy_png)
                compare_png = str(self.out_dir / "diagram_v3.compare.png")
                _write_proxy_compare(self.original, proxy_png, compare_png)

        result = verifier.verify(self.ir, str(self.image_path), rendered_png)
        strategy.apply_defect_strategy(self.ir)
        result["defects"] = self.ir.get("defects", [])
        result["metrics"] = self.ir.get("metrics", result.get("metrics", {}))
        if compare_png:
            self._run_visual_review(compare_png)
        self.log(f"[Planner] verification: {result['metrics']}")
        self._write_diagnostics()
        self._save_ir(f"ir_{self.ir['round']:02d}_verified.json")
        return result

    def _run_visual_review(self, compare_png: str) -> None:
        """Ask a visual critic what is visibly wrong in the compare image."""
        if self.ir is None:
            return
        review_vlm = self.vlm
        if os.environ.get("I2E_SKIP_VLM_VISUAL_REVIEW") == "1":
            review_vlm = None
        try:
            compare_hash = _file_hash(Path(compare_png))
        except Exception:
            compare_hash = ""
        if compare_hash and compare_hash in self._visual_review_cache:
            review = copy.deepcopy(self._visual_review_cache[compare_hash])
            review["cache_hit"] = True
            review["source"] = compare_png
        else:
            review = visual_review.review(
                compare_png,
                vlm=review_vlm,
                canvas_width=self.original.width,
                canvas_height=self.original.height,
                strategy_plan=self.strategy_plan or self.ir.get("strategy_plan") or {},
            )
            if compare_hash:
                self._visual_review_cache[compare_hash] = copy.deepcopy(review)
        visual_review.attach_to_ir(self.ir, review)
        (self.out_dir / f"visual_review_{self.ir.get('round', 0):02d}.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False, default=_jsonable))
        (self.out_dir / "visual_review_latest.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False, default=_jsonable))
        if review.get("status") == "ok":
            self.log("[Planner] visual review: "
                     f"{len(review.get('defects', []))} visual defects; "
                     f"{review.get('summary', '')[:160]}")
        else:
            self.log("[Planner] visual review unavailable: "
                     f"{review.get('reason', 'unknown')}")

    def accept_or_rollback(self, patch_id: str | None = None,
                           force: str | None = None) -> str:
        """Decide whether to keep the latest patch based on verification metrics.

        Args:
            patch_id: patch to evaluate. Defaults to the last patch.
            force: if set, "accept" or "rollback" regardless of metrics.

        Returns:
            "accept" or "rollback".
        """
        if not self.ir or not self.ir["patches"]:
            return "accept"
        patch = self.ir["patches"][-1] if patch_id is None else next(
            (p for p in self.ir["patches"] if p["patch_id"] == patch_id), None)
        if patch is None:
            raise ValueError(f"patch not found: {patch_id}")

        before = patch["metrics_before"]
        after = patch["metrics_after"]
        decision = "accept"
        reason = "metrics improved or stable"

        # Semantic validation: a patch must not destroy the meaning of the
        # elements it touches, and content-agent patches should actually fix
        # their target defects.  This runs before pixel metrics so that false
        # positives like "formula -> 0" are rejected even if residual drops.
        ir_before = None
        if patch.get("round", 0) > 0:
            ir_before = self._find_snapshot(patch["round"] - 1)
        sem = semantic.validate_patch(self.ir, patch, ir_before)

        if force in ("accept", "rollback"):
            decision = force
            reason = f"forced {force}"
        elif not sem.get("ok", True):
            decision = "rollback"
            reason = f"semantic check failed: {sem.get('reason', '')}"
        elif after.get("native_fraction_count", 1.0) < 1.0:
            decision = "rollback"
            reason = "native fraction dropped below 1.0"
        elif _target_defects_still_present(self.ir, patch) and not _has_real_metric_gain(before, after):
            decision = "rollback"
            reason = "target defect unchanged"
        elif (after.get("visual_delta", 0) >
              before.get("visual_delta", 0) + 0.015):
            decision = "rollback"
            reason = "visual delta worsened"
        elif after.get("critical_defect_count", 0) > before.get("critical_defect_count", 0):
            cov_gain = (after.get("coverage_explained", 0) -
                        before.get("coverage_explained", 0))
            visual_gain = before.get("visual_delta", 1.0) - after.get("visual_delta", 1.0)
            text_gain = after.get("text_accuracy", 0.0) - before.get("text_accuracy", 0.0)
            layout_gain = (
                before.get("text_layout_error", 1.0)
                - after.get("text_layout_error", 1.0)
            )
            template_gain = (
                before.get("text_template_error", 1.0)
                - after.get("text_template_error", 1.0)
            )
            critical_debt = (
                after.get("critical_defect_count", 0)
                - before.get("critical_defect_count", 0)
            )
            # Large structural patches can temporarily raise residual counts
            # because new native objects are unstyled, but should buy real ink
            # coverage.  Small gains with more critical defects are noise.
            text_layout_pass = (
                patch.get("agent") == "TextLayoutAgent"
                and critical_debt <= 1
                and visual_gain >= 0.001
                and layout_gain >= 0.006
                and text_gain >= 0.0
            )
            template_slot_pass = (
                patch.get("agent") == "TemplateSlotAgent"
                and critical_debt <= 1
                and visual_gain >= 0.0005
                and template_gain >= 0.006
                and text_gain >= -0.005
            )
            if cov_gain < 0.08 and not text_layout_pass and not template_slot_pass:
                decision = "rollback"
                reason = "critical defect count increased"
        elif after.get("defect_count", 0) > before.get("defect_count", 0) * 1.5 + 2:
            decision = "rollback"
            reason = "defect count grew significantly"

        patch["decision"] = decision
        patch["reason"] = reason

        if decision == "rollback":
            # Restore the IR from before this patch.
            backup_round = max(0, int(patch.get("round", self.ir["round"])) - 1)
            backup = self._patch_preimages.get(patch["patch_id"])
            if backup is None:
                backup = self._find_snapshot(backup_round)
            if backup is not None:
                IR.restore(self.ir, backup)
                self._snapshots = [(r, s) for r, s in self._snapshots
                                   if r <= self.ir.get("round", backup_round)]
                self.log(f"[Planner] rolled back to round {backup_round}: {reason}")
                self.render_and_verify()
            else:
                self.log(f"[Planner] rollback requested but no snapshot found for round {backup_round}")
            # Track repeated failures so we don't loop forever on defects no
            # agent can actually fix.  Use a blackboard-level counter because
            # the Verifier rebuilds the defects list every round.
            rollbacks = self.ir.setdefault("defect_rollbacks", {})
            for did in patch.get("expected_fixes", []):
                rollbacks[did] = rollbacks.get(did, 0) + 1
                for d in self.ir.get("defects", []):
                    if d.get("id") == did:
                        d["status"] = "skipped"
                        route_key = _route_fingerprint(d, patch.get("agent", ""))
                        self._failed_routes[route_key] = (
                            self._failed_routes.get(route_key, 0) + 1
                        )
            defect = patch.get("defect")
            if defect:
                route_key = _route_fingerprint(defect, patch.get("agent", ""))
                self._failed_routes[route_key] = (
                    self._failed_routes.get(route_key, 0) + 1
                )
                self.log(f"[Planner] defect {defect.get('id')} rolled back; skipping this route")
        else:
            self.log(f"[Planner] accepted patch {patch['patch_id']}: {reason}")

        self._save_ir(f"ir_{self.ir['round']:02d}_{decision}.json")
        return decision

    def run(self) -> dict:
        """Run the full Planner loop.

        Current Phase 0+1 implementation:
          1. plan() — decompose and migrate to v3 IR.
          2. Build/verify baseline.
          3. Stop if passed or max rounds reached.
        """
        self.plan()
        result = self.render_and_verify()

        if result.get("passed") and not _has_visual_review_defects(self.ir):
            self.ir["status"] = "accepted"
            self.log("[Planner] reconstruction passed on first attempt")
            self._save_ir("ir_final.json")
            return self.ir
        if result.get("passed"):
            self.log("[Planner] scalar verifier passed, but visual review "
                     "still has region defects; running proposal planner")

        proposal_result = self.run_proposal_phase()
        if proposal_result.get("accepted", 0):
            cleanup = component_cleanup.apply(self.ir, log=self.log)
            if cleanup.get("removed"):
                self._save_ir("ir_00_component_cleanup.json")
            result = self.render_and_verify()
            if result.get("passed"):
                self.ir["status"] = "accepted"
                self.log("[Planner] reconstruction passed after proposal phase")
                self._save_ir("ir_final.json")
                return self.ir

        # Fallback repair loop: single-defect repair is no longer the primary
        # architecture.  It remains useful for small residual defects after the
        # region proposal planner has tried multi-agent candidates.
        for _ in range(self.max_rounds):
            if result.get("passed"):
                break

            # Pick the most severe defect and route to an agent.
            repair = self._next_repair_task()
            if not repair:
                self.log("[Planner] no actionable defects; stopping")
                break

            agent_name = repair["suggested_agent"]
            self._agent_attempts[agent_name] = self._agent_attempts.get(agent_name, 0) + 1
            if agent_name not in self.agents:
                self.log(f"[Planner] no registered agent for {agent_name}; skipping")
                # mark as handled so we don't loop forever
                for d in self.ir["defects"]:
                    if d["id"] == repair["id"]:
                        d["status"] = "skipped"
                continue

            patch = self.run_round(agent_name, defect=repair,
                                   expected_fixes=[repair.get("id", "")])
            patch["defect"] = copy.deepcopy(repair)

            # If the agent made no changes for this defect, skip it in future
            # rounds to avoid looping forever on unfixable issues.
            if not patch.get("changed"):
                self.log(f"[Planner] {agent_name} made no changes for "
                         f"{repair.get('type')} defect {repair.get('id')}; skipping")
                route_key = _route_fingerprint(repair, agent_name)
                self._failed_routes[route_key] = self._failed_routes.get(route_key, 0) + 1
                backup = self._patch_preimages.get(patch["patch_id"])
                if backup is not None:
                    IR.restore(self.ir, backup)
                    self._snapshots = [(r, s) for r, s in self._snapshots
                                       if r <= self.ir.get("round", 0)]
                for d in self.ir["defects"]:
                    if d.get("id") == repair.get("id"):
                        d["status"] = "skipped"
                continue

            result = self.render_and_verify()
            if self.ir.get("patches"):
                self.ir["patches"][-1]["metrics_after"] = copy.deepcopy(
                    self.ir.get("metrics", {})
                )
            decision = self.accept_or_rollback()
            if decision == "rollback":
                result = {"passed": False, "metrics": self.ir.get("metrics", {})}

            if self.ir["status"] == "accepted":
                break

        # Final verification so the reported metrics reflect the actual accepted
        # state (especially important after rollbacks).
        if not result.get("passed"):
            result = self.render_and_verify()

        self.ir["status"] = "accepted" if result.get("passed") else "failed"
        self._save_ir("ir_final.json")
        return self.ir

    def run_proposal_phase(self) -> dict:
        """Plan region tasks and let multiple agents compete/cooperate.

        This is the planner's main orchestration step: build a region task
        graph, collect sandboxed proposals from assigned agents, render/verify
        candidates, and commit only proposals that improve the blackboard.
        """
        if self.ir is None:
            raise RuntimeError("call plan() before run_proposal_phase()")
        graph = task_graph.build(self.ir)
        (self.out_dir / "task_graph.json").write_text(
            json.dumps(graph, indent=2, ensure_ascii=False, default=_jsonable))
        self.log("[Planner] proposal task graph: "
                 f"tasks={graph.get('summary', {}).get('tasks', 0)} "
                 f"regions={graph.get('summary', {}).get('regions', 0)}")
        if not graph.get("tasks"):
            return {"accepted": 0, "tasks": []}
        report = proposal_orchestrator.run(
            ir=self.ir,
            original=self.original,
            image_path=self.image_path,
            out_dir=self.out_dir,
            agents=self.agents,
            task_graph=graph,
            log=self.log,
            max_tasks=8,
        )
        self._save_ir("ir_00_proposals.json")
        return report

    # Element types whose high-residual defects are most likely fixable by
    # content agents (TextAgent / FormulaAgent / ChartAgent / IconAgent).
    _CONTENT_ELEMENT_TYPES = {"text", "formula", "chart", "icon"}
    _OBJECT_AGENT_PRIORITY = {
        "SurfaceAgent": 0,
        "ChartAgent": 1,
        "ConnectorAgent": 2,
        "LayoutAgent": 3,
        "TextAgent": 4,
        "FormulaAgent": 5,
        "TextLayoutAgent": 6,
        "TemplateSlotAgent": 7,
        "StyleAgent": 8,
        "VectorizeAgent": 9,
        "IconAgent": 9,
        "ShapeAgent": 9,
    }

    def _next_repair_task(self) -> dict | None:
        """Select the highest-severity actionable defect.

        Content defects (text/formula/chart/icon) get a small priority boost
        so specialist agents get a chance before shape-style defects consume
        all repair rounds.
        """
        defects = [d for d in self.ir.get("defects", [])
                   if d.get("status") != "skipped"]
        defects = [d for d in defects if not self._route_is_quarantined(d)]
        if not defects:
            return None

        # If large regions of ink are unexplained, the IR is structurally
        # incomplete.  Fixing individual text/style residuals first is wasted
        # motion: the renderer is comparing a mostly-empty native deck against
        # a dense original.  Prioritize the largest missing regions until
        # coverage recovers.
        coverage = (self.ir.get("metrics") or {}).get("coverage_explained", 1.0)
        missing = [d for d in defects if d.get("type") == "missing_element"]
        object_repairs = [
            d for d in defects
            if d.get("type") in {"high_residual", "text_layout_mismatch", "text_template_mismatch"}
            and d.get("suggested_agent") in {
                "SurfaceAgent", "ChartAgent", "IconAgent", "ConnectorAgent",
                "TextAgent", "FormulaAgent", "TextLayoutAgent", "TemplateSlotAgent", "VectorizeAgent",
            }
        ]
        # Once most ink has an owner, the bottleneck is usually object
        # expression, not more boxes.  Running ShapeAgent forever improves the
        # recall metric while making a diagram of wrong rectangles.
        missing_without_owner = [d for d in missing if not d.get("element_id")]
        if missing_without_owner and coverage < 0.75:
            missing_without_owner.sort(key=lambda d: -_bbox_area(d.get("bbox", [])))
            return missing_without_owner[0]
        component_repairs = [
            d for d in defects
            if d.get("suggested_agent") == "LayoutAgent"
            and (d.get("strategy") or {}).get("method") in {
                "component_layout", "pipeline_context_layout",
            }
        ]
        if component_repairs:
            component_repairs.sort(key=lambda d: (
                self._agent_attempts.get("LayoutAgent", 0),
                -float(d.get("severity", 0)),
            ))
            return component_repairs[0]
        text_layout_repairs = [
            d for d in defects
            if d.get("type") == "text_layout_mismatch"
            and d.get("suggested_agent") == "TextLayoutAgent"
            and "TextLayoutAgent" in self.agents
            and not _protected_text_layout_defect(self.ir, d)
        ]
        template_slot_repairs = [
            d for d in defects
            if d.get("type") == "text_template_mismatch"
            and d.get("suggested_agent") == "TemplateSlotAgent"
            and "TemplateSlotAgent" in self.agents
        ]
        if template_slot_repairs and coverage >= 0.86 and (
            (
                self._agent_attempts.get("TextLayoutAgent", 0) >= 2
                and self._agent_attempts.get("TemplateSlotAgent", 0)
                < self._agent_attempts.get("TextLayoutAgent", 0)
            )
            or not text_layout_repairs
            or (
                self._agent_attempts.get("TemplateSlotAgent", 0)
                < self._agent_attempts.get("TextLayoutAgent", 0)
                and _max_severity(template_slot_repairs) >= _max_severity(text_layout_repairs) + 0.18
            )
        ):
            template_slot_repairs.sort(key=lambda d: (
                self._agent_attempts.get("TemplateSlotAgent", 0),
                _template_role_priority(d),
                -float(d.get("severity", 0)),
            ))
            return template_slot_repairs[0]
        if text_layout_repairs and coverage >= 0.86:
            text_layout_repairs.sort(key=lambda d: (
                self._agent_attempts.get("TextLayoutAgent", 0),
                -float(d.get("severity", 0)),
            ))
            return text_layout_repairs[0]
        if template_slot_repairs and coverage >= 0.86:
            template_slot_repairs.sort(key=lambda d: (
                self._agent_attempts.get("TemplateSlotAgent", 0),
                _template_role_priority(d),
                -float(d.get("severity", 0)),
            ))
            return template_slot_repairs[0]
        severe_repairs = [
            d for d in defects
            if d.get("type") == "high_residual"
            and float(d.get("severity", 0)) >= 0.70
            and d.get("suggested_agent") in {
                "StyleAgent", "TextAgent", "FormulaAgent",
                "ChartAgent", "IconAgent", "SurfaceAgent", "ConnectorAgent",
                "VectorizeAgent", "TextLayoutAgent", "TemplateSlotAgent",
            }
            and d.get("suggested_agent") in self.agents
        ]
        if severe_repairs:
            severe_repairs.sort(key=lambda d: (
                self._agent_attempts.get(d.get("suggested_agent", ""), 0),
                -float(d.get("severity", 0)),
            ))
            return severe_repairs[0]
        if object_repairs:
            object_repairs.sort(key=lambda d: (
                self._agent_attempts.get(d.get("suggested_agent", ""), 0),
                self._OBJECT_AGENT_PRIORITY.get(d.get("suggested_agent"), 99),
                -float(d.get("severity", 0)),
            ))
            return object_repairs[0]
        if missing_without_owner and coverage < 0.97:
            missing_without_owner.sort(key=lambda d: -_bbox_area(d.get("bbox", [])))
            return missing_without_owner[0]

        def _score(d: dict) -> float:
            severity = d.get("severity", 0)
            element_id = d.get("element_id")
            el_type = ""
            if element_id:
                el = next((e for e in self.ir.get("elements", [])
                           if e.get("id") == element_id), None)
                if el:
                    el_type = el.get("type", "")
            agent = d.get("suggested_agent", "")
            boost = 0.15 if el_type in self._CONTENT_ELEMENT_TYPES else 0.0
            boost += max(0.0, 0.12 - 0.015 * self._OBJECT_AGENT_PRIORITY.get(agent, 8))
            return severity + boost

        defects.sort(key=lambda d: -_score(d))
        return defects[0]

    def _route_is_quarantined(self, defect: dict) -> bool:
        agent = defect.get("suggested_agent", "")
        if self._failed_routes.get(_route_fingerprint(defect, agent), 0) >= 1:
            return True
        # Keep weak expression agents from monopolizing the loop.  They still
        # get a chance; repeated no-gain attempts force the planner to explore
        # other bottlenecks instead of cycling over regenerated defect ids.
        if agent in {"ConnectorAgent", "IconAgent"}:
            return self._agent_attempts.get(agent, 0) >= 2
        if agent == "StyleAgent":
            return self._agent_attempts.get(agent, 0) >= 3
        return False

    def _write_diagnostics(self) -> None:
        """Persist a small planner-readable diagnosis after each verification."""
        if not self.ir:
            return
        from collections import Counter, defaultdict

        defects = self.ir.get("defects", [])
        by_agent = Counter(d.get("suggested_agent", "") for d in defects)
        by_type = Counter(d.get("type", "") for d in defects)
        examples: dict[str, list[dict]] = defaultdict(list)
        elements = {e.get("id"): e for e in self.ir.get("elements", [])}
        for d in sorted(defects, key=lambda x: -float(x.get("severity", 0))):
            agent = d.get("suggested_agent", "")
            if len(examples[agent]) >= 5:
                continue
            el = elements.get(d.get("element_id", ""))
            examples[agent].append({
                "defect": d.get("id"),
                "type": d.get("type"),
                "severity": d.get("severity"),
                "bbox": d.get("bbox"),
                "element_id": d.get("element_id"),
                "element_type": el.get("type") if el else None,
                "text": (el.get("text") or "")[:80] if el else "",
                "route_key": _route_fingerprint(d, agent),
            })
        diag = {
            "round": self.ir.get("round", 0),
            "status": self.ir.get("status"),
            "metrics": self.ir.get("metrics", {}),
            "defects_by_agent": dict(by_agent),
            "defects_by_type": dict(by_type),
            "agent_attempts": dict(self._agent_attempts),
            "failed_routes": dict(self._failed_routes),
            "strategy": {
                "regions": len((self.strategy_plan or {}).get("regions", [])),
                "candidate_policy": (self.strategy_plan or {}).get("candidate_policy", {}),
            },
            "examples": dict(examples),
        }
        (self.out_dir / "diagnostics.json").write_text(
            json.dumps(diag, indent=2, ensure_ascii=False))

    def _push_snapshot(self) -> None:
        self._snapshots.append((self.ir["round"], IR.snapshot(self.ir)))

    def _find_snapshot(self, round_num: int) -> dict | None:
        for r, snap in reversed(self._snapshots):
            if r == round_num:
                return snap
        for r, snap in reversed(self._snapshots):
            if r <= round_num:
                return snap
        return None

    def _save_ir(self, name: str) -> None:
        IR.save(self.ir, str(self.out_dir / name))


def _jsonable(o: Any) -> Any:
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _write_proxy_compare(original: Image.Image, rendered_png: str | Path,
                         compare_png: str | Path) -> None:
    """Write an original/proxy/diff comparison image for visual review."""
    from PIL import ImageChops, ImageDraw, ImageOps

    rendered = Image.open(rendered_png).convert("RGB")
    orig = original.convert("RGB")
    if rendered.size != orig.size:
        rendered = rendered.resize(orig.size)
    diff = ImageOps.autocontrast(ImageChops.difference(orig, rendered))
    label_h = 48
    w, h = orig.size
    out = Image.new("RGB", (w * 3, h + label_h), "white")
    for idx, img in enumerate((orig, rendered, diff)):
        out.paste(img, (idx * w, label_h))
    draw = ImageDraw.Draw(out)
    for idx, label in enumerate(("original", "native proxy", "difference boosted")):
        draw.text((idx * w + 24, 16), label, fill=(0, 0, 0))
    out.save(compare_png)


def _bbox_area(bbox: list | tuple) -> float:
    if len(bbox) != 4:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def _is_structure_type(e: dict) -> bool:
    return e.get("type") in {
        "container", "shape", "rect", "rounded_rect", "oval", "diamond",
        "hexagon", "parallelogram", "arrow", "line", "icon", "chart",
        "surface", "dotcloud", "freeform",
    }


def _iou(a: list | tuple, b: list | tuple) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union else 0.0


def _stable_entity_id(e: dict, idx: int) -> str:
    bbox = e.get("bbox") or [0, 0, 0, 0]
    coords = "_".join(str(int(float(v))) for v in bbox[:4])
    typ = str(e.get("type") or "entity")
    return f"{typ}_{coords}_{idx}"


def _processed_quality_score(entities: list[dict]) -> float:
    from collections import Counter

    counts = Counter(e.get("type") for e in entities)
    text_count = counts.get("text", 0)
    structure_count = sum(
        counts.get(t, 0)
        for t in ("rounded_rect", "rect", "container", "icon", "arrow",
                  "chart", "dotcloud", "surface")
    )
    noisy_text = sum(
        1 for e in entities
        if e.get("type") == "text" and _is_noisy_text(e.get("text", ""))
    )
    # Good framework runs have rich structure and about 96 text lines.  A large
    # excess of tiny OCR fragments is worse than a slightly smaller text set.
    text_penalty = abs(text_count - 96) * 0.025
    noise_penalty = noisy_text * 0.018
    completeness = _content_completeness_score(entities)
    return (
        structure_count * 0.035
        + min(text_count, 100) * 0.01
        + completeness
        - text_penalty
        - noise_penalty
    )


def _content_completeness_score(entities: list[dict]) -> float:
    """Reward usable specialist payloads and penalize half-processed shells."""
    score = 0.0
    for e in entities:
        t = e.get("type")
        if t == "chart":
            spec = e.get("chart") or e.get("content") or {}
            kind = str(spec.get("kind") or spec.get("type") or "").lower()
            series = spec.get("series") or spec.get("values") or spec.get("points")
            if kind and kind != "none" and series:
                score += 0.10
            else:
                score -= 0.22
        elif t == "icon":
            icon = e.get("icon") or e.get("content") or {}
            kind = str(icon.get("kind") or icon.get("name") or "").lower()
            if kind and kind not in {"none", "unknown"}:
                score += 0.025
            else:
                score -= 0.05
        elif t in ("surface", "dotcloud"):
            payload = any(e.get(k) for k in ("dots", "paths", "wave_bands", "streamlines"))
            score += 0.08 if payload else -0.08
        elif t == "formula":
            txt = str(e.get("text") or e.get("latex") or "")
            if any(ch in txt for ch in "=≈~<>+-*/|⟨⟩βγτθ∇_^()[]{}°"):
                score += 0.025
            elif len(txt) > 5:
                score -= 0.035
    return score


def _initial_representation_score(ir: dict, plan: dict | None) -> dict[str, Any]:
    """Score whether an initial candidate satisfies planned native methods."""
    if not plan:
        return {"average": 1.0, "minimum": 1.0, "scores": {}}
    score_fns = {
        "procedural_3d_surface": proposal_orchestrator._procedural_surface_structure_score,
        "chart": proposal_orchestrator._q0_coverage_structure_score,
        "pipeline_context_row": proposal_orchestrator._pipeline_context_structure_score,
        "auditor_method_cards": proposal_orchestrator._auditor_cards_structure_score,
        "failure_summary_panel": proposal_orchestrator._failure_summary_structure_score,
        "component_card_row": proposal_orchestrator._action_cards_structure_score,
        "bottom_mini_surface": proposal_orchestrator._bottom_mini_surface_structure_score,
        "cross_panel_bridge": proposal_orchestrator._cross_panel_bridge_structure_score,
    }
    scores: dict[str, float] = {}
    for region in plan.get("regions", []):
        kind = str(region.get("kind") or "")
        fn = score_fns.get(kind)
        if not fn:
            continue
        rid = str(region.get("id") or kind)
        task = {
            "kind": kind,
            "region_id": rid,
            "bbox": region.get("bbox"),
            "representation": region.get("representation") or {},
            "locked_method": region.get("primary_method"),
        }
        try:
            scores[rid] = round(float(fn(ir, task)), 4)
        except Exception:
            scores[rid] = 0.0
    if not scores:
        return {"average": 1.0, "minimum": 1.0, "scores": {}}
    values = list(scores.values())
    return {
        "average": round(sum(values) / len(values), 4),
        "minimum": round(min(values), 4),
        "scores": scores,
    }


def _candidate_rank(metrics: dict, representation_score: dict[str, Any] | None = None) -> tuple:
    """Sort key for initial candidate selection.

    Rendered quality is the gate; representation satisfaction is a tie-breaker.

    A blank/under-covered deck can have a deceptively low pixel delta because
    large white regions match the original background.  Conversely, a
    representation-rich candidate can be visually disastrous.  The planner
    should not choose either failure mode just because one scalar looks good.
    """
    rep = representation_score or {}
    rep_avg = float(rep.get("average", 1.0))
    rep_min = float(rep.get("minimum", 1.0))
    visual = float(metrics.get("visual_delta", 1.0))
    coverage = float(metrics.get("coverage_explained", 0.0))
    critical = int(metrics.get("critical_defect_count", 9999))
    defects = int(metrics.get("defect_count", 9999))
    coverage_bucket = 2 if coverage < 0.55 else 1 if coverage < 0.85 else 0
    visual_bucket = int(visual / 0.05)
    return (
        coverage_bucket,
        visual_bucket,
        critical,
        defects,
        round(visual, 4),
        round(1.0 - rep_min, 4),
        round(1.0 - rep_avg, 4),
        -float(metrics.get("text_accuracy", 0.0)),
        -coverage,
    )


def _should_try_residual_completion(metrics: dict) -> bool:
    """Gate expensive residual-completion candidates to genuinely bad renders."""
    return (
        float(metrics.get("visual_delta", 1.0)) > 0.18
        or float(metrics.get("coverage_explained", 0.0)) < 0.96
        or int(metrics.get("critical_defect_count", 0)) > 0
    )


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:48]


def _is_noisy_text(text: Any) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if len(s) <= 3:
        allowed = {"x1", "x2", "x3", "ci", "q0", "0°", "1.0", "0.5", "0.0"}
        return s.lower() not in allowed
    alpha = sum(ch.isalpha() for ch in s)
    if len(s) <= 5 and alpha <= 2:
        return True
    return False


def _target_defects_still_present(ir: dict, patch: dict) -> bool:
    expected = [d for d in patch.get("expected_fixes", []) if d]
    if not expected:
        return False
    current = {d.get("id") for d in ir.get("defects", [])}
    return any(d in current for d in expected)


def _protected_text_layout_defect(ir: dict, defect: dict) -> bool:
    if defect.get("type") != "text_layout_mismatch":
        return False
    element_id = defect.get("element_id")
    if not element_id:
        return False
    el = next((e for e in ir.get("elements", []) if e.get("id") == element_id), None)
    if not el:
        return False
    role = str((((el.get("ext") or {}).get("typography") or {}).get("role")) or "")
    return role in {
        "slide_title",
        "solution_title",
        "section_title",
        "subtitle",
        "caption",
        "process_title",
        "auditor_title",
        "auditor_group_label",
        "chart_title",
        "chart_title_q",
        "chart_title_sub",
        "chart_title_rest",
        "failure_title",
        "action_title",
        "action_report_title",
        "checklist_body",
        "covariate_label",
        "covariate_text",
        "covariate_math",
        "axis_math",
        "vector_label",
        "surface_vector_math",
        "surface_theta_math",
        "ci_axis_label",
        "risk_label",
        "risk_label_math",
        "risk_q_math",
    }


def _template_role_priority(defect: dict) -> int:
    role = str(defect.get("template_role") or "")
    if role in {"slide_title", "solution_title", "subtitle", "caption"}:
        return 0
    if role in {"action_title", "action_report_title", "checklist_body"}:
        return 1
    if role in {"process_title", "auditor_title", "auditor_group_label", "failure_title"}:
        return 2
    if role in {"chart_title", "chart_title_q", "chart_title_rest"}:
        return 3
    if role == "chart_title_sub":
        return 6
    if role in {"ci_axis_label", "risk_label", "risk_label_math", "risk_q_math"}:
        return 4
    if role in {"covariate_label", "covariate_text"}:
        return 5
    if role in {"covariate_math", "axis_math", "surface_vector_math", "surface_theta_math"}:
        return 7
    if role.startswith("chart_title"):
        return 3
    return 8


def _max_severity(defects: list[dict]) -> float:
    return max((float(d.get("severity", 0.0)) for d in defects), default=0.0)


def _has_visual_review_defects(ir: dict | None) -> bool:
    if not ir:
        return False
    review = ir.get("visual_review") or {}
    if review.get("status") not in {"ok", "degraded"}:
        return False
    return any(d.get("region_id") for d in review.get("defects", []))


def _route_fingerprint(defect: dict, agent: str) -> str:
    """Stable-ish key for a failed repair route across verifier rebuilds."""
    bbox = defect.get("bbox") or [0, 0, 0, 0]
    if len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        # Bucket by center and size so tiny rerender jitter does not resurrect
        # the same failed route as a new task.
        cx = int(round(((x0 + x1) / 2) / 24))
        cy = int(round(((y0 + y1) / 2) / 24))
        bw = int(round(max(0.0, x1 - x0) / 24))
        bh = int(round(max(0.0, y1 - y0) / 24))
        loc = f"{cx}:{cy}:{bw}:{bh}"
    else:
        loc = "unknown"
    return "|".join([
        str(agent or defect.get("suggested_agent") or ""),
        str(defect.get("type") or ""),
        str(defect.get("element_id") or ""),
        loc,
    ])


def _has_real_metric_gain(before: dict, after: dict) -> bool:
    visual_gain = before.get("visual_delta", 1.0) - after.get("visual_delta", 1.0)
    coverage_gain = after.get("coverage_explained", 0.0) - before.get("coverage_explained", 0.0)
    text_gain = after.get("text_accuracy", 0.0) - before.get("text_accuracy", 0.0)
    text_layout_gain = (
        before.get("text_layout_error", 1.0)
        - after.get("text_layout_error", 1.0)
    )
    text_layout_count_gain = (
        before.get("text_layout_mismatch_count", 0)
        - after.get("text_layout_mismatch_count", 0)
    )
    text_template_gain = (
        before.get("text_template_error", 1.0)
        - after.get("text_template_error", 1.0)
    )
    text_template_count_gain = (
        before.get("text_template_mismatch_count", 0)
        - after.get("text_template_mismatch_count", 0)
    )
    connector_gain = after.get("connector_accuracy", 0.0) - before.get("connector_accuracy", 0.0)
    critical_gain = (
        before.get("critical_defect_count", 0)
        - after.get("critical_defect_count", 0)
    )
    defect_gain = before.get("defect_count", 0) - after.get("defect_count", 0)
    return (
        visual_gain >= 0.003
        or (visual_gain > 0.00005 and critical_gain >= 1)
        or critical_gain >= 1
        or defect_gain >= 2
        or coverage_gain >= 0.01
        or text_gain >= 0.01
        or (text_layout_gain >= 0.0005 and visual_gain > -0.002 and text_gain > -0.01)
        or (text_layout_count_gain >= 1 and visual_gain > -0.002 and text_gain > -0.01)
        or (text_template_gain >= 0.0008 and visual_gain > -0.003 and text_gain > -0.015)
        or (text_template_count_gain >= 1 and visual_gain > -0.003 and text_gain > -0.015)
        or connector_gain >= 0.02
    )


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _suggest_agent_for_blocker(blocker: dict) -> str:
    t = blocker.get("element_type", "")
    if t == "raster_crop":
        return "VectorizeAgent"
    if t in ("arrow", "line"):
        return "ConnectorAgent"
    if t in ("text", "formula"):
        return "TextAgent"
    if t == "chart":
        return "ChartAgent"
    if t == "icon":
        return "IconAgent"
    return "LayoutAgent"


def _ensure_proxy_image(ir: dict, image_path: Path, original: Image.Image) -> None:
    ir.setdefault("image", {
        "path": str(image_path),
        "width": original.width,
        "height": original.height,
    })


def run(image_path: str, out_dir: str = "work/diagram2ppt/v3_out",
        log=print) -> dict:
    """Convenience entry point for the full Planner loop."""
    planner = Planner(image_path, out_dir, log=log)
    # In Phase 0+1 we register no repair agents; the Planner runs one-shot
    # decompose → build → verify to validate the core loop.
    return planner.run()
