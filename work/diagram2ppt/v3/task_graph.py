"""Region task graph for multi-agent diagram reconstruction.

The v3 planner should not treat reconstruction as a single queue of unrelated
defects.  It first partitions the slide into semantic regions, then assigns a
small set of specialist agents to each region.  Agents can propose changes
against the same task; the planner decides which proposal or merged candidate
is committed to the global IR.
"""
from __future__ import annotations

from typing import Any

from . import rendering_methods, representation_plan


REGION_AGENT_ROLES = {
    "procedural_3d_surface": [
        "ProceduralSurfaceAgent",
        "StyleAgent",
    ],
    "chart": [
        "ChartAgent",
        "VectorizeAgent",
        "StyleAgent",
    ],
    "component_card_row": [
        "ActionCardAgent",
        "LayoutAgent",
        "TextAgent",
        "TextLayoutAgent",
        "TemplateSlotAgent",
        "IconAgent",
        "StyleAgent",
    ],
    "pipeline_context_row": [
        "PipelineContextAgent",
        "LayoutAgent",
        "TextAgent",
        "TextLayoutAgent",
        "TemplateSlotAgent",
        "IconAgent",
        "ConnectorAgent",
        "VectorizeAgent",
    ],
    "auditor_method_cards": [
        "AuditorCardAgent",
        "LayoutAgent",
        "ChartAgent",
        "IconAgent",
        "TextAgent",
        "TextLayoutAgent",
        "TemplateSlotAgent",
        "ConnectorAgent",
        "StyleAgent",
    ],
    "failure_summary_panel": [
        "FailureSummaryAgent",
        "TextAgent",
        "TextLayoutAgent",
        "TemplateSlotAgent",
        "IconAgent",
        "StyleAgent",
    ],
    "bottom_mini_surface": [
        "BottomMiniSurfaceAgent",
        "SurfaceAgent",
        "TextAgent",
        "TextLayoutAgent",
        "TemplateSlotAgent",
        "IconAgent",
        "StyleAgent",
    ],
}


def build(ir: dict, max_ownerless_tasks: int = 6) -> dict:
    """Build a planner-readable region task DAG from the current blackboard."""
    regions = list((ir.get("strategy_plan") or {}).get("regions", []))
    defects = list(ir.get("defects") or [])
    tasks: list[dict[str, Any]] = []

    for visual_defect in _visual_defects(ir):
        agents = _dedupe([
            str(a) for a in visual_defect.get("suggested_agents", []) if a
        ])
        if not agents:
            agents = ["LayoutAgent", "StyleAgent"]
        task = {
            "id": f"task_visual_{visual_defect.get('id')}",
            "kind": "visual_region_defect",
            "bbox": _clean_bbox(visual_defect.get("bbox")),
            "region_id": visual_defect.get("region_id", ""),
            "objective": visual_defect.get("visual_problem") or "fix visible reconstruction failure",
            "expected_native_expression": visual_defect.get("expected_native_expression", ""),
            "agent_roles": agents,
            "element_ids": [],
            "defect_ids": [],
            "primary_defect": None,
            "visual_defect": visual_defect,
            "success_metrics": ["visual_review", "visual_delta", "defect_count"],
            "status": "planned",
        }
        _attach_region_representation(task, regions)
        tasks.append(rendering_methods.apply_policy_to_task(task))

    for region in regions:
        kind = str(region.get("kind") or "region")
        task_defects = _defects_for_region(region, defects)
        roles = REGION_AGENT_ROLES.get(kind, _roles_from_defects(task_defects))
        if not roles:
            continue
        task = {
            "id": f"task_{region.get('id') or len(tasks)}",
            "kind": kind,
            "bbox": _clean_bbox(region.get("bbox")),
            "region_id": region.get("id"),
            "objective": _objective_for_region(kind),
            "agent_roles": roles,
            "element_ids": [str(eid) for eid in region.get("element_ids", []) if eid],
            "defect_ids": [str(d.get("id")) for d in task_defects if d.get("id")],
            "primary_defect": _primary_defect(task_defects),
            "success_metrics": _success_metrics_for_region(kind),
            "status": "planned",
        }
        representation_plan.apply_to_task(task, region)
        tasks.append(rendering_methods.apply_policy_to_task(task))

    covered_defects = {
        did for task in tasks for did in task.get("defect_ids", [])
    }
    ownerless = [
        d for d in defects
        if d.get("id") not in covered_defects
        and d.get("status") != "skipped"
        and d.get("suggested_agent")
        and d.get("bbox")
    ]
    ownerless.sort(key=lambda d: -float(d.get("severity", 0)))
    for idx, defect in enumerate(ownerless[:max_ownerless_tasks]):
        agent = str(defect.get("suggested_agent"))
        task = {
            "id": f"task_defect_{defect.get('id') or idx}",
            "kind": "defect_cluster",
            "bbox": _clean_bbox(defect.get("bbox")),
            "region_id": "",
            "objective": "resolve severe residual or missing native element without raster fallback",
            "agent_roles": _dedupe([agent, _fallback_agent(agent, defect)]),
            "element_ids": [str(defect.get("element_id"))] if defect.get("element_id") else [],
            "defect_ids": [str(defect.get("id"))] if defect.get("id") else [],
            "primary_defect": defect,
            "success_metrics": ["visual_delta", "defect_count", "target_defect"],
            "status": "planned",
        }
        _attach_region_representation(task, regions)
        tasks.append(rendering_methods.apply_policy_to_task(task))

    tasks.sort(key=_task_sort_key)
    tasks = _dedupe_equivalent_region_tasks(tasks)
    return {
        "version": "task-graph-v2-representation-aware",
        "round": ir.get("round", 0),
        "tasks": tasks,
        "representation_plan": (ir.get("strategy_plan") or {}).get("representation_plan", {}),
        "summary": {
            "tasks": len(tasks),
            "regions": len(regions),
            "defects": len(defects),
        },
    }


def _attach_region_representation(task: dict[str, Any],
                                  regions: list[dict[str, Any]]) -> None:
    """Attach the closest overlapping representation contract to a task.

    Visual fallback tasks are rough priors.  When the planner has a semantic
    strategy region for the same region_id, that strategy bbox is authoritative
    and must drive agent geometry and proposal evidence.
    """
    bbox = task.get("bbox")
    region_id = str(task.get("region_id") or "")
    best: tuple[float, dict[str, Any]] | None = None
    for region in regions:
        if region_id and region_id == str(region.get("id") or ""):
            if region.get("bbox") and task.get("kind") == "visual_region_defect":
                task["strategy_bbox"] = _clean_bbox(region.get("bbox"))
            elif region.get("bbox"):
                task["bbox"] = _clean_bbox(region.get("bbox"))
            representation_plan.apply_to_task(task, region)
            return
        if not bbox or not region.get("bbox"):
            continue
        overlap = _bbox_overlap_fraction(bbox, region.get("bbox"))
        if overlap <= 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, region)
    if best and best[0] >= 0.18:
        if best[1].get("bbox") and task.get("kind") == "visual_region_defect":
            task["strategy_bbox"] = _clean_bbox(best[1].get("bbox"))
        elif best[1].get("bbox"):
            task["bbox"] = _clean_bbox(best[1].get("bbox"))
        representation_plan.apply_to_task(task, best[1])


def _dedupe_equivalent_region_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Avoid spending proposal slots on the same method/region twice.

    Visual-review tasks are planner-level instructions about the same semantic
    regions that strategy tasks describe.  When both exist, prefer the visual
    task because it carries the visible failure statement and method policy.
    """
    kept: list[dict[str, Any]] = []
    visual_regions: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("kind") == "visual_region_defect":
            kept.append(task)
            visual_regions.append(task)
            continue
        if _covered_by_visual_task(task, visual_regions):
            continue
        kept.append(task)
    return kept


def _covered_by_visual_task(task: dict[str, Any],
                            visual_tasks: list[dict[str, Any]]) -> bool:
    method = str(task.get("locked_method") or "")
    bbox = task.get("bbox")
    if not method or not bbox:
        return False
    for visual in visual_tasks:
        if method != str(visual.get("locked_method") or ""):
            continue
        vb = visual.get("bbox")
        if not vb:
            continue
        if _bbox_overlap_fraction(bbox, vb) >= 0.16:
            return True
    return False


def _defects_for_region(region: dict, defects: list[dict]) -> list[dict]:
    bbox = region.get("bbox")
    element_ids = {str(eid) for eid in region.get("element_ids", []) if eid}
    out = []
    for defect in defects:
        if defect.get("status") == "skipped":
            continue
        if defect.get("element_id") and str(defect.get("element_id")) in element_ids:
            out.append(defect)
            continue
        if bbox and defect.get("bbox") and _bbox_overlap_fraction(defect["bbox"], bbox) >= 0.18:
            out.append(defect)
    return sorted(out, key=lambda d: -float(d.get("severity", 0)))


def _roles_from_defects(defects: list[dict]) -> list[str]:
    roles = [str(d.get("suggested_agent")) for d in defects if d.get("suggested_agent")]
    return _dedupe(roles)


def _primary_defect(defects: list[dict]) -> dict | None:
    return defects[0] if defects else None


def _fallback_agent(agent: str, defect: dict) -> str:
    if agent in {"IconAgent", "ConnectorAgent", "ShapeAgent"}:
        return "VectorizeAgent"
    if defect.get("type") == "missing_element":
        return "LayoutAgent"
    return "StyleAgent"


def _task_sort_key(task: dict) -> tuple:
    kind = str(task.get("kind") or "")
    region_id = str(task.get("region_id") or "")
    if kind == "chart" and region_id == "q0_coverage_charts":
        priority = 4
    elif kind == "chart":
        priority = 8
    else:
        priority = {
            "visual_region_defect": 0,
            "procedural_3d_surface": 0,
            "pipeline_context_row": 1,
            "auditor_method_cards": 2,
            "failure_summary_panel": 3,
            "component_card_row": 5,
            "bottom_mini_surface": 6,
            "defect_cluster": 7,
        }.get(task.get("kind"), 9)
    severity = 0.0
    defect = task.get("primary_defect") or {}
    if defect:
        severity = float(defect.get("severity", 0))
    visual_rank = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }.get(((task.get("visual_defect") or {}).get("severity") or "").lower(), 4)
    visual_region_rank = {
        "left_surface": 0,
        "pipeline_context": 1,
        "auditor_cards": 2,
        "q0_coverage_charts": 3,
        "action_cards": 4,
        "failure_summary": 5,
        "bottom_mini_surface": 6,
    }.get(str((task.get("visual_defect") or {}).get("region_id") or ""), 9)
    return (priority, visual_rank, visual_region_rank, -severity)


def _objective_for_region(kind: str) -> str:
    if kind == "procedural_3d_surface":
        return "rebuild the scientific 3D manifold as native procedural/vector geometry"
    if kind == "pipeline_context_row":
        return "rebuild the top process row as aligned native cards, text, icons, and connectors"
    if kind == "auditor_method_cards":
        return "rebuild the five CATE-CI Auditor method cards as native grouped components with internal diagrams"
    if kind == "component_card_row":
        return "rebuild repeated action cards as a component system with editable shapes/text/icons"
    if kind == "chart":
        return "rebuild chart axes, series, labels, and residual traces as editable objects"
    if kind == "failure_summary_panel":
        return "rebuild the failure summary as a native icon-and-text panel"
    if kind == "bottom_mini_surface":
        return "rebuild the mini manifold and cheap-nuisance checklist as native components"
    return "improve this semantic region while preserving native editability"


def _success_metrics_for_region(kind: str) -> list[str]:
    base = ["visual_delta", "critical_defect_count", "defect_count"]
    if kind in {"component_card_row", "pipeline_context_row", "failure_summary_panel", "bottom_mini_surface"}:
        return base + ["text_accuracy", "layout_alignment"]
    if kind == "auditor_method_cards":
        return base + ["text_accuracy", "chart_structure", "layout_alignment"]
    if kind == "procedural_3d_surface":
        return base + ["coverage_explained", "native_freeform_quality"]
    if kind == "chart":
        return base + ["chart_structure", "text_accuracy"]
    return base


def _clean_bbox(bbox: Any) -> list[float]:
    if not bbox or len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(v) for v in bbox]


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


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _visual_defects(ir: dict) -> list[dict]:
    review = ir.get("visual_review") or {}
    if review.get("status") not in {"ok", "degraded"}:
        return []
    always_include = {"failure_summary", "bottom_mini_surface"}
    defects = [
        d for d in review.get("defects", [])
        if d.get("severity") in {"critical", "high"}
        or d.get("region_id") in always_include
    ]
    return defects[:10]
