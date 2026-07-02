"""Unified executable audit tasks (P5 of the Decompiler plan).

Today two subsystems invent their own defect shapes: the ``verifier`` emits
per-element residual/text defects, and ``visual_review`` emits per-region
reconstruction findings. Downstream repair has to special-case each. This
module collapses both into a single, executable ``AuditTask`` so the loop is a
clean ``render → audit → route → refine``:

    {
      "task_id": "...",
      "type": "refine_geometry | refine_text | rebuild_component | apply_fallback",
      "component_id": "...",
      "element_id": "...",
      "source_error": "...",
      "suggested_agents": [...],
      "severity": 0.0-1.0,
      "origin": "verifier | visual_review",
      "acceptance": {...}   # what "done" means for this task
    }

Deterministic and offline: it reads an IR dict (and optional components) only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

TASK_TYPES = ["refine_geometry", "refine_text", "rebuild_component", "apply_fallback"]

_TEXT_AGENTS = {"TextLayoutAgent", "TextAgent", "TemplateSlotAgent"}
_REBUILD_AGENTS = {"ChartAgent", "ProceduralSurfaceAgent", "PipelineContextAgent"}
_FALLBACK_STRATEGIES = {"demote", "fallback", "faithful_crop"}

_SEVERITY_WORDS = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}

_ACCEPTANCE = {
    "refine_geometry": {"residual_max": 0.45},
    "refine_text": {"text_accuracy_min": 0.9},
    "rebuild_component": {"visual_review_defects": 0},
    "apply_fallback": {"documented_fallback": True},
}


def _severity_to_float(sev: Any) -> float:
    if isinstance(sev, (int, float)):
        return round(float(sev), 4)
    return _SEVERITY_WORDS.get(str(sev).lower(), 0.5)


def _verifier_task_type(defect: dict) -> str:
    strategy = defect.get("strategy")
    # Real strategy routing stores a dict ({method, region_id, fallback_methods,
    # ...}); older/synthetic defects may store a bare string.
    if isinstance(strategy, dict):
        if set(strategy.get("fallback_methods") or []) & _FALLBACK_STRATEGIES:
            return "apply_fallback"
    elif str(strategy or "") in _FALLBACK_STRATEGIES:
        return "apply_fallback"
    t = str(defect.get("type", ""))
    agent = str(defect.get("suggested_agent", ""))
    if t in ("text_layout_mismatch", "text_template_mismatch") or agent in _TEXT_AGENTS:
        return "refine_text"
    if agent in _REBUILD_AGENTS:
        return "rebuild_component"
    return "refine_geometry"


def _element_to_component(components: Optional[list]) -> tuple[dict, dict]:
    """Return (element_id -> component_id, region_id -> component_id) maps."""
    by_element: dict = {}
    by_region: dict = {}
    for comp in components or []:
        cid = comp.get("id")
        for eid in comp.get("element_ids") or []:
            by_element[eid] = cid
        region_id = (comp.get("provenance") or {}).get("region_id")
        if region_id:
            by_region[region_id] = cid
    return by_element, by_region


def unify_tasks(ir: dict, components: Optional[list] = None) -> list:
    """Build the unified executable task list from an IR's defects."""
    by_element, by_region = _element_to_component(components)
    tasks: list = []

    for defect in ir.get("defects") or []:
        if defect.get("status") == "skipped":
            continue
        ttype = _verifier_task_type(defect)
        eid = defect.get("element_id")
        tasks.append({
            "task_id": f"task_{defect.get('id', len(tasks))}",
            "type": ttype,
            "component_id": by_element.get(eid),
            "element_id": eid,
            "source_error": defect.get("reason"),
            "suggested_agents": [defect["suggested_agent"]] if defect.get("suggested_agent") else [],
            "severity": _severity_to_float(defect.get("severity")),
            "origin": "verifier",
            "acceptance": dict(_ACCEPTANCE[ttype]),
        })

    visual_review = (ir.get("visual_review") or {}).get("defects") or []
    for vr in visual_review:
        label = str(vr.get("region_label", "")).lower()
        ttype = "refine_text" if label in ("text", "caption", "label") else "rebuild_component"
        region_id = vr.get("region_id")
        tasks.append({
            "task_id": f"task_{vr.get('id', len(tasks))}_vr",
            "type": ttype,
            "component_id": by_region.get(region_id, region_id),
            "element_id": None,
            "source_error": vr.get("visual_problem"),
            "expected_native_expression": vr.get("expected_native_expression"),
            "suggested_agents": vr.get("suggested_agents") or [],
            "severity": _severity_to_float(vr.get("severity")),
            "origin": "visual_review",
            "acceptance": dict(_ACCEPTANCE[ttype]),
        })

    tasks.sort(key=lambda t: -t["severity"])
    return tasks


def write_audit_tasks(tasks: list, out_dir) -> dict:
    payload = {
        "schema": "audit-tasks-v1",
        "count": len(tasks),
        "types": TASK_TYPES,
        "type_summary": dict(sorted(Counter(t["type"] for t in tasks).items())),
        "tasks": tasks,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_tasks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unify verifier + visual_review defects into executable audit tasks.")
    ap.add_argument("run_dir", help="v3 output dir with ir_final.json (+ optional components.json)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    ir = json.loads((run_dir / "ir_final.json").read_text())
    components = None
    comp_path = run_dir / "components.json"
    if comp_path.exists():
        components = json.loads(comp_path.read_text()).get("components")
    tasks = unify_tasks(ir, components)
    payload = write_audit_tasks(tasks, run_dir)
    print(f"audit tasks: {payload['count']} → {run_dir / 'audit_tasks.json'}")
    for ttype, n in payload["type_summary"].items():
        print(f"  {ttype}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
