"""Template slot repair agent.

TextLayoutAgent fixes ordinary glyph ink by calibrating individual text boxes.
Protected semantic anchors are different: titles, captions, chart titles, and
component headings are generated from role templates, so their correction has
to live in the typography layer and be replayed whenever the IR is rebuilt.
"""
from __future__ import annotations

import os
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class TemplateSlotAgent(Agent):
    """Repair role-owned text slots from verifier ink diagnostics."""

    name = "TemplateSlotAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect") or {}
        if defect.get("type") == "text_template_mismatch":
            batch = _template_defects(ir, limit=_batch_limit(), seed=defect)
        else:
            batch = _template_defects(ir, limit=_batch_limit())

        changed: list[str] = []
        for item in batch:
            el = IR.get_element(ir, item.get("element_id"))
            if el and el.get("type") in {"text", "formula"}:
                changed.extend(self._apply_template_adjustment(ir, el, item.get("text_layout") or {}))
        return _dedupe(changed)

    def _apply_template_adjustment(self, ir: dict, el: dict, diag: dict) -> list[str]:
        if not diag or _adjustment_count(el) >= _max_adjustments(el):
            return []
        bbox = el.get("bbox")
        if not bbox:
            return []

        dx, dy = [float(v) for v in (diag.get("shift_px") or [0.0, 0.0])[:2]]
        font_scale = float(diag.get("font_scale") or 1.0)
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        shift_gain, scale_lo, scale_hi = _step_profile(el)

        step_dx = _clamp(dx * shift_gain, -w * 0.18, w * 0.18)
        step_dy = _clamp(dy * shift_gain, -h * 0.24, h * 0.24)
        step_scale = _clamp(font_scale, scale_lo, scale_hi)

        ext = el.setdefault("ext", {})
        adjust = ext.setdefault("typography_slot_adjustment", {})
        base_bbox = [float(v) for v in (adjust.get("base_bbox") or bbox)[:4]]
        base_font = float(adjust.get("base_font_size") or el.get("font_size") or 12.0)
        prev_dx = float(adjust.get("dx", 0.0))
        prev_dy = float(adjust.get("dy", 0.0))
        prev_scale = float(adjust.get("font_scale", 1.0))
        new_dx = _clamp(prev_dx + step_dx, -w * 0.24, w * 0.24)
        new_dy = _clamp(prev_dy + step_dy, -h * 0.32, h * 0.32)
        new_scale = _clamp(prev_scale * step_scale, 0.78, 1.28)
        adjust["base_bbox"] = [round(v, 3) for v in base_bbox]
        adjust["base_font_size"] = round(base_font, 3)
        adjust["dx"] = round(new_dx, 3)
        adjust["dy"] = round(new_dy, 3)
        adjust["font_scale"] = round(new_scale, 4)
        if _allow_color_adjustment(el) and float(diag.get("color_distance") or 0.0) >= 70.0:
            adjust["text_color"] = diag.get("original_color")
        adjust["source"] = self.name
        adjust["last_layout_error"] = diag.get("layout_error")
        adjust["count"] = int(adjust.get("count") or 0) + 1

        el["bbox"] = [
            round(base_bbox[0] + new_dx, 3),
            round(base_bbox[1] + new_dy, 3),
            round(base_bbox[2] + new_dx, 3),
            round(base_bbox[3] + new_dy, 3),
        ]
        el["font_size"] = round(base_font * new_scale, 2)
        if adjust.get("text_color"):
            el["text_color"] = adjust["text_color"]

        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "template_slot_adjustment",
            "round": ir.get("round", 0),
            "role": _role(el),
            "shift_px": [round(step_dx, 2), round(step_dy, 2)],
            "font_scale_step": round(step_scale, 4),
        })
        return [el["id"]]


def _template_defects(ir: dict, limit: int, seed: dict | None = None) -> list[dict]:
    candidates: list[dict] = []
    for d in ir.get("defects", []):
        if (
            d.get("suggested_agent") != TemplateSlotAgent.name
            or d.get("type") != "text_template_mismatch"
            or not d.get("element_id")
            or not d.get("text_layout")
        ):
            continue
        el = IR.get_element(ir, d.get("element_id"))
        if not el or _adjustment_count(el) >= _max_adjustments(el):
            continue
        candidates.append(d)
    if not candidates:
        return []
    seed_el = IR.get_element(ir, seed.get("element_id")) if seed else None
    if seed_el and _adjustment_count(seed_el) >= _max_adjustments(seed_el):
        seed_el = None
    high_priority = [d for d in candidates if _role_priority_for_defect(ir, d) <= 2]
    seed_priority = _role_priority(_role(seed_el)) if seed_el else 99
    if high_priority and seed_priority > 2:
        high_priority.sort(key=lambda d: (_role_priority_for_defect(ir, d), -float(d.get("severity", 0.0))))
        return high_priority[:max(1, limit)]
    if seed and seed_el:
        seed_family = _role_family(seed_el)
        seed_region = _region_id(seed_el, seed)
        candidates.sort(key=lambda d: _batch_rank(ir, d, seed_family, seed_region))
    else:
        candidates.sort(key=lambda d: (_role_priority_for_defect(ir, d), -float(d.get("severity", 0.0))))
    return candidates[:max(1, limit)]


def _batch_rank(ir: dict, defect: dict, seed_family: str, seed_region: str) -> tuple:
    el = IR.get_element(ir, defect.get("element_id"))
    if not el:
        return (9, 0.0)
    same_family = _role_family(el) == seed_family
    same_region = _region_id(el, defect) == seed_region
    return (
        0 if same_family and same_region else
        1 if same_family else
        2 if same_region else
        3,
        _role_priority(_role(el)),
        _adjustment_count(el),
        -float(defect.get("severity", 0.0)),
    )


def _role_priority_for_defect(ir: dict, defect: dict) -> int:
    el = IR.get_element(ir, defect.get("element_id"))
    return _role_priority(_role(el)) if el else 99


def _role_priority(role: str) -> int:
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
        return 1
    return 8


def _role_family(el: dict) -> str:
    role = _role(el)
    if role in {"slide_title", "solution_title", "subtitle", "caption"}:
        return "global_template"
    if role.startswith("chart_title"):
        return "chart_title"
    if role.startswith("action_"):
        return "action_title"
    if role.startswith("auditor_"):
        return "auditor_template"
    if role.startswith("risk_") or role == "ci_axis_label":
        return "surface_labels"
    if role.startswith("covariate_") or role == "axis_math":
        return "surface_axes"
    return role


def _region_id(el: dict, defect: dict | None = None) -> str:
    strategy = (defect or {}).get("strategy") or ((el.get("ext") or {}).get("strategy") or {})
    return str(strategy.get("region_id") or "")


def _role(el: dict) -> str:
    return str((((el.get("ext") or {}).get("typography") or {}).get("role")) or "")


def _adjustment_count(el: dict) -> int:
    adjust = (el.get("ext") or {}).get("typography_slot_adjustment") or {}
    return int(adjust.get("count") or 0)


def _max_adjustments(el: dict) -> int:
    role = _role(el)
    if role in {"slide_title", "solution_title", "subtitle", "caption"}:
        return 2
    if role.startswith("chart_title"):
        return 2
    if role in {"action_title", "action_report_title", "checklist_body"}:
        return 1
    if role in {"covariate_math", "axis_math", "surface_vector_math", "ci_axis_label"}:
        return 1
    return 2


def _step_profile(el: dict) -> tuple[float, float, float]:
    role = _role(el)
    if role in {"slide_title", "solution_title"}:
        return (0.46, 0.92, 1.08)
    if role in {"subtitle", "caption"}:
        return (0.48, 0.92, 1.08)
    if role.startswith("chart_title"):
        return (0.50, 0.90, 1.10)
    if role in {"action_title", "action_report_title"}:
        return (0.22, 0.98, 1.035)
    if role == "checklist_body":
        return (0.24, 0.98, 1.04)
    if role in {"covariate_math", "axis_math", "surface_vector_math", "surface_theta_math", "ci_axis_label"}:
        return (0.34, 0.94, 1.06)
    return (0.42, 0.92, 1.08)


def _allow_color_adjustment(el: dict) -> bool:
    ext = el.get("ext") or {}
    if ext.get("text_color_locked"):
        return False
    contract = ext.get("text_contract") or {}
    if isinstance(contract, dict) and contract.get("text_color_locked"):
        return False
    role = _role(el)
    return role not in {
        "action_title",
        "action_report_title",
        "covariate_math",
        "axis_math",
        "surface_vector_math",
        "surface_theta_math",
        "ci_axis_label",
        "checklist_body",
    }


def _batch_limit() -> int:
    raw = os.environ.get("I2E_TEMPLATE_SLOT_BATCH", "4")
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 4


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
