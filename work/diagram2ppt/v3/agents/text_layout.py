"""Text layout calibration agent.

This agent consumes verifier evidence from ``text_layout_audit``.  It does
not re-OCR content; it adjusts the editable text slot so rendered glyph ink
lands closer to the source image.
"""
from __future__ import annotations

import os
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class TextLayoutAgent(Agent):
    """Repair font scale, bbox offset, and color from rendered evidence."""

    name = "TextLayoutAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect") or {}
        if defect.get("type") == "text_layout_mismatch":
            batch = _text_layout_defects(ir, limit=_batch_limit(), seed=defect)
            changed: list[str] = []
            for item in batch:
                el = IR.get_element(ir, item.get("element_id"))
                if el and el.get("type") in {"text", "formula"}:
                    changed.extend(self._apply_diagnosis(ir, el, item.get("text_layout") or {}))
            return _dedupe(changed)
        if defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el and el.get("type") in {"text", "formula"}:
                return self._apply_diagnosis(ir, el, defect.get("text_layout") or {})

        changed: list[str] = []
        for d in _text_layout_defects(ir, limit=_batch_limit()):
            el = IR.get_element(ir, d["element_id"])
            if el and el.get("type") in {"text", "formula"}:
                changed.extend(self._apply_diagnosis(ir, el, d.get("text_layout") or {}))
        return _dedupe(changed)

    def _apply_diagnosis(self, ir: dict, el: dict, diag: dict) -> list[str]:
        if not diag:
            return []
        if _protected_role(el):
            return []
        if _calibration_count(el) >= _max_calibration_count(el):
            return []
        bbox = el.get("bbox")
        if not bbox:
            return []

        dx, dy = [float(v) for v in (diag.get("shift_px") or [0.0, 0.0])[:2]]
        font_scale = float(diag.get("font_scale") or 1.0)
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)

        # Apply a damped correction.  The verifier remeasures after rendering,
        # so repeated rounds converge without overshooting.
        shift_gain, scale_lo, scale_hi = _step_profile(el)
        step_dx = _clamp(dx * shift_gain, -w * 0.16, w * 0.16)
        step_dy = _clamp(dy * shift_gain, -h * 0.20, h * 0.20)
        step_scale = _clamp(font_scale, scale_lo, scale_hi)

        el["bbox"] = [
            round(x0 + step_dx, 3),
            round(y0 + step_dy, 3),
            round(x1 + step_dx, 3),
            round(y1 + step_dy, 3),
        ]
        if el.get("font_size"):
            el["font_size"] = round(float(el["font_size"]) * step_scale, 2)
        allow_color = _allow_color_adjustment(el)
        if allow_color and diag.get("original_color") and float(diag.get("color_distance") or 0.0) >= 58.0:
            el["text_color"] = diag["original_color"]

        ext = el.setdefault("ext", {})
        calib = ext.setdefault("typography_calibration", {})
        prev_dx = float(calib.get("dx", 0.0))
        prev_dy = float(calib.get("dy", 0.0))
        prev_scale = float(calib.get("font_scale", 1.0))
        calib["dx"] = round(_clamp(prev_dx + step_dx, -w * 0.22, w * 0.22), 3)
        calib["dy"] = round(_clamp(prev_dy + step_dy, -h * 0.28, h * 0.28), 3)
        calib["font_scale"] = round(_clamp(prev_scale * step_scale, 0.72, 1.36), 4)
        if allow_color and diag.get("original_color") and float(diag.get("color_distance") or 0.0) >= 58.0:
            calib["text_color"] = diag["original_color"]
        calib["source"] = self.name
        calib["last_layout_error"] = diag.get("layout_error")
        calib["count"] = int(calib.get("count") or 0) + 1

        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "glyph_ink_calibration",
            "round": ir.get("round", 0),
            "shift_px": [round(step_dx, 2), round(step_dy, 2)],
            "font_scale_step": round(step_scale, 4),
        })
        return [el["id"]]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _text_layout_defects(ir: dict, limit: int, seed: dict | None = None) -> list[dict]:
    candidates = []
    for d in ir.get("defects", []):
        if (
            d.get("suggested_agent") != TextLayoutAgent.name
            or d.get("type") != "text_layout_mismatch"
            or not d.get("element_id")
            or not d.get("text_layout")
        ):
            continue
        el = IR.get_element(ir, d.get("element_id"))
        if el and (el.get("ext") or {}).get("typography_locked"):
            continue
        if el and _protected_role(el):
            continue
        if el and _calibration_count(el) >= _max_calibration_count(el):
            continue
        candidates.append(d)
    if not candidates:
        return []
    seed_el = IR.get_element(ir, seed.get("element_id")) if seed else None
    if seed and seed_el and not _protected_role(seed_el):
        seed_key = _role_family(seed_el)
        seed_region = _region_id(seed_el, seed)
        ranked = sorted(
            candidates,
            key=lambda d: _batch_rank(ir, d, seed_key, seed_region),
        )
        return ranked[:max(1, limit)]
    candidates.sort(key=lambda d: -float(d.get("severity", 0.0)))
    return candidates[:max(1, limit)]


def _batch_rank(ir: dict, defect: dict, seed_key: str, seed_region: str) -> tuple:
    el = IR.get_element(ir, defect.get("element_id"))
    if not el:
        return (9, 0.0)
    same_family = _role_family(el) == seed_key
    same_region = _region_id(el, defect) == seed_region
    calibrated = _calibration_count(el)
    return (
        0 if same_family and same_region else
        1 if same_family else
        2 if same_region else
        3,
        calibrated,
        -float(defect.get("severity", 0.0)),
    )


def _role_family(el: dict) -> str:
    role = _role(el)
    if role.startswith("chart_"):
        if role in {"chart_tick", "chart_axis_label", "chart_bar_label", "chart_bar_value", "chart_curve_label"}:
            return "chart_labels"
        return "chart_text"
    if role.startswith("action_"):
        return "action_text"
    if role.startswith("auditor_formula"):
        return "auditor_formula"
    if role.startswith("auditor_"):
        return "auditor_text"
    if role.startswith("failure_"):
        return "failure_text"
    if role in {"process_card", "process_title"}:
        return "pipeline_text"
    if role == "body":
        return "body"
    return role


def _region_id(el: dict, defect: dict | None = None) -> str:
    strategy = (defect or {}).get("strategy") or ((el.get("ext") or {}).get("strategy") or {})
    return str(strategy.get("region_id") or "")


def _role(el: dict) -> str:
    return str((((el.get("ext") or {}).get("typography") or {}).get("role")) or "")


def _calibration_count(el: dict) -> int:
    calib = (el.get("ext") or {}).get("typography_calibration") or {}
    return int(calib.get("count") or 0)


def _max_calibration_count(el: dict) -> int:
    role = _role(el)
    if role.startswith("chart_"):
        return 1
    if role in {"process_card", "body"}:
        return 1
    if role.startswith("action_"):
        return 2
    if role.startswith("auditor_formula"):
        return 2
    return 1


def _step_profile(el: dict) -> tuple[float, float, float]:
    role = _role(el)
    if role.startswith("chart_"):
        return (0.64, 0.88, 1.12)
    if role.startswith("action_"):
        return (0.64, 0.88, 1.14)
    if role.startswith("auditor_formula"):
        return (0.58, 0.90, 1.12)
    return (0.62, 0.88, 1.14)


def _protected_role(el: dict) -> bool:
    """Skip semantic anchor text whose crop contains non-text visual content."""
    if (el.get("ext") or {}).get("typography_locked"):
        return True
    typo = ((el.get("ext") or {}).get("typography") or {})
    role = str(typo.get("role") or "")
    protected = {
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
    return role in protected


def _allow_color_adjustment(el: dict) -> bool:
    ext = el.get("ext") or {}
    if ext.get("text_color_locked"):
        return False
    role = _role(el)
    return role not in {
        "action_body",
        "action_body_emphasis",
        "action_body_math",
        "action_report_body",
        "action_report_body_emphasis",
    }


def _batch_limit() -> int:
    raw = os.environ.get("I2E_TEXT_LAYOUT_BATCH", "5")
    try:
        return max(1, min(64, int(raw)))
    except ValueError:
        return 16


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
