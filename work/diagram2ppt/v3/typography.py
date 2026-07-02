"""Typography planning for native diagram reconstruction.

OCR gives ink boxes; PPTX needs editable text containers.  This module turns
text elements into role-aware typography slots so font size, color, alignment,
and position are controlled by the owning component instead of raw OCR bbox
geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextStyle:
    font: str = "Arial"
    size: float = 14.0
    color: str = "#333333"
    align: str = "center"
    bold: bool = False
    italic: bool = False


STYLE: dict[str, TextStyle] = {
    "slide_title": TextStyle("Times New Roman", 39, "#071a4d", "center", True),
    "solution_title": TextStyle("Times New Roman", 39, "#071a4d", "center", True),
    "section_title": TextStyle("Times New Roman", 34, "#071a4d", "center", True),
    "subtitle": TextStyle("Arial", 23, "#555555", "center", False, True),
    "axis_label": TextStyle("Arial", 18, "#111111", "center"),
    "covariate_label": TextStyle("Arial", 28, "#111111", "center"),
    "covariate_text": TextStyle("Arial", 28, "#111111", "center"),
    "covariate_math": TextStyle("Times New Roman", 38, "#111111", "center", False, True),
    "axis_math": TextStyle("Times New Roman", 22, "#111111", "center", False, True),
    "vector_label": TextStyle("Arial", 31, "#333333", "center"),
    "surface_vector_math": TextStyle("Times New Roman", 34, "#333333", "center", False, True),
    "surface_theta_math": TextStyle("Times New Roman", 24, "#111111", "center", False, True),
    "annotation": TextStyle("Arial", 13, "#333333", "center"),
    "risk_label": TextStyle("Arial", 22, "#c83322", "center"),
    "risk_label_math": TextStyle("Times New Roman", 23, "#c83322", "center"),
    "risk_q_math": TextStyle("Times New Roman", 25, "#c83322", "center", False, True),
    "ci_title": TextStyle("Arial", 23, "#7d1c16", "center"),
    "ci_axis_label": TextStyle("Times New Roman", 20, "#111111", "center"),
    "formula": TextStyle("Cambria Math", 18, "#111111", "center"),
    "formula_main": TextStyle("Cambria Math", 38, "#111111", "center"),
    "surface_formula_main": TextStyle("Cambria Math", 34, "#111111", "center"),
    "surface_formula_fraction": TextStyle("Cambria Math", 30, "#111111", "center"),
    "process_title": TextStyle("Arial", 25, "#555555", "center"),
    "process_card": TextStyle("Arial", 19.8, "#444444", "center"),
    "auditor_title": TextStyle("Arial", 25.5, "#222222", "center", True),
    "auditor_formula": TextStyle("Cambria Math", 31, "#222222", "center"),
    "auditor_formula_prefix": TextStyle("Cambria Math", 27, "#222222", "right"),
    "auditor_formula_fraction": TextStyle("Cambria Math", 25, "#222222", "center"),
    "auditor_group_label": TextStyle("Arial", 25, "#1c56b7", "center", False, True),
    "nuisance_label": TextStyle("Arial", 18, "#1f66d1", "center", False, True),
    "chart_title": TextStyle("Times New Roman", 36, "#071a4d", "center", True),
    "chart_title_q": TextStyle("Times New Roman", 36, "#071a4d", "right", True),
    "chart_title_sub": TextStyle("Times New Roman", 22, "#071a4d", "center", True),
    "chart_title_rest": TextStyle("Times New Roman", 36, "#071a4d", "left", True),
    "chart_axis_label": TextStyle("Times New Roman", 21, "#111111", "center"),
    "chart_curve_label": TextStyle("Arial", 20.5, "#333333", "center"),
    "chart_label": TextStyle("Arial", 16, "#333333", "center"),
    "chart_bar_label": TextStyle("Arial", 16.5, "#333333", "center"),
    "chart_bar_value": TextStyle("Times New Roman", 18, "#111111", "center"),
    "chart_tick": TextStyle("Times New Roman", 16.5, "#333333", "center"),
    "failure_title": TextStyle("Arial", 34, "#071a4d", "center", True),
    "failure_body": TextStyle("Arial", 25, "#111827", "left"),
    "failure_math": TextStyle("Times New Roman", 25, "#111827", "left", False, True),
    "action_title": TextStyle("Arial", 26, "#333333", "center", True),
    "action_report_title": TextStyle("Arial", 24, "#333333", "center", True),
    "action_body": TextStyle("Arial", 20.5, "#333333", "center"),
    "action_body_emphasis": TextStyle("Arial", 19.9, "#333333", "center", False, True),
    "action_body_math": TextStyle("Times New Roman", 20.2, "#333333", "center", False, True),
    "action_report_body": TextStyle("Arial", 19.2, "#333333", "center"),
    "action_report_body_emphasis": TextStyle("Arial", 19.2, "#333333", "center", False, True),
    "checklist_body": TextStyle("Arial", 18, "#111111", "left"),
    "caption": TextStyle("Times New Roman", 36, "#222222", "center", False, True),
    "body": TextStyle("Arial", 12, "#333333", "center"),
}

ACTION_COLORS = {
    "retain": "#16806e",
    "defer": "#9a5b13",
    "alert": "#cf3d28",
    "report": "#245591",
}

MIN_SIZE = {
    "slide_title": 31.0,
    "solution_title": 31.0,
    "section_title": 26.0,
    "subtitle": 17.0,
    "axis_label": 13.0,
    "covariate_text": 20.0,
    "covariate_math": 26.0,
    "axis_math": 15.0,
    "process_title": 17.0,
    "process_card": 13.5,
    "auditor_title": 19.0,
    "auditor_formula": 22.0,
    "auditor_formula_prefix": 19.0,
    "auditor_formula_fraction": 18.0,
    "chart_title": 26.0,
    "chart_title_q": 26.0,
    "chart_title_sub": 16.0,
    "chart_title_rest": 26.0,
    "chart_axis_label": 14.0,
    "chart_curve_label": 14.0,
    "chart_label": 11.0,
    "chart_bar_label": 12.5,
    "chart_bar_value": 12.5,
    "chart_tick": 12.0,
    "failure_title": 22.0,
    "failure_body": 16.0,
    "failure_math": 16.0,
    "action_title": 19.0,
    "action_report_title": 18.0,
    "action_body": 13.5,
    "action_body_emphasis": 13.5,
    "action_body_math": 14.0,
    "action_report_body": 13.0,
    "action_report_body_emphasis": 13.0,
    "checklist_body": 13.0,
    "caption": 22.0,
    "formula_main": 30.0,
    "surface_formula_main": 24.0,
    "surface_formula_fraction": 20.0,
    "vector_label": 23.0,
    "surface_vector_math": 24.0,
    "surface_theta_math": 16.0,
    "ci_title": 17.0,
    "ci_axis_label": 14.0,
    "nuisance_label": 13.0,
    "covariate_label": 18.0,
    "risk_label": 14.0,
    "risk_label_math": 15.0,
    "risk_q_math": 16.0,
}

ROLE_RENDERING = {
    "slide_title": {"fit_width_factor": 0.38, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "solution_title": {"fit_width_factor": 0.38, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "section_title": {"fit_width_factor": 0.40, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "subtitle": {"fit_width_factor": 0.38, "fit_height_factor": 0.84, "margin_px": [0, 0, 0, 0]},
    "caption": {"fit_width_factor": 0.34, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
    "vector_label": {"fit_width_factor": 0.36, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "surface_vector_math": {"fit_width_factor": 0.30, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "surface_theta_math": {"fit_width_factor": 0.30, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "covariate_text": {"fit_width_factor": 0.34, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "covariate_math": {"fit_width_factor": 0.32, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "axis_math": {"fit_width_factor": 0.32, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "risk_label": {"fit_width_factor": 0.42, "fit_height_factor": 0.84, "margin_px": [0, 0, 0, 0]},
    "risk_label_math": {"fit_width_factor": 0.38, "fit_height_factor": 0.84, "margin_px": [0, 0, 0, 0], "line_spacing": 0.88},
    "risk_q_math": {"fit_width_factor": 0.30, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0]},
    "ci_axis_label": {"fit_width_factor": 0.34, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "failure_body": {"fit_width_factor": 0.39, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92, "word_wrap": True},
    "failure_math": {"fit_width_factor": 0.30, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "action_body": {"fit_width_factor": 0.36, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "line_spacing": 0.94, "word_wrap": False},
    "action_body_emphasis": {"fit_width_factor": 0.36, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "word_wrap": False},
    "action_body_math": {"fit_width_factor": 0.31, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "word_wrap": False},
    "action_report_body": {"fit_width_factor": 0.34, "fit_height_factor": 0.84, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92, "word_wrap": False},
    "action_report_body_emphasis": {"fit_width_factor": 0.34, "fit_height_factor": 0.84, "margin_px": [0, 0, 0, 0], "word_wrap": False},
    "checklist_body": {"fit_width_factor": 0.42, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "line_spacing": 0.95},
    "auditor_title": {"fit_width_factor": 0.40, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "line_spacing": 0.90},
    "auditor_formula": {"fit_width_factor": 0.27, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
    "auditor_formula_prefix": {"fit_width_factor": 0.25, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
    "auditor_formula_fraction": {"fit_width_factor": 0.25, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
    "auditor_group_label": {"fit_width_factor": 0.34, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
    "process_card": {"fit_width_factor": 0.45, "fit_height_factor": 0.82, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92, "word_wrap": True},
    "chart_title": {"fit_width_factor": 0.36, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "chart_title_q": {"fit_width_factor": 0.36, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "chart_title_sub": {"fit_width_factor": 0.34, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "chart_title_rest": {"fit_width_factor": 0.36, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.92},
    "chart_axis_label": {"fit_width_factor": 0.31, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.90},
    "chart_curve_label": {"fit_width_factor": 0.34, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "line_spacing": 0.90},
    "chart_bar_label": {"fit_width_factor": 0.36, "fit_height_factor": 0.82, "margin_px": [0, 0, 0, 0], "line_spacing": 0.84},
    "chart_bar_value": {"fit_width_factor": 0.34, "fit_height_factor": 0.86, "margin_px": [0, 0, 0, 0], "line_spacing": 0.90},
    "chart_tick": {"fit_width_factor": 0.31, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0], "line_spacing": 0.90},
    "formula": {"fit_width_factor": 0.30, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
    "formula_main": {"fit_width_factor": 0.30, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "surface_formula_main": {"fit_width_factor": 0.28, "fit_height_factor": 0.90, "margin_px": [0, 0, 0, 0]},
    "surface_formula_fraction": {"fit_width_factor": 0.28, "fit_height_factor": 0.88, "margin_px": [0, 0, 0, 0]},
}


TYPOGRAPHY_CONTRACTS: dict[str, dict[str, Any]] = {
    "procedural_surface": {
        "required_role_prefixes": [
            "covariate",
            "axis",
            "surface_vector",
            "risk",
            "ci_",
        ],
        "min_role_fraction": 0.90,
        "max_overflow_fraction": 0.08,
        "color_policy": "semantic_surface_colors",
    },
    "chart_parser": {
        "required_role_prefixes": ["chart_"],
        "min_role_fraction": 0.92,
        "max_overflow_fraction": 0.06,
        "color_policy": "chart_series_and_axis_colors",
    },
    "pipeline_context_layout": {
        "required_roles": ["process_title", "process_card"],
        "min_role_fraction": 0.92,
        "max_overflow_fraction": 0.04,
        "color_policy": "neutral_process_text",
    },
    "auditor_card_layout": {
        "required_role_prefixes": ["auditor_"],
        "min_role_fraction": 0.92,
        "max_overflow_fraction": 0.06,
        "color_policy": "method_card_palette",
    },
    "component_layout": {
        "required_role_prefixes": ["action_"],
        "min_role_fraction": 0.92,
        "max_overflow_fraction": 0.05,
        "color_policy": "action_card_palette",
    },
    "failure_summary_layout": {
        "required_role_prefixes": ["failure_"],
        "min_role_fraction": 0.92,
        "max_overflow_fraction": 0.05,
        "color_policy": "summary_icon_text_palette",
    },
    "mini_surface_checklist": {
        "required_roles": ["checklist_body", "nuisance_label"],
        "min_role_fraction": 0.88,
        "max_overflow_fraction": 0.08,
        "color_policy": "mini_surface_checklist_palette",
    },
}


def contract_for_method(method: str | None) -> dict[str, Any]:
    """Return the typography contract attached to a rendering method."""
    if not method:
        return {}
    contract = TYPOGRAPHY_CONTRACTS.get(str(method), {})
    if not contract:
        return {}
    out = dict(contract)
    out["method"] = str(method)
    out["source"] = "TypographyPlanner"
    return out


def score_contract(ir: dict, task: dict) -> dict[str, Any]:
    """Score role/style control inside a task region."""
    contract = (
        task.get("typography_contract")
        or ((task.get("representation") or {}).get("typography_contract"))
        or contract_for_method(task.get("locked_method"))
    )
    if not contract:
        return {"score": 1.0, "texts": 0, "role_fraction": 1.0, "overflow_fraction": 0.0}
    bbox = task.get("bbox")
    texts = [
        e for e in ir.get("elements", [])
        if e.get("type") in {"text", "formula"}
        and e.get("bbox")
        and (not bbox or _bbox_overlap_fraction(e.get("bbox"), bbox) > 0.0 or _bbox_center_inside(e.get("bbox"), bbox))
    ]
    if not texts:
        return {"score": 0.0, "texts": 0, "role_fraction": 0.0, "overflow_fraction": 1.0}
    controlled = [e for e in texts if _role_matches_contract(_element_role(e), contract)]
    overflow = [e for e in texts if _text_likely_overflows(e)]
    role_fraction = len(controlled) / max(1, len(texts))
    overflow_fraction = len(overflow) / max(1, len(texts))
    min_role = float(contract.get("min_role_fraction") or 0.90)
    max_overflow = float(contract.get("max_overflow_fraction") or 0.08)
    role_score = min(1.0, role_fraction / max(0.01, min_role))
    overflow_score = 1.0 if overflow_fraction <= max_overflow else max(0.0, 1.0 - (overflow_fraction - max_overflow) * 4.0)
    return {
        "score": round(role_score * 0.72 + overflow_score * 0.28, 4),
        "texts": len(texts),
        "controlled": len(controlled),
        "overflow": len(overflow),
        "role_fraction": round(role_fraction, 4),
        "overflow_fraction": round(overflow_fraction, 4),
        "required_roles": list(contract.get("required_roles") or []),
        "required_role_prefixes": list(contract.get("required_role_prefixes") or []),
    }


def apply(ir: dict) -> dict[str, int]:
    """Apply typography roles and component text slots in-place."""
    elements = ir.get("elements", [])
    by_id = {str(e.get("id")): e for e in elements if e.get("id")}
    stats = {"styled": 0, "slotted": 0, "clamped": 0}
    for el in elements:
        if el.get("type") not in {"text", "formula"} or not el.get("bbox"):
            continue
        if (el.get("ext") or {}).get("typography_locked"):
            continue
        old = _signature(el)
        role = _role(el)
        slot = _slot_for(el, by_id, ir)
        if slot is not None:
            el["bbox"] = slot
            stats["slotted"] += 1
        style = _style_for(el, role)
        _apply_style(el, role, style)
        if _apply_slot_adjustment(el):
            stats["slotted"] += 1
        if _apply_calibration(el):
            stats["slotted"] += 1
        if _clamp_font_to_box(el):
            stats["clamped"] += 1
        if _signature(el) != old:
            el.setdefault("repair_history", []).append({
                "agent": "TypographyPlanner",
                "action": "role_style_slot",
                "role": role,
                "round": ir.get("round", 0),
            })
            stats["styled"] += 1
    if stats["styled"]:
        ir.setdefault("quality_gate", {}).setdefault("typography", []).append({
            "round": ir.get("round", 0),
            **stats,
        })
    return stats


def _role(el: dict) -> str:
    eid = str(el.get("id") or "")
    text = str(el.get("text") or el.get("latex") or "").strip()
    lower = text.lower()
    ext = el.get("ext") or {}
    contract = _text_contract(el)
    if contract.get("role"):
        return str(contract["role"])
    comp = str(ext.get("component") or "")
    comp_role = str(ext.get("component_role") or "")
    if comp == "generic_flow_pipeline" and comp_role == "title":
        return "process_title"
    if comp == "generic_flow_pipeline" and comp_role in {"text", "process_card"}:
        return "process_card"

    if eid == "proc_formula_alignment":
        return "surface_formula_main"
    if eid in {"proc_formula_alignment_lhs", "proc_formula_alignment_rhs"}:
        return "surface_formula_main"
    if eid.startswith("proc_formula_alignment_"):
        return "surface_formula_fraction"
    if eid.startswith("proc_covariate_label_text") or comp_role == "covariate_text":
        return "covariate_text"
    if eid.startswith("proc_covariate_label_x") or comp_role == "covariate_math":
        return "covariate_math"
    if eid.startswith("proc_axis_label_") or comp_role == "axis_math":
        return "axis_math"
    if lower.startswith("problem:"):
        return "slide_title"
    if lower.startswith("solution:"):
        return "solution_title"
    if text == "CATE-CI Auditor":
        return "section_title"
    if "lightweight" in lower:
        return "subtitle"
    if lower.startswith("covariate space"):
        return "covariate_label"
    if "detects geometry-induced" in lower:
        return "caption"
    if eid.startswith("pipeline_context_title"):
        return "process_title"
    if eid.startswith("pipeline_context_text_"):
        return "process_card"
    if eid.startswith("auditor_title_"):
        return "auditor_title"
    if eid == "auditor_formula_alignment_prefix":
        return "auditor_formula_prefix"
    if eid.startswith("auditor_formula_alignment_"):
        return "auditor_formula_fraction"
    if eid.startswith("auditor_formula_"):
        return "auditor_formula"
    if eid == "auditor_cheap_nuisance_label":
        return "auditor_group_label"
    if eid == "bottom_nuisance_label":
        return "nuisance_label"
    if eid == "chart_q0_title_q":
        return "chart_title_q"
    if eid == "chart_q0_title_sub":
        return "chart_title_sub"
    if eid == "chart_q0_title_rest":
        return "chart_title_rest"
    if eid.startswith("chart_q0_title"):
        return "chart_title"
    if eid in {"chart_q0_y_label", "chart_q0_x_label", "chart_q0_bar_y_label"}:
        return "chart_axis_label"
    if eid in {"chart_q0_orthogonal", "chart_q0_aligned_label"}:
        return "chart_curve_label"
    if eid.startswith("chart_q0_bar_label_"):
        return "chart_bar_label"
    if eid.startswith("chart_q0_bar_value_"):
        return "chart_bar_value"
    if eid.startswith("chart_q0_tick_") or eid.startswith("chart_q0_bar_tick_"):
        return "chart_tick"
    if eid in {"chart_q0_strong", "chart_q0_weak"}:
        return "chart_tick"
    if eid.startswith("chart_q0_") or comp == "q0_coverage_panel":
        return "chart_label"
    if eid == "failure_summary_title":
        return "failure_title"
    if eid == "failure_summary_q0_math" or comp_role == "failure_math":
        return "failure_math"
    if eid.startswith("failure_summary_text_"):
        return "failure_body"
    if eid.startswith("action_card_title_report") or comp_role == "report_title":
        return "action_report_title"
    if eid.startswith("action_card_title_") or comp_role == "title":
        return "action_title"
    if comp_role == "body_emphasis":
        return "action_body_emphasis"
    if comp_role == "body_math":
        return "action_body_math"
    if comp_role == "report_body_emphasis":
        return "action_report_body_emphasis"
    if eid.startswith("action_card_body_report") or comp_role == "report_body":
        return "action_report_body"
    if eid.startswith("action_card_body_") or comp_role == "body":
        return "action_body"
    if eid.startswith("bottom_check_text_"):
        return "checklist_body"
    if eid == "proc_ci_title":
        return "ci_title"
    if eid in {"proc_ci_hat_label", "proc_ci_mid_label", "proc_ci_true_label"}:
        return "ci_axis_label"
    if eid in {"proc_vec_beta_label", "proc_vec_gamma_label"}:
        return "surface_vector_math"
    if eid == "proc_vec_theta":
        return "surface_theta_math"
    if eid == "proc_risk_low_overlap_q0":
        return "risk_q_math"
    if eid.startswith("proc_risk_") and el.get("type") == "text":
        return "risk_label"
    if eid.startswith("proc_risk_"):
        return "risk_label"
    if text.strip() in {"X1", "X2", "X3", "x1", "x2", "x3"}:
        return "axis_label"
    if eid.startswith("proc_vec_") or eid.startswith("proc_axis_"):
        return "axis_label"
    if any(k in lower for k in ("low propensity", "weak overlap", "high heterogeneity")):
        return "risk_label"
    if el.get("type") == "formula":
        return "formula"
    if any(ch in text for ch in "βγτθ∇≈⟨⟩∥"):
        return "annotation"
    return "body"


def _style_for(el: dict, role: str) -> TextStyle:
    style = STYLE.get(role, STYLE["body"])
    contract = _text_contract(el)
    if contract:
        return TextStyle(
            str(contract.get("font") or style.font),
            float(contract.get("size") or style.size),
            str(contract.get("color") or style.color),
            str(contract.get("align") or style.align),
            bool(contract.get("bold", style.bold)),
            bool(contract.get("italic", style.italic)),
        )
    eid = str(el.get("id") or "")
    if role in {"action_title", "action_report_title"}:
        for key, color in ACTION_COLORS.items():
            if key in eid:
                return TextStyle(style.font, style.size, color, style.align, style.bold, style.italic)
    if role in {
        "action_body", "action_body_emphasis", "action_body_math",
        "action_report_body", "action_report_body_emphasis",
    }:
        return TextStyle(
            style.font,
            style.size,
            el.get("text_color") or style.color,
            style.align,
            style.bold,
            style.italic,
        )
    if role == "failure_body" and el.get("italic"):
        return TextStyle(style.font, style.size, style.color, style.align, style.bold, True)
    if role in {"risk_label", "risk_label_math", "risk_q_math"}:
        text = str(el.get("text") or "").lower()
        color = "#b84824" if "heterogeneity" in text else "#c83322"
        return TextStyle(style.font, style.size, color, style.align, style.bold, style.italic)
    if role in {"vector_label", "surface_vector_math", "surface_theta_math"}:
        color = el.get("text_color") or style.color
        return TextStyle(style.font, style.size, color, style.align, style.bold, style.italic)
    if role in {"chart_curve_label", "chart_bar_value"}:
        color = el.get("text_color") or style.color
        return TextStyle(style.font, style.size, color, style.align, style.bold, style.italic)
    return style


def _apply_style(el: dict, role: str, style: TextStyle) -> None:
    el["font"] = style.font
    el["font_size"] = style.size
    el["text_color"] = style.color
    el["align"] = style.align
    if style.bold:
        el["bold"] = True
    else:
        el.pop("bold", None)
    if style.italic:
        el["italic"] = True
    else:
        el.pop("italic", None)
    el.setdefault("ext", {}).setdefault("typography", {})
    render_contract = _render_contract(el)
    el["ext"]["typography"].update({
        "role": role,
        "source": "TypographyPlanner",
        **ROLE_RENDERING.get(role, {}),
        **render_contract,
    })


def _apply_calibration(el: dict) -> bool:
    """Apply rendered glyph-ink calibration on top of role slots/styles."""
    calib = (el.get("ext") or {}).get("typography_calibration") or {}
    if not calib:
        return False
    changed = False
    bbox = el.get("bbox")
    if bbox and (calib.get("dx") or calib.get("dy")):
        dx = float(calib.get("dx") or 0.0)
        dy = float(calib.get("dy") or 0.0)
        shifted = [
            float(bbox[0]) + dx,
            float(bbox[1]) + dy,
            float(bbox[2]) + dx,
            float(bbox[3]) + dy,
        ]
        if tuple(round(v, 3) for v in shifted) != tuple(round(float(v), 3) for v in bbox[:4]):
            el["bbox"] = shifted
            changed = True
    scale = float(calib.get("font_scale") or 1.0)
    if abs(scale - 1.0) >= 0.005:
        base = float(el.get("font_size") or 12.0)
        el["font_size"] = round(base * scale, 2)
        changed = True
    color = calib.get("text_color")
    if color and not _text_color_locked(el) and color != el.get("text_color"):
        el["text_color"] = color
        changed = True
    if changed:
        el.setdefault("ext", {}).setdefault("typography", {})["calibrated"] = True
    return changed


def _apply_slot_adjustment(el: dict) -> bool:
    """Replay TemplateSlotAgent corrections after role slots/styles."""
    adjust = (el.get("ext") or {}).get("typography_slot_adjustment") or {}
    if not adjust:
        return False
    changed = False
    bbox = adjust.get("base_bbox") or el.get("bbox")
    if bbox and (adjust.get("dx") or adjust.get("dy")):
        dx = float(adjust.get("dx") or 0.0)
        dy = float(adjust.get("dy") or 0.0)
        shifted = [
            float(bbox[0]) + dx,
            float(bbox[1]) + dy,
            float(bbox[2]) + dx,
            float(bbox[3]) + dy,
        ]
        if tuple(round(v, 3) for v in shifted) != tuple(round(float(v), 3) for v in bbox[:4]):
            el["bbox"] = shifted
            changed = True
    scale = float(adjust.get("font_scale") or 1.0)
    if abs(scale - 1.0) >= 0.005:
        base = float(adjust.get("base_font_size") or el.get("font_size") or 12.0)
        el["font_size"] = round(base * scale, 2)
        changed = True
    color = adjust.get("text_color")
    if color and not _text_color_locked(el) and color != el.get("text_color"):
        el["text_color"] = color
        changed = True
    if changed:
        el.setdefault("ext", {}).setdefault("typography", {})["slot_adjusted"] = True
    return changed


def _slot_for(el: dict, by_id: dict[str, dict], ir: dict) -> list[float] | None:
    eid = str(el.get("id") or "")
    text = str(el.get("text") or el.get("latex") or "").strip()
    lower = text.lower()
    ext = el.get("ext") or {}
    comp_role = str(ext.get("component_role") or "")
    canvas = ir.get("canvas") or {}
    cw = float(canvas.get("width_px") or 0)
    ch = float(canvas.get("height_px") or 0)
    if cw > 0 and ch > 0:
        if lower.startswith("problem:"):
            return [cw * 0.075, ch * 0.010, cw * 0.455, ch * 0.066]
        if lower.startswith("solution:"):
            return [cw * 0.565, ch * 0.010, cw * 0.940, ch * 0.066]
        if "lightweight" in lower and "auditing layer" in lower:
            return [cw * 0.552, ch * 0.052, cw * 0.952, ch * 0.092]
        if "detects geometry-induced" in lower:
            return [cw * 0.250, ch * 0.940, cw * 0.760, ch * 0.988]
    if eid.startswith("auditor_title_"):
        key = eid.removeprefix("auditor_title_")
        card = by_id.get(f"auditor_card_{key}")
        if card and card.get("bbox"):
            x0, y0, x1, y1 = _box(card)
            w, h = x1 - x0, y1 - y0
            return [x0 + w * 0.070, y0 + h * 0.070, x1 - w * 0.070, y0 + h * 0.265]
    if eid.startswith("auditor_formula_"):
        if eid.startswith("auditor_formula_alignment_"):
            return None
        key = eid.removeprefix("auditor_formula_")
        card = by_id.get(f"auditor_card_{key}")
        if card and card.get("bbox"):
            x0, y0, x1, y1 = _box(card)
            w, h = x1 - x0, y1 - y0
            return [x0 + w * 0.050, y0 + h * 0.265, x1 - w * 0.050, y0 + h * 0.445]
    if eid.startswith("action_card_title_"):
        key = eid.removeprefix("action_card_title_")
        card = by_id.get(f"action_card_{key}")
        if card and card.get("bbox"):
            x0, y0, x1, y1 = _box(card)
            w, h = x1 - x0, y1 - y0
            top = 0.345 if key != "report" else 0.330
            bottom = 0.535 if key != "report" else 0.555
            return [x0 + w * 0.07, y0 + h * top, x1 - w * 0.07, y0 + h * bottom]
    if eid.startswith("action_card_body_") and comp_role in {"body", "report_body"}:
        key = eid.removeprefix("action_card_body_")
        card = by_id.get(f"action_card_{key}")
        if card and card.get("bbox"):
            x0, y0, x1, y1 = _box(card)
            w, h = x1 - x0, y1 - y0
            top = 0.545 if key == "retain" else 0.525
            bottom = 0.070 if key == "report" else 0.060
            return [x0 + w * 0.075, y0 + h * top, x1 - w * 0.075, y1 - h * bottom]
    if eid.startswith("failure_summary_text_"):
        panel = by_id.get("failure_summary_panel")
        if panel and panel.get("bbox"):
            row = {"overlap": 0, "undercover": 1, "invisible": 2}
            key = eid.removeprefix("failure_summary_text_")
            idx = row.get(key)
            if idx is not None:
                x0, y0, x1, y1 = _box(panel)
                w, h = x1 - x0, y1 - y0
                cy = y0 + h * (0.36 + idx * 0.245)
                return [x0 + w * 0.205, cy - h * 0.112, x1 - w * 0.020, cy + h * 0.112]
    if eid.startswith("bottom_check_text_"):
        panel = by_id.get("bottom_checklist_panel")
        if panel and panel.get("bbox"):
            idx = int(eid.rsplit("_", 1)[-1])
            x0, y0, x1, y1 = _box(panel)
            h = y1 - y0
            cy = y0 + h * (0.18 + idx * 0.295)
            return [x0 + 46, cy - 19, x1 - 8, cy + 19]
    if eid == "pipeline_context_title":
        cards = [
            e for e in by_id.values()
            if str(e.get("id") or "").startswith("pipeline_context_card_")
            and e.get("bbox")
        ]
        if cards:
            x0 = min(float(e["bbox"][0]) for e in cards)
            y0 = min(float(e["bbox"][1]) for e in cards)
            x1 = max(float(e["bbox"][2]) for e in cards)
            return [x0, y0 - 58, x1, y0 - 12]
    if eid.startswith("pipeline_context_text_"):
        key = eid.removeprefix("pipeline_context_text_")
        card = by_id.get(f"pipeline_context_card_{key}")
        if card and card.get("bbox"):
            x0, y0, x1, y1 = _box(card)
            w, h = x1 - x0, y1 - y0
            return [x0 + w * 0.380, y0 + h * 0.235, x1 - w * 0.045, y0 + h * 0.795]
    return None


def _clamp_font_to_box(el: dict) -> bool:
    bbox = el.get("bbox")
    if not bbox:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    rotation = abs(float(el.get("rotation") or 0.0)) % 180
    if 70 <= rotation <= 110:
        w, h = h, w
    lines = [ln for ln in str(el.get("text") or el.get("latex") or "").splitlines() if ln] or [""]
    longest = max(len(ln) for ln in lines)
    current = float(el.get("font_size") or 12)
    role = ((el.get("ext") or {}).get("typography") or {}).get("role")
    min_size = float(MIN_SIZE.get(str(role), 6.0))
    width_factor = 0.48
    height_factor = 0.74
    if role in {
        "process_card", "auditor_title", "auditor_formula",
        "action_body", "action_body_emphasis", "action_body_math",
        "action_report_body", "action_report_body_emphasis", "checklist_body",
    }:
        width_factor = 0.44
        height_factor = 0.80
    render = ((el.get("ext") or {}).get("typography") or {})
    if render.get("fit_width_factor"):
        width_factor = float(render["fit_width_factor"])
    if render.get("fit_height_factor"):
        height_factor = float(render["fit_height_factor"])
    if role in {
        "chart_title", "chart_axis_label", "chart_curve_label",
        "chart_label", "chart_bar_label", "chart_bar_value", "chart_tick",
    } and "fit_width_factor" not in render:
        width_factor = 0.44
        height_factor = 0.80
    if role in {"formula", "formula_main"}:
        width_factor = 0.34
        height_factor = 0.86
    if role == "vector_label":
        width_factor = 0.42
        height_factor = 0.84
    height_limit = h / max(1, len(lines)) * height_factor
    width_limit = w / max(1, longest) / width_factor
    raw_limit = min(current, height_limit, width_limit)
    limit = max(min_size, min(current, raw_limit))
    if min_size > raw_limit:
        el.setdefault("ext", {}).setdefault("typography", {})["below_min_fit"] = True
    # Titles and math can sit a little tighter vertically.
    if role in {"slide_title", "solution_title", "formula", "formula_main", "caption"}:
        limit = max(float(MIN_SIZE.get(str(role), 8.0)),
                    min(current, h / max(1, len(lines)) * 0.86, width_limit))
    if abs(limit - current) < 0.2:
        return False
    el["font_size"] = round(limit, 2)
    el.setdefault("ext", {}).setdefault("typography", {})["clamped_to_box"] = True
    return True


def _box(el: dict) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in el["bbox"][:4])  # type: ignore[return-value]


def _text_color_locked(el: dict) -> bool:
    ext = el.get("ext") or {}
    if ext.get("text_color_locked"):
        return True
    contract = _text_contract(el)
    if contract.get("text_color_locked"):
        return True
    role = str((ext.get("typography") or {}).get("role") or "")
    return role in {
        "action_body",
        "action_body_emphasis",
        "action_body_math",
        "action_report_body",
        "action_report_body_emphasis",
    }


def _signature(el: dict) -> tuple[Any, ...]:
    return (
        tuple(round(float(v), 3) for v in el.get("bbox", [])[:4]),
        el.get("font"),
        el.get("font_size"),
        el.get("text_color"),
        el.get("align"),
        bool(el.get("bold")),
        bool(el.get("italic")),
    )


def _text_contract(el: dict) -> dict[str, Any]:
    contract = (el.get("ext") or {}).get("text_contract") or {}
    return contract if isinstance(contract, dict) else {}


def _render_contract(el: dict) -> dict[str, Any]:
    contract = _text_contract(el)
    keys = {
        "fit_width_factor",
        "fit_height_factor",
        "line_spacing",
        "word_wrap",
        "margin_px",
        "text_color_locked",
    }
    return {k: contract[k] for k in keys if k in contract and contract[k] is not None}


def _element_role(el: dict) -> str:
    return str((((el.get("ext") or {}).get("typography") or {}).get("role")) or "")


def _role_matches_contract(role: str, contract: dict[str, Any]) -> bool:
    if not role:
        return False
    required = {str(r) for r in contract.get("required_roles") or [] if r}
    if role in required:
        return True
    prefixes = tuple(str(p) for p in contract.get("required_role_prefixes") or [] if p)
    return bool(prefixes and role.startswith(prefixes))


def _text_likely_overflows(el: dict) -> bool:
    text = str(el.get("text") or el.get("latex") or "")
    if not text or not el.get("bbox"):
        return False
    x0, y0, x1, y1 = [float(v) for v in el["bbox"][:4]]
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    rotation = abs(float(el.get("rotation") or 0.0)) % 180
    if 70 <= rotation <= 110:
        w, h = h, w
    lines = [ln for ln in text.splitlines() if ln] or [text]
    size = float(el.get("font_size") or 12.0)
    longest = max(len(ln) for ln in lines)
    typo = ((el.get("ext") or {}).get("typography") or {})
    width_factor = float(
        typo.get("fit_width_factor")
        or (0.45 if el.get("type") == "formula" or _element_role(el) == "formula" else 0.52)
    )
    height_factor = float(typo.get("fit_height_factor") or 0.84)
    estimated_w = longest * size * width_factor
    estimated_h = len(lines) * size / max(0.5, height_factor)
    width_slack = 1.14 if el.get("type") == "formula" or _element_role(el) == "formula" else 1.04
    return estimated_w > w * width_slack or estimated_h > h * 1.08


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
