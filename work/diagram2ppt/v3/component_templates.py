"""Reusable component contracts for native diagram reconstruction.

Specialist agents should not privately decide typography, slot geometry, and
repair policy.  This module is the planner-visible contract for repeated
components: it describes where text belongs inside a component, what native
method owns it, and which visual properties are allowed to move during repair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TextContract:
    """Typography and repair policy for one component text role."""

    role: str
    font: str
    size: float
    color: str
    align: str = "center"
    bold: bool = False
    italic: bool = False
    fit_width_factor: float = 0.38
    fit_height_factor: float = 0.86
    line_spacing: float | None = None
    word_wrap: bool = False
    text_color_locked: bool = True

    def to_ext(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "role": self.role,
            "source": "ComponentTemplate",
            **data,
        }


@dataclass(frozen=True)
class ComponentTemplate:
    """Planner-level contract for a native component family."""

    name: str
    kind: str
    primary_method: str
    preferred_agent: str
    text: dict[str, TextContract]
    fallback_methods: tuple[str, ...] = ("text_style",)
    allowed_repair_agents: tuple[str, ...] = (
        "TextLayoutAgent",
        "TemplateSlotAgent",
        "StyleAgent",
    )
    native_expression: str = "editable native component"

    def strategy(self, region_id: str) -> dict[str, Any]:
        return {
            "region_id": region_id,
            "kind": self.kind,
            "primary_method": self.primary_method,
            "fallback_methods": list(self.fallback_methods),
            "preferred_agent": self.preferred_agent,
            "allowed_repair_agents": list(self.allowed_repair_agents),
            "native_expression": self.native_expression,
        }


ACTION_COLORS = {
    "retain": "#16806e",
    "defer": "#9a5b13",
    "alert": "#cf3d28",
    "report": "#245591",
}


TEMPLATES: dict[str, ComponentTemplate] = {
    "action_card": ComponentTemplate(
        name="action_card",
        kind="component_card_row",
        primary_method="component_layout",
        preferred_agent="ActionCardAgent",
        fallback_methods=("text_style", "native_trace"),
        native_expression=(
            "editable repeated decision cards with shared geometry, icons, "
            "semantic text colors, and routed connectors"
        ),
        text={
            "title": TextContract(
                "action_title", "Arial", 26.0, "#333333",
                align="center", bold=True, fit_width_factor=0.36,
                fit_height_factor=0.86,
            ),
            "report_title": TextContract(
                "action_report_title", "Arial", 24.0, "#333333",
                align="center", bold=True, fit_width_factor=0.36,
                fit_height_factor=0.86,
            ),
            "body": TextContract(
                "action_body", "Arial", 20.5, "#333333",
                align="center", fit_width_factor=0.36,
                fit_height_factor=0.86, line_spacing=0.94,
            ),
            "body_emphasis": TextContract(
                "action_body_emphasis", "Arial", 19.9, "#333333",
                align="center", italic=True, fit_width_factor=0.36,
                fit_height_factor=0.86,
            ),
            "body_math": TextContract(
                "action_body_math", "Times New Roman", 20.2, "#333333",
                align="center", italic=True, fit_width_factor=0.31,
                fit_height_factor=0.88,
            ),
            "report_body": TextContract(
                "action_report_body", "Arial", 19.2, "#245591",
                align="center", fit_width_factor=0.34,
                fit_height_factor=0.84, line_spacing=0.92,
            ),
            "report_body_emphasis": TextContract(
                "action_report_body_emphasis", "Arial", 19.2, "#245591",
                align="center", italic=True, fit_width_factor=0.34,
                fit_height_factor=0.84,
            ),
        },
    ),
    "failure_summary": ComponentTemplate(
        name="failure_summary",
        kind="failure_summary_panel",
        primary_method="failure_summary_layout",
        preferred_agent="FailureSummaryAgent",
        fallback_methods=("text_style", "icon_rebuild"),
        native_expression=(
            "editable summary panel with native icons, title, and locked "
            "readable text rows"
        ),
        text={
            "title": TextContract(
                "failure_title", "Arial", 34.0, "#071a4d",
                align="left", bold=True, fit_width_factor=0.40,
                fit_height_factor=0.88, line_spacing=0.92,
            ),
            "text": TextContract(
                "failure_body", "Arial", 25.0, "#111827",
                align="left", fit_width_factor=0.39,
                fit_height_factor=0.90, line_spacing=0.92,
                word_wrap=True,
            ),
            "math": TextContract(
                "failure_math", "Times New Roman", 25.0, "#111827",
                align="left", italic=True, fit_width_factor=0.30,
                fit_height_factor=0.90,
            ),
        },
    ),
    "coverage_chart_panel": ComponentTemplate(
        name="coverage_chart_panel",
        kind="chart",
        primary_method="chart_parser",
        preferred_agent="ChartAgent",
        fallback_methods=("native_trace", "text_style"),
        native_expression=(
            "editable chart panel with native axes, line series, bar series, "
            "ticks, title, labels, and semantic colors"
        ),
        text={
            "title_q": TextContract(
                "chart_title_q", "Times New Roman", 36.0, "#071a4d",
                align="right", bold=True, fit_width_factor=0.36,
                fit_height_factor=0.88, line_spacing=0.92,
            ),
            "title_sub": TextContract(
                "chart_title_sub", "Times New Roman", 22.0, "#071a4d",
                align="center", bold=True, fit_width_factor=0.34,
                fit_height_factor=0.88, line_spacing=0.92,
            ),
            "title_rest": TextContract(
                "chart_title_rest", "Times New Roman", 36.0, "#071a4d",
                align="left", bold=True, fit_width_factor=0.36,
                fit_height_factor=0.88, line_spacing=0.92,
            ),
            "axis_label": TextContract(
                "chart_axis_label", "Times New Roman", 21.0, "#111111",
                fit_width_factor=0.31, fit_height_factor=0.88,
                line_spacing=0.90,
            ),
            "curve_label": TextContract(
                "chart_curve_label", "Arial", 20.5, "#333333",
                fit_width_factor=0.34, fit_height_factor=0.86,
                line_spacing=0.90,
            ),
            "tick": TextContract(
                "chart_tick", "Times New Roman", 16.5, "#333333",
                fit_width_factor=0.31, fit_height_factor=0.88,
                line_spacing=0.90,
            ),
            "bar_label": TextContract(
                "chart_bar_label", "Arial", 16.5, "#333333",
                fit_width_factor=0.36, fit_height_factor=0.82,
                line_spacing=0.84,
            ),
            "bar_value": TextContract(
                "chart_bar_value", "Times New Roman", 18.0, "#111111",
                fit_width_factor=0.34, fit_height_factor=0.86,
                line_spacing=0.90,
            ),
        },
    ),
}


def template(name: str) -> ComponentTemplate:
    return TEMPLATES[name]


def text_contract(component: str, role: str, key: str = "") -> TextContract | None:
    tmpl = TEMPLATES.get(component)
    if not tmpl:
        return None
    contract = tmpl.text.get(role)
    if not contract:
        return None
    if component == "action_card":
        return _action_contract(contract, role, key)
    return contract


def apply_text_contract(
    el: dict[str, Any],
    component: str,
    role: str,
    key: str = "",
    **overrides: Any,
) -> None:
    contract = text_contract(component, role, key)
    if not contract:
        return
    contract_data = contract.to_ext()
    contract_data.update({k: v for k, v in overrides.items() if v is not None})
    el["font"] = str(contract_data.get("font") or contract.font)
    el["font_size"] = float(contract_data.get("size") or contract.size)
    el["text_color"] = str(contract_data.get("color") or contract.color)
    el["align"] = str(contract_data.get("align") or contract.align)
    bold = bool(contract_data.get("bold"))
    italic = bool(contract_data.get("italic"))
    if bold:
        el["bold"] = True
    else:
        el.pop("bold", None)
    if italic:
        el["italic"] = True
    else:
        el.pop("italic", None)
    ext = el.setdefault("ext", {})
    ext["component_template"] = component
    ext["text_contract"] = contract_data
    if bool(contract_data.get("text_color_locked")):
        ext["text_color_locked"] = True


def component_ext(component: str, role: str, key: str = "",
                  region_id: str = "", **overrides: Any) -> dict[str, Any]:
    tmpl = TEMPLATES[component]
    ext = {
        "component": component,
        "component_template": component,
        "component_key": key,
        "component_role": role,
        "strategy": tmpl.strategy(region_id or _default_region_id(component)),
    }
    contract = text_contract(component, role, key)
    if contract:
        contract_data = contract.to_ext()
        contract_data.update({k: v for k, v in overrides.items() if v is not None})
        ext["text_contract"] = contract_data
        if bool(contract_data.get("text_color_locked")):
            ext["text_color_locked"] = True
    return ext


def _default_region_id(component: str) -> str:
    return {
        "action_card": "region_action_cards",
        "failure_summary": "failure_summary",
        "coverage_chart_panel": "q0_coverage_charts",
    }.get(component, f"region_{component}")


def _action_contract(base: TextContract, role: str, key: str) -> TextContract:
    color = base.color
    if role in {"title", "report_title"} and key in ACTION_COLORS:
        color = ACTION_COLORS[key]
    if role in {"body", "report_body", "report_body_emphasis"}:
        if key == "retain" and role == "body":
            color = "#16806e"
        elif key == "defer" and role == "body":
            color = "#9a5b13"
        elif key == "alert" and role == "body":
            color = "#cf3d28"
        elif key == "report":
            color = "#245591"
    return TextContract(
        role=base.role,
        font=base.font,
        size=base.size,
        color=color,
        align=base.align,
        bold=base.bold,
        italic=base.italic,
        fit_width_factor=base.fit_width_factor,
        fit_height_factor=base.fit_height_factor,
        line_spacing=base.line_spacing,
        word_wrap=base.word_wrap,
        text_color_locked=base.text_color_locked,
    )
