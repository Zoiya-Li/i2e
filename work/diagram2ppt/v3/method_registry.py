"""Reusable method registry for native diagram reconstruction.

The planner should choose a reconstruction method before specialist agents
touch the IR.  This module is the single source of truth for method contracts,
agent constraints, and generic semantic routing.  It intentionally works from
region kind and visual problem text instead of per-image case names.
"""
from __future__ import annotations

from typing import Any


CONTRACTS: dict[str, dict[str, Any]] = {
    "procedural_surface": {
        "representation": "procedural_3d",
        "owner_agent": "ProceduralSurfaceAgent",
        "required_agents": ["ProceduralSurfaceAgent"],
        "forbid_agents": ["ShapeAgent", "SurfaceAgent", "VectorizeAgent"],
        "acceptance_policy": "method_locked_visual",
        "visual_evidence": [
            "local triptych improves or stays close",
            "surface has editable wave bands, contours, scatter clusters, axes, vector arrows, risk ring, and CI inset",
        ],
        "native_expression": "editable generated vector surface; no raster crop and no generic residual trace",
    },
    "chart_parser": {
        "representation": "native_chart",
        "component_template": "coverage_chart_panel",
        "owner_agent": "ChartAgent",
        "required_agents": ["ChartAgent"],
        "forbid_agents": ["ShapeAgent"],
        "acceptance_policy": "semantic_chart",
        "visual_evidence": [
            "axes, ticks, labels, line series, bars, and panel title are native and spatially complete",
        ],
        "native_expression": "editable chart axes, series, bars, ticks, and labels",
    },
    "pipeline_context_layout": {
        "representation": "flow_pipeline",
        "component_template": "process_pipeline",
        "owner_agent": "PipelineContextAgent",
        "required_agents": ["PipelineContextAgent"],
        "forbid_agents": ["ShapeAgent"],
        "acceptance_policy": "component_visual",
        "visual_evidence": [
            "process cards, icons, labels, and arrows are aligned as one row",
        ],
        "native_expression": "editable process-row cards, icons, text, and connectors",
    },
    "auditor_card_layout": {
        "representation": "method_card_system",
        "component_template": "auditor_card",
        "owner_agent": "AuditorCardAgent",
        "required_agents": ["AuditorCardAgent"],
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "acceptance_policy": "component_visual",
        "visual_evidence": [
            "cards, numbered badges, formulas, and internal mini diagrams are complete",
        ],
        "native_expression": "editable repeated method cards with internal diagrams",
    },
    "component_layout": {
        "representation": "action_card_system",
        "component_template": "action_card",
        "owner_agent": "ActionCardAgent",
        "required_agents": ["ActionCardAgent"],
        "forbid_agents": ["ShapeAgent"],
        "acceptance_policy": "component_visual",
        "visual_evidence": [
            "repeated cards share geometry, typography, icons, and connectors",
        ],
        "native_expression": "editable repeated action cards",
    },
    "failure_summary_layout": {
        "representation": "summary_panel",
        "component_template": "failure_summary",
        "owner_agent": "FailureSummaryAgent",
        "required_agents": ["FailureSummaryAgent"],
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "acceptance_policy": "component_visual",
        "visual_evidence": [
            "summary title, icons, and readable rows are present",
        ],
        "native_expression": "editable icon-and-text failure summary panel",
    },
    "mini_surface_checklist": {
        "representation": "mini_surface_checklist",
        "component_template": "mini_surface_checklist",
        "owner_agent": "BottomMiniSurfaceAgent",
        "required_agents": ["BottomMiniSurfaceAgent"],
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "acceptance_policy": "component_visual",
        "visual_evidence": [
            "mini manifold, vector arrows, dashed checklist, and checks are native",
        ],
        "native_expression": "editable mini manifold and checklist component",
    },
    "cross_panel_bridge": {
        "representation": "semantic_connector",
        "owner_agent": "CrossPanelBridgeAgent",
        "required_agents": ["CrossPanelBridgeAgent"],
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "acceptance_policy": "component_visual",
        "visual_evidence": [
            "a native wide arrow connects semantic panels",
        ],
        "native_expression": "editable block-arrow connector between semantic panels",
    },
}


METHOD_POLICIES: dict[str, dict[str, Any]] = {
    "procedural_surface": {
        "locked_method": "procedural_surface",
        "required_agents": ["ProceduralSurfaceAgent"],
        "fallback_agents": ["StyleAgent"],
        "acceptance_policy": "method_locked_visual",
        "forbid_agents": ["ShapeAgent", "SurfaceAgent", "VectorizeAgent"],
        "native_expression": (
            "editable generated surface bands, streamlines, scatter clusters, "
            "axis arrows, and labels; no raster crop or generic contour trace"
        ),
    },
    "chart_parser": {
        "locked_method": "chart_parser",
        "required_agents": ["ChartAgent"],
        "fallback_agents": [
            "StyleAgent", "TextAgent", "TextLayoutAgent", "TemplateSlotAgent",
        ],
        "acceptance_policy": "semantic_chart",
        "forbid_agents": ["ShapeAgent"],
        "native_expression": (
            "editable chart axes, ticks, labels, and series; trace only for "
            "minor residual series details"
        ),
    },
    "auditor_card_layout": {
        "locked_method": "auditor_card_layout",
        "required_agents": ["AuditorCardAgent"],
        "fallback_agents": [
            "LayoutAgent", "ChartAgent", "TextAgent", "TextLayoutAgent",
            "TemplateSlotAgent", "StyleAgent",
        ],
        "acceptance_policy": "component_visual",
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "native_expression": (
            "editable method-card components with text, icons, mini charts, "
            "and connectors as native primitives"
        ),
    },
    "pipeline_context_layout": {
        "locked_method": "pipeline_context_layout",
        "required_agents": ["PipelineContextAgent"],
        "fallback_agents": [
            "LayoutAgent", "IconAgent", "ConnectorAgent", "TextAgent",
            "TextLayoutAgent", "TemplateSlotAgent", "StyleAgent",
        ],
        "acceptance_policy": "component_visual",
        "forbid_agents": ["ShapeAgent"],
        "native_expression": "aligned editable process cards, icons, labels, and connectors",
    },
    "component_layout": {
        "locked_method": "component_layout",
        "required_agents": ["ActionCardAgent"],
        "fallback_agents": [
            "LayoutAgent", "IconAgent", "ConnectorAgent", "TextAgent",
            "TextLayoutAgent", "TemplateSlotAgent", "StyleAgent",
        ],
        "acceptance_policy": "component_visual",
        "forbid_agents": ["ShapeAgent"],
        "native_expression": "editable repeated card components with shared geometry and styles",
    },
    "failure_summary_layout": {
        "locked_method": "failure_summary_layout",
        "required_agents": ["FailureSummaryAgent"],
        "fallback_agents": [
            "TextAgent", "TextLayoutAgent", "TemplateSlotAgent", "IconAgent",
            "StyleAgent",
        ],
        "acceptance_policy": "component_visual",
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "native_expression": "editable summary panel with title, icons, and readable text rows",
    },
    "mini_surface_checklist": {
        "locked_method": "mini_surface_checklist",
        "required_agents": ["BottomMiniSurfaceAgent"],
        "fallback_agents": [
            "SurfaceAgent", "TextAgent", "TextLayoutAgent", "TemplateSlotAgent",
            "IconAgent", "StyleAgent",
        ],
        "acceptance_policy": "component_visual",
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "native_expression": "editable mini manifold, vector arrows, and checklist components",
    },
    "cross_panel_bridge": {
        "locked_method": "cross_panel_bridge",
        "required_agents": ["CrossPanelBridgeAgent"],
        "fallback_agents": ["ConnectorAgent", "StyleAgent"],
        "acceptance_policy": "component_visual",
        "forbid_agents": ["ShapeAgent", "VectorizeAgent"],
        "native_expression": "editable semantic connector between panels",
    },
}


METHOD_BY_KIND = {
    "procedural_3d_surface": "procedural_surface",
    "chart": "chart_parser",
    "auditor_method_cards": "auditor_card_layout",
    "pipeline_context_row": "pipeline_context_layout",
    "component_card_row": "component_layout",
    "failure_summary_panel": "failure_summary_layout",
    "bottom_mini_surface": "mini_surface_checklist",
    "cross_panel_bridge": "cross_panel_bridge",
}


# Legacy region aliases are only hints.  Generic text/kind routing below must
# work even when a VLM invents new region ids for a new diagram.
REGION_ID_HINTS = {
    "left_surface": "procedural_surface",
    "surface": "procedural_surface",
    "manifold": "procedural_surface",
    "coverage_chart": "chart_parser",
    "q0_coverage": "chart_parser",
    "pipeline": "pipeline_context_layout",
    "flow": "pipeline_context_layout",
    "auditor": "auditor_card_layout",
    "action_card": "component_layout",
    "summary": "failure_summary_layout",
    "mini_surface": "mini_surface_checklist",
    "bridge": "cross_panel_bridge",
}


TEXT_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "procedural_surface",
        ("3d", "surface", "manifold", "coordinate space", "vector field"),
        ("chart", "bar", "legend"),
    ),
    (
        "chart_parser",
        ("chart", "axis", "tick", "legend", "series", "subplot", "bar", "line plot"),
        (),
    ),
    (
        "pipeline_context_layout",
        ("pipeline", "flow", "connector", "arrow", "process row", "horizon"),
        ("summary",),
    ),
    (
        "auditor_card_layout",
        ("method card", "auditor", "card row", "numbered badge"),
        (),
    ),
    (
        "component_layout",
        ("action card", "repeated card", "card layout", "retain", "defer", "alert"),
        (),
    ),
    (
        "failure_summary_layout",
        ("failure summary", "summary panel", "warning", "failure mode"),
        (),
    ),
    (
        "mini_surface_checklist",
        ("mini surface", "checklist", "mini manifold"),
        (),
    ),
    (
        "cross_panel_bridge",
        ("cross-panel", "wide arrow", "panel bridge", "between panels"),
        (),
    ),
]


def contract_for_method(method: str | None) -> dict[str, Any]:
    """Return a detached contract dict for a method."""
    if not method:
        return {}
    contract = CONTRACTS.get(str(method), {})
    if not contract:
        return {}
    out = dict(contract)
    out["method"] = str(method)
    return out


def policy_for_method(method: str | None) -> dict[str, Any]:
    """Return a detached policy dict for a method."""
    if not method:
        return {}
    policy = dict(METHOD_POLICIES.get(str(method), {}))
    if policy:
        policy["method"] = str(method)
    return policy


def infer_method(
    kind: str | None = None,
    region_id: str | None = None,
    objective: str | None = None,
    expected_native_expression: str | None = None,
    visual_problem: str | None = None,
) -> str:
    """Infer a reconstruction method from generic semantics.

    The order is deliberate: stable region kinds are strongest; free-form VLM
    text comes next; region ids are only weak compatibility hints.
    """
    if kind and str(kind) in METHOD_BY_KIND:
        return METHOD_BY_KIND[str(kind)]

    text = " ".join(
        str(v or "")
        for v in (kind, objective, expected_native_expression, visual_problem)
    ).lower()
    for method, required, excluded in TEXT_RULES:
        if required and any(term in text for term in required):
            if excluded and any(term in text for term in excluded):
                continue
            return method

    rid = str(region_id or "").lower()
    for hint, method in REGION_ID_HINTS.items():
        if hint in rid:
            return method
    return ""


def policy_for_region(
    kind: str | None = None,
    region_id: str | None = None,
    objective: str | None = None,
    expected_native_expression: str | None = None,
    visual_problem: str | None = None,
) -> dict[str, Any]:
    method = infer_method(
        kind=kind,
        region_id=region_id,
        objective=objective,
        expected_native_expression=expected_native_expression,
        visual_problem=visual_problem,
    )
    return policy_for_method(method)
