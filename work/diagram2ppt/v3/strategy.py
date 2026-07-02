"""Planner-level reconstruction strategy decisions.

The planner should decide *how* each visual region is rebuilt before agents
start changing the IR.  This module is the explicit policy layer between
perception and candidate generation: it maps detected regions/elements to a
method family, preferred agent, and fallback chain.
"""
from __future__ import annotations

import os
from collections import Counter

from . import representation_plan


def plan_from_entities(entities: list[dict], width: int, height: int) -> dict:
    """Build a high-level reconstruction plan from detected entities."""
    regions: list[dict] = []
    counts = Counter(e.get("type") for e in entities)

    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        bw, bh = x1 - x0, y1 - y0
        typ = e.get("type")

        if typ == "surface" and bw > width * 0.24 and bh > height * 0.16 and x0 < width * 0.55:
            regions.append({
                "id": f"region_surface_{len(regions)}",
                "kind": "procedural_3d_surface",
                "bbox": [x0, y0, x1, y1],
                "element_ids": [str(e.get("id", ""))],
                "primary_method": "procedural_surface",
                "fallback_methods": ["surface_vector_trace", "residual_replacement"],
                "preferred_agent": "ProceduralSurfaceAgent",
                "reason": "large left scientific manifold is parametric, not a generic traced contour",
            })

        elif typ == "chart":
            regions.append({
                "id": f"region_chart_{len(regions)}",
                "kind": "chart",
                "bbox": [x0, y0, x1, y1],
                "element_ids": [str(e.get("id", ""))],
                "primary_method": "chart_parser",
                "fallback_methods": ["native_trace"],
                "preferred_agent": "ChartAgent",
                "reason": "charts need semantic axes/series when possible, trace only as fallback",
            })

    inferred_surfaces = _generic_scientific_surface_regions(entities, width, height, regions)
    regions.extend(inferred_surfaces)
    generic_flows = _generic_flow_pipeline_regions(entities, width, height, regions)
    regions.extend(generic_flows)
    action_cards = _action_card_regions(entities, width, height)
    regions.extend(action_cards)
    failure_regions = _failure_summary_regions(entities, width, height)
    regions.extend(failure_regions)
    q0_regions = _q0_coverage_regions(entities, width, height)
    regions.extend(q0_regions)
    bottom_regions = _bottom_mini_surface_regions(entities, width, height)
    regions.extend(bottom_regions)
    pipeline_regions = _pipeline_context_regions(entities, width, height)
    regions.extend(pipeline_regions)
    auditor_regions = _auditor_card_regions(entities, width, height)
    regions.extend(auditor_regions)
    bridge_regions = _cross_panel_bridge_regions(
        regions, entities, width, height)
    regions.extend(bridge_regions)
    representation_plan.attach_to_regions(regions)

    candidate_policy = {
        "try_procedural_surfaces": any(r["kind"] == "procedural_3d_surface" for r in regions),
        "try_component_motifs": bool(action_cards),
        "try_pipeline_context": bool(pipeline_regions or generic_flows),
        "try_auditor_cards": bool(auditor_regions),
        "try_failure_summary": bool(failure_regions),
        "try_q0_coverage": bool(q0_regions),
        "try_bottom_mini_surface": bool(bottom_regions),
        "try_cross_panel_bridge": bool(bridge_regions),
        "try_residual_replacement": _initial_residual_enabled(),
        "max_base_candidates": 4,
        "procedural_candidate_limit": 2,
        "motif_candidate_limit": 2,
        "residual_candidate_limit": 0 if not _initial_residual_enabled() else 5,
    }

    return {
        "version": "strategy-v1",
        "summary": {
            "entity_counts": dict(counts),
            "regions": len(regions),
        },
        "regions": regions,
        "representation_plan": representation_plan.from_regions(regions, width, height),
        "candidate_policy": candidate_policy,
    }


def _initial_residual_enabled() -> bool:
    """Residual freeform completion is an explicit fallback, not a default plan.

    It can improve pixel coverage while producing editable-but-unsemantic
    trace fragments.  The planner should try specialist native methods first;
    enable this only for diagnostic runs or inputs with no usable semantic
    reconstruction path.
    """
    return os.environ.get("I2E_ENABLE_INITIAL_RESIDUAL", "0") == "1"


def apply_ir_strategy(ir: dict, plan: dict | None) -> None:
    """Annotate IR elements with their planned reconstruction method."""
    if not plan:
        return
    ir["strategy_plan"] = plan
    elements = {str(e.get("id")): e for e in ir.get("elements", [])}
    for region in plan.get("regions", []):
        for eid in region.get("element_ids", []):
            el = elements.get(str(eid))
            if not el:
                continue
            el.setdefault("ext", {})["strategy"] = {
                "region_id": region.get("id"),
                "kind": region.get("kind"),
                "primary_method": region.get("primary_method"),
                "component_template": region.get("component_template")
                or (region.get("representation") or {}).get("component_template"),
                "fallback_methods": list(region.get("fallback_methods") or []),
                "preferred_agent": region.get("preferred_agent"),
                "representation": region.get("representation") or {},
            }
            if region.get("representation"):
                el.setdefault("ext", {})["representation"] = region["representation"]


def apply_defect_strategy(ir: dict) -> None:
    """Route defects according to element-level strategy annotations."""
    elements = {str(e.get("id")): e for e in ir.get("elements", [])}
    for defect in ir.get("defects", []):
        el = elements.get(str(defect.get("element_id") or ""))
        strategy = (el.get("ext") or {}).get("strategy") if el else None
        if not strategy:
            _route_ownerless_region_defect(ir, defect)
            continue
        if defect.get("type") == "text_layout_mismatch":
            defect["suggested_agent"] = "TextLayoutAgent"
            defect.setdefault("strategy", {}).update({
                "method": strategy.get("primary_method"),
                "region_id": strategy.get("region_id"),
                "component_template": strategy.get("component_template"),
                "fallback_methods": strategy.get("fallback_methods", []),
            })
            continue
        if defect.get("type") == "text_template_mismatch":
            defect["suggested_agent"] = "TemplateSlotAgent"
            defect.setdefault("strategy", {}).update({
                "method": strategy.get("primary_method"),
                "region_id": strategy.get("region_id"),
                "component_template": strategy.get("component_template"),
                "fallback_methods": strategy.get("fallback_methods", []),
            })
            continue
        preferred = strategy.get("preferred_agent")
        method = strategy.get("primary_method")
        if preferred:
            defect["suggested_agent"] = preferred
        defect.setdefault("strategy", {}).update({
            "method": method,
            "region_id": strategy.get("region_id"),
            "component_template": strategy.get("component_template"),
            "fallback_methods": strategy.get("fallback_methods", []),
        })


def _route_ownerless_region_defect(ir: dict, defect: dict) -> None:
    if defect.get("element_id") or not defect.get("bbox"):
        return
    for region in (ir.get("strategy_plan") or {}).get("regions", []):
        if region.get("kind") not in {
            "pipeline_context_row",
            "auditor_method_cards",
            "failure_summary_panel",
            "chart",
            "component_card_row",
            "bottom_mini_surface",
            "cross_panel_bridge",
        }:
            continue
        if _bbox_overlap_fraction(defect.get("bbox"), region.get("bbox")) < 0.25:
            continue
        representation = region.get("representation") or {}
        method = (
            representation.get("method")
            or region.get("primary_method")
            or representation_plan.contract_for_method(
                region.get("primary_method")).get("method")
        )
        defect["suggested_agent"] = (
            representation.get("owner_agent")
            or region.get("preferred_agent")
            or "LayoutAgent"
        )
        defect.setdefault("strategy", {}).update({
            "method": method,
            "region_id": region.get("id"),
            "component_template": region.get("component_template")
            or representation.get("component_template"),
            "fallback_methods": region.get("fallback_methods", []),
            "representation": representation,
        })
        return


def candidate_specs(plan: dict | None, processed_candidates: list[tuple[str, list[dict]]]) -> list[dict]:
    """Return candidate build specs directed by the reconstruction plan."""
    policy = (plan or {}).get("candidate_policy", {})
    proc_limit = int(policy.get("procedural_candidate_limit", 0))
    motif_limit = int(policy.get("motif_candidate_limit", 0))
    max_base = int(policy.get("max_base_candidates", len(processed_candidates)))

    specs: list[dict] = []
    for idx, (name, entities) in enumerate(processed_candidates[:max_base]):
        specs.append({
            "name": name,
            "entities": entities,
            "component_motifs": False,
            "procedural_surfaces": False,
            "residual_replacement": bool(policy.get("try_residual_replacement", True)),
        })
        if policy.get("try_procedural_surfaces") and idx < proc_limit:
            specs.append({
                "name": f"{name}_procedural",
                "entities": entities,
                "component_motifs": False,
                "procedural_surfaces": True,
                "residual_replacement": True,
            })
        if policy.get("try_component_motifs") and idx < motif_limit:
            specs.append({
                "name": f"{name}_motif",
                "entities": entities,
                "component_motifs": True,
                "procedural_surfaces": False,
                "residual_replacement": True,
            })
        if (
            policy.get("try_procedural_surfaces")
            and policy.get("try_component_motifs")
            and idx < min(proc_limit, motif_limit)
        ):
            specs.append({
                "name": f"{name}_procedural_motif",
                "entities": entities,
                "component_motifs": True,
                "procedural_surfaces": True,
                "residual_replacement": True,
            })
    return specs


def residual_allowed(plan: dict | None, spec: dict, index: int) -> bool:
    policy = (plan or {}).get("candidate_policy", {})
    if not spec.get("residual_replacement", True):
        return False
    return index < int(policy.get("residual_candidate_limit", 4))


def _generic_scientific_surface_regions(
    entities: list[dict],
    width: int,
    height: int,
    existing_regions: list[dict],
) -> list[dict]:
    """Infer a procedural 3D/scientific surface when CV missed it.

    This is intentionally semantic and geometric, not image-specific: labels
    such as covariate/vector/gradient/heterogeneity around a large upper-left
    scientific panel indicate that generic traces will not be adequate.
    """
    if any(r.get("kind") == "procedural_3d_surface" for r in existing_regions):
        return []
    markers = []
    keywords = (
        "covariate", "manifold", "surface", "gradient", "heterogeneity",
        "propensity", "overlap", "theta", "nabla", "vector",
        "vr(", "ve(", "x1", "x2", "x3", "∇",
    )
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if x0 > width * 0.62 or y0 > height * 0.72:
            continue
        text = str(e.get("text") or e.get("latex") or "").lower().replace(" ", "")
        spaced = str(e.get("text") or e.get("latex") or "").lower()
        if not text and e.get("type") not in {"formula", "surface", "dotcloud"}:
            continue
        formula_in_science_slot = (
            e.get("type") == "formula"
            and x0 < width * 0.48
            and y0 < height * 0.58
        )
        if any(k in text or k in spaced for k in keywords) or formula_in_science_slot:
            markers.append(e)
    if len(markers) < 3:
        return []

    x0 = max(0.0, min(float(e["bbox"][0]) for e in markers) - width * 0.08)
    y0 = max(height * 0.08, min(float(e["bbox"][1]) for e in markers) - height * 0.06)
    x1 = min(width * 0.53, max(float(e["bbox"][2]) for e in markers) + width * 0.18)
    y1 = min(height * 0.64, max(float(e["bbox"][3]) for e in markers) + height * 0.20)
    if (x1 - x0) * (y1 - y0) < width * height * 0.08:
        return []
    region = [x0, y0, x1, y1]
    if _overlaps_region(region, existing_regions, min_overlap=0.45):
        return []
    return [{
        "id": "region_scientific_surface_0",
        "kind": "procedural_3d_surface",
        "bbox": region,
        "element_ids": [str(e.get("id", "")) for e in markers],
        "primary_method": "procedural_surface",
        "fallback_methods": ["surface_vector_trace", "text_style"],
        "preferred_agent": "ProceduralSurfaceAgent",
        "reason": "scientific surface semantics require procedural native geometry, not residual tracing",
    }]


def _generic_flow_pipeline_regions(
    entities: list[dict],
    width: int,
    height: int,
    existing_regions: list[dict],
) -> list[dict]:
    """Infer multi-step flow/pipeline regions from repeated blocks."""
    out = []
    containers = [
        e for e in entities
        if e.get("bbox") and e.get("type") in {"container", "rounded_rect", "rect"}
    ]
    block_types = {"shape", "rounded_rect", "rect", "chart", "container"}
    for idx, container in enumerate(containers):
        cb = [float(v) for v in container["bbox"][:4]]
        cx0, cy0, cx1, cy1 = cb
        cw, ch = cx1 - cx0, cy1 - cy0
        if cw < width * 0.25 or ch < height * 0.14:
            continue
        if cy0 > height * 0.62:
            continue
        blocks = []
        for e in entities:
            if e is container or e.get("type") not in block_types or not e.get("bbox"):
                continue
            bbox = [float(v) for v in e["bbox"][:4]]
            if _bbox_overlap_fraction(bbox, cb) < 0.55:
                continue
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if bh < ch * 0.18 or bw > cw * 0.42:
                continue
            blocks.append(e)
        if len(blocks) < 3:
            continue
        centers = sorted((float(e["bbox"][0]) + float(e["bbox"][2])) / 2 for e in blocks)
        if centers[-1] - centers[0] < cw * 0.35:
            continue
        text_items = [
            e for e in entities
            if e.get("bbox") and e.get("type") in {"text", "formula"}
            and _bbox_overlap_fraction([float(v) for v in e["bbox"][:4]], cb) > 0.20
        ]
        flow_text = " ".join(
            str(e.get("text") or e.get("latex") or "").lower()
            for e in text_items
        )
        flow_keywords = (
            "architecture", "pipeline", "flow", "decomposition",
            "projection", "recombination", "preprocessing", "input",
            "output", "horizon", "component", "adaptive",
        )
        if cy0 > height * 0.25 and not any(k in flow_text for k in flow_keywords):
            continue
        region = [
            max(0.0, cx0 - width * 0.01),
            max(0.0, cy0 - height * 0.02),
            min(float(width), cx1 + width * 0.01),
            min(float(height), cy1 + height * 0.02),
        ]
        if _overlaps_region(region, existing_regions + out, min_overlap=0.50):
            continue
        out.append({
            "id": f"region_flow_pipeline_{idx}",
            "kind": "pipeline_context_row",
            "bbox": region,
            "element_ids": [str(e.get("id", "")) for e in blocks + text_items],
            "primary_method": "pipeline_context_layout",
            "component_template": "process_pipeline",
            "fallback_methods": ["shape_recovery", "text_style", "connector_rebuild"],
            "preferred_agent": "PipelineContextAgent",
            "reason": "repeated blocks in one container form a flow/pipeline component",
        })
    return out


def _action_card_regions(entities: list[dict], width: int, height: int) -> list[dict]:
    cards = []
    labels = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if y0 < height * 0.55:
            continue
        if e.get("type") == "rounded_rect" and 80 <= x1 - x0 <= 260 and y1 - y0 >= 180:
            cards.append(e)
        txt = str(e.get("text") or "").lower()
        if any(k in txt for k in ("retain", "defer", "alert", "reliability")):
            labels.append(e)
    if len(cards) + len(labels) < 3:
        return []
    x0 = min(float(e["bbox"][0]) for e in cards + labels if e.get("bbox"))
    y0 = min(float(e["bbox"][1]) for e in cards + labels if e.get("bbox"))
    x1 = max(float(e["bbox"][2]) for e in cards + labels if e.get("bbox"))
    y1 = max(float(e["bbox"][3]) for e in cards + labels if e.get("bbox"))
    return [{
        "id": "region_action_cards",
        "kind": "component_card_row",
        "bbox": [x0, y0, x1, y1],
        "element_ids": [str(e.get("id", "")) for e in cards + labels],
        "primary_method": "component_layout",
        "component_template": "action_card",
        "fallback_methods": ["text_style", "residual_replacement"],
        "preferred_agent": "LayoutAgent",
        "reason": "bottom action cards are repeated components with shared geometry",
    }]


def _failure_summary_regions(entities: list[dict], width: int,
                             height: int) -> list[dict]:
    """Detect the bottom-left failure-summary panel as one component."""
    items = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if not (x1 <= width * 0.42 and height * 0.54 <= y0 <= height * 0.92):
            continue
        text = str(e.get("text") or e.get("latex") or "").lower()
        typ = e.get("type")
        if any(k in text for k in (
            "failure", "summary", "low overlap", "honest ci",
            "undercover", "invisible", "monitoring", "effect variation",
        )):
            items.append(e)
            continue
        if typ in {"rounded_rect", "rect", "icon", "oval"} and x0 < width * 0.26:
            items.append(e)
    if len(items) < 2:
        return []
    x0 = 0.0
    y0 = height * 0.640
    x1 = width * 0.180
    y1 = height * 0.930
    return [{
        "id": "failure_summary",
        "kind": "failure_summary_panel",
        "bbox": [x0, y0, x1, y1],
        "element_ids": [str(e.get("id", "")) for e in items],
        "primary_method": "failure_summary_layout",
        "component_template": "failure_summary",
        "fallback_methods": ["text_style", "icon_rebuild"],
        "preferred_agent": "FailureSummaryAgent",
        "reason": "bottom-left failure summary is a fixed icon/text component, not scattered OCR fragments",
    }]


def _q0_coverage_regions(entities: list[dict], width: int,
                         height: int) -> list[dict]:
    """Detect the Q0 coverage panel as one semantic chart component."""
    items = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if not (width * 0.20 <= x0 <= width * 0.55
                and height * 0.54 <= y0 <= height * 0.91):
            continue
        text = str(e.get("text") or e.get("latex") or "").lower()
        typ = e.get("type")
        if any(k in text for k in (
            "q0", "q₀", "coverage", "orthogonal", "aligned",
            "overlap quantile", "strong", "weak",
        )):
            items.append(e)
            continue
        if typ in {"chart", "line", "arrow", "dotcloud", "rect"} and width * 0.24 <= x0 <= width * 0.49:
            items.append(e)
    if len(items) < 3:
        return []
    # The chart agent owns the complete rounded panel, not only the line plot
    # crop.  This bbox is the component input region from which the native
    # panel, line chart, and bar chart slots are derived.
    x0 = width * 0.180
    y0 = height * 0.640
    x1 = width * 0.480
    y1 = height * 0.930
    return [{
        "id": "q0_coverage_charts",
        "kind": "chart",
        "bbox": [x0, y0, x1, y1],
        "element_ids": [str(e.get("id", "")) for e in items],
        "primary_method": "chart_parser",
        "component_template": "coverage_chart_panel",
        "fallback_methods": ["native_trace", "text_style"],
        "preferred_agent": "ChartAgent",
        "reason": "Q0 coverage collapse is a two-chart panel that must be rebuilt as one native chart component",
    }]


def _bottom_mini_surface_regions(entities: list[dict], width: int,
                                 height: int) -> list[dict]:
    """Detect the lower mini-manifold plus cheap-nuisance checklist component."""
    items = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if not (width * 0.30 <= x0 <= width * 0.68
                and height * 0.55 <= y0 <= height * 0.90):
            continue
        text = str(e.get("text") or e.get("latex") or "").lower()
        typ = e.get("type")
        if any(k in text for k in (
            "cheap nuisance", "zero retraining", "estimator",
            "agnostic", "overhead", "retraining",
        )):
            items.append(e)
            continue
        if typ in {"surface", "dotcloud", "arrow", "line", "icon"} and width * 0.31 <= x0 <= width * 0.58:
            items.append(e)
    if len(items) < 3:
        return []
    return [{
        "id": "bottom_mini_surface",
        "kind": "bottom_mini_surface",
        # This slot starts after the Q0 coverage panel and ends before the
        # action-card row.  Keeping it right-shifted prevents the mini
        # manifold/checklist agent from overwriting the Q0 chart.
        "bbox": [width * 0.485, height * 0.615, width * 0.705, height * 0.870],
        "element_ids": [str(e.get("id", "")) for e in items],
        "primary_method": "mini_surface_checklist",
        "component_template": "mini_surface_checklist",
        "fallback_methods": ["surface_vector_trace", "text_style"],
        "preferred_agent": "BottomMiniSurfaceAgent",
        "reason": "bottom mini-manifold and checklist are one native component, not separate text/shape fragments",
    }]


def _pipeline_context_regions(entities: list[dict], width: int,
                              height: int) -> list[dict]:
    """Detect the top existing-causal-pipeline row as a component."""
    items = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if not (width * 0.48 <= x0 <= width * 0.98 and height * 0.11 <= y0 <= height * 0.32):
            continue
        text = str(e.get("text") or "").lower()
        typ = e.get("type")
        if typ in {"icon", "arrow", "line"} or any(k in text for k in (
            "raw", "tables", "feature", "engineering", "cate",
            "estimator",
        )):
            items.append(e)
    if len(items) < 8:
        return []
    # Use the semantic top-row slot rather than noisy icon/text union.
    x0 = width * 0.492
    y0 = height * 0.120
    x1 = width * 0.982
    y1 = height * 0.340
    return [{
        "id": "region_pipeline_context",
        "kind": "pipeline_context_row",
        "bbox": [x0, y0, x1, y1],
        "element_ids": [str(e.get("id", "")) for e in items],
        "primary_method": "pipeline_context_layout",
        "component_template": "process_pipeline",
        "fallback_methods": ["shape_recovery", "text_style", "native_trace"],
        "preferred_agent": "LayoutAgent",
        "reason": "top causal decision pipeline is a repeated row of process cards",
    }]


def _auditor_card_regions(entities: list[dict], width: int,
                          height: int) -> list[dict]:
    """Detect the five CATE-CI Auditor method cards as one component row."""
    items = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if not (width * 0.48 <= x0 <= width * 0.99
                and height * 0.30 <= y0 <= height * 0.66):
            continue
        text = str(e.get("text") or e.get("latex") or "").lower()
        typ = e.get("type")
        if typ in {"rounded_rect", "rect", "oval", "chart", "icon", "surface", "dotcloud"}:
            items.append(e)
            continue
        if any(k in text for k in (
            "propensity", "surrogate", "heterogeneity", "alignment",
            "segment", "flag", "model", "cate", "score",
        )):
            items.append(e)
    if len(items) < 8:
        return []
    # Use the semantic row slot, not the union of detected internals.  The
    # union frequently includes lower connectors/action-card overlap, which
    # makes the auditor agent generate cards into the action row.
    x0 = width * 0.492
    y0 = height * 0.300
    x1 = width * 0.982
    y1 = height * 0.665
    return [{
        "id": "region_auditor_cards",
        "kind": "auditor_method_cards",
        "bbox": [x0, y0, x1, y1],
        "element_ids": [str(e.get("id", "")) for e in items],
        "primary_method": "auditor_card_layout",
        "component_template": "auditor_card",
        "fallback_methods": ["chart_parser", "icon_rebuild", "text_style"],
        "preferred_agent": "LayoutAgent",
        "reason": "middle method cards are a repeated component row with internal native mini-diagrams",
    }]


def _cross_panel_bridge_regions(regions: list[dict], entities: list[dict],
                                width: int, height: int) -> list[dict]:
    """Plan the semantic arrow from the problem panel into the solution system.

    Large cross-panel arrows are relational structure.  OCR/CV often drops them
    because they are a pale gradient and overlap otherwise empty whitespace, so
    the planner should infer the connector from the left problem region and the
    right solution component system.
    """
    has_left_problem = any(
        r.get("kind") == "procedural_3d_surface"
        and float((r.get("bbox") or [width, 0, width, 0])[0]) < width * 0.50
        for r in regions
    )
    has_solution_system = any(
        r.get("kind") in {
            "pipeline_context_row",
            "auditor_method_cards",
            "component_card_row",
        }
        for r in regions
    )
    if not (has_left_problem and has_solution_system):
        return []

    bridge_like = []
    for e in entities:
        bbox = e.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        if not (width * 0.38 <= x0 <= width * 0.54
                and height * 0.30 <= y0 <= height * 0.62):
            continue
        if e.get("type") in {"arrow", "line", "polygon", "path", "freeform"}:
            bridge_like.append(e)

    return [{
        "id": "region_cross_panel_bridge",
        "kind": "cross_panel_bridge",
        "bbox": [
            width * 0.425,
            height * 0.365,
            width * 0.505,
            height * 0.520,
        ],
        "element_ids": [str(e.get("id", "")) for e in bridge_like],
        "primary_method": "cross_panel_bridge",
        "component_template": "cross_panel_bridge",
        "fallback_methods": ["connector_rebuild", "shape_recovery"],
        "preferred_agent": "CrossPanelBridgeAgent",
        "reason": "problem-to-solution arrow is a semantic cross-panel connector",
    }]


def _overlaps_region(bbox: list[float], regions: list[dict],
                     min_overlap: float = 0.35) -> bool:
    for region in regions:
        rb = region.get("bbox")
        if rb and _bbox_overlap_fraction(bbox, rb) >= min_overlap:
            return True
    return False


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
