"""Visual review for rendered reconstruction.

Metrics are useful gates, but they are the wrong object to optimize directly.
This module asks a visual critic to inspect the comparison image and produce a
region-level defect inventory.  The planner then turns those visual defects
into task-graph work items.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from . import method_registry


# NOTE: these are FRAMEWORK.PNG-SPECIFIC fixture priors (left 3D manifold,
# failure summary, Q0 coverage, pipeline/auditor/action cards). They are a
# LAST-RESORT FALLBACK ONLY. Generic review derives regions from the
# strategy_plan (see `_semantic_regions_px`, which falls back to a generic
# whole-slide region, not to this list). Do not extend this fixture for new
# diagrams — add strategy regions instead, or move it to tests/fixtures.
REGION_PRIORS = [
    {
        "id": "left_surface",
        "label": "left 3D covariate manifold",
        "bbox_norm": [0.00, 0.10, 0.46, 0.64],
        "default_agents": ["ProceduralSurfaceAgent", "StyleAgent"],
    },
    {
        "id": "failure_summary",
        "label": "bottom-left failure summary panel",
        "bbox_norm": [0.00, 0.64, 0.18, 0.93],
        "default_agents": ["FailureSummaryAgent", "TextAgent", "IconAgent", "StyleAgent"],
    },
    {
        "id": "q0_coverage_charts",
        "label": "Q0 coverage chart panel",
        "bbox_norm": [0.18, 0.64, 0.48, 0.93],
        "default_agents": ["ChartAgent", "TextAgent", "StyleAgent"],
    },
    {
        "id": "pipeline_context",
        "label": "top causal decision pipeline row",
        "bbox_norm": [0.50, 0.10, 0.98, 0.30],
        "default_agents": ["LayoutAgent", "IconAgent", "ConnectorAgent", "TextAgent"],
    },
    {
        "id": "auditor_cards",
        "label": "five CATE-CI Auditor method cards",
        "bbox_norm": [0.50, 0.30, 0.98, 0.64],
        "default_agents": ["LayoutAgent", "ChartAgent", "IconAgent", "TextAgent", "StyleAgent"],
    },
    {
        "id": "bottom_mini_surface",
        "label": "bottom mini manifold and checklist",
        "bbox_norm": [0.485, 0.595, 0.715, 0.858],
        "default_agents": ["BottomMiniSurfaceAgent", "SurfaceAgent", "TextAgent", "IconAgent", "StyleAgent"],
    },
    {
        "id": "action_cards",
        "label": "retain/defer/alert/report action cards",
        "bbox_norm": [0.70, 0.64, 0.99, 0.94],
        "default_agents": ["LayoutAgent", "IconAgent", "ConnectorAgent", "TextAgent", "StyleAgent"],
    },
]


def review(
    compare_png: str | Path,
    vlm: Any | None = None,
    canvas_width: int | float = 0,
    canvas_height: int | float = 0,
    strategy_plan: dict | None = None,
) -> dict:
    """Return a visual defect inventory for a compare image.

    ``compare_png`` is expected to show original and reconstruction together.
    The VLM is intentionally asked for visual differences, not score estimates.
    """
    compare_png = Path(compare_png)
    width, height = _image_size(compare_png)
    canvas_width = float(canvas_width or width)
    canvas_height = float(canvas_height or height / 2)

    semantic_regions = _semantic_regions_px(
        strategy_plan or {},
        canvas_width,
        canvas_height,
    )

    base = {
        "version": "visual-review-v1",
        "status": "unavailable",
        "source": str(compare_png),
        "instruction": "Use visual defects, not numeric metrics, as iteration targets.",
        "regions": semantic_regions,
        "defects": [],
    }

    if vlm is None:
        base["status"] = "degraded"
        base["reason"] = "no VLM client attached; using deterministic visual-region fallback"
        base["summary"] = (
            "VLM visual review unavailable; using current strategy regions "
            "so the planner remains method-directed."
        )
        base["defects"] = _fallback_visual_defects(
            canvas_width,
            canvas_height,
            semantic_regions,
        )
        return base

    prompt = _prompt(semantic_regions)
    try:
        max_edge = int(os.environ.get("I2E_VISUAL_REVIEW_MAX_EDGE", "1100"))
        max_tokens = int(os.environ.get("I2E_VISUAL_REVIEW_MAX_TOKENS", "2200"))
        text = vlm.chat(prompt, str(compare_png), max_edge=max_edge, max_tokens=max_tokens)
        data = _parse_json(text)
    except Exception as exc:
        base["status"] = "degraded"
        base["reason"] = f"{type(exc).__name__}: {exc}"
        base["summary"] = (
            "VLM visual review unavailable; using current strategy regions "
            "so the planner does not fall back to metric-only iteration."
        )
        base["defects"] = _fallback_visual_defects(
            canvas_width,
            canvas_height,
            semantic_regions,
        )
        return base

    defects = []
    for idx, raw in enumerate(data.get("defects", [])):
        if not isinstance(raw, dict):
            continue
        region_id = str(raw.get("region_id") or raw.get("region") or "").strip()
        prior = _prior_by_id(region_id, semantic_regions)
        agents = raw.get("suggested_agents") or raw.get("agents") or []
        if isinstance(agents, str):
            agents = [agents]
        if prior and not agents:
            agents = prior["default_agents"]
        severity = str(raw.get("severity") or "medium").lower()
        defects.append({
            "id": str(raw.get("id") or f"visual_defect_{idx:02d}_{region_id or 'region'}"),
            "region_id": region_id,
            "region_label": str(raw.get("region_label") or (prior or {}).get("label") or region_id),
            "bbox": _bbox_from_raw(raw, prior, canvas_width, canvas_height),
            "severity": severity if severity in {"critical", "high", "medium", "low"} else "medium",
            "visual_problem": str(raw.get("visual_problem") or raw.get("problem") or ""),
            "expected_native_expression": str(raw.get("expected_native_expression") or raw.get("expected") or ""),
            "suggested_agents": _dedupe([str(a) for a in agents if a]),
            "blocking_reason": str(raw.get("blocking_reason") or ""),
        })

    base.update({
        "status": "ok",
        "summary": str(data.get("summary") or ""),
        "defects": defects,
        "raw_text": text[:6000],
    })
    return base


def attach_to_ir(ir: dict, review_result: dict) -> None:
    ir["visual_review"] = review_result


def _prompt(regions_px: list[dict]) -> str:
    regions = "\n".join(
        f"- {r['id']}: {r['label']}, expected agents={', '.join(r['default_agents'])}"
        for r in regions_px
    )
    return f"""You are the visual planning critic for an Image-to-native-PPTX reconstruction system.

The image shows the original slide and the reconstructed slide together. Do not optimize or report numeric metrics. Inspect the picture like a designer/engineer and list the visible reconstruction failures that should drive the next agent iteration.

Known semantic regions:
{regions}

Return STRICT JSON only:
{{
  "summary": "one sentence visual diagnosis",
  "defects": [
    {{
      "id": "short_stable_id",
      "region_id": "one known region id",
      "severity": "critical|high|medium|low",
      "visual_problem": "what is visibly wrong in the reconstruction",
      "expected_native_expression": "what editable PPT primitives or procedural code should rebuild it",
      "suggested_agents": ["Planner", "LayoutAgent", "ProceduralSurfaceAgent", "SurfaceAgent", "ChartAgent", "TextAgent", "IconAgent", "ConnectorAgent", "StyleAgent", "VectorizeAgent"],
      "blocking_reason": "why metric-only iteration would miss or mishandle this"
    }}
  ]
}}

Focus on structural/semantic visual failures: wrong region method, missing internal geometry, bad z-order, bad grouping, huge or tiny text, corrupted charts/icons/connectors, and places where procedural drawing is required.
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _semantic_regions_px(
    strategy_plan: dict,
    width: float,
    height: float,
) -> list[dict]:
    regions = []
    for idx, region in enumerate(strategy_plan.get("regions", []) or []):
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        method = (
            region.get("primary_method")
            or (region.get("representation") or {}).get("method")
            or method_registry.infer_method(
                kind=region.get("kind"),
                region_id=region.get("id"),
                objective=region.get("reason"),
            )
        )
        policy = method_registry.policy_for_method(method)
        contract = method_registry.contract_for_method(method)
        owner = (
            contract.get("owner_agent")
            or (region.get("representation") or {}).get("owner_agent")
            or policy.get("required_agents", ["LayoutAgent"])[0]
        )
        fallback_agents = list(policy.get("fallback_agents") or [])
        default_agents = _dedupe([owner] + fallback_agents + ["TextLayoutAgent", "StyleAgent"])
        label = str(
            region.get("label")
            or region.get("kind")
            or region.get("id")
            or f"semantic region {idx}"
        )
        regions.append({
            "id": str(region.get("id") or f"region_{idx}"),
            "label": label,
            "kind": region.get("kind"),
            "method": method,
            "bbox": [float(v) for v in bbox],
            "default_agents": default_agents,
            "expected_native_expression": (
                contract.get("native_expression")
                or policy.get("native_expression")
                or "editable native PPT primitives"
            ),
            "visual_evidence": contract.get("visual_evidence") or [],
        })
    if regions:
        return regions
    return _global_fallback_region(width, height)


def _global_fallback_region(width: float, height: float) -> list[dict]:
    return [{
        "id": "whole_slide",
        "label": "whole slide reconstruction",
        "kind": "unknown",
        "method": "",
        "bbox": [0.0, 0.0, float(width), float(height)],
        "default_agents": ["LayoutAgent", "TextAgent", "TextLayoutAgent", "StyleAgent"],
        "expected_native_expression": "editable native PPT primitives for the whole slide",
        "visual_evidence": ["visible regions are native and readable"],
    }]


def _region_priors_px(width: float, height: float) -> list[dict]:
    out = []
    for prior in REGION_PRIORS:
        x0, y0, x1, y1 = prior["bbox_norm"]
        item = dict(prior)
        item["bbox"] = [x0 * width, y0 * height, x1 * width, y1 * height]
        out.append(item)
    return out


def _prior_by_id(region_id: str, regions: list[dict] | None = None) -> dict | None:
    for prior in regions or REGION_PRIORS:
        if prior["id"] == region_id:
            return prior
    return None


def _bbox_from_raw(raw: dict, prior: dict | None,
                   width: float, height: float) -> list[float]:
    bbox = raw.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        vals = [float(v) for v in bbox]
        if max(vals) <= 1.5:
            return [vals[0] * width, vals[1] * height,
                    vals[2] * width, vals[3] * height]
        return vals
    if prior:
        if prior.get("bbox"):
            return [float(v) for v in prior["bbox"]]
        x0, y0, x1, y1 = prior["bbox_norm"]
        return [x0 * width, y0 * height, x1 * width, y1 * height]
    return [0.0, 0.0, width, height]


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fallback_visual_defects(
    width: float,
    height: float,
    regions_px: list[dict] | None = None,
) -> list[dict]:
    """Degraded visual plan when provider access fails.

    This is deliberately strategy/method oriented rather than metric oriented.
    It keeps the planner focused on the current image's semantic regions until
    a VLM or human/agent visual seed is available.
    """
    regions = regions_px or _global_fallback_region(width, height)
    out = []
    for idx, region in enumerate(regions):
        method = str(region.get("method") or "")
        expected = str(region.get("expected_native_expression") or "editable native PPT primitives")
        problem = _fallback_problem_for_method(method, region)
        out.append({
            "id": f"fallback_visual_{idx:02d}_{region['id']}",
            "region_id": region["id"],
            "region_label": region.get("label") or region["id"],
            "bbox": [float(v) for v in region.get("bbox", [0, 0, width, height])],
            "severity": "high",
            "visual_problem": problem,
            "expected_native_expression": expected,
            "suggested_agents": list(region.get("default_agents") or ["LayoutAgent", "StyleAgent"]),
            "blocking_reason": (
                "numeric residuals do not encode the correct reconstruction "
                "method for this current-image region"
            ),
        })
    return out


def _fallback_problem_for_method(method: str, region: dict) -> str:
    kind = str(region.get("kind") or "region").replace("_", " ")
    if method == "procedural_surface":
        return "procedural surface region needs native generated geometry rather than residual fragments"
    if method == "chart_parser":
        return "chart region needs native axes, series, ticks, labels, and legend reconstruction"
    if method == "pipeline_context_layout":
        return "flow or architecture region needs coherent native process cards, labels, icons, and connectors"
    if method == "auditor_card_layout":
        return "repeated method-card region needs grouped native cards with internal mini-diagrams"
    if method == "component_layout":
        return "repeated component-card region needs shared geometry, readable text slots, and native icons"
    if method == "failure_summary_layout":
        return "summary panel needs native icon-and-text structure with readable typography"
    if method == "mini_surface_checklist":
        return "mini-surface/checklist region needs native surface, checks, text, and connector structure"
    if method == "cross_panel_bridge":
        return "bridge region needs a native connector that links the semantic panels"
    return f"{kind} needs native editable reconstruction based on current image strategy"
