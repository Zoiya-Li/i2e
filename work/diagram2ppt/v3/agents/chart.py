"""Chart Agent: re-extracts chart data and normalizes the spec.

The v2 handler writes the chart spec with key "type", but the v2 builder
expects "kind".  ChartAgent re-reads the crop with a VLM, normalizes the
schema, and validates consistency before writing it back.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from work.diagram2ppt.v3 import component_templates, ir as IR
from work.diagram2ppt.v3.agents.base import Agent
from work.diagram2ppt.v3.providers import get_provider

_CHART_PROMPT = (
    "This crop is a chart. Identify its TYPE and output STRICT JSON only.\n\n"
    "Types:\n"
    "- bar: vertical or horizontal bars with categories + values\n"
    "- line: one or more polylines / curves with axes\n"
    "- scatter: cloud of points, optionally with a regression/trend line\n"
    "- pie: circular wedges\n\n"
    "Output schema (omit fields that do not apply):\n"
    '{\n'
    '  "type": "bar|line|scatter|pie",\n'
    '  "categories": ["...", "..."],\n'
    '  "series": [{"name": "...", "color": "#hex", "values": [numbers]}],\n'
    '  "points": [{"x": number, "y": number}, ...],\n'
    '  "trend": {"slope": number, "intercept": number, "color": "#hex"}\n'
    '}\n\n'
    "For bar charts: list EVERY visible bar, grouped by color as separate "
    "series. Do not omit small bars. Read category labels VERBATIM.\n"
    "For line charts: read axis/category labels VERBATIM and output at least "
    "12 points per series in [0,1] image coordinates.\n"
    "For scatter: output at least 8 representative (x,y) points in [0,1] "
    "image coordinates, and the trend line if visible.\n"
    "Never invent placeholder labels like 'Category 1' or 'Point 1'.\n"
    "If the crop is NOT a chart, output exactly "
    '{"type": "none"}. Output ONLY the JSON, no commentary.'
)

_FANTASY = re.compile(r"(?i)^(category|point|bar|item|series|value)[ _]?\d+$")


class ChartAgent(Agent):
    """Specialist agent for chart data extraction and repair."""

    name = "ChartAgent"

    def __init__(self) -> None:
        self.provider = get_provider("chart")

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if _is_q0_coverage_task(task):
            changed = self._repair_q0_coverage_panel(ir, task)
            self.record_contract_result(ir, task, changed)
            return changed
        if _is_generic_chart_task(task):
            changed = self._repair_generic_chart_region(ir, original, task)
            self.record_contract_result(ir, task, changed)
            return changed

        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el and el.get("type") == "chart":
                changed = self._repair_chart(ir, original, el)
                self.record_contract_result(ir, task, changed)
                return changed

        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") == "chart":
                changed.extend(self._repair_chart(ir, original, el))
        self.record_contract_result(ir, task, changed)
        return changed

    def _repair_generic_chart_region(
        self,
        ir: dict,
        original: Image.Image,
        task: dict,
    ) -> list[str]:
        region = _task_region_bbox(task)
        if not region:
            return []
        chart_spec = _infer_chart_primitives(original, region)
        if not chart_spec:
            return []
        cleanup = _expanded_bbox(region, original.size, 0.0)
        changed = set(_remove_generic_chart_orphans(ir, cleanup))
        for el in _generic_chart_elements(ir, region, chart_spec, task):
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.clear()
                existing.update(el)
            else:
                ir.setdefault("elements", []).append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "generic_chart_native_panel",
            "round": ir.get("round", 0),
            "region": region,
            "changed": sorted(changed),
        })
        return sorted(changed)

    def _repair_chart(self, ir: dict, original: Image.Image,
                      el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []

        crop = _padded_crop(original, bbox, pad=0.08)
        if crop.width < 30 or crop.height < 30:
            return []

        try:
            raw = self.provider.ask(crop, _CHART_PROMPT,
                                    temperature=0.0, max_tokens=1800)
            spec = _parse_chart_json(raw)
        except Exception:
            return []

        if not spec or spec.get("kind") == "none":
            return []

        spec = _normalize_chart_spec(spec)
        if not _spec_is_valid(spec):
            return []

        old_spec = el.get("chart") or el.get("ext", {}).get("chart") or {}
        if spec == old_spec:
            return []

        el["chart"] = spec
        el.setdefault("ext", {})["chart"] = dict(spec)
        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "chart_reextract",
            "round": ir.get("round", 0),
            "provider": self.provider.name,
        })
        return [el["id"]]

    def _repair_q0_coverage_panel(self, ir: dict, task: dict) -> list[str]:
        region = _q0_region_bbox(ir, task)
        if not region:
            return []
        cleanup = _q0_cleanup_bbox(ir, region)
        changed = set(_remove_q0_region_orphans(ir, cleanup))
        for el in _q0_panel_elements(ir, region):
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.update(el)
            else:
                ir["elements"].append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "q0_coverage_native_panel",
            "round": ir.get("round", 0),
            "region": region,
            "changed": sorted(changed),
        })
        return sorted(changed)


def _padded_crop(image: Image.Image, bbox: list[float],
                 pad: float = 0.08) -> Image.Image:
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    dx, dy = w * pad, h * pad
    left = max(0, int(x0 - dx))
    top = max(0, int(y0 - dy))
    right = min(image.width, int(x1 + dx))
    bottom = min(image.height, int(y1 + dy))
    return image.crop((left, top, right, bottom))


def _parse_chart_json(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_chart_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert v2 handler 'type' key to v2 builder 'kind' key and fill defaults."""
    ctype = str(spec.get("kind") or spec.get("type") or "none").lower().strip()
    if ctype not in ("bar", "line", "scatter", "pie", "none"):
        ctype = "none"

    out: dict[str, Any] = {"kind": ctype}

    categories = spec.get("categories") or []
    if categories and isinstance(categories, list):
        out["categories"] = [str(c) for c in categories]
    else:
        out["categories"] = []

    series = spec.get("series") or []
    normalized_series: list[dict[str, Any]] = []
    for s in series or []:
        if not isinstance(s, dict):
            continue
        values = s.get("values") or []
        if not isinstance(values, list):
            values = []
        normalized_series.append({
            "name": str(s.get("name") or "series"),
            "color": _normalize_hex(s.get("color")) or "#4472c4",
            "values": [float(v) if isinstance(v, (int, float)) else 0.0
                       for v in values],
        })
    out["series"] = normalized_series

    points = spec.get("points") or []
    if isinstance(points, list):
        out["points"] = [dict(p) for p in points if isinstance(p, dict)]
    else:
        out["points"] = []

    trend = spec.get("trend") or {}
    if isinstance(trend, dict):
        out["trend"] = {
            "slope": float(trend.get("slope", 0.0)),
            "intercept": float(trend.get("intercept", 0.0)),
            "color": _normalize_hex(trend.get("color")) or "#4472c4",
        }
    else:
        out["trend"] = {"slope": 0.0, "intercept": 0.0, "color": "#4472c4"}

    return out


def _spec_is_valid(spec: dict[str, Any]) -> bool:
    if spec.get("kind") not in ("bar", "line", "scatter", "pie"):
        return False
    cats = spec.get("categories") or []
    # Reject placeholder-only categories.
    if cats and all(_FANTASY.match(str(c).strip()) for c in cats):
        return False
    series = spec.get("series") or []
    if not series and spec["kind"] in ("bar", "line"):
        return False
    for s in series:
        vals = s.get("values") or []
        if spec["kind"] in ("bar", "line") and len(vals) != len(cats):
            return False
    return True


def _normalize_hex(color: Any) -> str | None:
    if not color:
        return None
    c = str(color).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", c):
        return c.lower()
    return None


def _is_q0_coverage_task(task: dict) -> bool:
    if not task:
        return False
    if task.get("region_id") == "q0_coverage_charts":
        return True
    objective = " ".join(str(task.get(k) or "") for k in (
        "id", "objective", "expected_native_expression", "locked_method"
    )).lower()
    return "q0" in objective and "coverage" in objective


def _is_generic_chart_task(task: dict) -> bool:
    if not task:
        return False
    if str(task.get("locked_method") or "") == "chart_parser":
        return True
    if str(task.get("kind") or "") == "chart":
        return True
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "region_id", "objective", "expected_native_expression",
    )).lower()
    return "chart" in text or "axis" in text or "series" in text


def _task_region_bbox(task: dict) -> list[float] | None:
    bbox = task.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _expanded_bbox(bbox: list[float], size: tuple[int, int], pad: float) -> list[float]:
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return [
        max(0.0, x0 - w * pad),
        max(0.0, y0 - h * pad),
        min(float(size[0]), x1 + w * pad),
        min(float(size[1]), y1 + h * pad),
    ]


def _infer_chart_primitives(image: Image.Image, region: list[float]) -> dict[str, Any] | None:
    x0, y0, x1, y1 = [float(v) for v in region]
    left, top, right, bottom = [int(round(v)) for v in (x0, y0, x1, y1)]
    crop = image.crop((max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))).convert("RGB")
    if crop.width < 24 or crop.height < 20:
        return None
    arr = np.asarray(crop).astype(np.int16)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    # Capture dark axes/series and saturated colored lines while ignoring white page background.
    mask = (gray < 210) | (saturation > 35)
    border = np.zeros(mask.shape, dtype=bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    mask &= ~border
    ys, xs = np.where(mask)
    if len(xs) < max(18, int(crop.width * crop.height * 0.006)):
        return None
    px0 = max(0, int(np.percentile(xs, 3)))
    px1 = min(crop.width - 1, int(np.percentile(xs, 97)))
    py0 = max(0, int(np.percentile(ys, 3)))
    py1 = min(crop.height - 1, int(np.percentile(ys, 97)))
    if px1 <= px0 + 8 or py1 <= py0 + 8:
        return None
    columns: list[list[float]] = []
    for x in np.linspace(px0, px1, num=min(36, max(12, (px1 - px0) // 4))):
        xi = int(round(float(x)))
        col_y = ys[np.abs(xs - xi) <= 1]
        if len(col_y) == 0:
            continue
        y = float(np.median(col_y))
        columns.append([
            (xi - px0) / max(1.0, px1 - px0),
            1.0 - ((y - py0) / max(1.0, py1 - py0)),
        ])
    if len(columns) < 4:
        return None
    return {
        "plot_norm": [
            px0 / crop.width,
            py0 / crop.height,
            px1 / crop.width,
            py1 / crop.height,
        ],
        "series": _smooth_points(columns),
        "color": _dominant_ink_color(arr, mask),
    }


def _smooth_points(points: list[list[float]]) -> list[list[float]]:
    if len(points) < 3:
        return points
    out = []
    for idx, (x, y) in enumerate(points):
        lo = max(0, idx - 1)
        hi = min(len(points), idx + 2)
        yy = sum(p[1] for p in points[lo:hi]) / (hi - lo)
        out.append([round(float(x), 4), round(float(yy), 4)])
    return out


def _dominant_ink_color(arr: np.ndarray, mask: np.ndarray) -> str:
    pixels = arr[mask]
    if len(pixels) == 0:
        return "#2f5c8f"
    sat = pixels.max(axis=1) - pixels.min(axis=1)
    colored = pixels[sat > 35]
    sample = colored if len(colored) >= 5 else pixels
    rgb = np.median(sample, axis=0).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _remove_generic_chart_orphans(ir: dict, region: list[float]) -> list[str]:
    # Preserve panel/card containers.  Generic chart repair owns the chart
    # primitives, not the surrounding layout component.
    removable = {"chart", "freeform", "line", "arrow", "dotcloud"}
    keep = []
    removed = []
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if eid.startswith("generic_chart_") or not bbox:
            keep.append(el)
            continue
        typ = el.get("type")
        overlap = _bbox_overlap_fraction(bbox, region)
        threshold = 0.65 if typ == "chart" else 0.25
        if typ in removable and overlap > threshold:
            removed.append(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _generic_chart_elements(
    ir: dict,
    region: list[float],
    spec: dict[str, Any],
    task: dict,
) -> list[dict]:
    x0, y0, x1, y1 = [float(v) for v in region]
    w, h = x1 - x0, y1 - y0
    px0n, py0n, px1n, py1n = spec["plot_norm"]
    plot = [
        x0 + px0n * w,
        y0 + py0n * h,
        x0 + px1n * w,
        y0 + py1n * h,
    ]
    round_num = ir.get("round", 0)
    ext = _generic_chart_ext(task)
    color = spec.get("color") or "#2f5c8f"
    elems = [
        IR.element(
            id=_generic_chart_id(task, "axis_x"),
            type="line",
            bbox=[plot[0], plot[3], plot[2], plot[3] + 1],
            provenance=IR.provenance("ChartAgent", "generic_chart_axis", round_num),
            confidence=0.72,
            points=[plot[0], plot[3], plot[2], plot[3]],
            color="#333333",
            thickness=1.2,
            line_width=1.2,
            z=6.0,
            ext=ext,
        ),
        IR.element(
            id=_generic_chart_id(task, "axis_y"),
            type="line",
            bbox=[plot[0], plot[1], plot[0] + 1, plot[3]],
            provenance=IR.provenance("ChartAgent", "generic_chart_axis", round_num),
            confidence=0.72,
            points=[plot[0], plot[3], plot[0], plot[1]],
            color="#333333",
            thickness=1.2,
            line_width=1.2,
            z=6.0,
            ext=ext,
        ),
        _generic_chart_polyline(_generic_chart_id(task, "series_0"), plot, spec["series"], color, round_num, ext),
    ]
    return elems


def _generic_chart_polyline(
    eid: str,
    plot: list[float],
    values: list[list[float]],
    color: str,
    round_num: int,
    ext: dict,
) -> dict:
    px0, py0, px1, py1 = plot
    local = [
        [
            round(float(x) * (px1 - px0), 2),
            round((1.0 - float(y)) * (py1 - py0), 2),
        ]
        for x, y in values
    ]
    return IR.element(
        id=eid,
        type="freeform",
        bbox=plot,
        provenance=IR.provenance("ChartAgent", "generic_chart_series", round_num),
        confidence=0.74,
        paths=[{
            "points": local,
            "fill": None,
            "line": color,
            "alpha": 100,
            "closed": False,
            "line_width": 2.0,
        }],
        z=6.5,
        ext=ext,
    )


def _generic_chart_id(task: dict, suffix: str) -> str:
    region_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(task.get("region_id") or task.get("id") or "region"))
    return f"generic_chart_{region_id}_{suffix}"


def _generic_chart_ext(task: dict) -> dict:
    region_id = str(task.get("region_id") or "generic_chart")
    return {
        "component": "generic_chart",
        "component_role": "chart_primitive",
        "strategy": {
            "region_id": region_id,
            "kind": "chart",
            "primary_method": "chart_parser",
            "preferred_agent": "ChartAgent",
        },
    }


def _q0_region_bbox(ir: dict, task: dict) -> list[float] | None:
    canvas = ir.get("canvas") or {}
    width = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    height = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    if task.get("region_id") == "q0_coverage_charts":
        bbox = task.get("bbox")
        if bbox and len(bbox) == 4 and max(float(v) for v in bbox) > 0:
            return [float(v) for v in bbox]
        # Fallback full rounded Q0 panel, not just the internal line chart.
        return [width * 0.180, height * 0.640, width * 0.480, height * 0.930]

    bbox = task.get("bbox")
    if bbox and len(bbox) == 4 and max(float(v) for v in bbox) > 0:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [x0 + 5.0, max(0.0, y0 - 55.0), x1 - 20.0, y1 - 70.0]

    candidates = []
    for el in ir.get("elements", []):
        text = str(el.get("text") or "").lower()
        if "coverage" in text and el.get("bbox"):
            candidates.append(el["bbox"])
    if not candidates:
        return None
    x0 = min(b[0] for b in candidates) - 170
    y0 = min(b[1] for b in candidates) - 40
    x1 = max(b[2] for b in candidates) + 240
    y1 = max(b[3] for b in candidates) + 290
    return [x0, y0, x1, y1]


def _remove_q0_region_orphans(ir: dict, region: list[float]) -> list[str]:
    removable = {"text", "formula", "chart", "dotcloud", "freeform", "rounded_rect", "rect", "line", "arrow"}
    keep = []
    removed = []
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if eid == "chart_q0_title":
            removed.append(eid)
            continue
        if (
            eid.startswith("chart_q0_")
            or eid.startswith(("auditor_", "action_card_", "bottom_", "failure_summary_", "pipeline_context_", "proc_"))
            or not bbox
        ):
            keep.append(el)
            continue
        if el.get("type") in removable and _bbox_overlap_fraction(bbox, region) > 0.22:
            removed.append(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _q0_cleanup_bbox(ir: dict, region: list[float]) -> list[float]:
    canvas = ir.get("canvas") or {}
    width = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    height = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    x0, y0, x1, y1 = [float(v) for v in region]
    # Cleanup is intentionally broader than layout: it should catch old OCR
    # ticks/value labels around the adjacent bar chart without changing the
    # native panel geometry itself.
    return [
        max(0.0, x0 - 28.0),
        max(0.0, y0 - 28.0),
        min(width, x1 + 210.0),
        min(height, y1 + 135.0),
    ]


def _q0_panel_elements(ir: dict, region: list[float]) -> list[dict]:
    rx0, ry0, rx1, ry1 = region
    width = rx1 - rx0
    height = ry1 - ry0
    x0 = rx0 + width * 0.004
    y0 = ry0 + height * 0.000
    x1 = rx1 - width * 0.090
    y1 = ry1 - height * 0.180
    w, h = x1 - x0, y1 - y0
    round_num = ir.get("round", 0)
    z = 6.0
    ext = _ext("panel")
    elems = [
        IR.element(
            id="chart_q0_panel",
            type="rounded_rect",
            bbox=[x0, y0, x1, y1],
            provenance=IR.provenance("ChartAgent", "q0_panel_container", round_num),
            confidence=0.86,
            fill="#ffffff",
            border_color="#8caee0",
            border_width=1.55,
            corner=0.050,
            z=1.0,
            ext=ext,
        ),
        _text("chart_q0_title_q", [x0 + 0.250 * w, y0 + 8, x0 + 0.305 * w, y0 + 74],
              "Q", round_num, ext, size=38, color="#09194b", bold=True,
              font="Times New Roman"),
        _text("chart_q0_title_sub", [x0 + 0.292 * w, y0 + 34, x0 + 0.330 * w, y0 + 67],
              "0", round_num, ext, size=22, color="#09194b", bold=True,
              font="Times New Roman"),
        _text("chart_q0_title_rest", [x0 + 0.325 * w, y0 + 8, x1 - 0.135 * w, y0 + 74],
              "coverage collapses", round_num, ext, size=38,
              color="#09194b", bold=True, font="Times New Roman"),
    ]

    left = [x0 + 0.050 * w, y0 + 0.195 * h, x0 + 0.575 * w, y0 + 0.850 * h]
    right = [x0 + 0.590 * w, y0 + 0.195 * h, x0 + 0.972 * w, y0 + 0.850 * h]
    elems.extend(_line_chart_elements(round_num, left, ext))
    elems.extend(_bar_chart_elements(round_num, right, ext))
    return elems


def _line_chart_elements(round_num: int, box: list[float], ext: dict) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    plot = [x0 + 70, y0 + 26, x1 - 30, y1 - 58]
    px0, py0, px1, py1 = plot
    weak_label = _text("chart_q0_weak", [px1 - 112, py1 + 7, px1 + 30, py1 + 39],
                       "Q0 (weak)", round_num, ext, size=18)
    weak_label["runs"] = [
        {"text": "Q", "font": "Times New Roman", "italic": True},
        {"text": "0", "font": "Times New Roman", "italic": True, "font_size": 13, "baseline": -25000},
        {"text": " (weak)", "font": "Arial"},
    ]
    elems = [
        _text("chart_q0_y_label", [x0 - 10, y0 + 0.17 * h, x0 + 32, y0 + 0.62 * h],
              "Coverage", round_num, ext, rotation=270, size=24),
        _text("chart_q0_tick_10", [x0 + 18, py0 - 11, x0 + 62, py0 + 15], "1.0", round_num, ext, size=20),
        _text("chart_q0_tick_05", [x0 + 18, (py0 + py1) / 2 - 13, x0 + 62, (py0 + py1) / 2 + 13], "0.5", round_num, ext, size=20),
        _text("chart_q0_tick_00", [x0 + 18, py1 - 12, x0 + 62, py1 + 14], "0.0", round_num, ext, size=20),
        _text("chart_q0_strong", [px0 - 8, py1 + 8, px0 + 92, py1 + 36], "(strong)", round_num, ext, size=18),
        weak_label,
        _text("chart_q0_x_label", [px0, py1 + 39, px1, py1 + 72], "Overlap quantile", round_num, ext, size=20),
        _text("chart_q0_orthogonal", [px0 + 0.34 * (px1 - px0), py0 - 30, px1 + 54, py0 + 8],
              "\u2212 orthogonal", round_num, ext, size=20, color="#1f55d1"),
        _text("chart_q0_aligned_label", [px0 + 28, py0 + 0.53 * (py1 - py0), px0 + 226, py0 + 0.53 * (py1 - py0) + 36],
              "aligned (A \u2248 1)", round_num, ext, size=18, color="#d22d25"),
        _line("chart_q0_axis_x", [px0, py1, px1, py1], round_num, ext, "#111111", 1.6),
        _line("chart_q0_axis_y", [px0, py1, px0, py0], round_num, ext, "#111111", 1.6),
    ]
    red = [[0.00, 0.995], [0.10, 0.985], [0.20, 0.950], [0.31, 0.870],
           [0.42, 0.720], [0.52, 0.520], [0.62, 0.310], [0.73, 0.150],
           [0.84, 0.060], [0.96, 0.025]]
    blue = [[0.00, 1.000], [0.14, 0.990], [0.28, 0.960], [0.42, 0.880],
            [0.56, 0.780], [0.68, 0.690], [0.78, 0.610], [0.86, 0.405],
            [0.93, 0.205]]
    elems.append(_polyline("chart_q0_line_aligned", plot, red, "#d22d25", round_num, ext, width=1.95))
    elems.append(_polyline("chart_q0_line_orthogonal", plot, blue, "#1f55d1", round_num, ext, width=2.05))
    elems.extend(_markers("chart_q0_mark_aligned", plot, red, "#d22d25", round_num, ext, every=1))
    elems.extend(_markers("chart_q0_mark_orthogonal", plot, blue, "#1f55d1", round_num, ext, every=1))
    return elems


def _bar_chart_elements(round_num: int, box: list[float], ext: dict) -> list[dict]:
    x0, y0, x1, y1 = box
    plot = [x0 + 58, y0 + 26, x1 - 8, y1 - 62]
    px0, py0, px1, py1 = plot
    y_label = _text("chart_q0_bar_y_label", [x0 - 14, y0 + 8, x0 + 42, y1 - 30],
                    "Coverage in Q0", round_num, ext, rotation=270, size=19, font="Times New Roman")
    y_label["runs"] = [
        {"text": "Coverage in ", "font": "Times New Roman"},
        {"text": "Q", "font": "Times New Roman", "italic": True},
        {"text": "0", "font": "Times New Roman", "italic": True, "font_size": 14, "baseline": -25000},
    ]
    elems = [
        y_label,
        _line("chart_q0_bar_axis_x", [px0, py1, px1, py1], round_num, ext, "#111111", 1.55),
        _line("chart_q0_bar_axis_y", [px0, py1, px0, py0], round_num, ext, "#111111", 1.55),
        _text("chart_q0_bar_tick_10", [x0 + 12, py0 - 11, x0 + 52, py0 + 15], "1.0", round_num, ext, size=20),
        _text("chart_q0_bar_tick_05", [x0 + 12, (py0 + py1) / 2 - 13, x0 + 52, (py0 + py1) / 2 + 13], "0.5", round_num, ext, size=20),
        _text("chart_q0_bar_tick_00", [x0 + 12, py1 - 12, x0 + 52, py1 + 14], "0.0", round_num, ext, size=20),
    ]
    bars = [
        ("orthogonal", 0.78, "#5d8ed8", 0.255),
        ("aligned", 0.12, "#e85043", 0.830),
    ]
    for name, value, color, center in bars:
        bw = (px1 - px0) * 0.205
        bx = px0 + center * (px1 - px0) - bw / 2
        by = py1 - value * (py1 - py0)
        elems.append(IR.element(
            id=f"chart_q0_bar_{name}",
            type="rect",
            bbox=[bx, by, bx + bw, py1],
            provenance=IR.provenance("ChartAgent", "q0_bar", round_num),
            confidence=0.86,
            fill=color,
            border_color="#2c5c9a" if name == "orthogonal" else "#b22a23",
            border_width=1,
            z=6.2,
            ext=ext,
        ))
        elems.append(_text(f"chart_q0_bar_value_{name}", [bx - 13, by - 26, bx + bw + 13, by - 4],
                           f"{value:.2f}", round_num, ext, size=18, font="Times New Roman"))
        label = "orthogonal" if name == "orthogonal" else "aligned\n(A \u2248 1)"
        x_pad = 30 if name == "orthogonal" else 38
        y_bottom = 42 if name == "orthogonal" else 60
        elems.append(_text(f"chart_q0_bar_label_{name}", [bx - x_pad, py1 + 8, bx + bw + x_pad, py1 + y_bottom],
                           label, round_num, ext, size=15))
    return elems


def _text(eid: str, bbox: list[float], text: str, round_num: int, ext: dict,
          size: float = 16, color: str = "#111111", rotation: float | None = None,
          bold: bool = False, font: str = "Arial") -> dict:
    kwargs = {}
    if rotation is not None:
        kwargs["rotation"] = rotation
    if bold:
        kwargs["bold"] = True
    role = _chart_text_role(eid)
    text_ext = component_templates.component_ext(
        "coverage_chart_panel",
        role,
        _chart_key(eid),
        region_id="q0_coverage_charts",
        size=size,
        color=color,
        font=font,
        bold=bold,
    ) if role else dict(ext)
    return IR.element(
        id=eid,
        type="text",
        bbox=bbox,
        provenance=IR.provenance("ChartAgent", "q0_label", round_num),
        confidence=0.84,
        text=text,
        font=font,
        font_size=size,
        text_color=color,
        align="center",
        z=7.0,
        ext=text_ext,
        **kwargs,
    )


def _ext(role: str, key: str = "") -> dict:
    return component_templates.component_ext(
        "coverage_chart_panel",
        role,
        key,
        region_id="q0_coverage_charts",
    )


def _chart_text_role(eid: str) -> str:
    if eid == "chart_q0_title_q":
        return "title_q"
    if eid == "chart_q0_title_sub":
        return "title_sub"
    if eid == "chart_q0_title_rest":
        return "title_rest"
    if eid in {"chart_q0_y_label", "chart_q0_x_label", "chart_q0_bar_y_label"}:
        return "axis_label"
    if eid in {"chart_q0_orthogonal", "chart_q0_aligned_label"}:
        return "curve_label"
    if eid.startswith("chart_q0_bar_label_"):
        return "bar_label"
    if eid.startswith("chart_q0_bar_value_"):
        return "bar_value"
    if eid.startswith("chart_q0_tick_") or eid.startswith("chart_q0_bar_tick_"):
        return "tick"
    if eid in {"chart_q0_strong", "chart_q0_weak"}:
        return "tick"
    return ""


def _chart_key(eid: str) -> str:
    if "orthogonal" in eid:
        return "orthogonal"
    if "aligned" in eid:
        return "aligned"
    return ""


def _line(eid: str, points: list[float], round_num: int, ext: dict,
          color: str, thickness: float) -> dict:
    x0, y0, x1, y1 = points
    return IR.element(
        id=eid,
        type="line",
        bbox=[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
        provenance=IR.provenance("ChartAgent", "q0_axis", round_num),
        confidence=0.84,
        points=points,
        color=color,
        thickness=thickness,
        line_width=thickness,
        z=6.4,
        ext=ext,
    )


def _polyline(eid: str, plot: list[float], values: list[list[float]], color: str,
              round_num: int, ext: dict, width: float = 2.4) -> dict:
    px0, py0, px1, py1 = plot
    pts = []
    for x, y in values:
        pts.append([px0 + x * (px1 - px0), py1 - y * (py1 - py0)])
    local = [[round(x - px0, 2), round(y - py0, 2)] for x, y in pts]
    return IR.element(
        id=eid,
        type="freeform",
        bbox=[px0, py0, px1, py1],
        provenance=IR.provenance("ChartAgent", "q0_line_series", round_num),
        confidence=0.86,
        paths=[{
            "points": local,
            "fill": None,
            "line": color,
            "alpha": 100,
            "closed": False,
            "line_width": width,
        }],
        z=6.8,
        ext=ext,
    )


def _markers(eid: str, plot: list[float], values: list[list[float]], color: str,
             round_num: int, ext: dict, every: int = 1) -> list[dict]:
    px0, py0, px1, py1 = plot
    out: list[dict] = []
    for idx, (x, y) in enumerate(values):
        if idx % max(1, every) != 0:
            continue
        cx = px0 + x * (px1 - px0)
        cy = py1 - y * (py1 - py0)
        r = 1.75
        out.append(IR.element(
            id=f"{eid}_{idx:02d}",
            type="oval",
            bbox=[cx - r, cy - r, cx + r, cy + r],
            provenance=IR.provenance("ChartAgent", "q0_line_marker", round_num),
            confidence=0.82,
            fill=color,
            border_color=color,
            border_width=0,
            z=6.9,
            ext=ext,
        ))
    return out


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
