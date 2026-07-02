"""BottomMiniSurfaceAgent: native mini manifold plus checklist."""
from __future__ import annotations

import math
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class BottomMiniSurfaceAgent(Agent):
    """Rebuild the lower-right mini manifold and cheap-nuisance checklist."""

    name = "BottomMiniSurfaceAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if not _is_bottom_task(task):
            return []
        region = _region_bbox(ir, task)
        changed = set(_remove_orphans(ir, region))
        for el in _mini_elements(ir, region):
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.clear()
                existing.update(el)
            else:
                ir.setdefault("elements", []).append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "native_bottom_mini_surface_transaction",
            "round": ir.get("round", 0),
            "changed": sorted(changed),
        })
        return sorted(changed)


def _is_bottom_task(task: dict) -> bool:
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "kind", "region_id", "locked_method", "objective",
    )).lower()
    return "bottom_mini_surface" in text or "mini manifold" in text or "checklist" in text


def _region_bbox(ir: dict, task: dict) -> list[float]:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    # Canonical figure slot: after the Q0 coverage panel and before the action
    # cards.  Do not derive this from noisy visual/OCR boxes; that previously
    # caused the mini-surface to drift into the Q0 chart or RETAIN/DEFER cards.
    return [w * 0.485, h * 0.615, w * 0.705, h * 0.870]


def _mini_elements(ir: dict, region: list[float]) -> list[dict]:
    x0, y0, x1, y1 = region
    r = ir.get("round", 0)
    w, h = x1 - x0, y1 - y0
    surf = [x0 + w * 0.02, y0 + h * 0.22, x0 + w * 0.47, y0 + h * 0.84]
    sx0, sy0, sx1, sy1 = surf
    sw, sh = sx1 - sx0, sy1 - sy0
    dots = []
    for i in range(22):
        t = i / 21
        band = i % 3
        cx = sw * (0.14 + 0.72 * t)
        cy = sh * (0.58 - 0.20 * math.sin(t * math.pi) + (band - 1) * 0.045)
        dots.append({"cx": cx, "cy": cy, "r": 1.35 if i % 4 else 1.8, "color": "#6d9fbd"})
    curves = []
    for j in range(8):
        pts = []
        for i in range(30):
            t = i / 29
            pts.append([
                sw * t,
                sh * (0.24 + j * 0.075 + 0.040 * math.sin(t * math.pi * 2.1 + j * 0.35)),
            ])
        curves.append(pts)
    elements: list[dict] = [
        IR.element(
            id="bottom_mini_surface_surface",
            type="surface",
            bbox=surf,
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_surface", r),
            confidence=0.82,
            dots=dots,
            wave_bands={
                "curves": curves,
                "fills": ["#e9f4fa", "#deedf6", "#d8e7f1", "#e9ecef"],
            },
            streamlines=curves[1:-1],
            fill="#dbeef8",
            z=2.0,
            ext=_ext("surface"),
        ),
        IR.element(
            id="bottom_mini_axis_x3",
            type="arrow",
            bbox=[sx0 + sw * 0.06, sy0 + sh * 0.48, sx0 + sw * 0.06, sy0 + sh * 0.08],
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_axis", r),
            confidence=0.80,
            start=[sx0 + sw * 0.06, sy0 + sh * 0.48],
            end=[sx0 + sw * 0.06, sy0 + sh * 0.08],
            color="#222222",
            thickness=2,
            z=5.0,
            ext=_ext("axis"),
        ),
        IR.element(
            id="bottom_mini_axis_x1",
            type="arrow",
            bbox=[sx0 + sw * 0.06, sy0 + sh * 0.48, sx0 + sw * 0.40, sy0 + sh * 0.67],
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_axis", r),
            confidence=0.80,
            start=[sx0 + sw * 0.06, sy0 + sh * 0.48],
            end=[sx0 + sw * 0.40, sy0 + sh * 0.67],
            color="#222222",
            thickness=2,
            z=5.0,
            ext=_ext("axis"),
        ),
        IR.element(
            id="bottom_mini_axis_x2",
            type="arrow",
            bbox=[sx0 + sw * 0.06, sy0 + sh * 0.48, sx0 + sw * 0.31, sy0 + sh * 0.22],
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_axis", r),
            confidence=0.80,
            start=[sx0 + sw * 0.06, sy0 + sh * 0.48],
            end=[sx0 + sw * 0.31, sy0 + sh * 0.22],
            color="#222222",
            thickness=2,
            z=5.0,
            ext=_ext("axis"),
        ),
        IR.element(
            id="bottom_mini_vec_beta",
            type="arrow",
            bbox=[sx0 + sw * 0.36, sy0 + sh * 0.70, sx0 + sw * 0.72, sy0 + sh * 0.34],
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_vector", r),
            confidence=0.82,
            start=[sx0 + sw * 0.36, sy0 + sh * 0.70],
            end=[sx0 + sw * 0.72, sy0 + sh * 0.34],
            color="#1f66d1",
            thickness=4,
            z=6.0,
            ext=_ext("vector"),
        ),
        IR.element(
            id="bottom_mini_vec_gamma",
            type="arrow",
            bbox=[sx0 + sw * 0.36, sy0 + sh * 0.70, sx0 + sw * 0.88, sy0 + sh * 0.45],
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_vector", r),
            confidence=0.82,
            start=[sx0 + sw * 0.36, sy0 + sh * 0.70],
            end=[sx0 + sw * 0.88, sy0 + sh * 0.45],
            color="#1c8b72",
            thickness=4,
            z=6.0,
            ext=_ext("vector"),
        ),
        IR.element(
            id="bottom_mini_origin_dot",
            type="oval",
            bbox=[
                sx0 + sw * 0.36 - 5,
                sy0 + sh * 0.70 - 5,
                sx0 + sw * 0.36 + 5,
                sy0 + sh * 0.70 + 5,
            ],
            provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_vector_origin", r),
            confidence=0.82,
            fill="#111111",
            border_color="#111111",
            border_width=1,
            z=6.5,
            ext=_ext("vector_origin"),
        ),
    ]
    elements.extend([
        _axis_label(ir, "x3", [sx0 + sw * 0.02, sy0 + sh * 0.01, sx0 + sw * 0.14, sy0 + sh * 0.12]),
        _axis_label(ir, "x2", [sx0 + sw * 0.31, sy0 + sh * 0.15, sx0 + sw * 0.43, sy0 + sh * 0.26]),
        _axis_label(ir, "x1", [sx0 + sw * 0.39, sy0 + sh * 0.62, sx0 + sw * 0.51, sy0 + sh * 0.74]),
    ])
    check_x0 = x0 + w * 0.49
    check_x1 = x0 + w * 0.88
    check_y0 = y0 + h * 0.24
    check_y1 = y0 + h * 0.76
    elements.append(IR.element(
        id="bottom_checklist_panel",
        type="rounded_rect",
        bbox=[check_x0 - 10, check_y0 + 2, check_x1, check_y1 + 10],
        provenance=IR.provenance("BottomMiniSurfaceAgent", "checklist_panel", r),
        confidence=0.82,
        fill="#ffffff",
        border_color="#9aa0a6",
        border_width=1.2,
        corner=0.08,
        dash=True,
        z=1.5,
        ext=_ext("checklist_panel"),
    ))
    rows = ["zero retraining", "estimator-agnostic", "O(np) overhead"]
    for idx, text in enumerate(rows):
        cy = check_y0 + (idx + 0.7) * (check_y1 - check_y0) / 3.25
        elements.extend([
            IR.element(
                id=f"bottom_check_{idx}",
                type="icon",
                bbox=[check_x0 + 6, cy - 11, check_x0 + 28, cy + 11],
                provenance=IR.provenance("BottomMiniSurfaceAgent", "check_icon", r),
                confidence=0.84,
                icon={"kind": "check", "color": "#2b9c6a"},
                z=6.0,
                ext=_ext("check"),
            ),
            IR.element(
                id=f"bottom_check_text_{idx}",
                type="text",
                bbox=[check_x0 + 36, cy - 15, check_x1 - 6, cy + 15],
                provenance=IR.provenance("BottomMiniSurfaceAgent", "check_text", r),
                confidence=0.86,
                text=text,
                font="Arial",
                font_size=13,
                text_color="#333333",
                align="left",
                z=6.0,
                ext=_ext("check_text"),
            ),
        ])
    return elements


def _axis_label(ir: dict, label: str, bbox: list[float]) -> dict:
    return IR.element(
        id=f"bottom_mini_axis_label_{label}",
        type="text",
        bbox=bbox,
        provenance=IR.provenance("BottomMiniSurfaceAgent", "mini_axis_label", ir.get("round", 0)),
        confidence=0.78,
        text=label,
        font="Times New Roman",
        font_size=14,
        italic=True,
        text_color="#111111",
        align="center",
        z=6.0,
        ext=_ext("axis_label"),
    )


def _remove_orphans(ir: dict, region: list[float]) -> set[str]:
    removable = {"text", "formula", "rounded_rect", "rect", "icon", "arrow", "line", "freeform", "dotcloud", "surface"}
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if (
            eid.startswith("bottom_")
            or eid.startswith("auditor_")
            or eid.startswith("action_card_")
            or eid.startswith("chart_q0_")
            or eid.startswith("failure_summary_")
            or eid.startswith("pipeline_context_")
            or eid.startswith("proc_")
            or not bbox
        ):
            keep.append(el)
            continue
        if el.get("type") in removable and _center_inside(bbox, region):
            removed.add(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _center_inside(bbox: list[float], region: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rx0, ry0, rx1, ry1 = [float(v) for v in region]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _ext(role: str) -> dict:
    return {
        "component": "bottom_mini_surface",
        "component_role": role,
        "strategy": {
            "region_id": "region_bottom_mini_surface",
            "kind": "bottom_mini_surface",
            "primary_method": "mini_surface_checklist",
            "fallback_methods": ["surface_vector_trace", "text_style"],
            "preferred_agent": "BottomMiniSurfaceAgent",
        },
    }
