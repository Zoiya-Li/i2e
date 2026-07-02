"""ProceduralSurfaceAgent: native generated 3D manifold reconstruction."""
from __future__ import annotations

import copy
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR, procedural_surface
from work.diagram2ppt.v3.agents.base import Agent


class ProceduralSurfaceAgent(Agent):
    """Build large 3D surface regions with deterministic native geometry."""

    name = "ProceduralSurfaceAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        seed = _ensure_surface_seed(ir, task)
        removed = _remove_surface_orphans(ir, task)
        before = {
            str(e.get("id")): _surface_signature(e)
            for e in ir.get("elements", [])
            if e.get("type") == "surface"
        }
        before_generated = {
            str(e.get("id")) for e in ir.get("elements", [])
            if str(e.get("id", "")).startswith(("proc_axis_", "proc_ci_", "proc_risk_"))
        }
        stats = procedural_surface.apply(ir)
        if not any(stats.values()) and not removed and not seed:
            self.record_contract_result(ir, task, [], status="no_procedural_change")
            return []

        changed: list[str] = list(removed)
        if seed:
            changed.append(seed)
        for e in ir.get("elements", []):
            eid = str(e.get("id"))
            if e.get("type") == "surface" and before.get(eid) != _surface_signature(e):
                changed.append(eid)
            if eid.startswith(("proc_axis_", "proc_ci_", "proc_risk_")) and eid not in before_generated:
                changed.append(eid)
        self.record_contract_result(ir, task, changed)
        return changed


def _ensure_surface_seed(ir: dict, task: dict) -> str:
    """Create a surface element from a method contract when perception missed it."""
    method = str(task.get("locked_method") or ((task.get("representation") or {}).get("method") or ""))
    if method != "procedural_surface":
        return ""
    region = _surface_region_bbox(ir, task)
    if not region:
        return ""
    for el in ir.get("elements", []):
        if el.get("type") == "surface" and _bbox_overlap_fraction(el.get("bbox"), region) > 0.35:
            return ""
    seed_id = f"surface_seed_{abs(hash(tuple(round(float(v), 1) for v in region))) % 1000000}"
    surface = IR.element(
        id=seed_id,
        type="surface",
        bbox=region,
        provenance=IR.provenance(
            "ProceduralSurfaceAgent",
            "surface_seed_from_method_contract",
            ir.get("round", 0),
        ),
        confidence=0.72,
        z=-0.5,
        fill="#ffffff",
        ext={
            "procedural_surface_seed": True,
            "strategy": {
                "kind": task.get("kind"),
                "primary_method": method,
                "region_id": task.get("region_id"),
                "representation": task.get("representation") or {},
            },
        },
    )
    ir.setdefault("elements", []).append(surface)
    ir.setdefault("history", []).append({
        "agent": ProceduralSurfaceAgent.name,
        "action": "surface_seed_from_method_contract",
        "round": ir.get("round", 0),
        "surface_id": seed_id,
        "region": region,
    })
    return seed_id


def _surface_signature(el: dict) -> tuple:
    bbox = tuple(round(float(v), 3) for v in (el.get("bbox") or []))
    curves = (el.get("wave_bands") or {}).get("curves") or []
    curve_anchor = None
    if curves and curves[0] and curves[-1]:
        curve_anchor = (
            tuple(round(float(v), 3) for v in curves[0][0]),
            tuple(round(float(v), 3) for v in curves[0][-1]),
            tuple(round(float(v), 3) for v in curves[-1][0]),
            tuple(round(float(v), 3) for v in curves[-1][-1]),
        )
    return (
        bbox,
        len((el.get("wave_bands") or {}).get("curves") or []),
        len(el.get("streamlines") or []),
        len(el.get("dots") or []),
        len(el.get("paths") or []),
        curve_anchor,
        copy.deepcopy((el.get("ext") or {}).get("procedural_surface")),
    )


def _remove_surface_orphans(ir: dict, task: dict) -> list[str]:
    """Remove native trace fragments that compete with the procedural surface.

    Text, formulas, arrows, and the CI inset card are preserved.  The goal is
    not to delete content broadly; it is to enforce that the manifold itself is
    expressed by one procedural native payload instead of a pile of residual
    freeforms and duplicated dotclouds.
    """
    region = _surface_region_bbox(ir, task)
    if not region:
        return []

    protected_boxes = [
        e.get("bbox") for e in ir.get("elements", [])
        if e.get("type") in {"rounded_rect", "chart"} and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), region) > 0.35
    ]
    keep: list[dict] = []
    removed: list[str] = []
    for el in ir.get("elements", []):
        typ = el.get("type")
        bbox = el.get("bbox")
        if not bbox or typ == "surface":
            keep.append(el)
            continue
        if _bbox_overlap_fraction(bbox, region) < 0.22:
            keep.append(el)
            continue
        if typ == "freeform" and not _inside_any(bbox, protected_boxes):
            removed.append(str(el.get("id")))
            continue
        if typ == "dotcloud" and not _inside_any(bbox, protected_boxes):
            removed.append(str(el.get("id")))
            continue
        if typ in {"path", "polygon"} and not _inside_any(bbox, protected_boxes):
            removed.append(str(el.get("id")))
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
        ir.setdefault("repair_history", []).append({
            "agent": ProceduralSurfaceAgent.name,
            "action": "remove_surface_orphans",
            "removed": removed,
            "round": ir.get("round", 0),
        })
    return removed


def _surface_region_bbox(ir: dict, task: dict) -> list[float] | None:
    bbox = task.get("bbox")
    if bbox and len(bbox) == 4 and max(float(v) for v in bbox) > 0:
        return [float(v) for v in bbox]
    surfaces = []
    width = float((ir.get("canvas") or {}).get("width_px") or (ir.get("image") or {}).get("width") or 1)
    for el in ir.get("elements", []):
        if el.get("type") != "surface" or not el.get("bbox"):
            continue
        x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
        if x0 < width * 0.52 and x1 - x0 > 500 and y1 - y0 > 220:
            surfaces.append([x0, y0, x1, y1])
    if not surfaces:
        return None
    return max(surfaces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _inside_any(bbox: list[float], boxes: list[list[float] | None]) -> bool:
    return any(box and _bbox_overlap_fraction(bbox, box) > 0.75 for box in boxes)


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
