"""Cleanup pass for regions rebuilt by component agents.

Component agents intentionally replace noisy OCR/CV fragments with coherent
native groups.  After a proposal is accepted, stale fragments from the old
perception layer can remain in the same region and visually double-print text
or connectors.  This pass removes only unowned leftovers inside regions that
already have a generated component owner.
"""
from __future__ import annotations

from typing import Callable


OWNER_PREFIXES = (
    "pipeline_context_",
    "auditor_",
    "failure_summary_",
    "chart_q0_",
    "bottom_",
    "action_card_",
)

CLEANABLE_TYPES = {
    "text",
    "formula",
    "icon",
    "line",
    "arrow",
    "dotcloud",
    "freeform",
    "rect",
    "rounded_rect",
    "oval",
}


def apply(ir: dict, log: Callable[[str], None] | None = None) -> dict:
    """Remove stale unowned fragments inside accepted component regions."""
    regions = _component_regions(ir)
    if not regions:
        return {"removed": 0, "regions": 0, "removed_ids": []}

    kept: list[dict] = []
    removed: list[str] = []
    for el in ir.get("elements", []):
        eid = str(el.get("id", ""))
        bbox = el.get("bbox")
        if not bbox or eid.startswith(OWNER_PREFIXES):
            kept.append(el)
            continue
        if not _is_cleanable(el):
            kept.append(el)
            continue
        if _is_left_surface_annotation(el, ir):
            kept.append(el)
            continue
        owner = _containing_component_region(bbox, regions)
        if owner is None:
            kept.append(el)
            continue
        removed.append(eid)

    if removed:
        ir["elements"] = kept
        entry = {
            "agent": "ComponentCleanup",
            "action": "remove_stale_component_fragments",
            "round": ir.get("round", 0),
            "removed": removed,
        }
        ir.setdefault("repair_history", []).append(entry)
        ir.setdefault("component_cleanup", []).append(entry)
        if log:
            log("[ComponentCleanup] removed "
                f"{len(removed)} stale component fragments")
    return {"removed": len(removed), "regions": len(regions), "removed_ids": removed}


def _component_regions(ir: dict) -> list[dict]:
    grouped: dict[str, list[list[float]]] = {p: [] for p in OWNER_PREFIXES}
    for el in ir.get("elements", []):
        eid = str(el.get("id", ""))
        bbox = el.get("bbox")
        if not bbox:
            continue
        for prefix in OWNER_PREFIXES:
            if eid.startswith(prefix):
                grouped[prefix].append([float(v) for v in bbox])
                break

    regions: list[dict] = []
    for prefix, boxes in grouped.items():
        if len(boxes) < _minimum_owned_elements(prefix):
            continue
        box = _union(boxes)
        pad_x, pad_y = _padding(prefix, box)
        regions.append({
            "prefix": prefix,
            "bbox": [
                box[0] - pad_x,
                box[1] - pad_y,
                box[2] + pad_x,
                box[3] + pad_y,
            ],
        })
    return regions


def _minimum_owned_elements(prefix: str) -> int:
    if prefix in {"failure_summary_", "bottom_"}:
        return 5
    if prefix == "chart_q0_":
        return 8
    if prefix == "action_card_":
        return 10
    return 12


def _padding(prefix: str, bbox: list[float]) -> tuple[float, float]:
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    if prefix == "action_card_":
        return w * 0.025, h * 0.05
    if prefix == "bottom_":
        return w * 0.035, h * 0.06
    if prefix == "chart_q0_":
        return w * 0.035, h * 0.08
    return w * 0.025, h * 0.045


def _is_cleanable(el: dict) -> bool:
    if el.get("type") not in CLEANABLE_TYPES:
        return False
    eid = str(el.get("id", ""))
    if eid.startswith(("proc_", "e0")):
        return False
    ext = el.get("ext") or {}
    if ext.get("procedural_surface") or ext.get("procedural_surface_axis"):
        return False
    if ext.get("procedural_surface_axis_label") or ext.get("procedural_surface_ci"):
        return False
    return True


def _is_left_surface_annotation(el: dict, ir: dict) -> bool:
    bbox = el.get("bbox")
    if not bbox:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    width = float((ir.get("canvas") or {}).get("width_px")
                  or (ir.get("image") or {}).get("width") or 1)
    # Preserve labels, formulas, and arrows around the left 3D manifold.  Those
    # are semantic annotations, not component leftovers.
    return x1 < width * 0.53 and y0 < 820 and el.get("type") in {
        "text", "formula", "arrow", "line",
    }


def _containing_component_region(bbox: list[float], regions: list[dict]) -> dict | None:
    cx = (float(bbox[0]) + float(bbox[2])) / 2
    cy = (float(bbox[1]) + float(bbox[3])) / 2
    for region in regions:
        rb = region["bbox"]
        if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
            # Also require meaningful overlap so a nearby global caption is not
            # removed just because its center touches padded region bounds.
            if _bbox_overlap_fraction(bbox, rb) >= 0.20:
                return region
    return None


def _union(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


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
