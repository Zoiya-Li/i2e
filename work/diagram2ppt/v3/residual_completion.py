"""Residual-driven native vector completion.

This module fills a gap between verification and repair: when the current
native deck leaves substantial non-text ink unexplained, add editable
PowerPoint freeform paths derived from the residual.  It is not a raster crop
fallback; the output is an explicit set of vector paths in the IR.
"""
from __future__ import annotations

from typing import Callable

from PIL import Image


def add_residual_freeforms(
    ir: dict,
    original: Image.Image,
    rendered_png_path: str,
    *,
    defects: list[dict] | None = None,
    replace_unreliable: bool = False,
    max_regions: int = 16,
    log: Callable[[str], None] = print,
) -> int:
    """Add native freeform elements for residual non-text graphics.

    Text and formulas are deliberately excluded so the pipeline keeps them as
    editable text/equations instead of silently converting glyphs to paths.
    """
    try:
        import cv2
        import numpy as np
        from work.diagram2ppt.v2.native_trace import extract_paths
    except Exception as exc:
        log(f"[ResidualCompletion] unavailable: {exc}")
        return 0

    rendered = Image.open(rendered_png_path).convert("RGB")
    if rendered.size != original.size:
        rendered = rendered.resize(original.size, Image.LANCZOS)

    orig = np.asarray(original.convert("RGB"))
    rend = np.asarray(rendered)
    h, w = orig.shape[:2]

    hsv = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    diff = np.linalg.norm(orig.astype(float) - rend.astype(float), axis=2)

    if replace_unreliable:
        removed = _remove_unreliable_complex_elements(ir, defects or [])
        if removed:
            log(f"[ResidualCompletion] removed {removed} unreliable complex elements")

    # Colored or dark original ink that the current render does not match.
    ink = (gray < 238) | ((sat > 18) & (val < 252))
    mask = (ink & (diff > 30)).astype("uint8") * 255

    target_regions = _target_regions(defects or [], w, h, include_complex=replace_unreliable)
    if target_regions:
        target_mask = np.zeros((h, w), dtype="uint8")
        for box in target_regions:
            x0, y0, x1, y1 = _clamp_box(box, w, h, pad=6)
            if x1 > x0 and y1 > y0:
                target_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, target_mask)

    for box in _text_exclusion_boxes(ir):
        x0, y0, x1, y1 = _clamp_box(box, w, h, pad=3)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = max(120, int(w * h * 0.00006))
    comps: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < min_area or bw < 4 or bh < 4:
            continue
        if bw * bh > w * h * 0.26:
            continue
        comps.append((x, y, bw, bh, area))
    comps.sort(key=lambda item: item[4], reverse=True)

    added = 0
    existing_ids = {str(e.get("id")) for e in ir.get("elements", [])}
    for x, y, bw, bh, area in comps[:max_regions]:
        x0, y0, x1, y1 = _clamp_box([x, y, x + bw, y + bh], w, h, pad=4)
        bbox = [float(x0), float(y0), float(x1), float(y1)]
        if _covered_by_existing_residual_freeform(bbox, ir.get("elements", [])):
            continue

        crop = original.crop((x0, y0, x1, y1))
        local_excludes = _local_text_excludes(ir, bbox)
        pale = _should_trace_as_pale(crop, area)
        paths = extract_paths(
            crop,
            exclude=local_excludes,
            max_paths=80 if area > 6000 else 36,
            min_area=max(8.0, crop.width * crop.height * (0.00035 if pale else 0.0008)),
            epsilon_frac=0.007 if area > 6000 else 0.012,
            pale=pale,
        )
        if not paths:
            continue
        for path in paths:
            path["source"] = "residual_completion"
            if pale:
                path["closed"] = False
                path["fill"] = None
                path["line_width"] = max(0.35, float(path.get("line_width", 0.35)))
                path["alpha"] = min(int(path.get("alpha", 35)), 42)

        base_id = f"residual_freeform_{added:02d}_{x0}_{y0}_{x1}_{y1}"
        eid = base_id
        suffix = 1
        while eid in existing_ids:
            eid = f"{base_id}_{suffix}"
            suffix += 1
        existing_ids.add(eid)
        ir.setdefault("elements", []).append({
            "id": eid,
            "type": "freeform",
            "status": "native",
            "bbox": bbox,
            "confidence": 0.62,
            "provenance": {
                "agent": "ResidualCompletion",
                "action": "original_render_residual_to_native_paths",
                "round": ir.get("round", 0),
            },
            "repair_history": [],
            "defects": [],
            "paths": paths,
            "fill": "",
            "border_color": "",
            "z": 4.5 if area < 9000 else 1.5,
            "ext": {
                "residual_area": int(area),
                "pale_trace": bool(pale),
            },
        })
        added += 1

    if added:
        ir.setdefault("quality_gate", {}).setdefault("residual_completion", []).append({
            "round": ir.get("round", 0),
            "added_freeforms": added,
        })
        log(f"[ResidualCompletion] added {added} native freeform residual regions")
    return added


def _text_exclusion_boxes(ir: dict) -> list[list[float]]:
    return [
        e["bbox"] for e in ir.get("elements", [])
        if e.get("type") in ("text", "formula") and e.get("bbox")
    ]


def _target_regions(defects: list[dict], w: int, h: int,
                    include_complex: bool = False) -> list[list[float]]:
    """Prefer verifier-localized missing ink over global residual scanning."""
    regions: list[list[float]] = []
    for defect in defects:
        bbox = defect.get("bbox") or []
        if len(bbox) != 4:
            continue
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
            0.0, float(bbox[3]) - float(bbox[1]))
        if area < w * h * 0.00045:
            continue
        if defect.get("type") == "missing_element":
            # If the verifier already found an owner, let that specialist own it.
            if defect.get("element_id"):
                continue
            regions.append([float(v) for v in bbox[:4]])
        elif include_complex and defect.get("type") == "high_residual":
            if float(defect.get("severity", 0.0)) < 0.55:
                continue
            regions.append([float(v) for v in bbox[:4]])
    return regions


def _remove_unreliable_complex_elements(ir: dict, defects: list[dict]) -> int:
    complex_types = {"surface", "dotcloud", "chart", "icon", "arrow", "line", "freeform"}
    elements = {e.get("id"): e for e in ir.get("elements", [])}
    remove_ids: set[str] = set()
    for defect in defects:
        if defect.get("type") not in {"high_residual", "connector_mismatch"}:
            continue
        if float(defect.get("severity", 0.0)) < 0.55:
            continue
        eid = defect.get("element_id")
        el = elements.get(eid)
        if not el or el.get("type") not in complex_types:
            continue
        if (el.get("ext") or {}).get("procedural_surface"):
            continue
        remove_ids.add(str(eid))
    if not remove_ids:
        return 0
    ir["elements"] = [
        e for e in ir.get("elements", [])
        if str(e.get("id")) not in remove_ids
    ]
    ir.setdefault("quality_gate", {}).setdefault("residual_replacement", []).append({
        "round": ir.get("round", 0),
        "removed": sorted(remove_ids),
    })
    return len(remove_ids)


def _local_text_excludes(ir: dict, bbox: list[float]) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    out: list[list[float]] = []
    for box in _text_exclusion_boxes(ir):
        bx0, by0, bx1, by1 = [float(v) for v in box[:4]]
        ix0, iy0 = max(x0, bx0), max(y0, by0)
        ix1, iy1 = min(x1, bx1), min(y1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        out.append([ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0])
    return out


def _clamp_box(box: list[float], w: int, h: int, pad: int = 0) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    )


def _should_trace_as_pale(crop: Image.Image, area: int) -> bool:
    import numpy as np

    arr = np.asarray(crop.convert("RGB"))
    if arr.size == 0:
        return False
    med = np.median(arr.reshape(-1, 3), axis=0)
    bright = float(med.min()) > 165
    return bool(area > 4500 and bright)


def _covered_by_existing_residual_freeform(bbox: list[float], elements: list[dict]) -> bool:
    for el in elements:
        if el.get("type") != "freeform" or not el.get("bbox"):
            continue
        if _iou(bbox, el["bbox"]) > 0.72:
            return True
    return False


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)
