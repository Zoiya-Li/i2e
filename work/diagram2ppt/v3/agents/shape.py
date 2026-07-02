"""Shape/Icon Agent: detects missing rectangles, cards, arrows, and icons.

Phase 1 uses the locally cached Grounding DINO tiny model via the local_model
provider. Later this can be replaced or augmented with YOLO-World / SAM.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent
from work.diagram2ppt.v3.providers.local import LocalModelProvider


class ShapeAgent(Agent):
    """Specialist agent for geometry detection and repair."""

    name = "ShapeAgent"

    def __init__(self) -> None:
        self.provider = LocalModelProvider(model_name="grounding-dino-tiny")

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("type") == "missing_element":
            bbox = defect.get("bbox")
            if bbox:
                return self._detect_in_region(ir, original, bbox)

        # Default: scan the whole image for missing shapes.
        return self._detect_in_region(ir, original,
                                      [0, 0, original.width, original.height])

    def _detect_in_region(self, ir: dict, original: Image.Image,
                          region: list[float]) -> list[str]:
        x0, y0, x1, y1 = region
        crop = original.crop((max(0, int(x0)), max(0, int(y0)),
                              min(original.width, int(x1)),
                              min(original.height, int(y1))))
        if crop.width < 10 or crop.height < 10:
            return []

        prompt = (
            "rectangle. rounded rectangle. card. box. panel. "
            "arrow. icon. text box. chart. button. badge."
        )
        try:
            detections = self.provider.detect(crop, prompt,
                                              threshold=0.18, nms_threshold=0.35)
        except Exception:
            detections = []

        existing = [e for e in ir.get("elements", []) if "bbox" in e]
        changed: list[str] = []
        for d in detections:
            cx0, cy0, cx1, cy1 = d["bbox"]
            # Map crop coordinates back to original image coordinates.
            abs_bbox = [cx0 + x0, cy0 + y0, cx1 + x0, cy1 + y0]

            # Skip if it overlaps heavily with an existing element.
            if self._has_overlap(abs_bbox, existing, iou_threshold=0.5):
                continue

            label = (d.get("label") or "").lower()
            el_type = self._label_to_type(label)
            if el_type is None:
                continue

            el_id = f"shape_{len(ir['elements'])}_{int(abs_bbox[0])}_{int(abs_bbox[1])}"
            el = IR.element(
                id=el_id,
                type=el_type,
                bbox=abs_bbox,
                provenance=IR.provenance(self.name, "grounding_dino",
                                         ir.get("round", 0)),
                confidence=float(d.get("score", 0.0)),
            )
            # Minimal style inference from crop.
            style = _infer_region_style(original, abs_bbox)
            el["fill"] = style.get("fill", "")
            el["border_color"] = style.get("border_color", "#777777")
            el["border_width"] = style.get("border_width", 1)
            ir["elements"].append(el)
            changed.append(el_id)

        # Deterministic fallback: VLM/detector structure passes are unstable on
        # framework diagrams.  When the missing region is full of thin rounded
        # rectangles/cards, recover them directly from border pixels.
        for abs_bbox in _detect_rectangular_regions(crop, region):
            if self._has_overlap(abs_bbox, existing, iou_threshold=0.45):
                continue
            if self._has_overlap(abs_bbox, [e for e in ir.get("elements", [])
                                            if "bbox" in e], iou_threshold=0.45):
                continue
            el_id = f"cv_shape_{len(ir['elements'])}_{int(abs_bbox[0])}_{int(abs_bbox[1])}"
            el = IR.element(
                id=el_id,
                type="rounded_rect",
                bbox=abs_bbox,
                provenance=IR.provenance(self.name, "cv_rectangular_region",
                                         ir.get("round", 0)),
                confidence=0.72,
            )
            style = _infer_region_style(original, abs_bbox)
            el["fill"] = style.get("fill", "")
            el["border_color"] = style.get("border_color", "#777777")
            el["border_width"] = style.get("border_width", 1)
            el["corner"] = 0.18
            ir["elements"].append(el)
            changed.append(el_id)
        return changed

    @staticmethod
    def _has_overlap(bbox: list[float], elements: list[dict],
                     iou_threshold: float = 0.5) -> bool:
        for el in elements:
            eb = el["bbox"]
            if _iou(bbox, eb) > iou_threshold:
                return True
        return False

    @staticmethod
    def _label_to_type(label: str) -> str | None:
        if any(k in label for k in ("rounded rectangle", "rounded_rect")):
            return "rounded_rect"
        if any(k in label for k in ("rectangle", "box", "card", "panel", "button", "badge")):
            return "rect"
        if "arrow" in label:
            return "arrow"
        if "icon" in label:
            return "icon"
        if "text box" in label:
            return "text"
        if "chart" in label:
            return "chart"
        return None


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = max(0, ax1 - ax0) * max(0, ay1 - ay0) + max(0, bx1 - bx0) * max(0, by1 - by0) - inter
    return inter / union if union else 0.0


def _detect_rectangular_regions(crop: Image.Image,
                                region: list[float]) -> list[list[float]]:
    """Detect card/panel outlines in a crop via deterministic CV.

    This is deliberately conservative: it returns medium/large rectangular
    contours only, avoiding text glyph boxes.  Coordinates are absolute image
    pixels.
    """
    arr = np.asarray(crop.convert("RGB"))
    if arr.size == 0:
        return []
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Text produces many small edges; close thin border gaps so card outlines
    # become connected components.
    edges = cv2.Canny(gray, 45, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((9, 9), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rx0, ry0, rx1, ry1 = region
    out: list[list[float]] = []
    crop_area = max(1, crop.width * crop.height)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 45 or h < 35:
            continue
        area = w * h
        if area < 0.0025 * crop_area or area > 0.75 * crop_area:
            continue
        # Rectangular outlines have contour area close to their bbox envelope.
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        rectangularity = hull_area / max(1.0, float(area))
        if rectangularity < 0.55:
            continue
        # Reject very skinny text/axis fragments.
        aspect = w / max(1, h)
        if aspect > 8 or aspect < 0.12:
            continue
        out.append([float(rx0 + x), float(ry0 + y),
                    float(rx0 + x + w), float(ry0 + y + h)])

    out.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    deduped: list[list[float]] = []
    for b in out:
        if any(_iou(b, o) > 0.55 for o in deduped):
            continue
        deduped.append(b)
    return deduped[:24]


def _infer_region_style(original: Image.Image, bbox: list[float]) -> dict:
    arr = np.asarray(original.convert("RGB"))
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.width, x1), min(original.height, y1)
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        return {"fill": "", "border_color": "#777777", "border_width": 1}
    h, w = crop.shape[:2]
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.abs(crop.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    ink = dist > 45
    # Border color: sample ink near the perimeter; fall back to all ink.
    perim = np.zeros((h, w), dtype=bool)
    band = max(2, min(8, min(h, w) // 8))
    perim[:band, :] = True
    perim[-band:, :] = True
    perim[:, :band] = True
    perim[:, -band:] = True
    border_px = crop[ink & perim]
    if len(border_px) < 8:
        border_px = crop[ink]
    if len(border_px) >= 3:
        col = np.median(border_px, axis=0).astype(int)
        border_color = "#%02x%02x%02x" % tuple(col)
    else:
        border_color = "#777777"
    # Fill color: center median if it is not just white page background.
    cx0, cy0 = int(w * 0.28), int(h * 0.28)
    cx1, cy1 = int(w * 0.72), int(h * 0.72)
    center = crop[cy0:max(cy0 + 1, cy1), cx0:max(cx0 + 1, cx1)]
    fill_rgb = np.median(center.reshape(-1, 3), axis=0).astype(int)
    fill = ""
    if np.abs(fill_rgb.astype(int) - np.array([255, 255, 255])).sum() > 20:
        fill = "#%02x%02x%02x" % tuple(fill_rgb)
    return {"fill": fill, "border_color": border_color, "border_width": 1}
