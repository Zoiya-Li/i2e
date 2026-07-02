"""Style Agent: infers fill, border, and text appearance from the original image.

Many high-residual defects are not content errors but style mismatches
(fill color, border color/width, text color/size) that content agents cannot
fix.  StyleAgent samples the original crop and updates the native style fields
so the rendered PPTX matches the source.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class StyleAgent(Agent):
    """Specialist agent for visual appearance repair."""

    name = "StyleAgent"

    SHAPE_TYPES = {"rect", "rounded_rect", "oval", "diamond", "hexagon",
                   "parallelogram"}

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el is None:
                return []
            if el and el.get("type") in self.SHAPE_TYPES:
                return self._repair_shape_style(ir, original, el)
            if el and el.get("type") == "text":
                return self._repair_text_style(ir, original, el)

        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") in self.SHAPE_TYPES:
                changed.extend(self._repair_shape_style(ir, original, el))
            elif el.get("type") == "text":
                changed.extend(self._repair_text_style(ir, original, el))
        return changed

    def _repair_shape_style(self, ir: dict, original: Image.Image,
                            el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []

        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(original.width, int(x1))
        y1 = min(original.height, int(y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return []

        crop = original.crop((x0, y0, x1, y1))
        img = np.array(crop)

        style = _infer_shape_style(img)
        updated = False
        for key in ("fill", "border_color", "border_width"):
            if style.get(key) is not None and style[key] != el.get(key):
                el[key] = style[key]
                updated = True

        if updated:
            el.setdefault("repair_history", []).append({
                "agent": self.name,
                "action": "shape_style_inference",
                "round": ir.get("round", 0),
            })
            return [el["id"]]
        return []

    def _repair_text_style(self, ir: dict, original: Image.Image,
                           el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []

        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(original.width, int(x1))
        y1 = min(original.height, int(y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return []

        crop = original.crop((x0, y0, x1, y1))
        img = np.array(crop)

        style = _infer_text_style(img)
        updated = False
        typography_controlled = bool(
            ((el.get("ext") or {}).get("typography") or {}).get("source")
        )
        if typography_controlled:
            return []
        keys = ("text_color", "font_size")
        for key in keys:
            if style.get(key) is not None and style[key] != el.get(key):
                el[key] = style[key]
                updated = True

        if updated:
            el.setdefault("repair_history", []).append({
                "agent": self.name,
                "action": "text_style_inference",
                "round": ir.get("round", 0),
                "typography_controlled": typography_controlled,
            })
            return [el["id"]]
        return []


def _infer_shape_style(img: np.ndarray) -> dict[str, Any]:
    """Infer fill color, border color, and border width from a shape crop."""
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return {}

    # Work in LAB for perceptually uniform distance.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # K-means to separate a small number of dominant colors.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=3)
    order = np.argsort(-counts)

    # Background is usually the dominant color touching the image border.
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_labels = labels.reshape(h, w)[border_mask]
    bg_label = int(np.bincount(border_labels, minlength=3).argmax())

    # The second-most-common non-background color is likely fill or border.
    fill_label = None
    border_label = None
    for idx in order:
        if idx == bg_label:
            continue
        if fill_label is None:
            fill_label = int(idx)
        elif border_label is None:
            border_label = int(idx)
            break

    # Convert centers back to RGB hex.
    centers_rgb = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8),
                               cv2.COLOR_LAB2RGB).reshape(-1, 3)

    def _hex(idx: int | None) -> str | None:
        if idx is None:
            return None
        r, g, b = centers_rgb[idx]
        return f"#{r:02x}{g:02x}{b:02x}"

    # Determine fill: compare center pixel color to background.
    cx, cy = w // 2, h // 2
    center_label = int(labels.reshape(h, w)[cy, cx])

    result: dict[str, Any] = {}

    if center_label != bg_label and fill_label is not None and center_label == fill_label:
        fill_hex = _hex(fill_label)
        if fill_hex:
            result["fill"] = fill_hex
    else:
        result["fill"] = ""

    # Border: look for a strong edge around the perimeter.
    if border_label is not None:
        result["border_color"] = _hex(border_label)
    elif fill_label is not None and center_label == bg_label:
        # Hollow shape with no second color: border is the non-bg center color
        # if any.
        result["border_color"] = _hex(fill_label)

    # Estimate border width from Canny edges.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if edges.any():
        # Sample edge thickness along the four sides.
        thicknesses = []
        for row in (edges[0], edges[-1], edges[:, 0], edges[:, -1]):
            row = np.asarray(row).ravel()
            runs = []
            cur = 0
            for v in row:
                if v:
                    cur += 1
                else:
                    if cur:
                        runs.append(cur)
                        cur = 0
            if cur:
                runs.append(cur)
            if runs:
                thicknesses.append(np.median(runs))
        if thicknesses:
            result["border_width"] = max(1, int(round(np.median(thicknesses))))

    return result


def _infer_text_style(img: np.ndarray) -> dict[str, Any]:
    """Infer text color and approximate font size from a text crop."""
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return {}

    result: dict[str, Any] = {}

    # Text color: darkest/lightest dominant cluster opposite to background.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=2)

    # Background is the more frequent color touching the border.
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    bg_label = int(np.bincount(labels.reshape(h, w)[border_mask],
                               minlength=2).argmax())
    text_label = 1 - bg_label

    centers_rgb = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8),
                               cv2.COLOR_LAB2RGB).reshape(-1, 3)
    r, g, b = centers_rgb[text_label]
    result["text_color"] = f"#{r:02x}{g:02x}{b:02x}"

    # Font size estimate from height of text contours.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Determine whether text is dark-on-light or light-on-dark.
    bg_l = centers[bg_label][0]
    text_l = centers[text_label][0]
    if text_l < bg_l:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours
               if cv2.boundingRect(c)[3] > 3]
    if heights:
        # Convert pixel height to PowerPoint font size (pt ≈ px * 0.75).
        result["font_size"] = max(8, int(round(np.median(heights) * 0.75)))

    return result
