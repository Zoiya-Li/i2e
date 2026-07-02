"""Raster-local ink → editable native vector paths.

This is not a screenshot fallback: callers store simplified contours that the
PPTX builder renders as PowerPoint freeform shapes.  It is meant for pictograms,
surface shading fragments, and other non-text visual marks that are too specific
for a small library of template icons.
"""
from __future__ import annotations

from typing import Iterable

from PIL import Image


def extract_paths(
    crop: Image.Image,
    *,
    exclude: Iterable[Iterable[float]] | None = None,
    max_paths: int = 40,
    min_area: float = 12.0,
    epsilon_frac: float = 0.012,
    pale: bool = False,
) -> list[dict]:
    """Return editable closed contours in crop-local coordinates.

    Each path is {"points": [[x, y], ...], "fill": "#rrggbb", "alpha": pct}.
    The mask is intentionally color-aware instead of pure dark-ink thresholding
    so pale blue manifold washes and colored technical icons can survive.
    """
    import cv2
    import numpy as np

    arr = np.asarray(crop.convert("RGB"))
    if arr.size == 0:
        return []
    h, w = arr.shape[:2]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    corner = np.concatenate([
        arr[: max(1, h // 12), : max(1, w // 12)].reshape(-1, 3),
        arr[: max(1, h // 12), -max(1, w // 12):].reshape(-1, 3),
        arr[-max(1, h // 12):, : max(1, w // 12)].reshape(-1, 3),
        arr[-max(1, h // 12):, -max(1, w // 12):].reshape(-1, 3),
    ])
    bg = np.median(corner, axis=0)
    dist = np.linalg.norm(arr.astype(float) - bg.reshape(1, 1, 3), axis=2)

    if pale:
        mask = ((dist > 9) & (val < 252)) | ((sat > 14) & (val < 252))
        close_k = 5
    else:
        mask = ((dist > 24) & (val < 250)) | ((sat > 28) & (val < 248)) | (val < 210)
        close_k = 3
    mask = mask.astype("uint8") * 255

    if exclude:
        for box in exclude:
            try:
                x0, y0, x1, y1 = [int(round(v)) for v in box]
            except Exception:
                continue
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paths: list[dict] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 2 or bh < 2:
            continue
        peri = max(1.0, cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        fill_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(fill_mask, [cnt], -1, 1, -1)
        pixels = arr[fill_mask.astype(bool)]
        if len(pixels) == 0:
            continue
        med = np.median(pixels, axis=0)
        if med.min() > 238 and (med.max() - med.min()) < 10:
            continue
        color = "#%02x%02x%02x" % tuple(int(v) for v in med)
        alpha = 24 if pale and med.min() > 185 else (45 if pale else 100)
        fill = None if pale else color
        paths.append({
            "points": [[float(px), float(py)] for px, py in approx],
            "fill": fill,
            "line": color,
            "alpha": alpha,
            "area": round(area, 2),
        })
        if len(paths) >= max_paths:
            break
    return paths
