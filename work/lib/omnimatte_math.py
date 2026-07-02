"""Pure numpy helpers for omnimatte layer construction. No IO, no GPU."""
from __future__ import annotations
import numpy as np
import cv2


def delta_alpha(before: np.ndarray, after: np.ndarray,
                smooth_sigma: float = 1.5, thresh: float = 0.08) -> np.ndarray:
    """Alpha (uint8 HxW) = normalized magnitude of the RGB change between `before`
    (object present) and `after` (object + its effects removed). Smooths speckle and
    drops changes below `thresh` (fraction of 255)."""
    b = before.astype(np.float32)
    a = after.astype(np.float32)
    mag = np.abs(b - a).mean(axis=2) / 255.0           # 0..1 per pixel
    if smooth_sigma > 0:
        mag = cv2.GaussianBlur(mag, (0, 0), smooth_sigma)
    mag[mag < thresh] = 0.0
    m = mag.max()
    if m > 0:
        mag = mag / m
    return (np.clip(mag, 0, 1) * 255).astype(np.uint8)


def build_layer(original: np.ndarray, before: np.ndarray, after: np.ndarray,
                obj_bbox: tuple[int, int, int, int], clamp_pad: int = 24):
    """Return (rgba_crop, (x0,y0,x1,y1)). RGB = original pixels; alpha = delta_alpha
    but zeroed outside the object's bbox dilated by `clamp_pad` (so a layer cannot carry
    an unrelated distant change). Crop is the alpha's tight bounding box."""
    H, W = original.shape[:2]
    bx0, by0, bx1, by1 = obj_bbox
    cx0, cy0 = max(0, bx0 - clamp_pad), max(0, by0 - clamp_pad)
    cx1, cy1 = min(W, bx1 + clamp_pad), min(H, by1 + clamp_pad)
    alpha = delta_alpha(before, after)
    clamp = np.zeros((H, W), np.uint8)
    clamp[cy0:cy1, cx0:cx1] = alpha[cy0:cy1, cx0:cx1]
    ys, xs = np.where(clamp > 0)
    if xs.size == 0:                       # nothing survived -> degenerate 1px layer
        return np.zeros((1, 1, 4), np.uint8), (bx0, by0, bx0 + 1, by0 + 1)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    rgba = np.dstack([original[y0:y1, x0:x1], clamp[y0:y1, x0:x1]])
    return rgba, (x0, y0, x1, y1)
