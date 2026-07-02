"""Render-diff scoring: per-element residual + unexplained-ink coverage.

Two independent signals drive the loop:

  element_residual   "is THIS element drawn right?"  1-SSIM between the
                     original and the render inside the element's bbox.
  coverage           "what did we MISS entirely?"  ink pixels in the original
                     that fall inside no element bbox, clustered into
                     candidate regions for the identify pass.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

INK_THRESHOLD = 240          # gray < this = ink on a white-background diagram
MIN_COMPONENT_FRAC = 5e-4    # ignore unexplained blobs smaller than this


def _gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float64)


def element_residual(original: Image.Image, rendered: Image.Image,
                     bbox: list, pad: float = 0.05,
                     exclude: list | None = None) -> float:
    """1 - SSIM of the bbox region (padded slightly to catch border misses).

    Regions are downsampled to <=160px and lightly blurred so antialiasing
    and font differences don't drown out real geometry/color errors.

    `exclude` lists child bboxes (image coords) to neutralize — a container
    must be judged on its own shell, not on the nested elements inside it
    (those carry their own residuals).
    """
    from skimage.metrics import structural_similarity as ssim
    import cv2

    w, h = original.size
    x0, y0, x1, y1 = bbox
    px = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = int(max(0, x0 - px[0])); y0 = int(max(0, y0 - px[1]))
    x1 = int(min(w, x1 + px[0])); y1 = int(min(h, y1 + px[1]))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0

    a = _gray(original.crop((x0, y0, x1, y1)))
    b = _gray(rendered.crop((x0, y0, x1, y1)))
    for ex in exclude or []:
        ex0 = int(max(0, ex[0] - x0)); ey0 = int(max(0, ex[1] - y0))
        ex1 = int(min(x1 - x0, ex[2] - x0)); ey1 = int(min(y1 - y0, ex[3] - y0))
        if ex1 > ex0 and ey1 > ey0:
            b[ey0:ey1, ex0:ex1] = a[ey0:ey1, ex0:ex1]  # zero diff there

    edge = max(a.shape)
    if edge > 160:
        f = 160 / edge
        size = (max(8, int(a.shape[1] * f)), max(8, int(a.shape[0] * f)))
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (3, 3), 1.0)
    b = cv2.GaussianBlur(b, (3, 3), 1.0)

    side = min(a.shape)
    if side < 7:  # SSIM window won't fit on sliver bboxes — mean-diff fallback
        return float(round(np.abs(a - b).mean() / 255.0, 4))
    win = min(7, side if side % 2 == 1 else side - 1)
    score = ssim(a, b, data_range=255.0, win_size=win)
    return float(round(1.0 - score, 4))


def text_residual(original: Image.Image, rendered: Image.Image,
                  bbox: list, pad: float = 0.1) -> float:
    """Edge-overlap residual for TEXT: 1 - F1 of dilated Canny edges.

    SSIM punishes every font difference, demoting perfectly usable text boxes
    to screenshots (30 false demotions on the framework.png run). Edge-F1
    with dilation is tolerant to glyph style while failing missing or
    hallucinated text. KNOWN BLIND SPOT: text displaced into a busy region
    can score ~0.4 from coincidental edge overlap — placement errors are
    caught upstream by the crop-refine pass and the coverage scan, not here.
    Calibration (framework.png, k=5, T=0.62): rescues 24/30 false demotions,
    keeps 53/55 true natives.
    """
    import cv2

    w, h = original.size
    x0, y0, x1, y1 = bbox
    px = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = int(max(0, x0 - px[0])); y0 = int(max(0, y0 - px[1]))
    x1 = int(min(w, x1 + px[0])); y1 = int(min(h, y1 + px[1]))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0

    kernel = np.ones((5, 5), np.uint8)
    ea = cv2.dilate(cv2.Canny(_gray(original.crop((x0, y0, x1, y1))).astype(np.uint8),
                              50, 150), kernel) > 0
    eb = cv2.dilate(cv2.Canny(_gray(rendered.crop((x0, y0, x1, y1))).astype(np.uint8),
                              50, 150), kernel) > 0
    if not ea.any() and not eb.any():
        return 0.0
    inter = float(np.logical_and(ea, eb).sum())
    f1 = 2 * inter / (float(ea.sum()) + float(eb.sum()))
    return float(round(1.0 - f1, 4))


def children_of(el: dict, elements: list, containment: float = 0.75) -> list:
    """Bboxes of other elements mostly inside `el` (for shell scoring/punch)."""
    if "bbox" not in el:
        return []
    x0, y0, x1, y1 = el["bbox"]
    out = []
    for o in elements:
        if o is el or "bbox" not in o:
            continue
        ox0, oy0, ox1, oy1 = o["bbox"]
        oarea = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
        if oarea == 0 or oarea >= (x1 - x0) * (y1 - y0):
            continue
        ix = max(0.0, min(x1, ox1) - max(x0, ox0))
        iy = max(0.0, min(y1, oy1) - max(y0, oy0))
        if ix * iy / oarea >= containment:
            out.append(o["bbox"])
    return out


def coverage(original: Image.Image, ir: dict,
             inflate: float = 0.03) -> dict:
    """Find original-image ink that no element bbox explains.

    Returns {"explained_frac": float, "missing": [{"bbox": [...], "area": int}]}
    with missing regions sorted largest-first (loop feeds them to identify).
    """
    import cv2

    w, h = original.size
    g = _gray(original)
    ink = (g < INK_THRESHOLD).astype(np.uint8)
    total_ink = int(ink.sum())
    if total_ink == 0:
        return {"explained_frac": 1.0, "missing": []}

    covered = np.zeros((h, w), dtype=bool)
    shapes = {e["id"]: e for e in ir["elements"] if "bbox" in e}
    for el in ir["elements"]:
        box = _cover_box(el, shapes)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        dx = max((x1 - x0) * inflate, 4)
        dy = max((y1 - y0) * inflate, 4)
        covered[int(max(0, y0 - dy)):int(min(h, y1 + dy)),
                int(max(0, x0 - dx)):int(min(w, x1 + dx))] = True

    orphan = ink.copy()
    orphan[covered] = 0
    explained = 1.0 - orphan.sum() / total_ink

    # cluster orphan ink; close small gaps so one figure = one region
    kernel = np.ones((9, 9), np.uint8)
    blob = cv2.morphologyEx(orphan, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)

    min_area = MIN_COMPONENT_FRAC * w * h
    missing = []
    for i in range(1, n):  # 0 = background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        missing.append({"bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                        "area": int(area)})
    missing.sort(key=lambda m: -m["area"])
    return {"explained_frac": float(round(explained, 4)), "missing": missing}


def connector_ink_fraction(original: Image.Image,
                           start: tuple, end: tuple,
                           samples: int = 40, radius: int = 4) -> float:
    """Fraction of points along the segment with ink nearby in the original.

    A connector the VLM hallucinated has no line pixels under it; a real one
    does (allowing small geometric error via the search radius).
    """
    g = _gray(original)
    h, w = g.shape
    hits = 0
    for i in range(samples):
        t = (i + 0.5) / samples
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        if x1 > x0 and y1 > y0 and g[y0:y1, x0:x1].min() < 200:
            hits += 1
    return hits / samples


def prune_connectors(original: Image.Image, ir: dict,
                     min_ink: float = 0.35, log=print) -> int:
    # 0.35 calibrated on framework.png: real-but-offset arrows score ~0.45,
    # hallucinated radiators ~0.10. A dropped real arrow is a 2-second manual
    # re-add; a kept fake arrow is a visual lie.
    """Drop arrow/line elements with no ink support in the original image."""
    from .render import _center, _edge_point

    shapes = {e["id"]: e for e in ir["elements"] if "bbox" in e}
    keep, dropped = [], 0
    for el in ir["elements"]:
        if el["type"] not in ("arrow", "line"):
            keep.append(el)
            continue
        src = shapes.get(el.get("from_id") or "")
        dst = shapes.get(el.get("to_id") or "")
        if src and dst:
            start = _edge_point(src, _center(dst))
            end = _edge_point(dst, _center(src))
        elif el.get("points"):
            p = el["points"]
            start, end = (p[0], p[1]), (p[2], p[3])
        else:
            dropped += 1   # dangling: nothing to attach, nothing to draw
            continue
        frac = connector_ink_fraction(original, start, end)
        if frac >= min_ink:
            keep.append(el)
        else:
            dropped += 1
            log(f"  [prune] connector {el.get('id','?')} ink={frac:.2f}")
    ir["elements"] = keep
    return dropped


def _cover_box(el: dict, shapes: dict) -> list | None:
    """Rectangle of pixels this element accounts for, or None.

    Connectors have no bbox; cover the rectangle spanned by their endpoints
    (over-covers diagonals — acceptable: coverage is a recall heuristic, the
    per-element residual is the precision signal).
    """
    if "bbox" in el:
        return el["bbox"]
    if el.get("type") not in ("arrow", "line"):
        return None
    src = shapes.get(el.get("from_id") or "")
    dst = shapes.get(el.get("to_id") or "")
    if src and dst:
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = src["bbox"], dst["bbox"]
        acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
        return [min(acx, bcx), min(acy, bcy), max(acx, bcx), max(acy, bcy)]
    if el.get("points"):
        x0, y0, x1, y1 = el["points"]
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    return None
