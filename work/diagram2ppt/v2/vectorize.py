"""Ban-screenshots mode: every remaining crop becomes drawn objects.

Policy (user decision 2026-06-12): the deck ships ZERO pictures. Editability
beats pixel fidelity — a stylized but draggable/recolorable approximation is
worth more in a PPT than a perfect screenshot nobody can touch. (Note this
INVERTS the poster-era faithful-pixels doctrine; different product contract.)

Strategies, tried in order per crop:
  icon       small pictograms → VLM classifies kind → MSO autoshape or a
             unicode glyph textbox
  dotcloud   scatter thumbnails / point clouds → cv2 blob extraction
             (deterministic) → one tiny oval per dot
  silhouette big organic art (3D surfaces) → largest-contour freeform with
             the region's dominant color + a dotcloud overlay
Forced shapes: demoted ovals/rects/texts are restored native with their last
refined fields, marked ext.forced (the visual error is accepted, reviewable).
"""
from __future__ import annotations

import json
import re

import numpy as np

ICON_PROMPT = (
    "This small image is an icon/pictogram from a technical diagram. "
    "Look at the SHAPE only. Identify its concrete visual TYPE from the list below (be specific):\n"
    "- database: stacked cylinders / disk-drive symbol\n"
    "- gear: cogwheel with teeth\n"
    "- scatter: cloud of dots, often with a trend/regression line\n"
    "- line: smooth curve / line-chart / bell curve / normal distribution\n"
    "- warning: triangle with exclamation mark\n"
    "- hourglass: sand timer\n"
    "- shield: shield shape, optionally with a checkmark inside\n"
    "- document: sheet of paper with lines or a small chart\n"
    "- check: tick / checkmark\n"
    "- cross: X / cancel mark\n"
    "- arrow: directional arrow head or block arrow\n"
    "- other: anything that does not match the above\n\n"
    "Examples:\n"
    '{"kind":"gear","color":"#6b7a8d","glyph":"⚙"}\n'
    '{"kind":"scatter","color":"#4472c4","glyph":"📊"}\n'
    '{"kind":"line","color":"#6b7a8d","glyph":"📈"}\n'
    '{"kind":"database","color":"#6b7a8d","glyph":"🗄"}\n\n'
    'Output STRICT JSON: {"kind": "...", "color": "#hex of the dominant color", '
    '"glyph": "single unicode character that best represents it"}. '
    'Use "scatter" for point-cloud icons, "line" for curve icons, "gear" for '
    'cog wheels. Output ONLY the JSON.'
)

ICON_SECOND_PROMPT = (
    "This small image is an icon. Look carefully at its SHAPE. Choose the "
    "closest match from: database, gear, scatter, line, warning, hourglass, "
    "shield, document, check, cross, arrow. If it is genuinely none of these, "
    "say other. Output ONLY JSON: {\"kind\":\"...\",\"color\":\"#hex\"}"
)

ICON_AREA_MAX = 10_000   # px²: above this it's not an icon
SILHOUETTE_AREA_MIN = 20_000  # px²: below this, dots alone read better
DOT_AREA_RANGE = (3, 400)  # px² per blob to count as a "dot"
MAX_DOTS = 150


# -- crop analysis (deterministic, cv2) -------------------------------------

def _ink_mask(arr: np.ndarray, threshold: int = 60) -> np.ndarray:
    """Pixels that differ from the local background (border median)."""
    import cv2
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
    mask = (dist > threshold).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _apply_excludes(mask: np.ndarray, exclude: list | None) -> np.ndarray:
    """Zero out regions of OTHER native elements (crop-local boxes) — their
    arrows/labels/formulas pollute the surface mask (seen in sil_debug)."""
    for x0, y0, x1, y1 in exclude or []:
        mask[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 0
    return mask


def extract_dots(crop, max_dots: int = MAX_DOTS,
                 exclude: list | None = None,
                 round_only: bool = False,
                 ink_threshold: int = 60) -> list[dict]:
    """Blob-detect scatter points: [{cx, cy, r, color}] in crop coords.

    round_only filters to circle-like blobs (aspect ~1, decent fill ratio) —
    without it, fragments of lines/glyph edges register as 'dots' and litter
    the canvas (the scattered-circles complaint)."""
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    mask = _apply_excludes(_ink_mask(arr, threshold=ink_threshold), exclude)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (DOT_AREA_RANGE[0] <= area <= DOT_AREA_RANGE[1]):
            continue
        if round_only:
            bw_ = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_ = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = bw_ / max(1, bh_)
            fill = area / max(1, bw_ * bh_)
            if not (0.45 <= aspect <= 2.2 and fill >= 0.45 and area <= 150):
                continue
        ys, xs = np.where(labels == i)
        px = arr[ys, xs].astype(int)
        lum = px.sum(axis=1)
        core = px[lum <= np.percentile(lum, 30)]  # antialiased rims wash color
        color = np.median(core if len(core) >= 3 else px, axis=0).astype(int)
        dots.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                     "r": max(1.5, float(np.sqrt(area / np.pi))),
                     "color": "#%02x%02x%02x" % tuple(color),
                     "area": area})
    dots.sort(key=lambda d: -d["area"])
    return dots[:max_dots]


def extract_silhouette(crop, max_points: int = 60,
                       exclude: list | None = None) -> dict | None:
    """Largest ink contour as a SMOOTH polygon + its dominant color.

    A pale gradient surface fragments the ink mask; raw contours come out
    spiky (seen in the v3.1 PowerPoint render). Blur + a generous close heal
    the fragments, and uniform resampling (not coarser approxPolyDP) keeps
    the outline smooth at the point budget.
    """
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    mask = _apply_excludes(_ink_mask(arr, threshold=30), exclude)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    blur = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    mask = (blur > 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.05 * mask.size:
        return None
    eps = 0.004 * cv2.arcLength(big, True)
    poly = cv2.approxPolyDP(big, eps, True).reshape(-1, 2)
    if len(poly) > max_points:   # uniform subsample preserves smoothness
        idx = np.linspace(0, len(poly) - 1, max_points).astype(int)
        poly = poly[idx]
    inside = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(inside, [big], -1, 1, -1)
    ys, xs = np.where(inside > 0)
    # overall median: pale surfaces stay pale (the outline supplies contrast;
    # strong-pixel sampling turned them into dark slabs — seen in v3.2)
    color = np.median(arr[ys, xs], axis=0).astype(int)
    return {"points": [[float(x), float(y)] for x, y in poly],
            "fill": "#%02x%02x%02x" % tuple(color)}


# -- the pass ----------------------------------------------------------------

SURFACE_AREA_MIN = 60_000   # px²: above this an organic crop is a 3D surface


def vectorize_pass(ir: dict, original, vlm, log=print) -> dict:
    """Convert EVERY remaining raster_crop to drawn objects. Returns stats."""
    from .loop import _padded_crop

    stats = {"forced_shapes": 0, "icons": 0, "dotclouds": 0,
             "silhouettes": 0, "surfaces": 0}
    for el in list(ir["elements"]):
        if el["type"] != "raster_crop" or "bbox" not in el:
            continue
        ot = el.get("ext", {}).get("original_type") or ""
        x0, y0, x1, y1 = el["bbox"]
        area = (x1 - x0) * (y1 - y0)

        # 1. shapes/text that failed refinement: force-draw the last fields
        if ot in ("rect", "rounded_rect", "oval", "diamond", "hexagon",
                  "parallelogram", "text"):
            el["type"] = ot
            el["status"] = "native"
            el["ext"]["forced"] = True
            stats["forced_shapes"] += 1
            continue

        crop, _ = _padded_crop(original, el["bbox"], pad=0.0)
        excl = _local_children(el, ir["elements"])

        # 2. LARGE organic surface → flowing wave bands (the manifold).
        # Detected by size, modelled as horizontal bands so it can never blob.
        if area >= SURFACE_AREA_MIN and _vectorize_surface(el, crop, ir, log=log):
            stats["surfaces"] += 1
            continue

        dots = extract_dots(crop, exclude=excl)

        # 3. scatter thumbnail (mini plot) — dots win over "icon": a 117x65
        # thumbnail is icon-SIZED but is a point cloud, and a glyph there
        # renders as a giant β (seen in the v3 snapshot)
        if len(dots) >= 6:
            el["type"] = "dotcloud"
            el["dots"] = [dict(d, r=min(max(d["r"], 1.8), 2.6)) for d in dots[:24]]
            stats["dotclouds"] += 1
            el["status"] = "native"
            el["ext"]["expert"] = "vectorize"
            continue

        # 4. icons: VLM classification → autoshape/glyph
        if area <= ICON_AREA_MAX:
            kind, color, glyph = _classify_icon(crop, vlm, log)
            el["type"] = "icon"
            el["icon"] = {"kind": kind, "color": color, "glyph": glyph}
            el["status"] = "native"
            el["ext"]["expert"] = "icon"
            stats["icons"] += 1
            continue

        # 5. medium organic art with few dots: silhouette
        sil = extract_silhouette(crop, exclude=excl)
        el["type"] = "dotcloud"
        el["dots"] = dots
        if sil:
            el["silhouette"] = sil
            el["style"] = surface_style(crop, sil)
            stats["silhouettes"] += 1
        else:
            stats["dotclouds"] += 1
        el["status"] = "native"
        el["ext"]["expert"] = "vectorize"

    log(f"[vectorize] {json.dumps(stats)}")
    return stats


def _vectorize_surface(el: dict, crop, ir: dict, log=print) -> bool:
    """Model a large organic crop as flowing wave bands + crest-bound dots.

    All parameters derive from the crop; no hardcoded coordinates. Returns
    False (caller falls through to silhouette) if the wave model fails.
    """
    nx0, ny0, nx1, ny1 = el["bbox"]
    ch = ny1 - ny0
    # containers/annotations within the crop pollute envelope + dot detection
    env_excl, dot_excl = [], []
    for o in ir["elements"]:
        if o is el or "bbox" not in o or o.get("status") != "native":
            continue
        is_ann = o["type"] in ("text", "formula", "rect", "rounded_rect", "chart") \
            or (o["type"] == "oval" and o.get("fill"))
        ex0, ey0, ex1, ey1 = o["bbox"]
        lb = [max(ex0 - nx0, 0), max(ey0 - ny0, 0),
              min(ex1 - nx0, nx1 - nx0), min(ey1 - ny0, ny1 - ny0)]
        if lb[2] <= lb[0] or lb[3] <= lb[1]:
            continue
        if is_ann:
            dot_excl.append([lb[0] - 3, lb[1] - 3, lb[2] + 3, lb[3] + 3])
        # the envelope must ignore UI in the top band + any big container
        if is_ann and ((ey0 + ey1) / 2 <= ny0 + 0.35 * ch
                       or (ex1 - ex0) * (ey1 - ey0) > 3000):
            env_excl.append([lb[0] - 4, lb[1] - 4, lb[2] + 4, lb[3] + 4])

    wb = extract_wave_bands(crop, exclude=env_excl, smooth_win=81)
    if not wb or len(wb.get("curves", [])) < 2:
        return False

    _tint_bands(wb)
    el["type"] = "dotcloud"
    el["wave_bands"] = wb
    el["streamlines"] = _synth_flow_lines(wb)
    el["style"] = {"light": wb["fills"][0], "dark": "#b6c2d4"}

    # crest dots: deep-ink (separates from pale wash), round, in top bands
    import numpy as np
    dots = extract_dots(crop, exclude=dot_excl, round_only=True,
                        max_dots=400, ink_threshold=130)
    xs0 = np.array([p[0] for p in wb["curves"][0]])
    top = np.asarray(wb["curves"][0], float)
    band2 = np.asarray(wb["curves"][min(2, len(wb["curves"]) - 1)], float)

    def hex2rgb(hx):
        return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))

    kept = []
    for d in dots:
        r, g, b = hex2rgb(d["color"])
        sat = max(r, g, b) - min(r, g, b)
        ytop = float(np.interp(d["cx"], xs0, top[:, 1]))
        yb2 = float(np.interp(d["cx"], xs0, band2[:, 1]))
        if sat > 50 or (ytop - 6 <= d["cy"] <= yb2):   # crest band, or red cluster
            if sat < 35:
                t = 0.45
                d["color"] = "#%02x%02x%02x" % (
                    int(r * (1 - t) + 90 * t), int(g * (1 - t) + 108 * t),
                    int(b * (1 - t) + 140 * t))
            d["r"] = min(max(d["r"], 1.6), 2.6)
            kept.append(d)
    el["dots"] = kept
    el["status"] = "native"
    el["ext"]["expert"] = "surface"
    log(f"  [surface] {el['id']}: {len(wb['curves'])} bands, {len(kept)} crest dots")
    return True


def _tint_bands(wb: dict) -> None:
    """Alternate light/shade blue-gray so pale-median fills don't go invisible."""
    def hex2rgb(hx):
        return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
    for i in range(len(wb["fills"])):
        ref = (214, 224, 238) if i % 2 == 0 else (188, 202, 222)
        r, g, b = hex2rgb(wb["fills"][i])
        t = 0.75
        wb["fills"][i] = "#%02x%02x%02x" % (
            int(r * (1 - t) + ref[0] * t), int(g * (1 - t) + ref[1] * t),
            int(b * (1 - t) + ref[2] * t))


def _synth_flow_lines(wb: dict) -> list:
    """Flow lines = curves interpolated between adjacent band boundaries —
    guaranteed to follow the wave (extracted iso-contours die in the veil)."""
    import numpy as np
    curves = [np.asarray(c, float) for c in wb["curves"]]
    out = []
    for a, b in zip(curves[:-1], curves[1:]):
        for t in (0.33, 0.66):
            m = a.copy()
            m[:, 1] = a[:, 1] * (1 - t) + b[:, 1] * t
            out.append([[float(x), float(y)] for x, y in m])
    return out


def smooth_closed(points: list, samples: int = 140) -> list:
    """Dense Catmull-Rom resampling of a closed polygon — rendered as line
    segments it is indistinguishable from a true curve, so the freeform
    builder needs no bezier XML surgery."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 4:
        return [list(map(float, p)) for p in P]
    per = max(2, samples // n)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for t in np.linspace(0.0, 1.0, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append([float(q[0]), float(q[1])])
    return out


def surface_style(crop, sil: dict) -> dict:
    """Light/dark gradient stops sampled from the actual surface shading."""
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    mask = np.zeros(arr.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(sil["points"], np.int32)], 1)
    px = arr[mask > 0]
    if len(px) < 50:
        return {"light": sil.get("fill", "#e4e9ef"), "dark": "#b9c5d6"}
    lum = px.astype(int).sum(axis=1)
    light = np.median(px[lum >= np.percentile(lum, 70)], axis=0).astype(int)
    dark = np.median(px[lum <= np.percentile(lum, 25)], axis=0).astype(int)
    return {"light": "#%02x%02x%02x" % tuple(light),
            "dark": "#%02x%02x%02x" % tuple(dark)}


def extract_surface_layers(crop, sil: dict, exclude: list | None = None,
                           n_levels: int = 4, max_points: int = 70) -> list:
    """Posterize the surface's luminance field into stacked vector relief
    layers (terraced-topography style). Brightness IS the height map of a
    shaded 3D surface: drawing iso-luminance bands back-to-front with
    progressively lighter fills recreates the ridge/valley depth that a
    single flat gradient cannot ('多维空间的立体感').

    Returns [{points, fill}] ordered base→highlight (draw in order).
    """
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    g = cv2.GaussianBlur(np.asarray(crop.convert("L"), np.float32), (21, 21), 0)
    base = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(base, [np.asarray(sil["points"], np.int32)], 1)
    for ex in exclude or []:
        pass  # silhouette already excluded annotations
    inside = g[base > 0]
    if inside.size < 200:
        return []
    layers = []
    levels = np.percentile(inside, np.linspace(18, 92, n_levels))
    prev_level = None
    for lv in levels:
        m = ((g > lv) & (base > 0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        soft = cv2.GaussianBlur(m.astype(np.float32), (15, 15), 0)
        m = (soft > 0.45).astype(np.uint8)        # dense bands ≈ continuous shading
        band = ((g > (prev_level if prev_level is not None else -1))
                & (g <= lv) & (base > 0))
        prev_level = lv
        px = arr[band] if band.sum() > 50 else arr[base > 0]
        fill = "#%02x%02x%02x" % tuple(np.median(px, axis=0).astype(int))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            if cv2.contourArea(c) < 0.008 * g.size:
                continue
            eps = 0.006 * cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(poly) > max_points:
                idx = np.linspace(0, len(poly) - 1, max_points).astype(int)
                poly = poly[idx]
            if len(poly) >= 4:
                polys.append([[float(x), float(y)] for x, y in poly])
        if polys:
            layers.append({"polys": polys, "fill": fill})
    return layers


def extract_wave_bands(crop, exclude: list | None = None,
                       n_curves: int = 3, smooth_win: int = 41) -> dict | None:
    """Model a flowing-wave surface as what it IS: horizontal bands between
    smooth single-valued curves y=f(x).

    Iso-luminance CONTOURS produced blobs (the angry-user render): pixel
    statistics can't give elegance. Column-scanning the envelope and the
    shading transitions, then smoothing each into a function of x, yields
    clean flowing curves BY CONSTRUCTION — every band spans the full width
    and can never blob.

    Returns {"curves": [top, c1..ck, bottom] each [[x,y]...], "fills": [...]}
    """
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    g = cv2.GaussianBlur(np.asarray(crop.convert("L"), np.float32), (31, 31), 0)
    raw_ink = _apply_excludes(_ink_mask(arr, threshold=22), exclude)
    # envelope from ink DENSITY, not ink: discrete dots floating above the
    # crest dragged the top envelope into a dozen random bumps (the native-res
    # '一坨'). A wide blur erases sparse dots; only the continuous surface
    # wash survives the threshold.
    density = cv2.GaussianBlur(raw_ink.astype(np.float32) * 255, (61, 61), 0)
    ink = (density > 80).astype(np.uint8)
    H, W = ink.shape

    def smooth(vals):
        v = np.asarray(vals, float)
        ok = ~np.isnan(v)
        if ok.sum() < 10:
            return None
        v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
        k = np.ones(smooth_win) / smooth_win
        pad = np.pad(v, smooth_win // 2, mode="edge")
        return np.convolve(pad, k, mode="valid")[:len(v)]

    top = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    for x in range(W):
        ys = np.flatnonzero(ink[:, x])
        if len(ys) > 4:
            top[x], bot[x] = ys[0], ys[-1]
    top, bot = smooth(top), smooth(bot)
    if top is None or bot is None:
        return None

    inside_vals = g[ink > 0]
    levels = np.percentile(inside_vals, np.linspace(30, 70, n_curves))
    curves = [top]
    for lv in levels:
        c = np.full(W, np.nan)
        for x in range(W):
            y0, y1 = int(top[x]), int(bot[x])
            if y1 - y0 < 6:
                continue
            col = g[y0:y1, x]
            hits = np.flatnonzero(col < lv)
            if len(hits):
                c[x] = y0 + hits[0]
        c = smooth(c)
        if c is not None:
            curves.append(np.clip(c, top, bot))
    curves.append(bot)

    fills = []
    for a, b in zip(curves[:-1], curves[1:]):
        m = np.zeros((H, W), bool)
        for x in range(0, W, 2):
            m[int(min(a[x], b[x])):int(max(a[x], b[x])), x] = True
        px = arr[m]
        fills.append("#%02x%02x%02x" % tuple(
            np.median(px, axis=0).astype(int)) if len(px) > 30 else "#dde5ee")

    step = max(1, W // 90)
    xs = list(range(0, W, step))
    return {"curves": [[[float(x), float(c[x])] for x in xs] for c in curves],
            "fills": fills}


def extract_streamlines(crop, sil: dict, max_lines: int = 10) -> list:
    """Interior shading iso-contours = the 'flow lines' that make a manifold
    read as a surface instead of a blob. Open smoothed polylines (crop px)."""
    import cv2
    from skimage import measure
    g = cv2.GaussianBlur(np.asarray(crop.convert("L"), np.float32), (15, 15), 0)
    mask = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(sil["points"], np.int32)], 1)
    mask = cv2.erode(mask, np.ones((13, 13), np.uint8))  # interior weave only
    inside = g[mask > 0]
    if inside.size < 100:
        return []
    out = []
    for level in np.percentile(inside, np.linspace(20, 90, 9)):
        for c in measure.find_contours(g, level):
            pts = c[:, ::-1]                       # (row,col) -> (x,y)
            keep = [p for p in pts
                    if mask[min(mask.shape[0] - 1, int(p[1])),
                            min(mask.shape[1] - 1, int(p[0]))] > 0]
            if len(keep) < 30:
                continue
            sub = np.asarray(keep)[::max(1, len(keep) // 50)]
            k = np.ones(5) / 5                     # light smoothing
            xs = np.convolve(sub[:, 0], k, mode="valid")
            ys = np.convolve(sub[:, 1], k, mode="valid")
            out.append([[float(x), float(y)] for x, y in zip(xs, ys)])
    out.sort(key=len, reverse=True)
    return out[:max_lines]


def measure_thickness(original, points: list) -> float:
    """Median ink run-length perpendicular to the segment (px). Lets a fat
    block arrow render as a block arrow instead of a 1.5pt connector."""
    arr = np.asarray(original.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    x0, y0, x1, y1 = points
    dx, dy = x1 - x0, y1 - y0
    L = max(1.0, np.hypot(dx, dy))
    nx, ny = -dy / L, dx / L          # unit normal
    runs = []
    for t in np.linspace(0.25, 0.75, 9):   # middle of the shaft, not the head
        px, py = x0 + dx * t, y0 + dy * t
        hits = [s for s in range(-30, 31)
                if 0 <= int(px + nx * s) < w and 0 <= int(py + ny * s) < h
                and arr[int(py + ny * s), int(px + nx * s)] < 210]
        # span, not count: gradient/hollow strokes have sparse dark pixels
        runs.append(hits[-1] - hits[0] + 1 if hits else 0)
    return float(np.median(runs))


def detect_fat_arrows(original, bbox: list, exclude: list | None = None,
                      max_n: int = 2) -> list[dict]:
    """Find the dominant saturated straight strokes in a region (the β/γ
    arrows the VLM treats as 'plot content'). Deterministic: HSV saturation
    mask + Hough. Returns [{points(global), color, thickness}]."""
    import cv2
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = np.asarray(original.convert("RGB"))[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 90) & (hsv[..., 2] > 60)).astype(np.uint8) * 255
    mask = _apply_excludes(mask, exclude)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    min_len = int(0.18 * max(crop.shape[:2]))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=60,
                            minLineLength=min_len, maxLineGap=15)
    if lines is None:
        return []
    segs = sorted((l[0] for l in lines),
                  key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    out: list[dict] = []
    for sx0, sy0, sx1, sy1 in segs:
        ang = np.degrees(np.arctan2(sy1 - sy0, sx1 - sx0)) % 180
        mid = ((sx0 + sx1) / 2, (sy0 + sy1) / 2)
        if any(abs(ang - a["_ang"]) % 180 < 12 and
               np.hypot(mid[0] - a["_mid"][0], mid[1] - a["_mid"][1]) < 60
               for a in out):
            continue   # same stroke, shorter segment
        n = 12
        xs = np.linspace(sx0, sx1, n).astype(int).clip(0, crop.shape[1] - 1)
        ys = np.linspace(sy0, sy1, n).astype(int).clip(0, crop.shape[0] - 1)
        color = np.median(crop[ys, xs], axis=0).astype(int)
        pts = [x0 + sx0, y0 + sy0, x0 + sx1, y0 + sy1]
        out.append({"points": [float(v) for v in pts],
                    "color": "#%02x%02x%02x" % tuple(color),
                    "thickness": measure_thickness(original, pts),
                    "_ang": ang, "_mid": mid})
        if len(out) >= max_n:
            break
    for a in out:
        a.pop("_ang"); a.pop("_mid")
    return out


def _local_children(el: dict, elements: list) -> list:
    """Native elements inside el's bbox, as crop-local boxes (inflated 2px)."""
    from .diff import children_of
    natives = [o for o in elements
               if o is not el and o.get("status") == "native" and "bbox" in o]
    x0, y0 = el["bbox"][0], el["bbox"][1]
    return [[bx0 - x0 - 2, by0 - y0 - 2, bx1 - x0 + 2, by1 - y0 + 2]
            for bx0, by0, bx1, by1 in children_of(el, natives, containment=0.6)]


def _classify_icon(crop, vlm, log) -> tuple[str, str, str]:
    try:
        raw = vlm.chat(ICON_PROMPT, crop, max_edge=512)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        d = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        kind = str(d.get("kind", "other"))
        # Second opinion for vague classifications.
        if kind == "other":
            raw2 = vlm.chat(ICON_SECOND_PROMPT, crop, max_edge=512)
            raw2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw2.strip())
            d2 = json.loads(raw2[raw2.find("{"):raw2.rfind("}") + 1])
            kind = str(d2.get("kind", "other"))
        return (kind, str(d.get("color", "#555555")),
                str(d.get("glyph", "◆"))[:2])
    except Exception as e:
        log(f"  [icon] classify failed ({e}) — generic glyph")
        return ("other", "#555555", "◆")
