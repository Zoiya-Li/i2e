"""Faithful-AND-editable painterly surfaces via SVG gradient bands.

The all-native PPTX redraw of the manifold looked crude (flat fills, PPT has
no gradient mesh); the faithful image layer looked right but wasn't editable.
SVG breaks that dichotomy: a smooth surface is reconstructed as many thin
bezier bands, each with its OWN vertical gradient sampled from the original,
softened by a blur — continuous shading that nonetheless is fully editable
(every band path, every gradient stop, every scatter dot is a vector object
you can drag in Figma / Illustrator).

manifold_svg(crop) returns an SVG <g> group (defs + bands + dots + flow
lines) positioned at the crop's origin, to drop into a whole-figure SVG.
"""
from __future__ import annotations

import numpy as np


def _smooth(vals, win):
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    if ok.sum() < 4:
        return None
    v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
    k = np.ones(win) / win
    return np.convolve(np.pad(v, win // 2, mode="edge"), k, mode="valid")[:len(v)]


def extract_gradient_bands(crop, n_bands: int = 18, exclude=None):
    """Slice the surface into n_bands smooth bands, each with top+bottom
    colors for a vertical gradient. Returns (bands, W, H) where each band is
    {top: [[x,y]...], bot: [[x,y]...], c_top: hex, c_bot: hex}."""
    import cv2

    arr = np.asarray(crop.convert("RGB"))
    H, W = arr.shape[:2]
    g = cv2.GaussianBlur(np.asarray(crop.convert("L"), np.float32), (31, 31), 0)
    bg = float(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
    ink = (g < bg - 8).astype(np.uint8)
    if exclude:
        for x0, y0, x1, y1 in exclude:
            ink[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 0
    dens = cv2.GaussianBlur(ink.astype(np.float32) * 255, (61, 61), 0)
    body = (dens > 70).astype(np.uint8)

    top = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    for x in range(W):
        ys = np.flatnonzero(body[:, x])
        if len(ys) > 4:
            top[x], bot[x] = ys[0], ys[-1]
    top, bot = _smooth(top, 81), _smooth(bot, 81)
    if top is None or bot is None:
        return None, W, H

    # band boundaries = ISO-LUMINANCE curves, which undulate with the ridges
    # (even horizontal slicing flattened the waves). For each level, the y in
    # each column where the shading first crosses it → a wavy curve.
    inside_lum = []
    for x in range(0, W, 2):
        a, b = int(top[x]), int(bot[x])
        if b > a:
            inside_lum.append(g[a:b, x])
    if not inside_lum:
        return None, W, H
    inside = np.concatenate(inside_lum)
    levels = np.percentile(inside, np.linspace(8, 96, n_bands - 1))

    curves = [top]
    for lv in levels:
        c = np.full(W, np.nan)
        for x in range(W):
            a, b = int(top[x]), int(bot[x])
            if b - a < 4:
                continue
            hits = np.flatnonzero(g[a:b, x] <= lv)
            if len(hits):
                c[x] = a + hits[0]
        # require enough columns to actually cross this level, else the
        # sparse hits interpolate into a dark wedge artifact
        if np.isfinite(c).sum() < 0.5 * W:
            continue
        cs = _smooth(c, 71)
        if cs is not None:
            curves.append(np.clip(cs, top, bot))
    # order boundaries by mean depth (keeps undulation, avoids crossings that
    # would invert a band) instead of forcing pointwise monotonicity, which
    # flattened the waves into terraces
    mids = sorted(curves[1:], key=lambda c: float(np.mean(c)))
    curves = [top] + mids + [bot]

    xs = np.arange(W)
    bands = []
    for i in range(len(curves) - 1):
        ya, yb = curves[i], curves[i + 1]
        c_top = _band_color(arr, xs, ya, ya + (yb - ya) * 0.5)
        c_bot = _band_color(arr, xs, yb - (yb - ya) * 0.5, yb)
        step = max(1, W // 90)
        sx = list(range(0, W, step)) + [W - 1]
        bands.append({
            "top": [[int(x), float(ya[x])] for x in sx],
            "bot": [[int(x), float(yb[x])] for x in sx],
            "c_top": c_top, "c_bot": c_bot,
        })
    return bands, W, H


def _band_color(arr, xs, y_lo, y_hi):
    H, W = arr.shape[:2]
    cols = []
    for x in range(0, W, 3):
        a, b = int(max(0, y_lo[x])), int(min(H, y_hi[x]))
        if b > a:
            cols.append(arr[a:b, x].reshape(-1, 3))
    if not cols:
        return "#dfe7ef"
    px = np.concatenate(cols).astype(int)
    lum = px.sum(axis=1)
    # the surface is pale blue shading over white, with dark ink (dots,
    # arrows, glyphs) scattered on it. Sample the MIDDLE luminance band so we
    # get the shading, excluding both white background AND dark ink (the dark
    # ink is what produced the black wedge).
    lo, hi = np.percentile(lum, 25), np.percentile(lum, 80)
    mid = px[(lum >= lo) & (lum <= hi) & (lum > 360)]
    use = mid if len(mid) > 15 else px[lum > 360]
    if len(use) < 5:
        return "#e3e9f1"
    c = np.median(use, axis=0)
    shadow = np.array([182, 197, 217])
    c = (c * 0.78 + shadow * 0.22)
    c = np.clip(c, [150, 165, 188], 255).astype(int)   # brightness floor
    return "#%02x%02x%02x" % tuple(c)


def _path(top, bot):
    """Closed smooth path: top curve left→right, bottom curve right→left."""
    def seg(pts):
        d = f"{pts[0][0]:.1f},{pts[0][1]:.1f}"
        # quadratic-smoothed polyline (midpoint control) reads as a curve
        for j in range(1, len(pts)):
            mx = (pts[j - 1][0] + pts[j][0]) / 2
            my = (pts[j - 1][1] + pts[j][1]) / 2
            d += f" Q{pts[j-1][0]:.1f},{pts[j-1][1]:.1f} {mx:.1f},{my:.1f}"
        d += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
        return d
    return f"M{seg(top)} L{seg(list(reversed(bot)))[1:]} Z"


def manifold_svg(crop, dots=None, flow=None, n_bands: int = 18,
                 exclude=None, idp: str = "mf") -> str | None:
    """SVG <g> for one painterly surface: gradient bands + dots + flow lines.
    Everything inside is an editable vector object."""
    bands, W, H = extract_gradient_bands(crop, n_bands, exclude=exclude)
    if not bands:
        return None

    defs, paths = [], []
    for i, bd in enumerate(bands):
        gid = f"{idp}g{i}"
        defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{bd["c_top"]}"/>'
            f'<stop offset="1" stop-color="{bd["c_bot"]}"/></linearGradient>')
        paths.append(f'<path d="{_path(bd["top"], bd["bot"])}" '
                     f'fill="url(#{gid})"/>')
    defs.append(f'<filter id="{idp}soft" x="-8%" y="-8%" width="116%" '
                f'height="116%"><feGaussianBlur stdDeviation="3.6"/></filter>')

    out = [f'<defs>{"".join(defs)}</defs>']
    out.append(f'<g filter="url(#{idp}soft)">{"".join(paths)}</g>')

    # ridge contour lines = the band boundaries, the curves that give the
    # surface its wave definition (editable strokes)
    ridges = []
    for i, bd in enumerate(bands):
        if i == 0:
            continue
        pts = bd["top"]
        d = "M" + " ".join(
            (f"{p[0]:.1f},{p[1]:.1f}" if j == 0 else
             f"Q{pts[j-1][0]:.1f},{pts[j-1][1]:.1f} "
             f"{(pts[j-1][0]+p[0])/2:.1f},{(pts[j-1][1]+p[1])/2:.1f}")
            for j, p in enumerate(pts))
        ridges.append(f'<path d="{d}" fill="none" stroke="#8497b0" '
                      f'stroke-width="0.7" opacity="0.6"/>')
    out.append(f'<g>{"".join(ridges)}</g>')

    if flow:
        seg = []
        for line in flow:
            if len(line) < 3:
                continue
            d = "M" + " L".join(f"{p[0]:.1f},{p[1]:.1f}" for p in line)
            seg.append(f'<path d="{d}" fill="none" stroke="#aebdd0" '
                       f'stroke-width="0.6" opacity="0.5"/>')
        out.append(f'<g>{"".join(seg)}</g>')

    if dots:
        cs = "".join(
            f'<circle cx="{d["cx"]:.1f}" cy="{d["cy"]:.1f}" '
            f'r="{max(1.4, min(d.get("r", 2), 2.6)):.1f}" '
            f'fill="{d["color"]}"/>' for d in dots)
        out.append(f'<g>{cs}</g>')

    return "".join(out)


def wrap_svg(width: int, height: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">{body}</svg>')
