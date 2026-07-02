"""Whole-figure SVG export — the format where 'faithful AND editable' both
hold for painterly content (PPTX can't do gradient meshes).

Every element becomes an editable SVG object: text→<text>, shapes→<rect>/
<ellipse>, arrows→<path>+marker, formulas→<text> (LaTeX source, editable),
charts→native <rect> bars, painterly surfaces→gradient bands (svg_surface).
Open the result in Figma / Illustrator and you can grab any band, dot, label,
or box. No element is a flat screenshot.
"""
from __future__ import annotations

import html

import numpy as np

from .build_pptx import _hex_to_rgb  # noqa: F401 (kept for parity)
from .svg_surface import manifold_svg, wrap_svg


def _esc(s):
    return html.escape(str(s or ""))


def _color(hex_str, default="none"):
    if not hex_str or str(hex_str).lower() in ("none", "", "transparent"):
        return default
    s = str(hex_str)
    return s if s.startswith("#") else default


def _hex_to_rgb(hx):
    hx = hx.lstrip("#")
    if len(hx) == 3:
        hx = "".join(c * 2 for c in hx)
    return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(max(0, min(255, v)))) for v in rgb)


def _ensure_contrast(hex_color, bg_hex="#ffffff", min_delta=45):
    """Darken/lighten a color so it differs from the background by at least
    min_delta in RGB Manhattan distance."""
    try:
        r, g, b = _hex_to_rgb(hex_color)
    except Exception:
        return hex_color
    br, bg, bb = _hex_to_rgb(bg_hex)
    delta = abs(r - br) + abs(g - bg) + abs(b - bb)
    if delta >= min_delta:
        return hex_color
    # compute luminance; darken light colors, lighten dark ones
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > 128:
        # darken
        scale = max(0.35, 1.0 - (min_delta - delta) / 255.0)
        r, g, b = r * scale, g * scale, b * scale
    else:
        # lighten toward a mid gray if very dark
        target = 128
        t = min(1.0, (min_delta - delta) / 255.0)
        r = r * (1 - t) + target * t
        g = g * (1 - t) + target * t
        b = b * (1 - t) + target * t
    return _rgb_to_hex((r, g, b))


def export_svg(ir: dict, original, out_path: str, log=print) -> dict:
    from .postprocess import dedup_overlapping
    dedup_overlapping(ir, log=lambda *a: None)   # kill duplicate labels first
    w, h = ir["image"]["width"], ir["image"]["height"]
    body, stats = [], {"shapes": 0, "texts": 0, "surfaces": 0,
                       "arrows": 0, "charts": 0}
    shapes = {e["id"]: e for e in ir["elements"] if "bbox" in e}
    ordered = sorted(ir["elements"], key=lambda e: e.get("z", 0))

    for el in ordered:
        t = el.get("type")
        if t in ("arrow", "line"):
            continue
        # NO embedded original pixels anywhere (user rule: 禁止拿原图截图).
        # dotcloud / surface / leftover raster_crop ALL get vectorized.
        if t in ("dotcloud", "surface", "raster_crop"):
            svg = _surface(el, original)
            if svg:
                body.append(svg)
                stats["surfaces"] += 1
            continue
        if t in ("rect", "rounded_rect", "oval", "diamond", "hexagon",
                 "parallelogram"):
            body.append(_shape(el))
            stats["shapes"] += 1
            if el.get("text"):
                body.append(_text(el))
        elif t in ("text", "formula"):
            body.append(_text(el))
            stats["texts"] += 1
        elif t == "chart":
            body.append(_chart(el))
            stats["charts"] += 1
        elif t == "icon":
            body.append(_icon(el))
            stats["shapes"] += 1

    for el in ordered:                       # arrows on top
        if el.get("type") in ("arrow", "line"):
            seg = _arrow(el, shapes)
            if seg:
                body.append(seg)
                stats["arrows"] += 1

    inner = (DEFS + '<rect width="%d" height="%d" fill="white"/>%s'
             % (w, h, "".join(body)))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" '
           f'height="{h}" viewBox="0 0 {w} {h}">{inner}</svg>')
    open(out_path, "w").write(svg)
    log(f"[svg] {out_path} {stats}")
    return stats


def _is_big(el):
    x0, y0, x1, y1 = el["bbox"]
    return (x1 - x0) * (y1 - y0) > 60_000


def _surface(el, original):
    """Vectorize an organic region — gradient bands + circle dots — at a band
    count scaled to its size. NEVER embeds original pixels."""
    from .vectorize import extract_dots
    x0, y0, x1, y1 = (int(v) for v in el["bbox"])
    crop = original.crop((x0, y0, x1, y1))
    area = (x1 - x0) * (y1 - y0)
    W, H = original.size
    img_area = W * H

    def hx(c):
        return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    dots = el.get("dots")
    if not dots:
        dots = [d for d in extract_dots(crop, round_only=True,
                                        ink_threshold=130, max_dots=300)
                if (lambda r, g, b: (r + g + b) / 3 < 195
                    or max(r, g, b) - min(r, g, b) > 50)(*hx(d["color"]))]

    # Pale background surfaces (large light gradient washes) must not obscure
    # foreground text and shapes. Render them with fewer bands and low opacity.
    arr = np.asarray(crop.convert("RGB"))
    median_color = np.median(arr.reshape(-1, 3), axis=0)
    luminance = float(np.mean(median_color))
    saturation = float(max(median_color) - min(median_color))
    is_pale = luminance > 215 and saturation < 45
    is_large = area > 0.08 * img_area

    if is_pale and is_large:
        n_bands = 3
        opacity = 0.25
    else:
        n_bands = 12 if area > 60_000 else (6 if area > 12_000 else 4)
        opacity = None

    g = manifold_svg(crop, dots=dots, flow=el.get("streamlines"),
                     n_bands=n_bands, idp=f"s{abs(hash(el['id'])) % 9999}")
    if g:
        style = f' opacity="{opacity:.2f}"' if opacity else ""
        return f'<g transform="translate({x0},{y0})"{style}>{g}</g>'
    # last resort is STILL vector: a soft tinted box + any dots, never a crop
    cs = "".join(f'<circle cx="{x0+d["cx"]:.1f}" cy="{y0+d["cy"]:.1f}" '
                 f'r="2" fill="{d["color"]}"/>' for d in dots)
    extra = ""
    trend = el.get("trend")
    if trend:
        (x1t, y1t), (x2t, y2t) = trend
        extra += (f'<line x1="{x0+x1t:.1f}" y1="{y0+y1t:.1f}" '
                  f'x2="{x0+x2t:.1f}" y2="{y0+y2t:.1f}" stroke="#d9534f" '
                  f'stroke-width="2"/>')
    for curve in el.get("curves", []):
        if len(curve) < 2:
            continue
        pts = " ".join(f"{x0+p[0]:.1f},{y0+p[1]:.1f}" for p in curve)
        extra += (f'<polyline points="{pts}" fill="none" stroke="#4472c4" '
                  f'stroke-width="2" stroke-linecap="round" '
                  f'stroke-linejoin="round"/>')
    return (f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
            f'rx="6" fill="#eef2f7"/>{cs}{extra}')


def _shape(el):
    # CraftEditor skill.md rules: stroke-width >= 2.5, dashed panels
    # dasharray "10 7", subtle drop-shadow on filled boxes (dx1 dy2 blur2).
    x0, y0, x1, y1 = el["bbox"]
    w, h = x1 - x0, y1 - y0
    fill = _color(el.get("fill"))
    stroke = _color(el.get("border_color"), "#888888"
                    if fill == "none" else "none")
    # Make sure light-gray borders on white remain visible.
    stroke = _ensure_contrast(stroke, bg_hex="#ffffff", min_delta=60)
    sw = min(2.0, max(1.0, el.get("border_width", 1) * 0.7))
    dash = ' stroke-dasharray="10 7"' if el.get("dash") else ""
    # Drop shadows on near-white fills just blur adjacent panels together.
    _light = False
    try:
        r, g, b = _hex_to_rgb(fill)
        _light = (r + g + b) / 3 > 245
    except Exception:
        _light = fill in ("none", "")
    shadow = (' filter="url(#boxsh)"'
              if fill != "none" and not el.get("dash") and not _light else "")
    t = el["type"]
    if t == "oval":
        return (f'<ellipse cx="{x0+w/2:.1f}" cy="{y0+h/2:.1f}" rx="{w/2:.1f}" '
                f'ry="{h/2:.1f}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw:.1f}"{dash}/>')
    if t == "rounded_rect" and el.get("header_fill"):
        rx = min(el.get("rx", 10.0), w / 2.0, h / 2.0)
        header_fill = _color(el["header_fill"])
        header_h = min(int(el.get("header_height", h * 0.22)), int(h * 0.45))
        # Body + colored top header strip.
        body = (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}"'
                f' rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"'
                f'{dash}{shadow}/>')
        header = (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{header_h:.1f}"'
                  f' rx="{rx:.1f}" fill="{header_fill}" stroke="none"/>')
        return body + header
    if t == "rounded_rect":
        rx = min(el.get("rx", 10.0), w / 2.0, h / 2.0)
        rx_attr = f' rx="{rx:.1f}"'
    else:
        rx_attr = ""
    return (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}"'
            f'{rx_attr} fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"'
            f'{dash}{shadow}/>')


def _text(el):
    # skill.md: DejaVu Sans (Unicode), text color #000/#333 never faded,
    # formulas in a math font, math variables italic.
    x0, y0, x1, y1 = el["bbox"]
    is_formula = el.get("type") == "formula"
    color = _color(el.get("text_color"), "#000000")
    # Diagrams are overwhelmingly light-background; light sampled text colors
    # (from anti-aliasing or faint ink) must be snapped to dark for readability.
    try:
        r, g, b = _hex_to_rgb(color)
        if (r + g + b) / 3 > 180:
            color = "#2a2a2a"
    except Exception:
        color = "#000000"
    fs = el.get("font_size") or max(9, (y1 - y0) * 0.6)
    weight = "bold" if el.get("bold") else "normal"
    style = ' font-style="italic"' if el.get("italic") else ""
    fam = el.get("font") or (
        "STIX Two Text, STIXGeneral, serif" if is_formula
        else "DejaVu Sans, Arial, sans-serif")
    if el.get("omml_lines"):
        lines = [_line_text(l) for l in el["omml_lines"]]
    else:
        lines = str(el.get("text") or el.get("latex") or "").split("\n")
    lines = [l for l in lines if l != ""] or [" "]
    n = max(1, len(lines))
    align = el.get("align", "center")
    if align == "left":
        anchor = "start"
        tx = x0 + fs * 0.15
    elif align == "right":
        anchor = "end"
        tx = x1 - fs * 0.15
    else:
        anchor = "middle"
        tx = (x0 + x1) / 2
    out = []
    for i, ln in enumerate(lines):
        ty = y0 + (y1 - y0) * (i + 0.5) / n + fs * 0.35
        out.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{fs:.0f}" '
                   f'font-family="{fam}" font-weight="{weight}"{style} '
                   f'fill="{color}" text-anchor="{anchor}">{_esc(ln)}</text>')
    return "".join(out)


def _line_text(l):
    return l.get("value", "") if l["kind"] == "text" else _omml_to_unicode(l.get("value", ""))


def _omml_to_unicode(omml):
    """Pull the run-text out of OMML in reading order — the math characters
    (β, ⟨⟩, fractions as a/b) without the XML tags. Better than blind tag
    stripping which dropped spacing."""
    import re
    if not omml:
        return ""
    # OMML fraction <m:f><m:num>..</m:num><m:den>..</m:den></m:f> → (num)/(den)
    omml = re.sub(r"<m:f>.*?<m:num>(.*?)</m:num>.*?<m:den>(.*?)</m:den>.*?</m:f>",
                  lambda m: f"({_omml_to_unicode(m.group(1))})/"
                            f"({_omml_to_unicode(m.group(2))})", omml,
                  flags=re.S)
    texts = re.findall(r"<m:t>(.*?)</m:t>", omml, flags=re.S)
    return _esc("".join(texts) if texts else re.sub(r"<[^>]+>", "", omml))


def _arrow(el, shapes):
    # Prefer the perimeter-snapped points computed by handlers.py.  They land
    # on the target shape's edge rather than its center.
    if el.get("points"):
        p = el["points"]
        s, d = (p[0], p[1]), (p[2], p[3])
    elif el.get("from_id") and el.get("to_id"):
        src = shapes.get(el.get("from_id") or "")
        dst = shapes.get(el.get("to_id") or "")
        if src and dst:
            s = ((src["bbox"][0] + src["bbox"][2]) / 2,
                 (src["bbox"][1] + src["bbox"][3]) / 2)
            d = ((dst["bbox"][0] + dst["bbox"][2]) / 2,
                 (dst["bbox"][1] + dst["bbox"][3]) / 2)
        else:
            return ""
    else:
        return ""
    # skill.md: arrows medium-gray (#808080), thin (1.5-2px), curved Q-bezier
    # rather than straight lines crossing boxes.
    color = _color(el.get("color"), "#808080")
    th = el.get("thickness", 2)
    head = ' marker-end="url(#ah)"' if el["type"] == "arrow" else ""
    mx, my = (s[0] + d[0]) / 2, (s[1] + d[1]) / 2
    # gentle bow perpendicular to the segment for a non-straight feel
    dx, dy = d[0] - s[0], d[1] - s[1]
    L = max(1.0, (dx * dx + dy * dy) ** 0.5)
    bow = min(18.0, L * 0.06)
    cxp, cyp = mx - dy / L * bow, my + dx / L * bow
    return (f'<path d="M{s[0]:.1f},{s[1]:.1f} Q{cxp:.1f},{cyp:.1f} '
            f'{d[0]:.1f},{d[1]:.1f}" stroke="{color}" '
            f'stroke-width="{max(1.6, th*0.45):.1f}" fill="none"{head}/>')


def _chart(el):
    spec = el.get("chart") or {}
    x0, y0, x1, y1 = el["bbox"]
    w, h = x1 - x0, y1 - y0
    ctype = (spec.get("type") or "bar").lower()
    cats = spec.get("categories", [])
    series = spec.get("series", [])
    points = spec.get("points", [])

    if ctype == "scatter":
        out = []
        xs = [p["x"] for p in points if isinstance(p, dict)]
        ys = [p["y"] for p in points if isinstance(p, dict)]
        if xs and ys:
            r = max(1.5, min(w, h) * 0.018)
            for p in points:
                if not isinstance(p, dict):
                    continue
                px = x0 + float(p["x"]) * w
                py = y1 - float(p["y"]) * h
                out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" '
                           f'fill="{_color(p.get("color"),"#4472c4")}"/>')
        else:
            # Fallback: draw as dotcloud if points missing
            return _surface(el, None)
        trend = spec.get("trend") or {}
        if trend and "slope" in trend:
            m, b = float(trend["slope"]), float(trend.get("intercept", 0))
            x1_ = x0 + w * 0.1
            y1_ = y1 - (m * 0.1 + b) * h
            x2_ = x0 + w * 0.9
            y2_ = y1 - (m * 0.9 + b) * h
            tc = _color(trend.get("color"), "#d9534f")
            out.append(f'<line x1="{x1_:.1f}" y1="{y1_:.1f}" x2="{x2_:.1f}" '
                       f'y2="{y2_:.1f}" stroke="{tc}" stroke-width="2"/>')
        return "".join(out)

    if ctype == "line":
        out = []
        for s in series:
            pts = s.get("values", [])
            if not pts:
                continue
            # Normalize list of dicts [{x,y}] into coordinate pairs
            if pts and isinstance(pts[0], dict):
                pts = [(float(p.get("x", 0)), float(p.get("y", 0))) for p in pts]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                coords = " ".join(
                    f"{x0+xs[i]*w:.1f},{y1-h*0.1-ys[i]*h*0.8:.1f}"
                    for i in range(len(pts)))
            else:
                pts = [float(v) for v in pts]
                xs = [i / max(1, len(pts) - 1) for i in range(len(pts))]
                vmax = max(pts) or 1
                coords = " ".join(
                    f"{x0+xs[i]*w:.1f},{y1-h*0.1-(pts[i]/vmax)*h*0.8:.1f}"
                    for i in range(len(pts)))
            c = _color(s.get("color"), "#4472c4")
            out.append(f'<polyline points="{coords}" fill="none" stroke="{c}" '
                       f'stroke-width="2.5" stroke-linecap="round" '
                       f'stroke-linejoin="round"/>')
        return "".join(out) if out else _shape(dict(el, type="rect", fill="none",
                                                     border_color="#bbb"))

    # bar (default)
    if not cats or not series:
        return _shape(dict(el, type="rect", fill="none", border_color="#bbb"))
    # Defensively flatten numeric values; ignore malformed dicts.
    def _num(v):
        try:
            return float(v)
        except Exception:
            return 0.0
    allv = [_num(v) for s in series for v in s.get("values", [])] or [1]
    vmax = max(allv) or 1
    out = [f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
           f'fill="white" stroke="#ccc"/>']
    n = len(cats) * max(1, len(series))
    bw = w / (n + len(cats))
    i = 0
    for ci in range(len(cats)):
        for s in series:
            vals = s.get("values", [])
            v = _num(vals[ci]) if ci < len(vals) else 0
            bh = (v / vmax) * h * 0.85
            bx = x0 + (i + ci * 0.5 + 0.5) * bw
            out.append(f'<rect x="{bx:.1f}" y="{y1-bh:.1f}" width="{bw*0.8:.1f}" '
                       f'height="{bh:.1f}" fill="{_color(s.get("color"),"#4472c4")}"/>')
            i += 1
    return "".join(out)


def _icon(el):
    # draw vector pictograms — unicode glyphs render as tofu in rsvg/Figma
    # when the font lacks it.  Deterministic SVG paths always render.
    ic = el.get("icon") or {}
    kind = ic.get("kind", "other")
    x0, y0, x1, y1 = el["bbox"]
    w, h = x1 - x0, y1 - y0
    c = _color(ic.get("color"), "#6b7a8d")
    cx, cy_ = x0 + w / 2, y0 + h / 2

    if kind == "database":
        rxt, rxb = w * 0.38, w * 0.38
        ry = h * 0.12
        ytop = y0 + h * 0.18
        ybot = y0 + h * 0.72
        return (
            f'<path d="M{x0+w*0.12:.1f},{ytop:.1f} '
            f'C{x0+w*0.12:.1f},{ytop-ry:.1f} {x1-w*0.12:.1f},{ytop-ry:.1f} '
            f'{x1-w*0.12:.1f},{ytop:.1f} '
            f'L{x1-w*0.12:.1f},{ybot:.1f} '
            f'C{x1-w*0.12:.1f},{ybot+ry:.1f} {x0+w*0.12:.1f},{ybot+ry:.1f} '
            f'{x0+w*0.12:.1f},{ybot:.1f} Z" fill="{c}"/>'
            f'<ellipse cx="{cx:.1f}" cy="{ytop:.1f}" rx="{w*0.38:.1f}" '
            f'ry="{ry:.1f}" fill="{c}"/>'
            f'<path d="M{x0+w*0.12:.1f},{ytop+h*0.18:.1f} '
            f'C{x0+w*0.12:.1f},{ytop+h*0.18-ry:.1f} '
            f'{x1-w*0.12:.1f},{ytop+h*0.18-ry:.1f} '
            f'{x1-w*0.12:.1f},{ytop+h*0.18:.1f}" fill="none" stroke="white" '
            f'stroke-width="{max(1.2,h*0.03):.1f}" opacity="0.7"/>')

    if kind == "gear":
        n = 8
        out_r = min(w, h) * 0.34
        in_r = out_r * 0.62
        pts = []
        for i in range(n * 2):
            ang = np.pi * i / n - np.pi / 2
            r = out_r if i % 2 == 0 else in_r
            pts.append(f"{cx + r * np.cos(ang):.1f},{cy_ + r * np.sin(ang):.1f}")
        d = "M" + " L".join(pts) + " Z"
        return (f'<path d="{d}" fill="{c}"/>'
                f'<circle cx="{cx:.1f}" cy="{cy_:.1f}" r="{out_r*0.25:.1f}" '
                f'fill="white"/>')

    if kind == "scatter":
        dots = [(0.25, 0.65), (0.40, 0.50), (0.55, 0.58), (0.70, 0.35),
                (0.82, 0.28), (0.35, 0.72), (0.62, 0.45)]
        r = max(1.5, min(w, h) * 0.045)
        ds = "".join(
            f'<circle cx="{x0+w*px:.1f}" cy="{y0+h*py:.1f}" r="{r:.1f}" '
            f'fill="{c}"/>' for px, py in dots)
        return (ds + f'<line x1="{x0+w*0.20:.1f}" y1="{y0+h*0.78:.1f}" '
                f'x2="{x1-w*0.15:.1f}" y2="{y0+h*0.22:.1f}" stroke="{c}" '
                f'stroke-width="{max(1.2,h*0.025):.1f}"/>')

    if kind == "line":
        pts = " ".join(
            f"{x0+w*t:.1f},{y0+h*(0.75-0.55*np.sin(np.pi*t)):.1f}"
            for t in [i/20 for i in range(21)])
        return (f'<polyline points="{pts}" fill="none" stroke="{c}" '
                f'stroke-width="{max(1.5,h*0.04):.1f}" stroke-linecap="round"/>')

    if kind == "bell":   # distribution curve
        pts = " ".join(
            f"{x0+t*w:.1f},{y1-h*0.15-np.exp(-((t-0.5)**2)/0.04)*h*0.6:.1f}"
            for t in [i/20 for i in range(21)])
        return f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>'

    if kind in ("warning", "alert"):
        return (f'<path d="M{cx:.1f},{y0+h*0.12:.1f} L{x1-w*0.08:.1f},{y1-h*0.12:.1f} '
                f'L{x0+w*0.08:.1f},{y1-h*0.12:.1f} Z" fill="{c}"/>'
                f'<circle cx="{cx:.1f}" cy="{y0+h*0.42:.1f}" r="{h*0.06:.1f}" '
                f'fill="white"/>'
                f'<rect x="{cx-h*0.045:.1f}" y="{y0+h*0.52:.1f}" '
                f'width="{h*0.09:.1f}" height="{h*0.22:.1f}" rx="1" fill="white"/>')

    if kind == "hourglass":
        return (
            f'<path d="M{x0+w*0.25:.1f},{y0+h*0.10:.1f} L{x1-w*0.25:.1f},'
            f'{y0+h*0.10:.1f} L{cx:.1f},{cy_:.1f} L{x1-w*0.25:.1f},'
            f'{y1-h*0.10:.1f} L{x0+w*0.25:.1f},{y1-h*0.10:.1f} Z" fill="{c}"/>'
            f'<rect x="{x0+w*0.22:.1f}" y="{y0+h*0.06:.1f}" width="{w*0.56:.1f}" '
            f'height="{h*0.06:.1f}" rx="1" fill="{c}"/>'
            f'<rect x="{x0+w*0.22:.1f}" y="{y1-h*0.12:.1f}" width="{w*0.56:.1f}" '
            f'height="{h*0.06:.1f}" rx="1" fill="{c}"/>')

    if kind == "shield":
        return (f'<path d="M{cx:.1f},{y0+h*0.10:.1f} '
                f'L{x1-w*0.12:.1f},{y0+h*0.22:.1f} '
                f'C{x1-w*0.12:.1f},{y0+h*0.65:.1f} {cx:.1f},{y1-h*0.08:.1f} '
                f'{cx:.1f},{y1-h*0.08:.1f} '
                f'C{cx:.1f},{y1-h*0.08:.1f} {x0+w*0.12:.1f},{y0+h*0.65:.1f} '
                f'{x0+w*0.12:.1f},{y0+h*0.22:.1f} Z" fill="{c}"/>'
                f'<path d="M{x0+w*0.38:.1f},{cy_:.1f} '
                f'L{cx:.1f},{y1-h*0.28:.1f} '
                f'L{x1-w*0.30:.1f},{y0+h*0.30:.1f}" fill="none" stroke="white" '
                f'stroke-width="{max(2,h*0.06):.1f}" stroke-linecap="round" '
                f'stroke-linejoin="round"/>')

    if kind == "document":
        return (f'<rect x="{x0+w*0.15:.1f}" y="{y0+h*0.12:.1f}" width="{w*0.70:.1f}" '
                f'height="{h*0.76:.1f}" rx="3" fill="{c}"/>'
                f'<rect x="{x0+w*0.25:.1f}" y="{y0+h*0.25:.1f}" width="{w*0.35:.1f}" '
                f'height="{h*0.05:.1f}" rx="1" fill="white"/>'
                f'<rect x="{x0+w*0.25:.1f}" y="{y0+h*0.38:.1f}" width="{w*0.50:.1f}" '
                f'height="{h*0.05:.1f}" rx="1" fill="white"/>'
                f'<rect x="{x0+w*0.55:.1f}" y="{y0+h*0.55:.1f}" width="{w*0.25:.1f}" '
                f'height="{h*0.22:.1f}" rx="1" fill="white"/>')

    if kind == "check":
        return (f'<path d="M{x0+w*0.22:.1f},{cy_:.1f} '
                f'L{x0+w*0.44:.1f},{y1-h*0.22:.1f} '
                f'L{x1-w*0.18:.1f},{y0+h*0.18:.1f}" fill="none" stroke="{c}" '
                f'stroke-width="{max(2.5,h*0.07):.1f}" stroke-linecap="round" '
                f'stroke-linejoin="round"/>')

    if kind == "cross":
        sw = max(2.5, h * 0.07)
        return (f'<line x1="{x0+w*0.25:.1f}" y1="{y0+h*0.25:.1f}" '
                f'x2="{x1-w*0.25:.1f}" y2="{y1-h*0.25:.1f}" stroke="{c}" '
                f'stroke-width="{sw:.1f}" stroke-linecap="round"/>'
                f'<line x1="{x1-w*0.25:.1f}" y1="{y0+h*0.25:.1f}" '
                f'x2="{x0+w*0.25:.1f}" y2="{y1-h*0.25:.1f}" stroke="{c}" '
                f'stroke-width="{sw:.1f}" stroke-linecap="round"/>')

    if kind == "arrow":
        return (f'<path d="M{x0+w*0.15:.1f},{cy_:.1f} L{x1-w*0.25:.1f},{cy_:.1f} '
                f'L{x1-w*0.25:.1f},{y0+h*0.25:.1f} L{x1-w*0.08:.1f},{cy_:.1f} '
                f'L{x1-w*0.25:.1f},{y1-h*0.25:.1f} L{x1-w*0.25:.1f},{cy_:.1f} Z" '
                f'fill="{c}"/>')

    # fallback: a simple rhomb
    return (f'<rect x="{x0+w*0.2:.1f}" y="{y0+h*0.2:.1f}" width="{w*0.6:.1f}" '
            f'height="{h*0.6:.1f}" rx="3" fill="none" stroke="{c}" '
            f'stroke-width="2"/>')


def cy(y0, y1):
    return f"{(y0+y1)/2:.1f}"


DEFS = ('<defs>'
        '<marker id="ah" markerWidth="8" markerHeight="6" refX="6" refY="3" '
        'orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#808080"/></marker>'
        '<filter id="boxsh" x="-10%" y="-10%" width="120%" height="130%">'
        '<feDropShadow dx="1" dy="2" stdDeviation="2" flood-opacity="0.2"/>'
        '</filter>'
        '</defs>')
