"""Render the diagram IR back to pixels with PIL.

This is the diff PROXY for the iterative loop, not the deliverable (the
deliverable is the PPTX). It only needs to be faithful enough that a wrong
bbox / fill / missing element shows up as residual against the original.
Fonts therefore approximate (Helvetica), and styling is flat.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

_TEXT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]

_FONT_CANDIDATES_BY_ROLE = {
    ("times", False, False): [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
    ],
    ("times", True, False): [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Times.ttc",
    ],
    ("times", False, True): [
        "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
        "/System/Library/Fonts/Times.ttc",
    ],
    ("times", True, True): [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
        "/System/Library/Fonts/Times.ttc",
    ],
    ("arial", False, False): [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    ("arial", True, False): [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    ("arial", False, True): [
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "/Library/Fonts/Arial Italic.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    ("arial", True, True): [
        "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
        "/Library/Fonts/Arial Bold Italic.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
}

_MATH_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/STIXGeneral.otf",
    "/System/Library/Fonts/Supplemental/STIXTwoText-Italic.ttf",
    "/System/Library/Fonts/Symbol.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Times.ttc",
]

_MATH_CHARS = set("βγτ∇≈θ₀₁₂₃<>≤≥≡∥⟨⟩")


def _font(size: int, math_text: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (_MATH_FONT_CANDIDATES + _TEXT_FONT_CANDIDATES
                  if math_text else _TEXT_FONT_CANDIDATES)
    for p in candidates:
        try:
            return ImageFont.truetype(p, max(10, int(size)))  # Helvetica.ttc div/0 below 10
        except OSError:
            continue
    return ImageFont.load_default()


def _font_for_text(text: str, size: int) -> ImageFont.FreeTypeFont:
    return _font(size, math_text=any(ch in _MATH_CHARS for ch in str(text)))


def _font_for_element(el: dict, text: str, size: int) -> ImageFont.FreeTypeFont:
    requested = str(el.get("font") or "").lower()
    family = "times" if "times" in requested or "cambria" in requested else "arial"
    candidates = _FONT_CANDIDATES_BY_ROLE.get(
        (family, bool(el.get("bold")), bool(el.get("italic"))),
        [],
    )
    if any(ch in _MATH_CHARS for ch in str(text)):
        candidates = _MATH_FONT_CANDIDATES + candidates
    for p in candidates:
        try:
            return ImageFont.truetype(p, max(10, int(size)))
        except OSError:
            continue
    return _font_for_text(text, size)


def _rgb(hex_str: str | None, default=None):
    if not hex_str or str(hex_str).lower() in ("none", "transparent", "null"):
        return default
    s = str(hex_str).lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return default
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def faithful_crop(original: Image.Image, el: dict, elements: list) -> Image.Image:
    """Crop of the original for a demoted element, with NATIVE children's
    regions patched to the surrounding color.

    Without the patch, a native text box placed over its own pixels inside a
    parent screenshot renders doubled — and editing the text leaves the stale
    pixels behind. The patch is a ring-median flat fill (diagram fills are
    flat; photographic crops rarely contain native children).
    """
    import numpy as np

    x0, y0, x1, y1 = (int(v) for v in el["bbox"])
    x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)  # never an empty crop
    crop = original.convert("RGB").crop((x0, y0, x1, y1))
    natives = [o for o in elements
               if o is not el and o.get("status") == "native" and "bbox" in o]
    if not natives:
        return crop

    from .diff import children_of
    child_boxes = children_of(el, natives, containment=0.9)
    if not child_boxes:
        return crop

    arr = np.asarray(crop).copy()
    ch, cw = arr.shape[:2]
    for cx0, cy0, cx1, cy1 in child_boxes:
        lx0 = max(0, int(cx0 - x0)); ly0 = max(0, int(cy0 - y0))
        lx1 = min(cw, int(cx1 - x0)); ly1 = min(ch, int(cy1 - y0))
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        rx0 = max(0, lx0 - 4); ry0 = max(0, ly0 - 4)
        rx1 = min(cw, lx1 + 4); ry1 = min(ch, ly1 + 4)
        ring = np.concatenate([
            arr[ry0:ly0, rx0:rx1].reshape(-1, 3),
            arr[ly1:ry1, rx0:rx1].reshape(-1, 3),
            arr[ly0:ly1, rx0:lx0].reshape(-1, 3),
            arr[ly0:ly1, lx1:rx1].reshape(-1, 3),
        ])
        if len(ring) == 0:
            continue
        arr[ly0:ly1, lx0:lx1] = np.median(ring, axis=0).astype(arr.dtype)
    return Image.fromarray(arr)


def render(ir: dict, original: Image.Image | None = None) -> Image.Image:
    """Render IR to an RGB canvas the size of the source image.

    `original` is required when any element is a raster_crop (the crop is
    taken from it — that is the fidelity guarantee).
    """
    w, h = ir["image"]["width"], ir["image"]["height"]
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    border_w = max(1, round(min(w, h) / 400))

    shapes = {e["id"]: e for e in ir["elements"] if "bbox" in e}
    ordered = sorted((e for e in ir["elements"]), key=lambda e: e.get("z", 0))

    # pass 1: boxed elements (shapes, text, crops) in z order
    for el in ordered:
        t = el["type"]
        if t in ("arrow", "line"):
            continue
        if t == "raster_crop":
            if original is None:
                raise ValueError(f"{t} {el['id']} needs the original image")
            x0, y0 = int(el["bbox"][0]), int(el["bbox"][1])
            canvas.paste(faithful_crop(original, el, ir["elements"]), (x0, y0))
        elif t == "formula":
            _draw_formula(draw, el)
        elif t == "chart":
            _draw_chart(draw, el)
        elif t == "text":
            _draw_text(draw, el, canvas)
        elif t == "icon":
            _draw_icon(draw, el)
        elif t == "freeform":
            _draw_local_paths(draw, el)
        elif t in ("dotcloud", "surface"):
            _draw_dotcloud(draw, el)
        else:
            _draw_shape(draw, el, border_w)
            if el.get("text"):
                _draw_text(draw, el, canvas)

    # pass 2: connectors on top
    for el in ordered:
        if el["type"] in ("arrow", "line"):
            _draw_connector(draw, el, shapes, border_w)

    return canvas


# -- shapes -----------------------------------------------------------------

def _draw_shape(draw: ImageDraw.ImageDraw, el: dict, border_w: int) -> None:
    x0, y0, x1, y1 = el["bbox"]
    fill = _rgb(el.get("fill"))
    outline = _rgb(el.get("border_color"))
    if fill is None and outline is None:
        outline = (51, 51, 51)  # invisible shapes don't exist in diagrams
    t = el["type"]
    if t == "oval":
        if fill is not None:
            draw.ellipse([x0, y0, x1, y1], fill=fill)
        if outline is not None:
            if el.get("dash"):
                _draw_dashed_ellipse(draw, [x0, y0, x1, y1], outline, border_w)
            else:
                draw.ellipse([x0, y0, x1, y1], outline=outline, width=border_w)
    elif t == "diamond":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)],
                     fill=fill, outline=outline, width=border_w)
    elif t == "hexagon":
        dx = (x1 - x0) * 0.2
        cy = (y0 + y1) / 2
        draw.polygon([(x0 + dx, y0), (x1 - dx, y0), (x1, cy),
                      (x1 - dx, y1), (x0 + dx, y1), (x0, cy)],
                     fill=fill, outline=outline, width=border_w)
    elif t == "parallelogram":
        dx = (x1 - x0) * 0.15
        draw.polygon([(x0 + dx, y0), (x1, y0), (x1 - dx, y1), (x0, y1)],
                     fill=fill, outline=outline, width=border_w)
    elif t == "rounded_rect":
        corner = el.get("corner")
        if corner is None:
            r = min(x1 - x0, y1 - y0) * 0.18
        else:
            try:
                c = float(corner)
            except (TypeError, ValueError):
                c = 0.18
            r = min(x1 - x0, y1 - y0) * (c if c <= 1 else c / max(1, min(x1 - x0, y1 - y0)))
        if el.get("dash") and outline is not None:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill)
            _draw_dashed_rect(draw, [x0, y0, x1, y1], outline, border_w)
        else:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=r,
                                   fill=fill, outline=outline, width=border_w)
    else:  # rect
        if el.get("dash") and outline is not None:
            draw.rectangle([x0, y0, x1, y1], fill=fill)
            _draw_dashed_rect(draw, [x0, y0, x1, y1], outline, border_w)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=border_w)


def _draw_dashed_ellipse(draw: ImageDraw.ImageDraw, box: list[float],
                         color: tuple[int, int, int], width: int) -> None:
    # PIL has no dashed ellipse primitive; draw short arc segments.
    dash_deg = 13
    gap_deg = 9
    angle = 0
    while angle < 360:
        draw.arc(box, start=angle, end=min(360, angle + dash_deg),
                 fill=color, width=max(1, width))
        angle += dash_deg + gap_deg


def _draw_dashed_rect(draw: ImageDraw.ImageDraw, box: list[float],
                      color: tuple[int, int, int], width: int) -> None:
    x0, y0, x1, y1 = [float(v) for v in box]
    dash = max(6.0, width * 5.0)
    gap = max(4.0, width * 3.0)
    for start, end in (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ):
        _draw_dashed_line(draw, start, end, color, width, dash, gap)


def _draw_dashed_line(draw: ImageDraw.ImageDraw, start: tuple[float, float],
                      end: tuple[float, float], color: tuple[int, int, int],
                      width: int, dash: float, gap: float) -> None:
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        seg_end = min(length, pos + dash)
        draw.line([
            (x0 + ux * pos, y0 + uy * pos),
            (x0 + ux * seg_end, y0 + uy * seg_end),
        ], fill=color, width=max(1, width))
        pos += dash + gap


def _draw_text(draw: ImageDraw.ImageDraw, el: dict,
               canvas: Image.Image | None = None) -> None:
    text = (el.get("text") or "").strip()
    if not text:
        return
    if el.get("rotation") and canvas is not None:
        _draw_rotated_text(draw, canvas, el)
        return
    x0, y0, x1, y1 = el["bbox"]
    bw, bh = x1 - x0, y1 - y0
    color = _rgb(el.get("text_color")) or (0, 0, 0)
    lines = text.split("\n") if "\n" in text else [text]
    runs = el.get("runs") if "\n" not in text else None

    size = el.get("font_size") or (bh / max(1, len(lines))) * 0.7
    size = max(10, min(int(size), max(10, int(bh))))
    font = _font_for_element(el, text, size)
    # shrink until the widest line fits
    typo = ((el.get("ext") or {}).get("typography") or {})
    width_factor = float(typo.get("fit_width_factor") or 0.50)
    width_slack = 0.98 if width_factor >= 0.48 else min(1.12, 0.98 * 0.50 / max(0.30, width_factor))
    while size > 10:
        widest = max(draw.textlength(ln, font=font) for ln in lines)
        if widest <= bw * width_slack:
            break
        size = int(size * 0.9)
        font = _font_for_element(el, text, size)

    line_h = size * float(typo.get("line_spacing") or 1.18)
    total_h = line_h * len(lines)
    ty = y0 + (bh - total_h) / 2
    align = str(el.get("align") or "").lower()
    if isinstance(runs, list) and runs:
        run_fonts = []
        run_widths = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            tmp = dict(el)
            tmp.update({k: v for k, v in run.items() if k != "text"})
            run_text = str(run.get("text") or "")
            run_size = int(run.get("font_size") or size)
            run_font = _font_for_element(tmp, run_text, run_size)
            run_fonts.append((run, run_text, run_font, run_size))
            run_widths.append(draw.textlength(run_text, font=run_font))
        total_w = sum(run_widths)
        if align in {"left", "start"}:
            tx = x0 + 2
        elif align in {"right", "end"}:
            tx = x1 - total_w - 2
        else:
            tx = x0 + (bw - total_w) / 2
        for run, run_text, run_font, run_size in run_fonts:
            tmp_color = _rgb(run.get("color") or run.get("text_color")) or color
            draw.text((tx, ty + _baseline_offset(run, run_size)), run_text, fill=tmp_color, font=run_font)
            tx += draw.textlength(run_text, font=run_font)
        return
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        if align in {"left", "start"}:
            tx = x0 + 2
        elif align in {"right", "end"}:
            tx = x1 - tw - 2
        else:
            tx = x0 + (bw - tw) / 2
        _draw_text_line(draw, ln, tx, ty, color, font, size)
        ty += line_h


def _draw_text_line(draw: ImageDraw.ImageDraw, text: str, x: float, y: float,
                    color: tuple[int, int, int], font: ImageFont.FreeTypeFont,
                    size: int) -> None:
    base, hats = _strip_combining_hats(text)
    draw.text((x, y), base, fill=color, font=font)
    if not hats:
        return
    for idx in hats:
        if idx < 0 or idx >= len(base):
            continue
        prefix_w = draw.textlength(base[:idx], font=font)
        char_w = draw.textlength(base[idx], font=font)
        cx = x + prefix_w + char_w * 0.50
        top = y + size * 0.18
        half = max(1.6, size * 0.075)
        height = max(1.4, size * 0.060)
        stroke = max(1, int(size * 0.035))
        draw.line((cx - half, top, cx, top - height), fill=color, width=stroke)
        draw.line((cx, top - height, cx + half, top), fill=color, width=stroke)


def _strip_combining_hats(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    hats: list[int] = []
    for ch in text:
        if ch == "\u0302":
            if out:
                hats.append(len(out) - 1)
            continue
        out.append(ch)
    return "".join(out), hats


def _baseline_offset(run: dict, size: int) -> float:
    try:
        baseline = float(run.get("baseline") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    # DrawingML baseline is in thousandths of font size. Negative means
    # subscript, positive means superscript.
    return -baseline / 100000.0 * float(size)


def _draw_formula(draw: ImageDraw.ImageDraw, el: dict) -> None:
    """Draw a native-ish formula preview for verification.

    PPTX output uses editable text boxes for formulas.  The proxy should not
    paste original pixels here, otherwise the planner gets misleading evidence
    and formula regions behave like screenshots during visual debugging.
    """
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return
    latex = str(el.get("latex") or (el.get("ext") or {}).get("latex") or "")
    text = str(el.get("text") or latex or "").strip()
    color = _rgb(el.get("text_color")) or (0, 0, 0)
    layout = ((el.get("ext") or {}).get("math_layout") or {})
    if layout:
        _draw_math_layout(draw, el, layout, color)
        return
    if "langle" in latex and "beta" in latex and "gamma" in latex:
        size = int(el.get("font_size") or min(34, bh * 0.42))
        size = max(12, min(size, int(bh * 0.44)))
        font = _font(size, math_text=True)
        small = _font(max(10, int(size * 0.68)), math_text=True)
        lhs = "A ="
        num = "|⟨β, γ⟩|"
        den = "∥β∥ ∥γ∥"
        rhs = "≈ 1"
        lhs_w = draw.textlength(lhs, font=font)
        num_w = draw.textlength(num, font=font)
        den_w = draw.textlength(den, font=small)
        rhs_w = draw.textlength(rhs, font=font)
        frac_w = max(num_w, den_w) + 8
        total_w = lhs_w + frac_w + rhs_w + 22
        tx = x0 + max(2.0, (bw - total_w) / 2)
        cy = y0 + bh * 0.55
        draw.text((tx, cy - size * 0.50), lhs, fill=color, font=font)
        fx = tx + lhs_w + 10
        draw.text((fx + (frac_w - num_w) / 2, cy - size * 1.30), num, fill=color, font=font)
        draw.line((fx, cy - size * 0.02, fx + frac_w, cy - size * 0.02),
                  fill=color, width=max(1, int(size * 0.055)))
        draw.text((fx + (frac_w - den_w) / 2, cy + size * 0.18), den, fill=color, font=small)
        draw.text((fx + frac_w + 12, cy - size * 0.50), rhs, fill=color, font=font)
        return

    preview = _latex_preview(text)
    temp = dict(el)
    temp["type"] = "text"
    temp["text"] = preview
    temp.setdefault("align", "center")
    _draw_text(draw, temp)


def _draw_math_layout(draw: ImageDraw.ImageDraw, el: dict, layout: dict,
                      color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    bw, bh = x1 - x0, y1 - y0
    tokens = [t for t in (layout.get("tokens") or []) if isinstance(t, dict)]
    if not tokens:
        return
    size = int(el.get("font_size") or min(34, bh * 0.56))
    size = max(10, min(size, int(max(10, bh * 0.78))))
    while size > 10:
        font = _font_for_element(el, "".join(str(t.get("text") or "") for t in tokens), size)
        total_w = sum(draw.textlength(str(t.get("text") or ""), font=font) for t in tokens)
        if total_w <= bw * 0.97 and size * 1.18 <= bh * 0.94:
            break
        size = int(size * 0.92)
    font = _font_for_element(el, "".join(str(t.get("text") or "") for t in tokens), size)
    widths = [draw.textlength(str(t.get("text") or ""), font=font) for t in tokens]
    total_w = sum(widths)
    align = str(el.get("align") or "").lower()
    if align in {"left", "start"}:
        tx = x0 + 2
    elif align in {"right", "end"}:
        tx = x1 - total_w - 2
    else:
        tx = x0 + (bw - total_w) / 2
    ty = y0 + (bh - size * 1.18) / 2
    for tok, width in zip(tokens, widths):
        txt = str(tok.get("text") or "")
        draw.text((tx, ty), txt, fill=color, font=font)
        if tok.get("accent") == "hat" and txt:
            char_w = draw.textlength(txt, font=font)
            cx = tx + char_w * 0.50
            top = ty + size * 0.15
            half = max(2.0, size * 0.12)
            height = max(1.8, size * 0.075)
            stroke = max(1, int(size * 0.045))
            draw.line((cx - half, top, cx, top - height), fill=color, width=stroke)
            draw.line((cx, top - height, cx + half, top), fill=color, width=stroke)
        tx += width


def _latex_preview(text: str) -> str:
    out = text
    replacements = {
        r"\beta": "β",
        r"\gamma": "γ",
        r"\tau": "τ",
        r"\nabla": "∇",
        r"\approx": "≈",
        r"\langle": "<",
        r"\rangle": ">",
        r"\|": "||",
        "{": "",
        "}": "",
        "$": "",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = out.replace(r"\frac", "")
    return " ".join(out.split())


def _draw_rotated_text(draw: ImageDraw.ImageDraw, canvas: Image.Image,
                       el: dict) -> None:
    text = (el.get("text") or "").strip()
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return
    color = _rgb(el.get("text_color")) or (0, 0, 0)
    size = int(el.get("font_size") or min(bw, bh) * 0.8)
    size = max(8, min(size, int(max(8, bw * 0.92))))
    font = _font_for_element(el, text, size)
    while size > 8:
        text_w = draw.textlength(text, font=font)
        if text_w <= bh * 0.96 and size * 1.25 <= bw * 0.98:
            break
        size = int(size * 0.9)
        font = _font_for_element(el, text, size)

    runs = el.get("runs") if "\n" not in text else None
    run_draws = []
    if isinstance(runs, list) and runs:
        total_w = 0.0
        max_size = size
        for run in runs:
            if not isinstance(run, dict):
                continue
            tmp_el = dict(el)
            tmp_el.update({k: v for k, v in run.items() if k != "text"})
            run_text = str(run.get("text") or "")
            run_size = int(run.get("font_size") or size)
            run_font = _font_for_element(tmp_el, run_text, run_size)
            run_w = draw.textlength(run_text, font=run_font)
            run_draws.append((run, run_text, run_font, run_size, run_w))
            total_w += run_w
            max_size = max(max_size, run_size)
        tmp_w = max(1, int(total_w + 8))
        tmp_h = max(1, int(max_size * 1.55 + 8))
    else:
        tmp_w = max(1, int(draw.textlength(text, font=font) + 8))
        tmp_h = max(1, int(size * 1.45 + 8))
    tmp = Image.new("RGBA", (tmp_w, tmp_h), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    if run_draws:
        tx = 4.0
        for run, run_text, run_font, run_size, run_w in run_draws:
            tmp_color = _rgb(run.get("color") or run.get("text_color")) or color
            ty = (tmp_h - run_size * 1.2) / 2 + _baseline_offset(run, run_size)
            td.text((tx, ty), run_text, fill=tmp_color + (255,), font=run_font)
            tx += run_w
    else:
        td.text((4, (tmp_h - size * 1.2) / 2), text, fill=color + (255,), font=font)
    angle = -float(el.get("rotation") or 0)
    rot = tmp.rotate(angle, expand=True)
    px = int(x0 + (bw - rot.width) / 2)
    py = int(y0 + (bh - rot.height) / 2)
    canvas.paste(rot, (px, py), rot)


def _wrap(text: str, bw: float, bh: float) -> list[str]:
    """Crude aspect-based wrap: aim for the line count the box suggests."""
    words = text.split()
    if len(words) <= 1 or bw <= 0:
        return [text]
    est_lines = max(1, min(len(words), round(math.sqrt(len(text) * 0.6 * bh / bw))))
    per = math.ceil(len(words) / est_lines)
    return [" ".join(words[i:i + per]) for i in range(0, len(words), per)]


def _draw_icon(draw: ImageDraw.ImageDraw, el: dict) -> None:
    """Proxy native pictograms for the diff loop."""
    x0, y0, x1, y1 = el["bbox"]
    icon = el.get("icon") or {}
    kind = str(icon.get("kind") or "").lower()
    variant = str(icon.get("variant") or "").lower()
    color = _rgb(icon.get("color")) or (85, 85, 85)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    if kind == "warning":
        pts = [(cx, y0 + h * 0.08), (x1 - w * 0.08, y1 - h * 0.08),
               (x0 + w * 0.08, y1 - h * 0.08)]
        fill = _blend(color, (255, 255, 255), 0.82 if variant == "solid" else 0.22)
        mark = (255, 255, 255) if variant == "solid" else color
        draw.polygon(pts, fill=fill, outline=color)
        _center_text(draw, "!", [x0, y0 + h * 0.24, x1, y1 - h * 0.08], mark, int(h * 0.52), bold=True)
        return
    if kind == "shield":
        pts = [(cx, y0 + h * 0.05), (x1 - w * 0.12, y0 + h * 0.22),
               (x1 - w * 0.18, y0 + h * 0.70), (cx, y1 - h * 0.06),
               (x0 + w * 0.18, y0 + h * 0.70), (x0 + w * 0.12, y0 + h * 0.22)]
        fill = _blend(color, (255, 255, 255), 0.72 if variant == "solid" else 0.18)
        draw.polygon(pts, fill=fill, outline=color)
        draw.line([(x0 + w * 0.30, cy), (cx - w * 0.05, y1 - h * 0.28),
                   (x1 - w * 0.24, y0 + h * 0.34)], fill=(255, 255, 255), width=max(2, int(min(w, h) * 0.08)))
        return
    if kind == "hourglass":
        lw = max(2, int(min(w, h) * 0.055))
        top = [(x0 + w * 0.24, y0 + h * 0.12), (x1 - w * 0.24, y0 + h * 0.12), (cx, cy)]
        bottom = [(cx, cy), (x1 - w * 0.24, y1 - h * 0.12), (x0 + w * 0.24, y1 - h * 0.12)]
        fill = _blend(color, (255, 255, 255), 0.16)
        draw.polygon(top, fill=fill, outline=color)
        draw.polygon(bottom, fill=fill, outline=color)
        draw.line([(x0 + w * 0.18, y0 + h * 0.10), (x1 - w * 0.18, y0 + h * 0.10)], fill=color, width=lw)
        draw.line([(x0 + w * 0.18, y1 - h * 0.10), (x1 - w * 0.18, y1 - h * 0.10)], fill=color, width=lw)
        draw.ellipse([cx - lw * 0.65, cy - lw * 0.65, cx + lw * 0.65, cy + lw * 0.65], fill=color)
        return
    if kind == "document":
        draw.rounded_rectangle([x0 + w * 0.18, y0 + h * 0.06, x1 - w * 0.16, y1 - h * 0.04],
                               radius=min(w, h) * 0.06, outline=color, width=max(2, int(min(w, h) * 0.04)))
        for i in range(3):
            yy = y0 + h * (0.28 + i * 0.17)
            draw.line([(x0 + w * 0.30, yy), (x1 - w * 0.30, yy)], fill=color, width=max(1, int(h * 0.035)))
        draw.rectangle([x0 + w * 0.58, y1 - h * 0.28, x0 + w * 0.66, y1 - h * 0.12], fill=color)
        draw.rectangle([x0 + w * 0.70, y1 - h * 0.38, x0 + w * 0.78, y1 - h * 0.12], fill=color)
        return
    if kind == "check":
        draw.ellipse([x0 + w * 0.12, y0 + h * 0.12, x1 - w * 0.12, y1 - h * 0.12], fill=color, outline=color)
        draw.line([(x0 + w * 0.32, cy), (cx - w * 0.04, y1 - h * 0.30),
                   (x1 - w * 0.26, y0 + h * 0.34)], fill=(255, 255, 255), width=max(2, int(min(w, h) * 0.08)))
        return
    if kind == "other":
        draw.ellipse([x0 + w * 0.10, y0 + h * 0.28, x1 - w * 0.10, y1 - h * 0.28],
                     outline=color, width=max(2, int(min(w, h) * 0.05)))
        draw.ellipse([cx - w * 0.10, cy - h * 0.10, cx + w * 0.10, cy + h * 0.10], fill=color)
        draw.line([(x0 + w * 0.08, y1 - h * 0.10), (x1 - w * 0.08, y0 + h * 0.10)],
                  fill=color, width=max(2, int(min(w, h) * 0.06)))
        return

    r = min(w, h) * 0.2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=color, width=2)
    glyph = (icon.get("glyph") or "◆")[:1]
    size = int(h * 0.6)
    font = _font(size)
    tw = draw.textlength(glyph, font=font)
    draw.text((x0 + (w - tw) / 2, y0 + (h - size * 1.2) / 2),
              glyph, fill=color, font=font)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], weight_a: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * weight_a + b[i] * (1.0 - weight_a)) for i in range(3))


def _center_text(draw: ImageDraw.ImageDraw, text: str, box: list[float],
                 color: tuple[int, int, int], size: int, bold: bool = False) -> None:
    x0, y0, x1, y1 = box
    font = _font(size)
    tw = draw.textlength(text, font=font)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - size * 1.2) / 2),
              text, fill=color, font=font)


def _draw_dotcloud(draw: ImageDraw.ImageDraw, el: dict) -> None:
    x0, y0 = el["bbox"][0], el["bbox"][1]
    if el.get("paths"):
        _draw_local_paths(draw, el)
    wb = el.get("wave_bands")
    if wb and len(wb.get("curves") or []) >= 2:
        curves = wb.get("curves") or []
        fills = wb.get("fills") or []
        for i in range(len(curves) - 1):
            upper = curves[i]
            lower = curves[i + 1]
            if len(upper) < 2 or len(lower) < 2:
                continue
            pts = [(x0 + px, y0 + py) for px, py in upper + lower[::-1]]
            fill = _rgb(fills[i] if i < len(fills) else None) or (220, 235, 243)
            draw.polygon(pts, fill=fill)
        for i, curve in enumerate(curves[1:-1], 1):
            if len(curve) < 2:
                continue
            color = _rgb(fills[min(i, len(fills) - 1)] if fills else None) or (150, 175, 195)
            line = tuple(max(0, c - 35) for c in color)
            pts = [(x0 + px, y0 + py) for px, py in curve]
            draw.line(pts, fill=line, width=1)
    sil = el.get("silhouette")
    if sil and len(sil.get("points", [])) >= 3:
        pts = [(x0 + p[0], y0 + p[1]) for p in sil["points"]]
        draw.polygon(pts, fill=_rgb(sil.get("fill")) or (200, 210, 225))
    for heat in el.get("heat_regions", []) or []:
        try:
            cx = float(heat.get("cx", 0.0))
            cy = float(heat.get("cy", 0.0))
            rx = float(heat.get("rx", 0.0))
            ry = float(heat.get("ry", 0.0))
        except Exception:
            continue
        if rx <= 1 or ry <= 1:
            continue
        color = _rgb(heat.get("color")) or (220, 235, 243)
        # PIL proxy is flat RGB, so approximate translucency by blending with white.
        opacity = max(0.0, min(1.0, float(heat.get("opacity", 45)) / 100.0))
        fill = tuple(int(255 * (1.0 - opacity) + c * opacity) for c in color)
        draw.ellipse([x0 + cx - rx, y0 + cy - ry,
                      x0 + cx + rx, y0 + cy + ry], fill=fill)
    stream_color = _rgb((el.get("style") or {}).get("dark")) or (170, 195, 210)
    for line in el.get("streamlines", []) or []:
        if len(line) < 2:
            continue
        pts = [(x0 + px, y0 + py) for px, py in line]
        draw.line(pts, fill=stream_color, width=1)
    for d in el.get("dots", []):
        c = _rgb(d.get("color")) or (60, 60, 60)
        r = d["r"]
        draw.ellipse([x0 + d["cx"] - r, y0 + d["cy"] - r,
                      x0 + d["cx"] + r, y0 + d["cy"] + r], fill=c)


def _draw_local_paths(draw: ImageDraw.ImageDraw, el: dict) -> None:
    x0, y0 = el["bbox"][0], el["bbox"][1]
    for path in sorted(el.get("paths") or [], key=lambda p: p.get("area", 0)):
        pts = path.get("points") or []
        if len(pts) < 2:
            continue
        xy = [(x0 + float(px), y0 + float(py)) for px, py in pts]
        fill = _rgb(path.get("fill"))
        line = _rgb(path.get("line")) or fill or (90, 90, 90)
        closed = bool(path.get("closed", True))
        if fill and closed and len(xy) >= 3:
            draw.polygon(xy, fill=fill)
        width = max(1, int(round(float(path.get("line_width", 1)))))
        if closed and len(xy) >= 3:
            draw.line(xy + [xy[0]], fill=line, width=width)
        else:
            draw.line(xy, fill=line, width=width)


def _draw_chart(draw: ImageDraw.ImageDraw, el: dict) -> None:
    paths = el.get("paths") or (el.get("chart") or {}).get("paths") \
        or (el.get("ext", {}).get("chart") or {}).get("paths")
    if paths:
        _draw_local_paths(draw, dict(el, paths=paths))
        return

    spec = _normalize_chart_spec(el.get("chart") or el.get("ext", {}).get("chart") or {})
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1:
        return
    px0, py0 = x0 + w * 0.12, y0 + h * 0.10
    px1, py1 = x1 - w * 0.08, y1 - h * 0.12
    draw.rectangle([x0, y0, x1, y1], fill="white")
    draw.line([(px0, py1), (px1, py1)], fill=(35, 35, 35), width=1)
    draw.line([(px0, py1), (px0, py0)], fill=(35, 35, 35), width=1)

    kind = spec.get("kind")
    if kind == "scatter":
        points = spec.get("points") or []
        for p in points:
            if not isinstance(p, dict):
                continue
            cx = px0 + float(p.get("x", 0.0)) * (px1 - px0)
            cy = py1 - float(p.get("y", 0.0)) * (py1 - py0)
            c = _rgb(p.get("color")) or (68, 114, 196)
            r = max(1.5, min(w, h) * 0.018)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        trend = spec.get("trend") or {}
        if trend:
            m = float(trend.get("slope", 0.0))
            b = float(trend.get("intercept", 0.0))
            c = _rgb(trend.get("color")) or (210, 80, 70)
            y_a = py1 - (m * 0.05 + b) * (py1 - py0)
            y_b = py1 - (m * 0.95 + b) * (py1 - py0)
            draw.line([(px0 + 0.05 * (px1 - px0), y_a),
                       (px0 + 0.95 * (px1 - px0), y_b)], fill=c, width=2)
        return

    series = spec.get("series") or []
    if kind == "line":
        for s in series:
            pts = _series_points(s)
            if len(pts) < 2:
                continue
            c = _rgb(s.get("color")) or (68, 114, 196)
            xy = [(px0 + x * (px1 - px0), py1 - y * (py1 - py0))
                  for x, y in pts]
            draw.line(xy, fill=c, width=2)
        return

    if kind == "bar":
        values = []
        for s in series:
            for v in s.get("values") or []:
                try:
                    values.append(float(v))
                except Exception:
                    values.append(0.0)
        vmax = max(values) if values else 1.0
        vmax = vmax or 1.0
        count = max(1, len(values))
        gap = (px1 - px0) / (count * 2 + 1)
        i = 0
        for s in series:
            c = _rgb(s.get("color")) or (68, 114, 196)
            for v in s.get("values") or []:
                try:
                    value = float(v)
                except Exception:
                    value = 0.0
                bx0 = px0 + gap * (1 + i * 2)
                bx1 = bx0 + gap * 1.35
                by = py1 - (value / vmax) * (py1 - py0)
                draw.rectangle([bx0, by, bx1, py1], fill=c, outline=c)
                i += 1


def _normalize_chart_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        return {"kind": "none", "categories": [], "series": [], "points": []}
    kind = str(spec.get("kind") or spec.get("type") or "none").lower()
    if kind not in ("bar", "line", "scatter", "pie"):
        kind = "line" if spec.get("series") else "none"
    return {
        "kind": kind,
        "categories": list(spec.get("categories") or []),
        "series": [s for s in (spec.get("series") or []) if isinstance(s, dict)],
        "points": [p for p in (spec.get("points") or []) if isinstance(p, dict)],
        "trend": spec.get("trend") if isinstance(spec.get("trend"), dict) else {},
    }


def _series_points(series: dict) -> list[tuple[float, float]]:
    raw_points = series.get("points") or []
    if raw_points and isinstance(raw_points[0], dict):
        return [
            (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
            for p in raw_points if isinstance(p, dict)
        ]
    values = series.get("values") or []
    if not values:
        return []
    vals = []
    for v in values:
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    vmax = max(vals) or 1.0
    return [(i / max(1, len(vals) - 1), v / vmax) for i, v in enumerate(vals)]


# -- connectors -------------------------------------------------------------

def _center(e: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = e["bbox"]
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _edge_point(e: dict, toward: tuple[float, float]) -> tuple[float, float]:
    cx, cy = _center(e)
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return (cx, cy)
    x0, y0, x1, y1 = e["bbox"]
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    sx = hw / abs(dx) if dx else float("inf")
    sy = hh / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return (cx + dx * s, cy + dy * s)


def _draw_connector(draw: ImageDraw.ImageDraw, el: dict,
                    shapes: dict, border_w: int) -> None:
    color = _rgb(el.get("color")) or (51, 51, 51)
    src = shapes.get(el.get("from_id") or "")
    dst = shapes.get(el.get("to_id") or "")
    if src and dst:
        start = _edge_point(src, _center(dst))
        end = _edge_point(dst, _center(src))
    elif el.get("points"):
        p = el["points"]
        start, end = (p[0], p[1]), (p[2], p[3])
    elif el.get("start") and el.get("end"):
        start = tuple(el["start"][:2])
        end = tuple(el["end"][:2])
    else:
        return  # dangling connector: nothing to draw

    line_w = max(1, int(round(float(el.get("line_width") or el.get("thickness") or border_w))))
    if el.get("dash"):
        _draw_dashed_line(draw, start, end, color, line_w,
                          dash=max(6.0, line_w * 5.0),
                          gap=max(4.0, line_w * 4.0))
    else:
        draw.line([start, end], fill=color, width=line_w)
    if el["type"] == "arrow":
        _arrowhead(draw, start, end, color, size=max(6, line_w * 4))


def _arrowhead(draw, start, end, color, size: float) -> None:
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (end[0] - size * math.cos(ang - 0.45),
            end[1] - size * math.sin(ang - 0.45))
    right = (end[0] - size * math.cos(ang + 0.45),
             end[1] - size * math.sin(ang + 0.45))
    draw.polygon([end, left, right], fill=color)
