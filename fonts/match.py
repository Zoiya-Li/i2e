"""Sample text color + pick the closest available CJK font/weight by ink density."""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

# Candidate CJK faces (path, index, label). Spans sans weights + a serif.
_CANDIDATE_FILES = [
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", "HiraginoSansGB"),
    ("/System/Library/Fonts/STHeiti Medium.ttc", "STHeiti-Medium"),
    ("/System/Library/Fonts/STHeiti Light.ttc", "STHeiti-Light"),
    ("/System/Library/Fonts/PingFang.ttc", "PingFang"),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", "Songti-serif"),
    ("/Library/Fonts/Arial Unicode.ttf", "ArialUnicode"),
]


def candidate_fonts() -> list[tuple[str, int, str]]:
    out = []
    for path, label in _CANDIDATE_FILES:
        if not os.path.exists(path):
            continue
        for i in range(12):
            try:
                ImageFont.truetype(path, 20, index=i)
                out.append((path, i, f"{label}#{i}"))
            except Exception:
                break
    return out


def _np():
    import numpy as np
    return np


def sample_text_color(image: Image.Image, bbox: dict, default=(245, 245, 240)) -> tuple[int, int, int]:
    """Estimate the glyph (text) color: pixels whose luminance is farthest from the
    region's background median. Works for light-on-dark and dark-on-light."""
    np = _np()
    x, y, w, h = (int(bbox[k]) for k in ("x", "y", "w", "h"))
    crop = np.asarray(image.convert("RGB").crop((x, y, x + max(1, w), y + max(1, h)))).reshape(-1, 3)
    if crop.size == 0:
        return default
    lum = crop.mean(1)
    bg_lum = np.median(lum)
    dist = np.abs(lum - bg_lum)
    sel = crop[dist >= np.percentile(dist, 85)]
    if sel.size == 0:
        return default
    return tuple(int(v) for v in sel.mean(0))


def _orig_ink(image: Image.Image, bbox: dict, thresh: int = 40) -> float:
    np = _np()
    x, y, w, h = (int(bbox[k]) for k in ("x", "y", "w", "h"))
    crop = _np().asarray(image.convert("RGB").crop((x, y, x + max(1, w), y + max(1, h)))).mean(2)
    if crop.size == 0:
        return 0.0
    return float((np.abs(crop - np.median(crop)) > thresh).mean())


def _rendered_ink(path: str, index: int, content: str, w: int, h: int) -> float:
    np = _np()
    lines = content.split("\n") or [""]
    size = max(8, int(h / len(lines) * 0.82))
    font = ImageFont.truetype(path, size, index=index)
    im = Image.new("L", (max(1, w), max(1, h)), 0)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((0, int(i * h / len(lines))), line, font=font, fill=255)
    return float((np.asarray(im) > 40).mean())


def match_font(image: Image.Image, bbox: dict, content: str) -> tuple[str, int] | None:
    """Pick the candidate face whose ink density best matches the original text
    (captures weight: bold faces have higher density)."""
    cands = candidate_fonts()
    if not cands or not content.strip():
        return None
    target = _orig_ink(image, bbox)
    w, h = int(bbox["w"]), int(bbox["h"])
    best, best_d = None, 1e9
    for path, index, _ in cands:
        try:
            d = abs(_rendered_ink(path, index, content, w, h) - target)
        except Exception:
            continue
        if d < best_d:
            best, best_d = (path, index), d
    return best


def match_text_style(image: Image.Image, element: dict) -> dict:
    """Populate a text element with a sampled color and the best-match font
    (path/index stored under text._font / _font_index for the renderer)."""
    if element.get("type") != "text":
        return element
    t = element.setdefault("text", {})
    r, g, b = sample_text_color(image, element["bbox"])
    t["color"] = f"#{r:02X}{g:02X}{b:02X}"
    m = match_font(image, element["bbox"], t.get("content", ""))
    if m:
        t["font_file"], t["font_index"] = m[0], m[1]
        t["font_family"] = os.path.basename(m[0])
    return element
