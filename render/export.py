"""Render an edited IR to a shipped PNG.

Two paths, auto-selected by `render()`:
- export_layered: when a reconstructed clean background exists, composite the
  layers (clean bg → raster cutouts at their bbox → text). Elements can MOVE and
  the old spot stays clean — a true layered re-render.
- export_png: fallback when there's no clean bg yet — cover the original baked
  text with a sampled bg color and redraw. Good for copy/localization only.

    python -m render.export <edited.ir.json> [--original <predicted.ir.json>] [--out out.png]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONTS = ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
          "/System/Library/Fonts/Hiragino Sans GB.ttc", "/Library/Fonts/Arial Unicode.ttf"]


def _load_font(size: int):
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _hex(s, default=(255, 255, 255)):
    if isinstance(s, str) and s.startswith("#") and len(s) >= 7:
        try:
            return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    return default


def _img_median(img: Image.Image):
    t = img.copy(); t.thumbnail((48, 48))
    px = list(t.getdata())
    return tuple(int(statistics.median(p[c] for p in px)) for c in range(3))


def _bg_color(img: Image.Image, bbox: dict, pad: int = 8):
    W, H = img.size
    x0, y0 = int(bbox["x"]), int(bbox["y"])
    x1, y1 = int(bbox["x"] + bbox["w"]), int(bbox["y"] + bbox["h"])
    pts = []
    for rx0, ry0, rx1, ry1 in [(x0, y0 - pad, x1, y0), (x0, y1, x1, y1 + pad),
                               (x0 - pad, y0, x0, y1), (x1, y0, x1 + pad, y1)]:
        rx0, ry0, rx1, ry1 = max(0, rx0), max(0, ry0), min(W, rx1), min(H, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        crop = img.crop((rx0, ry0, rx1, ry1)); crop.thumbnail((32, 32))
        pts.extend(list(crop.getdata()))
    if not pts:
        return _img_median(img)
    return tuple(int(statistics.median(p[c] for p in pts)) for c in range(3))


def _draw_text(draw: ImageDraw.ImageDraw, el: dict) -> None:
    t = el.get("text") or {}
    content = t.get("content", "")
    if not content:
        return
    b = el["bbox"]
    n = max(1, content.count("\n") + 1)               # size per line for multi-line boxes
    size = int(t.get("font_size_px") or max(10, b["h"] / n * 0.82))
    if t.get("font_file"):                            # matched font (path + face index)
        try:
            font = ImageFont.truetype(t["font_file"], size, index=int(t.get("font_index", 0)))
        except Exception:
            font = _load_font(size)
    else:
        font = _load_font(size)
    color = _hex(t.get("color"))
    align = t.get("align", "left")
    lh = (t.get("line_height") or 1.15) * size
    for i, line in enumerate(content.split("\n")):
        tw = draw.textlength(line, font=font)
        x = b["x"] + (b["w"] - tw) / 2 if align == "center" else \
            b["x"] + b["w"] - tw if align == "right" else b["x"]
        draw.text((x, b["y"] + i * lh), line, font=font, fill=color)


def _save(img: Image.Image, out_path: str | None) -> bytes:
    buf = io.BytesIO(); img.save(buf, "PNG")
    data = buf.getvalue()
    if out_path:
        Path(out_path).write_bytes(data)
    return data


def _bg_asset(ir: dict) -> str | None:
    for el in ir["elements"]:
        if el["type"] == "background":
            return (el.get("background") or {}).get("asset_ref")
    return None


def export_png(edited_ir: dict, original_ir: dict, image_path: str, out_path: str | None = None) -> bytes:
    """Fallback: cover original baked text (sampled bg color) + redraw."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for el in original_ir["elements"]:
        if el["type"] == "text":
            b = el["bbox"]
            draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], fill=_bg_color(img, b))
    for el in edited_ir["elements"]:
        if el["type"] == "text":
            _draw_text(draw, el)
    return _save(img, out_path)


def _cutout_path(el: dict) -> str | None:
    """Where this object's RGBA cutout lives — ext.cutout (any object type),
    else the type-specific raster path."""
    return ((el.get("ext") or {}).get("cutout")
            or (el.get("raster") or {}).get("asset_ref")
            or (el.get("logo") or {}).get("raster_ref"))


def export_layered(edited_ir: dict, fallback_image_path: str, out_path: str | None = None) -> bytes:
    """Composite layers over the reconstructed clean background: each object's
    cutout at its (possibly moved) bbox, then free-standing text. Text baked into
    an object (ext.baked) is skipped — it lives in the object's cutout."""
    bg = _bg_asset(edited_ir)
    base = Image.open(bg).convert("RGB") if (bg and Path(bg).exists()) else Image.open(fallback_image_path).convert("RGB")
    canvas = base.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    for el in sorted(edited_ir["elements"], key=lambda e: e.get("z", 0)):
        if el["type"] == "background":
            continue
        b = el["bbox"]
        if el["type"] == "text":
            ext = el.get("ext") or {}
            if ext.get("baked"):
                continue
            # unedited text -> composite the ORIGINAL pixels (pixel-perfect);
            # only re-typeset when the content was actually changed.
            crop = ext.get("text_crop")
            unchanged = ext.get("orig_content") is not None and (el.get("text") or {}).get("content") == ext["orig_content"]
            if crop and unchanged and Path(crop).exists():
                im = Image.open(crop).convert("RGBA").resize((max(1, int(b["w"])), max(1, int(b["h"]))))
                canvas.alpha_composite(im, (int(b["x"]), int(b["y"])))
            else:
                _draw_text(draw, el)
            continue
        cut = _cutout_path(el)
        if cut and Path(cut).exists():
            im = Image.open(cut).convert("RGBA").resize((max(1, int(b["w"])), max(1, int(b["h"]))))
            canvas.alpha_composite(im, (int(b["x"]), int(b["y"])))
    return _save(canvas.convert("RGB"), out_path)


def render(edited_ir: dict, original_ir: dict, image_path: str, out_path: str | None = None) -> bytes:
    """Pick the layered renderer when a clean background exists, else the fallback."""
    bg = _bg_asset(edited_ir)
    if bg and Path(bg).exists():
        return export_layered(edited_ir, image_path, out_path)
    return export_png(edited_ir, original_ir, image_path, out_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render an edited IR to PNG (Node ④).")
    ap.add_argument("ir", help="edited IR json")
    ap.add_argument("--original", help="predicted IR json (for the fallback path); defaults to <ir>")
    ap.add_argument("--out", help="output png (default: <ir>.png)")
    args = ap.parse_args(argv)
    edited = json.loads(Path(args.ir).read_text())
    original = json.loads(Path(args.original).read_text()) if args.original else edited
    out = args.out or str(Path(args.ir).with_suffix(".png"))
    render(edited, original, edited["source"]["original_image_ref"], out)
    print(f"OK rendered -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
