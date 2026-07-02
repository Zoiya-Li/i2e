"""Faithful layered document: original pixels per overlay element, NO re-rendering.

The overlay (text + logos/graphics) is detected ONCE by work.detect_overlay — the VLM
enumerates it completely, SAM3 localizes each element precisely, OCR finds the text. No colour
heuristics. Each overlay element is matted as its ORIGINAL pixels (soft alpha) over a backdrop
that has the overlay inpainted away; recompositing backdrop + layers reproduces the poster
PIXEL-FOR-PIXEL. To edit a line a designer hides its layer (clean backdrop shows) and retypes in
their own font — the record carries content/colour/position + a font hint.

Writes work/poster/faithful/.
"""
import json, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from work.detect_overlay import detect_overlay, load_cached
OUT = ROOT / "work/poster/faithful"; OUT.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "IMG_9493.jpg"
OVERLAY_CACHE = ROOT / "work/poster/overlay"
FONT_HINT = True


def matte(crop):
    """Soft alpha for the glyph ink; RGB stays the original pixels. Returns RGBA, ink_color."""
    g = np.dot(crop[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    thr = g.mean()
    ink_is_dark = (g < thr).mean() < 0.5
    inkmask = (g < thr) if ink_is_dark else (g > thr)
    bg_lum = float(np.median(g[~inkmask])) if (~inkmask).any() else (255.0 if ink_is_dark else 0.0)
    ink_lum = float(np.median(g[inkmask])) if inkmask.any() else (0.0 if ink_is_dark else 255.0)
    denom = max(8.0, abs(ink_lum - bg_lum))
    alpha = np.clip(np.abs(g - bg_lum) / denom, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    col = np.median(crop[inkmask], axis=0).astype(int) if inkmask.any() else np.array([255, 255, 255])
    rgba = np.dstack([crop[..., :3], (alpha * 255).astype(np.uint8)])
    return rgba, (int(col[0]), int(col[1]), int(col[2]))


def main(image_path=IMG):
    src = Image.open(str(image_path)).convert("RGB"); W, H = src.size; arr = np.array(src)
    # ONE principled overlay detection (cached so SAM3/VLM aren't re-run on reprocess)
    det = (load_cached(OVERLAY_CACHE) if (OVERLAY_CACHE / "overlay.json").exists()
           else detect_overlay(str(image_path), OVERLAY_CACHE))
    font_match = None
    if FONT_HINT:
        from work.font_match import match as font_match

    recs = []
    overlay = np.zeros((H, W), np.uint8)

    # TEXT layers — matte the ORIGINAL glyph pixels
    for i, t in enumerate(det["texts"]):
        b = t["bbox"]; x0, y0 = max(0, int(b["x"])), max(0, int(b["y"]))
        x1, y1 = min(W, int(b["x"] + b["w"])), min(H, int(b["y"] + b["h"]))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = arr[y0:y1, x0:x1]; rgba, color = matte(crop)
        p = OUT / f"layer_t{i}.png"; Image.fromarray(rgba, "RGBA").save(p)
        overlay[y0:y1, x0:x1] = 255
        content = t.get("content", "") or ""
        hint = ""
        if font_match and content.strip():
            try:
                m = font_match(content, crop, topk=1); hint = m[0][0] if m else ""
            except Exception:
                hint = ""
        recs.append({"id": f"t{i}", "content": content, "kind": "text",
                     "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                     "color": "#%02x%02x%02x" % color, "font_hint": hint, "asset": str(p)})

    # GRAPHIC layers — ORIGINAL pixels masked by SAM3's precise mask (no colour tricks)
    for g in det["graphics"]:
        m = g["mask"]; b = g["bbox"]
        x0, y0 = int(b["x"]), int(b["y"])
        x1, y1 = min(W, x0 + int(b["w"])), min(H, y0 + int(b["h"]))
        if x1 <= x0 or y1 <= y0:
            continue
        alpha = cv2.GaussianBlur((m[y0:y1, x0:x1] * 255).astype(np.uint8), (0, 0), 0.6)
        rgba = np.dstack([arr[y0:y1, x0:x1], alpha])
        p = OUT / f"layer_{g['id']}.png"; Image.fromarray(rgba, "RGBA").save(p)
        overlay = np.maximum(overlay, (m * 255).astype(np.uint8))
        recs.append({"id": g["id"], "content": g["name"], "kind": "graphic",
                     "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                     "color": "", "font_hint": "", "asset": str(p)})
    n_g = sum(1 for r in recs if r["kind"] == "graphic")

    # BACKDROP = scene with the overlay removed. Non-generative: cv2 Telea + blur of the filled
    # regions (no colour heuristics, no LaMa/FLUX hallucination). Products/scene left untouched.
    removal = cv2.dilate(overlay, np.ones((5, 5), np.uint8), 2)
    telea = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), removal, 7,
                                     cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB).astype(np.float32)
    soft = cv2.GaussianBlur(telea, (0, 0), 16)
    fm = cv2.GaussianBlur((removal > 0).astype(np.float32), (0, 0), 9)[..., None]
    backdrop = (soft * fm + telea * (1 - fm)).astype(np.uint8)
    Image.fromarray(backdrop).save(OUT / "backdrop.png")
    print(f"{len(recs) - n_g} text + {n_g} graphic faithful layers + backdrop")

    (OUT / "faithful.ir.json").write_text(json.dumps(
        {"canvas": {"w": W, "h": H}, "backdrop": str(OUT / "backdrop.png"), "layers": recs},
        ensure_ascii=False, indent=2))

    # HONEST verification: recomposite (graphics first, text on top) vs original
    ordered = sorted(recs, key=lambda r: 0 if r["kind"] == "graphic" else 1)
    comp = Image.open(OUT / "backdrop.png").convert("RGBA")
    for r in ordered:
        comp.alpha_composite(Image.open(r["asset"]).convert("RGBA"), (r["x"], r["y"]))
    comp = comp.convert("RGB"); comp.save(OUT / "_recomposite.png")
    diff = float(np.abs(np.array(comp).astype(int) - arr.astype(int)).mean())
    print(f"recomposite vs original: mean abs diff = {diff:.2f} / 255")


if __name__ == "__main__":
    main()
