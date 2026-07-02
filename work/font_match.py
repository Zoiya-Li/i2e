"""Visual font matching: for a text's actual glyphs, find the closest installed font by
shape IoU. Also doubles as a logo detector — if NO font matches well (low IoU), the element
is a logo / decorative script and should be kept as a graphic asset, not re-typed.

Experiment entry: compares a few IMG_9493 texts (original glyphs vs best-match render).
"""
import os, glob, json, sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

FONT_DIRS = ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
             "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]


def _enumerate_fonts():
    """All loadable (path,index,name) faces, tagged cjk=True if they have real CJK glyphs."""
    files = []
    for d in FONT_DIRS:
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            files += glob.glob(os.path.join(d, ext))
    faces = []
    for f in sorted(set(files)):
        for idx in range(10):
            try:
                ft = ImageFont.truetype(f, 48, index=idx)
            except Exception:
                break
            # CJK if 油 and 一 render as DIFFERENT glyphs (a missing-glyph font draws the
            # SAME .notdef box for both); real glyphs differ in shape and/or size
            try:
                a = np.array(ft.getmask("油")); b = np.array(ft.getmask("一"))
                same = (a.shape == b.shape and np.array_equal(a, b))
                cjk = a.size > 0 and not same
            except Exception:
                cjk = False
            nm = " ".join(p for p in ft.getname() if p)
            faces.append({"path": f, "index": idx, "name": nm, "cjk": cjk})
    return faces


def _norm_mask(img_gray, target_h=64):
    """Binarize to ink mask, tight-crop, resize to fixed height keeping aspect."""
    g = np.asarray(img_gray, np.float32)
    thr = g.mean()
    ink = (g < thr) if (g < thr).mean() < 0.5 else (g > thr)   # minority = ink
    ys, xs = np.where(ink)
    if xs.size == 0: return None
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    nw = max(1, int(target_h * w / h))
    return np.array(Image.fromarray((ink * 255).astype("uint8")).resize((nw, target_h))) > 127


def _render_mask(content, face, target_h=64):
    ft = ImageFont.truetype(face["path"], 80, index=face["index"])
    l, t, r, b = ft.getbbox(content)
    if r - l < 1 or b - t < 1: return None
    im = Image.new("L", (r - l + 4, b - t + 4), 0)
    ImageDraw.Draw(im).text((2 - l, 2 - t), content, font=ft, fill=255)
    return _norm_mask(np.array(im), target_h)


def _iou(a, b):
    """IoU of two binary masks after aligning to a common width."""
    if a is None or b is None: return 0.0
    w = max(a.shape[1], b.shape[1])
    def pad(m):
        out = np.zeros((m.shape[0], w), bool); out[:, :m.shape[1]] = m; return out
    A, B = pad(a), pad(b)
    inter = (A & B).sum(); uni = (A | B).sum()
    return inter / uni if uni else 0.0


_FACES = None
def match(content, crop_rgb, topk=1):
    """Return [(name, score, face)] best matches for the text's glyphs."""
    global _FACES
    if _FACES is None: _FACES = _enumerate_fonts()
    import re
    is_cjk = bool(re.search(r"[一-鿿]", content))
    cands = [f for f in _FACES if (f["cjk"] if is_cjk else not f["cjk"])]
    g = np.array(Image.fromarray(crop_rgb).convert("L"))
    target = _norm_mask(g)
    scored = []
    for f in cands:
        try:
            m = _render_mask(content, f)
            scored.append((f["name"], _iou(target, m), f))
        except Exception:
            continue
    scored.sort(key=lambda t: -t[1])
    return scored[:topk]


def main():
    ir = json.load(open(ROOT / "work/poster/poster.ir.json"))
    src = np.array(Image.open(str(ROOT / "IMG_9493.jpg")).convert("RGB"))
    texts = {(e.get("ext") or {}).get("orig_content", ""): e for e in ir["elements"] if e["type"] == "text"}
    # representative samples: headline, tagline, latin footer, the Haagen-Dazs logo, product name
    picks = ["一口清醒", "经典国货清凉新生", "MENTHOL REFRESHING", "Haagen-Dazs", "风油精薄荷冰淇淋", "FENGYOUJING"]
    rows = []
    for key in picks:
        el = next((texts[k] for k in texts if k.replace(" ", "") == key.replace(" ", "")), None)
        if not el: continue
        b = el["bbox"]; x0, y0 = int(b["x"]), int(b["y"]); x1, y1 = int(b["x"] + b["w"]), int(b["y"] + b["h"])
        crop = src[max(0, y0):y1, max(0, x0):x1]
        res = match(key, crop, topk=2)
        best = res[0] if res else ("none", 0.0, None)
        bname, bscore, bface = best
        second = res[1][0] if len(res) > 1 else "-"
        verdict = "LOGO/decorative (keep as asset)" if bscore < 0.55 else f"font≈ {bname}"
        print(f"  '{key[:16]:16s}'  best={bname:24s} IoU={bscore:.2f}  2nd={second:18s} -> {verdict}")
        rows.append((key, crop, best))
    # comparison grid: original crop | best-match render
    from PIL import ImageFont as IF
    TH = 70; W = 1000; pad = 12
    canvas = Image.new("RGB", (W, (TH + pad) * len(rows) + pad), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    try: lab = IF.truetype("/System/Library/Fonts/STHeiti Light.ttc", 15)
    except Exception: lab = IF.load_default()
    y = pad
    for key, crop, best in rows:
        ch = Image.fromarray(crop); s = TH / ch.height; ch = ch.resize((int(ch.width * s), TH))
        canvas.paste(ch, (pad, y))
        if best[2]:
            m = _render_mask(key, best[2], target_h=TH)
            if m is not None:
                mi = Image.fromarray((~m * 255).astype("uint8"))  # black text on white
                canvas.paste(mi.convert("RGB"), (520, y))
        d.text((pad, y + TH - 16), f"original", fill=(150, 0, 0), font=lab)
        d.text((520, y + TH - 16), f"match: {best[0]} (IoU {best[1]:.2f})", fill=(0, 110, 0), font=lab)
        y += TH + pad
    canvas.save(ROOT / "work/poster/_font_match.png")
    print("comparison -> work/poster/_font_match.png")


if __name__ == "__main__":
    main()
