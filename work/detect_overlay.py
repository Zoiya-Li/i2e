"""Principled overlay detection — ONE pass, no heuristic patches.

A poster = photographic SCENE + graphic OVERLAY. This finds the complete overlay:
  - the VLM enumerates EVERY element and classifies it (logo/graphic/product/natural) — this
    gives COMPLETENESS (it sees the gold ornaments, badge, icons, ×, that hardcoded queries miss);
  - SAM3, prompted with the VLM's overlay concepts, gives PRECISE masks (the VLM's own boxes are poor);
  - OCR gives the text boxes.
overlay = {logo, graphic} + text ; scene = {product, natural} (stays in the backdrop).

This replaces the old [hardcoded SAM3 queries + OCR + top-hat marks + gold/warm color detection]
patch stack: completeness from the VLM, precision from SAM3 — zero colour tricks.
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from label.vlm import identify_elements
from ocr.detect import get_text_detector

OVERLAY_KINDS = ("logo", "graphic")
SCENE_KINDS = ("product", "natural")
# a SMALL canonical backstop (the VLM already names specifics; too many queries floods SAM3's
# CPU postprocess). Keep total queries lean.
CANON = ["brand logo", "circular badge", "gold decorative ornament"]


def sam3_segment(image_path, queries, conf=0.30, min_area=200, max_area_frac=0.45):
    """Image-parameterized SAM3 concept segmentation -> [{label, score, mask(HxW bool), area}]."""
    import torch
    from ultralytics.models.sam.predict import SAM3SemanticPredictor
    rgb = np.array(Image.open(image_path).convert("RGB")); H, W = rgb.shape[:2]
    overrides = dict(model=str(ROOT / "sam3.pt"), task="segment", mode="predict",
                     imgsz=1024, conf=conf, save=False, verbose=False)
    p = SAM3SemanticPredictor(overrides=overrides)
    p.setup_model(); p.setup_source(image_path)
    im = None
    for batch in p.dataset:
        p.batch = batch; im = p.preprocess(batch[1]); features = p.get_im_features(im); break
    with torch.no_grad():
        preds = p._inference_features(features, text=queries)
    results = p.postprocess(preds, im, [rgb])
    r = results[0]; out = []
    if r.masks is None or len(r.masks.data) == 0:
        return out
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes.data.cpu().numpy() if r.boxes is not None else None
    for i in range(len(masks)):
        m = masks[i] > 0.5; a = int(m.sum())
        if a < min_area or a > max_area_frac * W * H:
            continue
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m.astype("uint8") * 255).resize((W, H))) > 127
        out.append({"label": queries[int(boxes[i][5])], "score": float(boxes[i][4]), "mask": m, "area": a})
    out.sort(key=lambda d: -d["score"])
    return out


def _nms(dets, iou_thr=0.5):
    kept = []
    for d in dets:
        dup = False
        for k in kept:
            inter = int(np.logical_and(d["mask"], k["mask"]).sum())
            if inter / (d["area"] + k["area"] - inter + 1e-6) > iou_thr:
                dup = True; break
        if not dup:
            kept.append(d)
    return kept


def detect_overlay(image_path, cache_dir=None):
    """Return {W,H, graphics:[{id,name,score,bbox,mask}], texts:[{content,bbox}], scene:[...]}."""
    W, H = Image.open(image_path).convert("RGB").size
    els = identify_elements(image_path)
    overlay_names = sorted({e["name"] for e in els if e["kind"] in OVERLAY_KINDS})
    queries = list(dict.fromkeys(overlay_names + CANON))[:10]   # lean: SAM3 CPU cost ~ #queries
    print(f"  VLM: {len(els)} elements, {len(overlay_names)} overlay concepts -> SAM3 with {len(queries)} queries")
    dets = _nms(sam3_segment(image_path, queries))
    texts = get_text_detector("rapid").detect(image_path)
    graphics = []
    for i, d in enumerate(dets):
        ys, xs = np.where(d["mask"])
        if xs.size == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        graphics.append({"id": f"g{i}", "name": d["label"], "score": round(d["score"], 3),
                         "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}, "mask": d["mask"]})
    result = {"W": W, "H": H, "graphics": graphics, "texts": texts,
              "scene": [e for e in els if e["kind"] in SCENE_KINDS]}
    if cache_dir:
        cd = Path(cache_dir); cd.mkdir(parents=True, exist_ok=True)
        for g in graphics:
            Image.fromarray((g["mask"] * 255).astype("uint8")).save(cd / f"mask_{g['id']}.png")
        (cd / "overlay.json").write_text(json.dumps(
            {"W": W, "H": H, "texts": texts,
             "graphics": [{k: g[k] for k in ("id", "name", "score", "bbox")} for g in graphics],
             "scene": result["scene"]}, ensure_ascii=False, indent=2))
    return result


def load_cached(cache_dir):
    """Reload a cached detection (incl. mask PNGs) without re-running VLM/SAM3."""
    cd = Path(cache_dir); meta = json.loads((cd / "overlay.json").read_text())
    for g in meta["graphics"]:
        g["mask"] = np.array(Image.open(cd / f"mask_{g['id']}.png").convert("L")) > 127
    return meta


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "IMG_9493.jpg")
    r = detect_overlay(img, cache_dir=ROOT / "work/poster/overlay")
    print(f"overlay = {len(r['graphics'])} graphics + {len(r['texts'])} texts ; scene = {len(r['scene'])}")
    for g in r["graphics"]:
        b = g["bbox"]; print(f"  GRAPHIC {g['name'][:34]:34s} {b['w']}x{b['h']}@({b['x']},{b['y']})")
