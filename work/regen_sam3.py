"""regen_sam3.py — single-stage SAM 3 semantic segmentation + LaMa background.

Replaces the entire VLM-list → GroundingDINO → SAM box-prompt → heuristic-dedup
pipeline with one call to SAM 3's Promptable Concept Segmentation: give it
descriptive text queries, get back masks for every matching instance, at SOTA
quality (LVIS zero-shot 47.0 mAP — 22% above previous best).
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("HF_HOME", os.path.abspath("work/hf_cache"))
for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "HF_ENDPOINT"):
    os.environ.pop(k, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMG = str(ROOT / "IMG_9493.jpg")
OUT = ROOT / "work" / "poster"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
W, H = Image.open(IMG).size

# Descriptive queries — better than bare nouns. Each maps to many instances.
QUERIES = [
    "small green glass perfume bottle",
    "green dark ice cream tub",
    "mint chocolate ice cream scoop",
    "fresh green mint leaf",
    "transparent clear ice cube",
    "brand logo",
    "circular limited edition badge",
    "small circular feature icon",
    "gold ornament label",
    "white text label on dark surface",
]


def run_sam3(queries, conf=0.30, min_area=300, max_area_frac=0.30):
    """Single SAM 3 call → list of (label, score, mask_full_canvas)."""
    import torch
    from ultralytics.models.sam.predict import SAM3SemanticPredictor
    overrides = dict(model=str(ROOT / "sam3.pt"), task="segment", mode="predict",
                     imgsz=1024, conf=conf, save=False, verbose=False)
    p = SAM3SemanticPredictor(overrides=overrides)
    p.setup_model(); p.setup_source(IMG)
    im = None
    for batch in p.dataset:
        p.batch = batch; im = p.preprocess(batch[1])
        features = p.get_im_features(im); break
    with torch.no_grad():
        preds = p._inference_features(features, text=queries)
    results = p.postprocess(preds, im, [np.array(Image.open(IMG).convert("RGB"))])
    out = []
    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        return out
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes.data.cpu().numpy() if r.boxes is not None else None
    canvas_area = W * H
    for i in range(len(masks)):
        m = masks[i] > 0.5
        a = int(m.sum())
        if a < min_area or a > max_area_frac * canvas_area:
            continue
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m.astype("uint8") * 255).resize((W, H))) > 127
        cls = int(boxes[i][5]); score = float(boxes[i][4])
        out.append({"label": queries[cls], "score": score, "mask": m, "area": a})
    out.sort(key=lambda d: -d["score"])
    return out


def dedup(dets, iou_thr=0.55):
    """NMS-style dedup: drop lower-score detections that heavily overlap a kept one."""
    kept = []
    for d in dets:
        skip = False
        for k in kept:
            inter = int(np.logical_and(d["mask"], k["mask"]).sum())
            iou = inter / (d["area"] + k["area"] - inter + 1e-6)
            if iou > iou_thr:
                skip = True; break
        if not skip:
            kept.append(d)
    return kept


def main():
    t = time.time()
    print("Running SAM 3 (single call, all queries)…")
    dets = run_sam3(QUERIES, conf=0.30)
    print(f"  raw detections: {len(dets)} ({time.time()-t:.0f}s)")
    dets = dedup(dets, iou_thr=0.55)
    print(f"  after dedup: {len(dets)}")
    for d in dets:
        ys, xs = np.where(d["mask"])
        print(f"    {d['score']:.2f}  {d['label']:42}  {xs.max()-xs.min()}x{ys.max()-ys.min()} at ({xs.min()},{ys.min()})")

    # Subtract higher-z (smaller area, drawn on top) masks from lower-z so cutouts
    # don't share pixels. Sort dets by area DESC so smaller (higher z) come later.
    dets.sort(key=lambda d: -d["area"])
    masks = [d["mask"] for d in dets]
    cleaned = []
    for i, m in enumerate(masks):
        sub = np.zeros_like(m, bool)
        for j in range(i + 1, len(masks)):
            sub |= masks[j]
        if sub.any():
            sub_d = cv2.dilate(sub.astype("uint8"), np.ones((7, 7), np.uint8), 1) > 0
            cleaned.append(m & ~sub_d)
        else:
            cleaned.append(m)

    # Build raw elements for assemble_ir
    raw = [{"type": "background", "name": "background", "confidence": 0.9,
            "bbox": {"x": 0, "y": 0, "w": W, "h": H}}]
    cleaned_per_id = {}
    for i, d in enumerate(dets):
        cm = cleaned[i]
        if cm.sum() < 200:
            continue
        ys, xs = np.where(cm)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        raw.append({"type": "raster", "name": d["label"], "confidence": d["score"],
                    "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                    "raster": {"kind": "product"}})
        cleaned_per_id[len(raw) - 1] = cm  # map raw index to cleaned mask

    # OCR text lines
    from ocr.detect import get_text_detector
    try:
        ocr = get_text_detector("rapid").detect(IMG)
    except Exception:
        ocr = []
    print(f"  OCR: {len(ocr)} text lines")
    for ln in ocr:
        b = ln["bbox"]
        raw.append({"type": "text", "name": "text", "confidence": float(ln.get("confidence", 0.7)),
                    "bbox": b, "text": {"content": ln.get("content", "")}})

    from extractor.assemble import assemble_ir, validate_ir
    ir = assemble_ir(raw, image_path=IMG, generator="jimeng",
                     provider_name="sam3", model_version="sam3-848M",
                     method="sam3-text-prompt")

    # Realize cutouts using cleaned masks
    src = Image.open(IMG).convert("RGBA")
    raw_idx = 0
    for el in ir["elements"]:
        if el["type"] == "raster":
            raw_idx += 1
            # find the next raw index that's raster (raw_idx 0 is bg)
            while raw_idx <= len(raw) and raw[raw_idx]["type"] != "raster":
                raw_idx += 1
            cm = cleaned_per_id.get(raw_idx)
            if cm is None or cm.sum() < 200:
                el.setdefault("ext", {})["dropped"] = True
                continue
            ys, xs = np.where(cm)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            rgba = src.copy(); rgba.putalpha(Image.fromarray((cm * 255).astype("uint8")))
            cut = rgba.crop((x0, y0, x1, y1))
            p = ASSETS / f"{el['id']}.png"; cut.save(p)
            el["raster"]["asset_ref"] = str(p); el["raster"]["mask_ref"] = str(p)
            el.setdefault("ext", {})["cutout"] = str(p)
            el["bbox"] = {"x": float(x0), "y": float(y0), "w": float(x1 - x0), "h": float(y1 - y0)}
            el["nbox"] = {"x": x0 / W, "y": y0 / H, "w": (x1 - x0) / W, "h": (y1 - y0) / H}
        elif el["type"] == "text":
            b = el["bbox"]; x0, y0 = max(0, int(b["x"])), max(0, int(b["y"]))
            x1, y1 = min(W, int(b["x"] + b["w"])), min(H, int(b["y"] + b["h"]))
            crop = src.convert("RGB").crop((x0, y0, x1, y1))
            p = ASSETS / f"{el['id']}_text.png"; crop.save(p)
            ext = el.setdefault("ext", {})
            ext["text_crop"] = str(p)
            ext["orig_content"] = (el.get("text") or {}).get("content", "")

    # BG plate: build mask from ORIGINAL (pre-subtraction) masks of every det,
    # not the cleaned cutouts. Otherwise pixels outside the cleaned region stay
    # un-inpainted in bg, leaving a "ghost" when the layer is removed.
    from inpaint.fill import get_inpainter
    bg_mask = np.zeros((H, W), dtype=np.uint8)
    for d in dets:                                     # original masks, pre-subtract
        bg_mask = np.maximum(bg_mask, (d["mask"] * 255).astype(np.uint8))
    for el in ir["elements"]:
        if el["type"] == "text":
            b = el["bbox"]; x0, y0 = max(0, int(b["x"])), max(0, int(b["y"]))
            x1, y1 = min(W, int(b["x"] + b["w"])), min(H, int(b["y"] + b["h"]))
            bg_mask[y0:y1, x0:x1] = 255
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3), np.uint8), 1)
    out_bg = str(ASSETS / "_bg.png")
    try:
        clean = get_inpainter("lama").fill(Image.open(IMG).convert("RGB"), Image.fromarray(bg_mask, "L"))
    except Exception:
        clean = get_inpainter("opencv").fill(Image.open(IMG).convert("RGB"), Image.fromarray(bg_mask, "L"))
    clean.save(out_bg)
    for el in ir["elements"]:
        if el.get("type") == "background":
            el.setdefault("background", {})["asset_ref"] = out_bg

    # AMODAL COMPLETION via LaMa: for each raster layer, fill the holes punched
    # by higher-z layer subtraction so the extracted object looks COMPLETE
    # (cup rim + label area filled, scoop bottom rounded). LaMa just extends
    # the object's own texture into the holes — no hallucinated objects.
    print("  amodal completion (LaMa) per object...")
    from segment.interactive import complete_occlusions
    completed = 0
    modal_snap = {}
    for el in ir["elements"]:
        if el["type"] == "raster":
            cp = (el.get("ext") or {}).get("cutout")
            if cp and os.path.exists(cp):
                modal_snap[el["id"]] = cp
    for el in ir["elements"]:
        if el["type"] != "raster" or el["id"] not in modal_snap:
            continue
        # occluders = other raster layers' modal cutouts that overlap this one
        eb = el["bbox"]
        occ = []
        for o in ir["elements"]:
            if o is el or o["type"] != "raster" or o["id"] not in modal_snap:
                continue
            ob = o["bbox"]
            ix = max(0, min(eb["x"]+eb["w"], ob["x"]+ob["w"]) - max(eb["x"], ob["x"]))
            iy = max(0, min(eb["y"]+eb["h"], ob["y"]+ob["h"]) - max(eb["y"], ob["y"]))
            if ix * iy > 0:
                occ.append({"cutout": modal_snap[o["id"]], "bbox": ob})
        if not occ:
            continue
        asp = eb["h"] / max(1.0, eb["w"])
        up = 0.35 if asp >= 0.75 else 0.0
        # SD-1.5 Fill, prompted with SAM 3's accurate label (not generic).
        # SAM 3 gives semantic-correct names ("dark green ice cream tub",
        # "mint chocolate ice cream scoop") at 0.92+ confidence, so SD has
        # the right intent — no more cabbage-roll cascades.
        label = el.get("name") or "product object"
        prompt = (f"a complete {label}, photorealistic product photograph, "
                  "sharp focus, high detail, consistent studio lighting, "
                  "matches the visible part exactly")
        try:
            res = complete_occlusions(IMG, modal_snap[el["id"]], eb, occ, str(ASSETS),
                                      f"{el['id']}_amodal", method="sd", up_frac=up, prompt=prompt)
        except Exception as ex:
            print(f"    complete {el['id']} failed: {ex}"); continue
        if res.get("ok"):
            el["raster"]["asset_ref"] = res["cutout"]
            el.setdefault("ext", {})["cutout"] = res["cutout"]
            b = res["bbox"]
            el["bbox"] = {k: float(b[k]) for k in ("x", "y", "w", "h")}
            el["nbox"] = {"x": b["x"]/W, "y": b["y"]/H, "w": b["w"]/W, "h": b["h"]/H}
            completed += 1
    print(f"  amodal-completed {completed} layers")

    validate_ir(ir)
    irp = OUT / "poster.ir.json"
    irp.write_text(json.dumps(ir, ensure_ascii=False, indent=2))
    from collections import Counter
    print(f"  IR -> {irp}", dict(Counter(e["type"] for e in ir["elements"])))


if __name__ == "__main__":
    main()
