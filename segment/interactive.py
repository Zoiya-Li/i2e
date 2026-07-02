"""Interactive, user-driven segmentation for the editor (Node ③).

SAM point/box prompts let the user carve a sub-part out of an existing layer
("split this further") or lasso a brand-new region. A single point returns SAM's
three nested granularities (subpart / part / whole) so the user picks how
thoroughly to cut. When a sub-part is split off its parent, the parent's surface
is LaMa-inpainted where the part was, so moving the new layer reveals a clean
parent (not a ghost).

Heavy (torch) — the SAM predictor and LaMa model load lazily on first use, then
the image is encoded once and reused for every subsequent prompt.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

_LOCK = threading.Lock()
_WEIGHTS = str(Path(__file__).resolve().parents[1] / "mobile_sam.pt")
_PRED = None          # SAMPredictor with the image already encoded
_PRED_IMG = None      # path of the encoded image
_PRED_IM = None       # preprocessed input tensor (needed by prompt_inference)
_LAMA = None
_LEVELS = ("subpart", "part", "whole")


def _predictor(image_path: str):
    """Lazy SAM predictor; encodes `image_path` once and caches its features."""
    global _PRED, _PRED_IMG, _PRED_IM
    if _PRED is not None and _PRED_IMG == image_path:
        return _PRED, _PRED_IM
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
    from ultralytics.models.sam import Predictor as SAMPredictor
    p = SAMPredictor(overrides=dict(model=_WEIGHTS, save=False, verbose=False, mode="predict"))
    p.setup_model(); p.setup_source(image_path)
    im = None
    for batch in p.dataset:
        p.batch = batch
        im = p.preprocess(batch[1])
        p.features = p.get_im_features(im)
        break
    _PRED, _PRED_IMG, _PRED_IM = p, image_path, im
    return p, im


def _scaled_masks(image_path, mode, points, labels, box, multimask):
    """Return (masks[N,H,W] bool, scores[N]) at the original image resolution."""
    import torch
    from ultralytics.utils import ops
    W, H = Image.open(image_path).size
    p, im = _predictor(image_path)
    with _LOCK, torch.no_grad():
        if mode == "box":
            preds = p.prompt_inference(im, bboxes=[[float(v) for v in box]], multimask_output=False)
        else:  # nest as ONE object's point group, so multiple points refine a single mask
            preds = p.prompt_inference(
                im, points=[[[float(x), float(y)] for x, y in points]],
                labels=[[int(v) for v in (labels or [1] * len(points))]],
                multimask_output=multimask)
        masks = ops.scale_masks(preds[0][None].float(), (H, W), padding=False)[0] > p.model.mask_threshold
        m = masks.detach().cpu().numpy()
        s = preds[1].detach().cpu().numpy().ravel()
    return m, s


def _clip(mask, clip_bbox, pad=8):
    if not clip_bbox:
        return mask
    H, W = mask.shape
    x0 = max(0, int(clip_bbox["x"]) - pad); y0 = max(0, int(clip_bbox["y"]) - pad)
    x1 = min(W, int(clip_bbox["x"] + clip_bbox["w"]) + pad); y1 = min(H, int(clip_bbox["y"] + clip_bbox["h"]) + pad)
    out = np.zeros_like(mask); out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def _save_candidate(image_path, mask, out_dir, uid, score):
    area = int(mask.sum())
    if area < 64:
        return None
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    src = Image.open(image_path).convert("RGBA")
    rgba = src.copy(); rgba.putalpha(Image.fromarray((mask * 255).astype("uint8")))
    cut = rgba.crop((x0, y0, x1, y1))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cutp = str(Path(out_dir) / f"{uid}.png"); maskp = str(Path(out_dir) / f"{uid}_mask.png")
    cut.save(cutp); Image.fromarray((mask * 255).astype("uint8")).save(maskp)
    return {"cutout": cutp, "mask_ref": maskp, "area": area, "score": round(float(score), 3),
            "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}}


def segment(image_path, mode, points=None, labels=None, box=None,
            clip_bbox=None, out_dir="/tmp", uid="seg") -> dict:
    """Returns {ok, candidates:[{level,cutout,mask_ref,bbox,area,score}, ...]}.
    Point mode -> up to 3 candidates (subpart/part/whole, small->large). Box -> 1."""
    npts = len(points or [])
    multi = (mode == "point" and npts == 1)  # 3 granularities only for a single point
    masks, scores = _scaled_masks(image_path, mode, points, labels, box, multi)
    order = list(np.argsort([m.sum() for m in masks]))  # small -> large
    cands = []
    for rank, i in enumerate(order):
        m = _clip(masks[i], clip_bbox)
        c = _save_candidate(image_path, m, out_dir, f"{uid}-{rank}", scores[i])
        if c:
            c["level"] = _LEVELS[min(rank, len(_LEVELS) - 1)] if multi else ("region" if mode == "box" else "refined")
            cands.append(c)
    if not cands:
        return {"ok": False, "error": "empty mask — try another point/box"}
    return {"ok": True, "candidates": cands}


def _lama():
    global _LAMA
    if _LAMA is None:
        with _LOCK:
            if _LAMA is None:
                from inpaint.fill import get_inpainter
                _LAMA = get_inpainter("lama")
    return _LAMA


_SD = None
_SD_PROMPT = ("professional product photograph, the complete object, photorealistic, "
              "sharp focus, high detail, consistent lighting")
_SD_NEG = "cropped, cut off, out of frame, border, watermark, text, blurry, deformed, extra objects"


_SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def _sd():
    global _SD
    if _SD is None:
        with _LOCK:
            if _SD is None:
                import torch
                from diffusers import AutoPipelineForInpainting
                # fp16 weights (small download), cast to fp32 for stable MPS inference
                p = AutoPipelineForInpainting.from_pretrained(
                    _SD_MODEL, variant="fp16", torch_dtype=torch.float32, safety_checker=None)
                _SD = p.to("mps"); _SD.set_progress_bar_config(disable=True)
    return _SD


def _sd_fill(crop_rgb, mask_l, prompt=None):
    """Generative inpaint of the masked region; returns same-size RGB (np).

    If env var I2E_FLUX_URL is set, send the request to that remote Flux-Fill
    server (much higher quality). Else fall back to local SD-1.5 on MPS."""
    import os
    remote = os.environ.get("I2E_FLUX_URL")
    if remote:
        return _flux_remote_fill(crop_rgb, mask_l, prompt, remote)
    import torch
    w, h = crop_rgb.size
    s = 512.0 / max(w, h)
    nw, nh = max(8, round(w * s) // 8 * 8), max(8, round(h * s) // 8 * 8)
    out = _sd()(prompt=prompt or _SD_PROMPT, negative_prompt=_SD_NEG,
                image=crop_rgb.resize((nw, nh)), mask_image=mask_l.resize((nw, nh)),
                num_inference_steps=30, guidance_scale=7.5, strength=1.0,
                generator=torch.Generator("mps").manual_seed(0)).images[0]
    return np.array(out.resize((w, h)).convert("RGB"))


def _flux_remote_fill(crop_rgb, mask_l, prompt, url, timeout=300):
    """POST (image, mask, prompt) to a remote Flux-Fill server, return np RGB."""
    import io
    import requests
    img_buf = io.BytesIO(); crop_rgb.convert("RGB").save(img_buf, "PNG")
    mask_buf = io.BytesIO(); mask_l.convert("L").save(mask_buf, "PNG")
    sess = requests.Session(); sess.trust_env = False   # ignore env proxies
    r = sess.post(
        url.rstrip("/") + "/fill",
        files={"image": ("image.png", img_buf.getvalue(), "image/png"),
               "mask":  ("mask.png",  mask_buf.getvalue(), "image/png")},
        data={"prompt": prompt or _SD_PROMPT, "negative_prompt": _SD_NEG,
              "steps": "28", "guidance": "30.0", "seed": "0", "max_edge": "1024"},
        timeout=timeout)
    r.raise_for_status()
    out = Image.open(io.BytesIO(r.content)).convert("RGB")
    if out.size != crop_rgb.size:
        out = out.resize(crop_rgb.size)
    return np.array(out)


def _place(cut, bb, H, W):
    """Rasterize a cutout's alpha onto a full-canvas boolean mask at its bbox. Scales the
    cutout to the bbox's w/h (so a resized layer's mask matches its box) and clips correctly
    when the box runs off the top/left of the canvas (negative x/y)."""
    m = np.zeros((H, W), bool)
    a = np.array(Image.open(cut).convert("RGBA"))[:, :, 3]
    bw = int(bb.get("w", a.shape[1])); bh = int(bb.get("h", a.shape[0]))
    if bw > 0 and bh > 0 and (a.shape[1], a.shape[0]) != (bw, bh):
        a = np.array(Image.fromarray(a).resize((bw, bh)))
    a = a > 10
    x, y = int(bb["x"]), int(bb["y"]); h, w = a.shape
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return m
    sx0, sy0 = dx0 - x, dy0 - y                  # skip the off-canvas top/left of the cutout
    m[dy0:dy1, dx0:dx1] = a[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return m


def complete_occlusions(image_path, target_cutout, target_bbox, occluders,
                        out_dir, uid, method="lama", up_frac=0.45, prompt=None) -> dict:
    """Amodal completion: paint the parts of `target` hidden behind higher layers.

    The silhouette is the convex hull of the visible object EXTENDED UPWARD into
    the occluder directly above it (e.g. a cup's rim hidden by the scoop on top),
    so structure that was fully occluded is included in the fill region. `method`:
    'sd' paints real structure (generative), 'lama' extends texture, 'cv2' fallback.
    Visible pixels are kept crisp — only the occluded region is generated."""
    import cv2
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    modal = _place(target_cutout, target_bbox, H, W)
    occ = np.zeros((H, W), bool)
    for o in occluders:
        occ |= _place(o["cutout"], o["bbox"], H, W)
    if modal.sum() < 16:
        return {"ok": False, "error": "empty target"}
    occ_d = cv2.dilate(occ.astype("uint8"), np.ones((9, 9), np.uint8), 1) > 0
    ys, xs = np.where(modal)
    xmin, xmax, top_v, bot_v = xs.min(), xs.max(), ys.min(), ys.max()
    h_v = max(1, bot_v - top_v)
    # upward band within the object's (slightly padded) x-span, reaching up by up_frac
    padx = int(0.06 * (xmax - xmin))
    y_up = max(0, top_v - int(up_frac * h_v))
    band = np.zeros((H, W), bool)
    band[y_up:top_v + 1, max(0, xmin - padx):min(W, xmax + padx + 1)] = True
    ext = band & occ_d  # the occluder sitting directly above the object (the hidden rim zone)
    pts = np.column_stack(np.where(modal | ext))[:, ::-1]  # (x,y)
    hull = cv2.convexHull(pts)
    amodal = np.zeros((H, W), "uint8"); cv2.fillConvexPoly(amodal, hull, 1); amodal = amodal > 0
    to_fill = amodal & (~modal) & occ_d
    if to_fill.sum() < 64:
        return {"ok": False, "error": "nothing occluded to complete"}
    full = modal | to_fill
    ays, axs = np.where(amodal)
    pad = 20
    x0, y0 = max(0, axs.min() - pad), max(0, ays.min() - pad)
    x1, y1 = min(W, axs.max() + 1 + pad), min(H, ays.max() + 1 + pad)
    region = img[y0:y1, x0:x1].copy()
    fillmask = to_fill[y0:y1, x0:x1]
    crop = Image.fromarray(region)
    mcrop = Image.fromarray((fillmask * 255).astype("uint8"))
    gen, used = None, method
    if method == "sd":
        try:
            gen = _sd_fill(crop, mcrop, prompt)
        except Exception:
            used = "lama"
    if gen is None and used == "lama":
        try:
            gen = np.array(_lama().fill(crop, mcrop).convert("RGB").resize(crop.size))
        except Exception:
            used = "cv2"
    if gen is None:
        bgr = cv2.inpaint(cv2.cvtColor(region, cv2.COLOR_RGB2BGR),
                          (fillmask * 255).astype("uint8"), 6, cv2.INPAINT_TELEA)
        gen = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); used = "cv2"
    region[fillmask] = gen[fillmask]  # keep visible crisp; only fill the occluded region
    out = img.copy(); out[y0:y1, x0:x1] = region
    rgba = np.dstack([out, (full * 255).astype("uint8")])
    fy, fx = np.where(full)
    bx0, by0, bx1, by1 = fx.min(), fy.min(), fx.max() + 1, fy.max() + 1
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = str(Path(out_dir) / f"{uid}.png")
    Image.fromarray(rgba[by0:by1, bx0:bx1], "RGBA").save(p)
    return {"ok": True, "cutout": p, "added_px": int(to_fill.sum()), "method": used,
            "bbox": {"x": int(bx0), "y": int(by0), "w": int(bx1 - bx0), "h": int(by1 - by0)}}


def bulk_segment(image_path, box, out_dir, uid, min_area=400,
                 max_area_frac=0.5) -> list:
    """Run SAM-everything DENSELY inside `box=(x0,y0,x1,y1)` of the original
    image; return every sub-object as {cutout, mask_ref, bbox, area}. Lets the
    user lasso a region and bulk-extract all the small bits in it (mint leaves,
    ice chunks, droplets) that the global pass missed."""
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
    from ultralytics import SAM
    src = Image.open(image_path).convert("RGB")
    W, H = src.size
    x0, y0, x1, y1 = (max(0, int(box[0])), max(0, int(box[1])),
                      min(W, int(box[2])), min(H, int(box[3])))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    crop_path = str(Path(out_dir) / f"_bulk_{uid}_crop.png")
    src.crop((x0, y0, x1, y1)).save(crop_path)
    sam = SAM(_WEIGHTS)
    with _LOCK:
        r = sam(crop_path, verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return []
    arr = r.masks.data.cpu().numpy() > 0.5
    region_area = (x1 - x0) * (y1 - y0)
    src_rgba = Image.open(image_path).convert("RGBA")
    out = []
    for i in range(arr.shape[0]):
        ml = arr[i]
        a = int(ml.sum())
        if a < min_area or a > region_area * max_area_frac:
            continue
        if ml.shape != (y1 - y0, x1 - x0):
            ml = np.array(Image.fromarray(ml.astype("uint8") * 255).resize((x1 - x0, y1 - y0))) > 127
        full = np.zeros((H, W), bool)
        full[y0:y1, x0:x1] = ml
        ys, xs = np.where(full)
        if len(xs) == 0:
            continue
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgba = src_rgba.copy()
        rgba.putalpha(Image.fromarray((full * 255).astype("uint8")))
        cropped = rgba.crop((bx0, by0, bx1, by1))
        cpath = str(Path(out_dir) / f"{uid}-{len(out)}.png")
        mpath = str(Path(out_dir) / f"{uid}-{len(out)}_mask.png")
        cropped.save(cpath)
        Image.fromarray((full * 255).astype("uint8")).save(mpath)
        out.append({"cutout": cpath, "mask_ref": mpath, "area": a,
                    "bbox": {"x": bx0, "y": by0, "w": bx1 - bx0, "h": by1 - by0}})
    # sort by area desc — bigger / more interesting first
    out.sort(key=lambda c: -c["area"])
    return out


def inpaint_parent(parent_cutout, parent_bbox, mask_ref, out_path, dilate=6, method="lama") -> str:
    """Remove the split-off part from the parent's surface: inpaint the parent
    cutout's RGB where the part's mask falls (within the parent's opaque area).
    Alpha (object shape) is preserved. LaMa by default; cv2 as fallback."""
    import cv2
    parent = Image.open(parent_cutout).convert("RGBA")
    pw, ph = parent.size
    px, py = int(parent_bbox["x"]), int(parent_bbox["y"])
    full = np.array(Image.open(mask_ref).convert("L")) > 127
    H, W = full.shape
    local = full[max(0, py):min(H, py + ph), max(0, px):min(W, px + pw)]
    lm = np.array(Image.fromarray((local * 255).astype("uint8")).resize((pw, ph))) > 127
    arr = np.array(parent)
    alpha = arr[:, :, 3] > 0
    hole = (np.logical_and(lm, alpha)).astype("uint8") * 255
    if dilate > 0:
        hole = cv2.dilate(hole, np.ones((dilate, dilate), np.uint8), 1)
        hole = (np.logical_and(hole > 0, alpha)).astype("uint8") * 255
    rgb = Image.fromarray(arr[:, :, :3], "RGB")
    if method == "lama":
        try:
            fixed = np.array(_lama().fill(rgb, Image.fromarray(hole, "L")).convert("RGB").resize((pw, ph)))
            arr[:, :, :3] = fixed
            Image.fromarray(arr, "RGBA").save(out_path)
            return out_path
        except Exception:
            pass  # fall through to cv2
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, hole, 5, cv2.INPAINT_TELEA)
    arr[:, :, :3] = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
    Image.fromarray(arr, "RGBA").save(out_path)
    return out_path
