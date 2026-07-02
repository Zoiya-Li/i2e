"""Craft ONE clean hero-region plate via a CRAFTER-style critic loop:
best-of-N steered fill (remote) -> GroundingDINO 'no confabulated object' critic (local)
-> pick -> Poisson-blend into the raw plate."""
import os, sys, json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from work import remote

ASSETS = ROOT / "work/poster/omni_assets"
SEEDS = [1, 7, 24, 42, 101, 202]
# objects whose presence in the hero region means confabulation
BAD = ["cup", "bowl", "fruit", "lime", "lemon", "food", "ice cream", "glass", "bottle", "vessel", "plate"]


def _iou_region(b, region):
    rx0, ry0, rx1, ry1 = region
    ix = max(0, min(b["x"] + b["w"], rx1) - max(b["x"], rx0))
    iy = max(0, min(b["y"] + b["h"], ry1) - max(b["y"], ry0))
    return ix * iy


def is_clean(detections, region, score_thr=0.35):
    """True iff no BAD-class detection overlaps the hero region above threshold."""
    for d in detections:
        if d["score"] < score_thr:
            continue
        if any(k in d["label"].lower() for k in BAD) and _iou_region(d["bbox"], region) > 0:
            return False
    return True


def hero_region(layers_json):
    """Union bbox of the hero objects (names containing 'tub'/'scoop'/'cup')."""
    js = json.load(open(layers_json))
    xs0 = ys0 = 10 ** 9
    xs1 = ys1 = 0
    for L in js["layers"]:
        if any(k in L["name"].lower() for k in ("ice cream tub", "ice cream scoop", "cup")):
            xs0 = min(xs0, L["x"])
            ys0 = min(ys0, L["y"])
            xs1 = max(xs1, L["x"] + L["w"])
            ys1 = max(ys1, L["y"] + L["h"])
    return (xs0, ys0, xs1, ys1)


def run_candidates(region):
    """Upload a hero-region mask + job, run remote best-of-N, pull candidates."""
    src = Image.open(str(ROOT / "IMG_9493.jpg"))
    W, H = src.size
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = region
    m[y0:y1, x0:x1] = 255
    mp = ASSETS / "hero_region_mask.png"
    Image.fromarray(m, "L").save(mp)
    remote.push(str(mp), f"{remote.REMOTE_ROOT}/test_inputs/hero_region_mask.png")
    remote.push(str(ROOT / "IMG_9493.jpg"), f"{remote.REMOTE_ROOT}/test_inputs/IMG_9493.jpg")
    remote.push(str(ROOT / "plate_fill_remote.py"), f"{remote.REMOTE_ROOT}/plate_fill_remote.py")
    job = {"image": "test_inputs/IMG_9493.jpg", "mask": "test_inputs/hero_region_mask.png",
           "seeds": SEEDS, "max_edge": 1024}
    (ASSETS / "plate_job.json").write_text(json.dumps(job))
    remote.push(str(ASSETS / "plate_job.json"), f"{remote.REMOTE_ROOT}/test_inputs/plate_job.json")
    gpu = remote.freest_gpu()
    log = remote.run(
        f"cd {remote.REMOTE_ROOT} && PYTHONPATH={remote.REMOTE_ROOT}/pylibs "
        f"CUDA_VISIBLE_DEVICES={gpu} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"python3 -u plate_fill_remote.py 2>&1 | grep -E 'loaded|CAND|PLATE_DONE|Error'",
        timeout=3600)
    assert "PLATE_DONE" in log, log
    cands = []
    for s in SEEDS:
        lp = ASSETS / f"plate_cand_{s}.png"
        remote.pull(f"{remote.REMOTE_ROOT}/test_inputs/plate_cand/cand_{s}.png", str(lp))
        cands.append(lp)
    return cands


def critic_pick(cands, region):
    """Return the first candidate the GroundingDINO critic deems clean (no BAD object
    in the hero region); fall back to the candidate with the fewest BAD detections."""
    from extractor.grounded import detect
    queries = list(set(BAD))
    scored = []
    for lp in cands:
        dets = detect(str(lp), queries, box_threshold=0.35, text_threshold=0.25)
        bad = [d for d in dets if any(k in d["label"].lower() for k in BAD)
               and _iou_region(d["bbox"], region) > 0 and d["score"] >= 0.35]
        if is_clean(dets, region):
            return lp, 0
        scored.append((len(bad), lp))
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][0]


def poisson_blend(plate_bgr, region):
    """Blend the hero region of `plate` into the raw plate seamlessly."""
    raw = cv2.imread(str(ASSETS / "raw_plate.png"))
    x0, y0, x1, y1 = region
    mask = np.zeros(raw.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 255
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    return cv2.seamlessClone(plate_bgr, raw, mask, center, cv2.NORMAL_CLONE)


def main():
    region = hero_region(ASSETS / "layers.json")
    print("hero region", region)
    cands = run_candidates(region)
    pick, nbad = critic_pick(cands, region)
    print(f"picked {pick.name} (bad_dets={nbad})")
    blended = poisson_blend(cv2.imread(str(pick)), region)
    out = ASSETS / "plate.png"
    cv2.imwrite(str(out), blended)
    print("wrote", out)


if __name__ == "__main__":
    main()
