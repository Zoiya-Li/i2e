"""Build RGBA omnimatte layers + raw plate for IMG_9493.

Orders objects front->back (smaller area first), runs the remote iterative peel,
pulls scene_0..scene_N, and constructs layers locally via lib.omnimatte_math."""
import os, sys, json, time
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from work import remote
from work.lib.omnimatte_math import build_layer

OUT = ROOT / "work/poster"
ASSETS = OUT / "omni_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
IR_IN = OUT / "poster.ir.json"


def ordered_objects(ir):
    """Raster objects with a cutout, ordered front->back (smaller bbox area first)."""
    objs = []
    for el in ir["elements"]:
        if el["type"] != "raster":
            continue
        cp = (el.get("ext") or {}).get("cutout")
        if not cp or not os.path.exists(cp):
            continue
        b = el["bbox"]
        objs.append((b["w"] * b["h"], el))
    objs.sort(key=lambda t: t[0])           # smaller area = nearer = peel first
    return [el for _, el in objs]


def full_mask(el, W, H):
    cp = el["ext"]["cutout"]
    a = (np.array(Image.open(cp).convert("RGBA").getchannel("A")) > 10).astype(np.uint8) * 255
    m = np.zeros((H, W), np.uint8)
    x, y = int(el["bbox"]["x"]), int(el["bbox"]["y"])
    ah, aw = a.shape
    x1, y1 = min(W, x + aw), min(H, y + ah)
    xx, yy = max(0, x), max(0, y)
    m[yy:y1, xx:x1] = a[:y1 - yy, :x1 - xx]
    return m


def main():
    ir = json.load(open(IR_IN))
    src = Image.open(str(ROOT / "IMG_9493.jpg")).convert("RGB")
    W, H = src.size
    objs = ordered_objects(ir)
    print(f"{len(objs)} objects, front->back")

    # 1) upload image + ordered masks; write the job spec
    remote.push(str(ROOT / "IMG_9493.jpg"), f"{remote.REMOTE_ROOT}/test_inputs/IMG_9493.jpg")
    mask_rels = []
    for i, el in enumerate(objs):
        mp = ASSETS / f"omask_{i}.png"
        Image.fromarray(full_mask(el, W, H), "L").save(mp)
        rel = f"test_inputs/omask_{i}.png"
        remote.push(str(mp), f"{remote.REMOTE_ROOT}/{rel}")
        mask_rels.append(rel)
    spec = {"image": "test_inputs/IMG_9493.jpg", "masks": mask_rels, "max_edge": 1024}
    (ASSETS / "omni_job.json").write_text(json.dumps(spec))
    remote.push(str(ASSETS / "omni_job.json"), f"{remote.REMOTE_ROOT}/test_inputs/omni_job.json")
    remote.push(str(ROOT / "omnimatte_remote.py"), f"{remote.REMOTE_ROOT}/omnimatte_remote.py")

    # 2) run remote peel (load-once)
    gpu = remote.freest_gpu()
    print(f"remote peel on GPU {gpu} ...")
    log = remote.run(
        f"cd {remote.REMOTE_ROOT} && PYTHONPATH={remote.REMOTE_ROOT}/pylibs "
        f"CUDA_VISIBLE_DEVICES={gpu} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"python3 -u omnimatte_remote.py 2>&1 | grep -E 'loaded|PEELED|OMNI_DONE|Error'",
        timeout=3600)
    print(log)
    assert "OMNI_DONE" in log, "remote peel did not finish"

    # 3) pull frames scene_0..scene_N
    frames = []
    for i in range(len(objs) + 1):
        lp = ASSETS / f"scene_{i}.png"
        remote.pull(f"{remote.REMOTE_ROOT}/test_inputs/omni_out/scene_{i}.png", str(lp))
        frames.append(np.array(Image.open(lp).convert("RGB")))

    # 4) build RGBA layers from consecutive deltas; last frame = raw plate
    layers = []
    original = frames[0]
    for i, el in enumerate(objs):
        before, after = frames[i], frames[i + 1]
        b = el["bbox"]
        bbox = (int(b["x"]), int(b["y"]), int(b["x"] + b["w"]), int(b["y"] + b["h"]))
        rgba, (x0, y0, x1, y1) = build_layer(original, before, after, bbox)
        p = ASSETS / f"layer_{el['id']}.png"
        Image.fromarray(rgba, "RGBA").save(p)
        layers.append({"id": el["id"], "name": el.get("name", ""), "z": i,
                       "asset": str(p), "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    Image.fromarray(frames[-1]).save(ASSETS / "raw_plate.png")
    (ASSETS / "layers.json").write_text(json.dumps({"layers": layers, "W": W, "H": H}, indent=2))
    print(f"built {len(layers)} layers + raw_plate -> {ASSETS}")


if __name__ == "__main__":
    main()
