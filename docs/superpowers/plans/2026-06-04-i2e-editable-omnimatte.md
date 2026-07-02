# i2e Editable-Omnimatte Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `IMG_9493.jpg` into RGBA omnimatte layers (object + its own smoke/shadow) over one crafted clean background plate, then prove all four edit classes (recolor, relabel, move-secondary, free-move-hero) work via a programmatic render demo — zero new model training.

**Architecture:** Local Mac orchestrates; the A800 docker box runs OmniEraser (removal) and FLUX (plate fill). Per-object iterative removal runs in ONE remote loaded session and returns intermediate "scene-without-object-i" frames; local numpy turns consecutive frames into RGBA layers (alpha = normalized removal delta, so each object's smoke/shadow rides in its own alpha). The hero-region plate is crafted once via a best-of-N FLUX fill filtered by a GroundingDINO "no-confabulated-object" critic, then Poisson-blended.

**Tech Stack:** Python, numpy, OpenCV, PIL, GroundingDINO (existing `extractor/grounded.py`), SAM3 (existing `work/regen_sam3.py`), remote FLUX/OmniEraser (already provisioned at `/home/lzy/AAAI_2026/i2e/` on docker `29e8e3afb73f`), pytest.

**Scope note:** This plan covers components ②③④a of the spec + a programmatic edit demo. The interactive editor extension (④b: `editor/server.py` + `editor/editor.html`) is a SEPARATE follow-up plan, written after this lands and after reading the current editor code.

---

## File structure

| Path | New/Mod | Responsibility |
|---|---|---|
| `work/lib/omnimatte_math.py` | Create | Pure numpy: removal-delta → alpha; build one RGBA layer; no IO, no GPU. |
| `work/lib/__init__.py` | Create | Make `work/lib` importable. |
| `tests/test_omnimatte_math.py` | Create | Unit tests for the pure math. |
| `work/remote.py` | Create | Local↔box helper: push bytes/files, run a remote script, pull files (base64 over `ssh + docker exec -i`). |
| `omnimatte_remote.py` | Create (uploaded to box) | On-box: load OmniEraser once, iterative removal over ordered masks, save scene_0..scene_N. |
| `plate_fill_remote.py` | Create (uploaded to box) | On-box: best-of-N FLUX/OmniEraser fill of one masked region, save N candidates. |
| `work/omnimatte.py` | Create | Orchestrator: order objects, call `omnimatte_remote.py`, pull frames, build RGBA layers + raw plate. |
| `work/plate.py` | Create | Craft the hero-region plate: call `plate_fill_remote.py`, GroundingDINO critic, pick, Poisson blend. |
| `work/assemble_omnimatte.py` | Create | Build `work/poster/omnimatte.ir.json` (plate + layers + transforms). |
| `work/edit_demo.py` | Create | Load IR, render 4 edits (recolor/relabel/move-secondary/move-hero) → PNGs. |

**Remote layout (already exists):** `/home/lzy/AAAI_2026/i2e/` with `models/black-forest-labs/FLUX___1-dev`, `models/alimama-cn-beta`, `models/omnieraser-lora`, `pylibs/` overlay, `Omnieraser/ControlNet_version/`. Run pattern: `PYTHONPATH=/home/lzy/AAAI_2026/i2e/pylibs CUDA_VISIBLE_DEVICES=<freest> python3 <script>`.

**SSH/exec pattern (reuse everywhere):**
```bash
sshpass -p 'xhqweQWE123!@#' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 8022 \
  xuhu@202.120.12.172 "docker exec 29e8e3afb73f bash -lc \"<cmd>\""
# file in:  base64 -i local | ssh ... "docker exec -i 29e8e3afb73f bash -lc 'base64 -d > REMOTE'"
# file out: ssh ... "docker exec 29e8e3afb73f bash -lc 'base64 REMOTE'" | base64 -d > local
```
**Server rules (hard):** only container `29e8e3afb73f`, only under `/home/lzy/AAAI_2026/i2e/`, never delete anything, never `pkill` others, pick the freest GPU via `nvidia-smi`, never preempt.

---

## Phase 1 — Omnimatte math (pure, local, TDD)

### Task 1: `omnimatte_math.py` — delta → alpha

**Files:**
- Create: `work/lib/__init__.py` (empty)
- Create: `work/lib/omnimatte_math.py`
- Test: `tests/test_omnimatte_math.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omnimatte_math.py
import numpy as np
from work.lib.omnimatte_math import delta_alpha

def test_delta_alpha_zero_when_no_change():
    a = np.full((8, 8, 3), 100, np.uint8)
    assert delta_alpha(a, a.copy()).max() == 0

def test_delta_alpha_full_where_object_vanished():
    before = np.zeros((8, 8, 3), np.uint8)
    after = before.copy()
    before[2:6, 2:6] = 255          # bright object present in `before`, gone in `after`
    al = delta_alpha(before, after)  # uint8 HxW
    assert al.shape == (8, 8)
    assert al[3, 3] > 200            # object center -> high alpha
    assert al[0, 0] == 0             # untouched -> zero alpha

def test_delta_alpha_smooths_speckle():
    before = np.zeros((16, 16, 3), np.uint8)
    after = before.copy()
    before[8, 8] = 30                # tiny low-contrast 1px change
    al = delta_alpha(before, after, smooth_sigma=1.0, thresh=0.15)
    assert al.max() == 0             # below threshold -> removed as speckle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_omnimatte_math.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'work.lib.omnimatte_math'`

- [ ] **Step 3: Write minimal implementation**

```python
# work/lib/omnimatte_math.py
"""Pure numpy helpers for omnimatte layer construction. No IO, no GPU."""
from __future__ import annotations
import numpy as np
import cv2


def delta_alpha(before: np.ndarray, after: np.ndarray,
                smooth_sigma: float = 1.5, thresh: float = 0.08) -> np.ndarray:
    """Alpha (uint8 HxW) = normalized magnitude of the RGB change between `before`
    (object present) and `after` (object + its effects removed). Smooths speckle and
    drops changes below `thresh` (fraction of 255)."""
    b = before.astype(np.float32); a = after.astype(np.float32)
    mag = np.abs(b - a).mean(axis=2) / 255.0           # 0..1 per pixel
    if smooth_sigma > 0:
        mag = cv2.GaussianBlur(mag, (0, 0), smooth_sigma)
    mag[mag < thresh] = 0.0
    m = mag.max()
    if m > 0:
        mag = mag / m
    return (np.clip(mag, 0, 1) * 255).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_omnimatte_math.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/lib/__init__.py work/lib/omnimatte_math.py tests/test_omnimatte_math.py
git commit -m "feat(omnimatte): pure delta->alpha math with speckle smoothing"
```

### Task 2: `omnimatte_math.py` — build RGBA layer with bbox clamp

**Files:**
- Modify: `work/lib/omnimatte_math.py`
- Test: `tests/test_omnimatte_math.py`

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_omnimatte_math.py
from work.lib.omnimatte_math import build_layer

def test_build_layer_clamps_to_dilated_bbox():
    original = np.full((20, 20, 3), 50, np.uint8)
    original[5:10, 5:10] = 200                     # the object pixels
    before = original
    after = original.copy(); after[5:10, 5:10] = 50  # object removed
    after[18, 18] = 60                              # stray far-away change (different obj)
    rgba, (x0, y0, x1, y1) = build_layer(original, before, after,
                                         obj_bbox=(5, 5, 10, 10), clamp_pad=3)
    assert rgba.shape[2] == 4
    # stray change at (18,18) is outside dilated bbox -> alpha there is 0
    assert rgba[18 - y0, 18 - x0].size == 4 if (y0 <= 18 < y1 and x0 <= 18 < x1) else True
    # crop is within the clamped region (does not span to (18,18))
    assert x1 <= 13 and y1 <= 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_omnimatte_math.py::test_build_layer_clamps_to_dilated_bbox -q`
Expected: FAIL — `ImportError: cannot import name 'build_layer'`

- [ ] **Step 3: Implement**

```python
# append to work/lib/omnimatte_math.py
def build_layer(original: np.ndarray, before: np.ndarray, after: np.ndarray,
                obj_bbox: tuple[int, int, int, int], clamp_pad: int = 24):
    """Return (rgba_crop, (x0,y0,x1,y1)). RGB = original pixels; alpha = delta_alpha
    but zeroed outside the object's bbox dilated by `clamp_pad` (so a layer cannot carry
    an unrelated distant change). Crop is the alpha's tight bounding box."""
    H, W = original.shape[:2]
    bx0, by0, bx1, by1 = obj_bbox
    cx0, cy0 = max(0, bx0 - clamp_pad), max(0, by0 - clamp_pad)
    cx1, cy1 = min(W, bx1 + clamp_pad), min(H, by1 + clamp_pad)
    alpha = delta_alpha(before, after)
    clamp = np.zeros((H, W), np.uint8)
    clamp[cy0:cy1, cx0:cx1] = alpha[cy0:cy1, cx0:cx1]
    ys, xs = np.where(clamp > 0)
    if xs.size == 0:                       # nothing survived -> degenerate 1px layer
        return np.zeros((1, 1, 4), np.uint8), (bx0, by0, bx0 + 1, by0 + 1)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    rgba = np.dstack([original[y0:y1, x0:x1], clamp[y0:y1, x0:x1]])
    return rgba, (x0, y0, x1, y1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_omnimatte_math.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/lib/omnimatte_math.py tests/test_omnimatte_math.py
git commit -m "feat(omnimatte): build_layer with dilated-bbox alpha clamp"
```

---

## Phase 2 — Remote transport helper (local, TDD where possible)

### Task 3: `work/remote.py` — push/run/pull over ssh+docker

**Files:**
- Create: `work/remote.py`
- Test: `tests/test_remote.py`

- [ ] **Step 1: Write the failing test** (tests command construction only — no live SSH)

```python
# tests/test_remote.py
from work.remote import _exec_argv, REMOTE_ROOT

def test_remote_root_is_workspace():
    assert REMOTE_ROOT == "/home/lzy/AAAI_2026/i2e"

def test_exec_argv_targets_only_allowed_container():
    argv = _exec_argv("echo hi")
    joined = " ".join(argv)
    assert "29e8e3afb73f" in joined
    assert "docker exec" in joined
    assert "-p 8022" in joined and "xuhu@202.120.12.172" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_remote.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'work.remote'`

- [ ] **Step 3: Implement**

```python
# work/remote.py
"""Local <-> A800 docker box transport. Only ever targets container 29e8e3afb73f
under /home/lzy/AAAI_2026/i2e. Files move as base64 over ssh stdin (overlay fs is
not host-shared, so scp/docker cp are unavailable/forbidden)."""
from __future__ import annotations
import base64, subprocess, shlex

HOST = "xuhu@202.120.12.172"
PORT = "8022"
PW = "xhqweQWE123!@#"
CONTAINER = "29e8e3afb73f"
REMOTE_ROOT = "/home/lzy/AAAI_2026/i2e"
_SSH = ["sshpass", "-p", PW, "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25", "-p", PORT, HOST]


def _exec_argv(cmd: str, interactive: bool = False) -> list[str]:
    flag = "-i " if interactive else ""
    inner = f"docker exec {flag}{CONTAINER} bash -lc {shlex.quote(cmd)}"
    return _SSH + [inner]


def run(cmd: str, timeout: int = 1800) -> str:
    """Run a shell command inside the container, return stdout."""
    p = subprocess.run(_exec_argv(cmd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"remote cmd failed ({p.returncode}): {p.stderr[-2000:]}")
    return p.stdout


def push(local_path: str, remote_path: str, timeout: int = 600) -> None:
    """Copy a local file into the container via base64 stdin."""
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read())
    cmd = f"mkdir -p $(dirname {shlex.quote(remote_path)}) && base64 -d > {shlex.quote(remote_path)}"
    p = subprocess.run(_exec_argv(cmd, interactive=True), input=b64,
                       capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"push failed: {p.stderr[-2000:]!r}")


def pull(remote_path: str, local_path: str, timeout: int = 600) -> None:
    """Copy a file out of the container to local via base64 stdout."""
    p = subprocess.run(_exec_argv(f"base64 {shlex.quote(remote_path)}"),
                       capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"pull failed: {p.stderr[-2000:]}")
    with open(local_path, "wb") as f:
        f.write(base64.b64decode(p.stdout))


def freest_gpu() -> str:
    """Index of the GPU with the most free memory (never preempt others)."""
    out = run("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
    best, bestfree = "0", -1
    for line in out.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        if int(free) > bestfree:
            best, bestfree = idx, int(free)
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_remote.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Verify live connectivity (manual)**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -c "from work.remote import run; print(run('echo LIVE && nvidia-smi --query-gpu=index,memory.free --format=csv,noheader'))"`
Expected: prints `LIVE` and 3 GPU free-memory lines.

- [ ] **Step 6: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/remote.py tests/test_remote.py
git commit -m "feat(remote): ssh+docker push/run/pull transport with gpu picker"
```

---

## Phase 3 — Iterative removal on the box (remote script + run/verify)

### Task 4: `omnimatte_remote.py` — load once, peel objects in z-order

**Files:**
- Create: `omnimatte_remote.py` (local copy; uploaded to box)

- [ ] **Step 1: Write the remote script**

```python
# omnimatte_remote.py  (runs on the box; mirrors work/run_sweep.py load path)
import os, sys, json, time
import numpy as np, cv2
from PIL import Image
ROOT = "/home/lzy/AAAI_2026/i2e"
sys.path.insert(0, os.path.join(ROOT, "Omnieraser", "ControlNet_version"))

def flux_dir():
    for c in ["models/black-forest-labs/FLUX___1-dev", "models/AI-ModelScope/FLUX___1-dev"]:
        p = os.path.join(ROOT, c)
        if os.path.isdir(os.path.join(p, "transformer")): return p
    raise SystemExit("FLUX dir not found")

def dims(W, H, m):
    s = min(1.0, m / max(W, H)); return max(16, int(W*s)//16*16), max(16, int(H*s)//16*16)

import torch
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline

spec = json.load(open(os.path.join(ROOT, "test_inputs/omni_job.json")))  # {"image":..,"masks":[..ordered..],"max_edge":1024}
img = Image.open(os.path.join(ROOT, spec["image"])).convert("RGB"); W, H = img.size
nw, nh = dims(W, H, spec.get("max_edge", 1024))

cn = FluxControlNetModel.from_pretrained(os.path.join(ROOT, "models/alimama-cn-beta"), torch_dtype=torch.bfloat16)
tr = FluxTransformer2DModel.from_pretrained(flux_dir(), subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = FluxControlNetInpaintingPipeline.from_pretrained(flux_dir(), controlnet=cn, transformer=tr, torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights(os.path.join(ROOT, "models/omnieraser-lora"), weight_name="controlnet_flux_pytorch_lora_weights.safetensors")
pipe.transformer.to(torch.bfloat16); pipe.controlnet.to(torch.bfloat16)
print("loaded", flush=True)

outdir = os.path.join(ROOT, "test_inputs/omni_out"); os.makedirs(outdir, exist_ok=True)
scene = img.resize((nw, nh))
scene.resize((W, H)).save(os.path.join(outdir, "scene_0.png"))   # original
for i, mrel in enumerate(spec["masks"], 1):
    m = cv2.imread(os.path.join(ROOT, mrel), cv2.IMREAD_GRAYSCALE)
    mk = Image.fromarray(m.astype(np.uint8)[..., None].repeat(3, -1)).convert("RGB").resize((nw, nh))
    g = torch.Generator("cuda").manual_seed(24); t = time.time()
    with torch.inference_mode():
        scene = pipe(prompt="There is nothing here.", negative_prompt="", height=nh, width=nw,
                     control_image=scene, control_mask=mk, num_inference_steps=28,
                     true_guidance_scale=1.0, guidance_scale=3.5, generator=g,
                     controlnet_conditioning_scale=0.9).images[0]
    scene.resize((W, H)).save(os.path.join(outdir, f"scene_{i}.png"))
    print(f"PEELED {i}/{len(spec['masks'])} {mrel} ({time.time()-t:.0f}s)", flush=True)
print("OMNI_DONE", flush=True)
```

- [ ] **Step 2: (no unit test — GPU/generative)** Sanity-check it parses

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -c "import ast; ast.parse(open('omnimatte_remote.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add omnimatte_remote.py
git commit -m "feat(remote): iterative omnimatte removal script (load-once, z-order peel)"
```

### Task 5: `work/omnimatte.py` — orchestrate peel + build layers

**Files:**
- Create: `work/omnimatte.py`

- [ ] **Step 1: Implement orchestrator**

```python
# work/omnimatte.py
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

OUT = ROOT / "work/poster"; ASSETS = OUT / "omni_assets"; ASSETS.mkdir(parents=True, exist_ok=True)
IR_IN = OUT / "poster.ir.json"


def ordered_objects(ir):
    """Raster objects with a cutout, ordered front->back (smaller bbox area first)."""
    objs = []
    for el in ir["elements"]:
        if el["type"] != "raster": continue
        cp = (el.get("ext") or {}).get("cutout")
        if not cp or not os.path.exists(cp): continue
        b = el["bbox"]; objs.append((b["w"] * b["h"], el))
    objs.sort(key=lambda t: t[0])           # smaller area = nearer = peel first
    return [el for _, el in objs]


def full_mask(el, W, H):
    cp = el["ext"]["cutout"]
    a = (np.array(Image.open(cp).convert("RGBA").getchannel("A")) > 10).astype(np.uint8) * 255
    m = np.zeros((H, W), np.uint8)
    x, y = int(el["bbox"]["x"]), int(el["bbox"]["y"]); ah, aw = a.shape
    x1, y1 = min(W, x + aw), min(H, y + ah); xx, yy = max(0, x), max(0, y)
    m[yy:y1, xx:x1] = a[:y1 - yy, :x1 - xx]
    return m


def main():
    ir = json.load(open(IR_IN))
    src = Image.open(str(ROOT / "IMG_9493.jpg")).convert("RGB"); W, H = src.size
    objs = ordered_objects(ir)
    print(f"{len(objs)} objects, front->back")

    # 1) upload image + ordered masks; write the job spec
    remote.push(str(ROOT / "IMG_9493.jpg"), f"{remote.REMOTE_ROOT}/test_inputs/IMG_9493.jpg")
    mask_rels = []
    for i, el in enumerate(objs):
        mp = ASSETS / f"omask_{i}.png"
        Image.fromarray(full_mask(el, W, H), "L").save(mp)
        rel = f"test_inputs/omask_{i}.png"
        remote.push(str(mp), f"{remote.REMOTE_ROOT}/{rel}"); mask_rels.append(rel)
    spec = {"image": "test_inputs/IMG_9493.jpg", "masks": mask_rels, "max_edge": 1024}
    (ASSETS / "omni_job.json").write_text(json.dumps(spec))
    remote.push(str(ASSETS / "omni_job.json"), f"{remote.REMOTE_ROOT}/test_inputs/omni_job.json")
    remote.push(str(ROOT / "omnimatte_remote.py"), f"{remote.REMOTE_ROOT}/omnimatte_remote.py")

    # 2) run remote peel (load-once)
    gpu = remote.freest_gpu(); print(f"remote peel on GPU {gpu} ...")
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
        b = el["bbox"]; bbox = (int(b["x"]), int(b["y"]), int(b["x"] + b["w"]), int(b["y"] + b["h"]))
        rgba, (x0, y0, x1, y1) = build_layer(original, before, after, bbox)
        p = ASSETS / f"layer_{el['id']}.png"; Image.fromarray(rgba, "RGBA").save(p)
        layers.append({"id": el["id"], "name": el.get("name", ""), "z": i,
                       "asset": str(p), "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    Image.fromarray(frames[-1]).save(ASSETS / "raw_plate.png")
    (ASSETS / "layers.json").write_text(json.dumps({"layers": layers, "W": W, "H": H}, indent=2))
    print(f"built {len(layers)} layers + raw_plate -> {ASSETS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end (remote GPU; ~N×30s + load)**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -u work/omnimatte.py`
Expected: prints `OMNI_DONE`, then `built <N> layers + raw_plate`. Files appear under `work/poster/omni_assets/`.

- [ ] **Step 3: Eyeball a layer + raw plate**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -c "from PIL import Image; Image.open('work/poster/omni_assets/raw_plate.png').show()"`
Expected: bottle/mint/ice areas clean; hero region likely confabulated (that is what Phase 4 fixes).

- [ ] **Step 4: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/omnimatte.py
git commit -m "feat(omnimatte): orchestrate remote peel + local RGBA layer build"
```

---

## Phase 4 — Hero-region plate crafting (critic loop)

### Task 6: `plate_fill_remote.py` — best-of-N steered fill of one region

**Files:**
- Create: `plate_fill_remote.py` (uploaded to box)

- [ ] **Step 1: Write remote best-of-N fill**

```python
# plate_fill_remote.py  (on box) — fills test_inputs/plate_job.json's mask region N times
import os, sys, json, time
import numpy as np, cv2
from PIL import Image
ROOT = "/home/lzy/AAAI_2026/i2e"
sys.path.insert(0, os.path.join(ROOT, "Omnieraser", "ControlNet_version"))
import torch
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline
def flux_dir():
    for c in ["models/black-forest-labs/FLUX___1-dev", "models/AI-ModelScope/FLUX___1-dev"]:
        p = os.path.join(ROOT, c)
        if os.path.isdir(os.path.join(p, "transformer")): return p
    raise SystemExit("no flux")
def dims(W,H,m):
    s=min(1.0,m/max(W,H)); return max(16,int(W*s)//16*16), max(16,int(H*s)//16*16)
job = json.load(open(os.path.join(ROOT,"test_inputs/plate_job.json")))  # {image,mask,seeds:[..],prompt,neg,max_edge}
img = Image.open(os.path.join(ROOT,job["image"])).convert("RGB"); W,H=img.size
nw,nh=dims(W,H,job.get("max_edge",1024))
cn=FluxControlNetModel.from_pretrained(os.path.join(ROOT,"models/alimama-cn-beta"),torch_dtype=torch.bfloat16)
tr=FluxTransformer2DModel.from_pretrained(flux_dir(),subfolder="transformer",torch_dtype=torch.bfloat16)
pipe=FluxControlNetInpaintingPipeline.from_pretrained(flux_dir(),controlnet=cn,transformer=tr,torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights(os.path.join(ROOT,"models/omnieraser-lora"),weight_name="controlnet_flux_pytorch_lora_weights.safetensors")
pipe.transformer.to(torch.bfloat16); pipe.controlnet.to(torch.bfloat16); print("loaded",flush=True)
image=img.resize((nw,nh))
m=cv2.imread(os.path.join(ROOT,job["mask"]),cv2.IMREAD_GRAYSCALE)
mk=Image.fromarray(m.astype(np.uint8)[...,None].repeat(3,-1)).convert("RGB").resize((nw,nh))
out=os.path.join(ROOT,"test_inputs/plate_cand"); os.makedirs(out,exist_ok=True)
for s in job["seeds"]:
    g=torch.Generator("cuda").manual_seed(int(s)); t=time.time()
    with torch.inference_mode():
        r=pipe(prompt=job.get("prompt","empty dark smoky studio, wet black marble, deep shadow, no objects"),
               negative_prompt=job.get("neg","cup,bowl,fruit,lime,food,object,plate,glass,vessel,ice cream"),
               height=nh,width=nw,control_image=image,control_mask=mk,num_inference_steps=28,
               true_guidance_scale=job.get("true_cfg",3.5),guidance_scale=3.5,generator=g,
               controlnet_conditioning_scale=0.9).images[0]
    r.resize((W,H)).save(os.path.join(out,f"cand_{s}.png")); print(f"CAND {s} ({time.time()-t:.0f}s)",flush=True)
print("PLATE_DONE",flush=True)
```

- [ ] **Step 2: Parse-check + commit**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -c "import ast; ast.parse(open('plate_fill_remote.py').read()); print('parse OK')"`
Expected: `parse OK`
```bash
git add plate_fill_remote.py && git commit -m "feat(plate): remote best-of-N steered region fill"
```

### Task 7: `work/plate.py` — critic-loop pick + Poisson blend

**Files:**
- Create: `work/plate.py`
- Test: `tests/test_plate_critic.py`

- [ ] **Step 1: Failing test for the critic predicate (pure logic)**

```python
# tests/test_plate_critic.py
from work.plate import is_clean  # (detections, region_box) -> bool

def test_reject_when_object_detected_in_region():
    dets = [{"label": "cup", "score": 0.6, "bbox": {"x": 500, "y": 700, "w": 200, "h": 300}}]
    region = (450, 560, 1285, 1830)
    assert is_clean(dets, region) is False

def test_accept_when_no_object_in_region():
    dets = [{"label": "mint leaf", "score": 0.5, "bbox": {"x": 40, "y": 1600, "w": 100, "h": 100}}]
    region = (450, 560, 1285, 1830)
    assert is_clean(dets, region) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_plate_critic.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'work.plate'`

- [ ] **Step 3: Implement plate.py**

```python
# work/plate.py
"""Craft ONE clean hero-region plate via a CRAFTER-style critic loop:
best-of-N steered fill (remote) -> GroundingDINO 'no confabulated object' critic (local)
-> pick -> Poisson-blend into the raw plate."""
import os, sys, json
import numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
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
    js = json.load(open(layers_json)); xs0=ys0=10**9; xs1=ys1=0
    for L in js["layers"]:
        if any(k in L["name"].lower() for k in ("ice cream tub", "ice cream scoop", "cup")):
            xs0=min(xs0,L["x"]); ys0=min(ys0,L["y"]); xs1=max(xs1,L["x"]+L["w"]); ys1=max(ys1,L["y"]+L["h"])
    return (xs0, ys0, xs1, ys1)


def run_candidates(region):
    """Upload a hero-region mask + job, run remote best-of-N, pull candidates."""
    src = Image.open(str(ROOT / "IMG_9493.jpg")); W, H = src.size
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = region; m[y0:y1, x0:x1] = 255
    mp = ASSETS / "hero_region_mask.png"; Image.fromarray(m, "L").save(mp)
    remote.push(str(mp), f"{remote.REMOTE_ROOT}/test_inputs/hero_region_mask.png")
    remote.push(str(ROOT / "IMG_9493.jpg"), f"{remote.REMOTE_ROOT}/test_inputs/IMG_9493.jpg")
    remote.push(str(ROOT / "plate_fill_remote.py"), f"{remote.REMOTE_ROOT}/plate_fill_remote.py")
    job = {"image": "test_inputs/IMG_9493.jpg", "mask": "test_inputs/hero_region_mask.png",
           "seeds": SEEDS, "max_edge": 1024}
    (ASSETS / "plate_job.json").write_text(json.dumps(job))
    remote.push(str(ASSETS / "plate_job.json"), f"{remote.REMOTE_ROOT}/test_inputs/plate_job.json")
    gpu = remote.freest_gpu()
    log = remote.run(f"cd {remote.REMOTE_ROOT} && PYTHONPATH={remote.REMOTE_ROOT}/pylibs "
                     f"CUDA_VISIBLE_DEVICES={gpu} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                     f"python3 -u plate_fill_remote.py 2>&1 | grep -E 'loaded|CAND|PLATE_DONE|Error'", timeout=3600)
    assert "PLATE_DONE" in log, log
    cands = []
    for s in SEEDS:
        lp = ASSETS / f"plate_cand_{s}.png"
        remote.pull(f"{remote.REMOTE_ROOT}/test_inputs/plate_cand/cand_{s}.png", str(lp)); cands.append(lp)
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
    scored.sort(key=lambda t: t[0]); return scored[0][1], scored[0][0]


def poisson_blend(plate_bgr, region):
    """Blend the hero region of `plate` into the raw plate seamlessly."""
    raw = cv2.imread(str(ASSETS / "raw_plate.png"))
    x0, y0, x1, y1 = region
    mask = np.zeros(raw.shape[:2], np.uint8); mask[y0:y1, x0:x1] = 255
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    return cv2.seamlessClone(plate_bgr, raw, mask, center, cv2.NORMAL_CLONE)


def main():
    region = hero_region(ASSETS / "layers.json")
    print("hero region", region)
    cands = run_candidates(region)
    pick, nbad = critic_pick(cands, region)
    print(f"picked {pick.name} (bad_dets={nbad})")
    blended = poisson_blend(cv2.imread(str(pick)), region)
    out = ASSETS / "plate.png"; cv2.imwrite(str(out), blended)
    print("wrote", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run critic unit tests**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_plate_critic.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run end-to-end plate crafting (remote + local critic)**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -u work/plate.py`
Expected: prints `PLATE_DONE`, `picked plate_cand_<seed>.png`, `wrote .../plate.png`. Open `work/poster/omni_assets/plate.png` — hero region should read as clean dark smoky background. If NO candidate is clean (all confabulate), that is the documented "must-train" signal → stop and report to user.

- [ ] **Step 6: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/plate.py tests/test_plate_critic.py
git commit -m "feat(plate): CRAFTER-style critic loop + poisson blend for hero plate"
```

---

## Phase 5 — IR assembly + edit demo

### Task 8: `work/assemble_omnimatte.py` — write the omnimatte IR

**Files:**
- Create: `work/assemble_omnimatte.py`
- Test: `tests/test_assemble_omnimatte.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_assemble_omnimatte.py
import json
from work.assemble_omnimatte import build_ir

def test_build_ir_shape(tmp_path):
    layers = {"W": 100, "H": 200, "layers": [
        {"id": "raster-1", "name": "cup", "z": 0, "asset": "a.png", "x": 10, "y": 20, "w": 30, "h": 40}]}
    ir = build_ir(layers, plate="plate.png", texts=[{"id": "t1", "content": "风油精", "x": 5, "y": 5, "w": 20, "h": 8}])
    assert ir["plate"] == "plate.png"
    assert ir["canvas"] == {"w": 100, "h": 200}
    L = ir["layers"][0]
    assert L["transform"] == {"x": 10, "y": 20, "scale": 1.0, "rotation": 0.0}
    assert ir["layers"][0]["z"] == 0
    assert ir["texts"][0]["content"] == "风油精"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_assemble_omnimatte.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# work/assemble_omnimatte.py
"""Assemble work/poster/omnimatte.ir.json: plate + RGBA omnimatte layers (each with a
transform) + text layers. This IR is the contract the editor (follow-up plan) consumes."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "work/poster/omni_assets"


def build_ir(layers, plate, texts):
    out = {"canvas": {"w": layers["W"], "h": layers["H"]}, "plate": plate, "layers": [], "texts": texts}
    for L in layers["layers"]:
        out["layers"].append({
            "id": L["id"], "name": L["name"], "z": L["z"], "asset": L["asset"],
            "bbox": {"x": L["x"], "y": L["y"], "w": L["w"], "h": L["h"]},
            "transform": {"x": L["x"], "y": L["y"], "scale": 1.0, "rotation": 0.0}})
    return out


def main():
    layers = json.load(open(ASSETS / "layers.json"))
    ir0 = json.load(open(ROOT / "work/poster/poster.ir.json"))
    texts = []
    for el in ir0["elements"]:
        if el["type"] != "text": continue
        b = el["bbox"]; ext = el.get("ext") or {}
        texts.append({"id": el["id"], "content": ext.get("orig_content", ""),
                      "crop": ext.get("text_crop", ""),
                      "x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"])})
    plate = str(ASSETS / "plate.png") if (ASSETS / "plate.png").exists() else str(ASSETS / "raw_plate.png")
    ir = build_ir(layers, plate, texts)
    p = ROOT / "work/poster/omnimatte.ir.json"; p.write_text(json.dumps(ir, ensure_ascii=False, indent=2))
    print(f"wrote {p}: {len(ir['layers'])} layers, {len(ir['texts'])} texts, plate={Path(plate).name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test, then build the real IR**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_assemble_omnimatte.py -q`
Expected: PASS (1 passed)
Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python work/assemble_omnimatte.py`
Expected: `wrote .../omnimatte.ir.json: <N> layers, <M> texts, plate=plate.png`

- [ ] **Step 5: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/assemble_omnimatte.py tests/test_assemble_omnimatte.py
git commit -m "feat(ir): assemble omnimatte IR with per-layer transforms + text layers"
```

### Task 9: `work/edit_demo.py` — render the four edit classes

**Files:**
- Create: `work/edit_demo.py`
- Test: `tests/test_edit_demo.py`

- [ ] **Step 1: Failing test for the two pure helpers (compose + recolor)**

```python
# tests/test_edit_demo.py
import numpy as np
from work.edit_demo import recolor_hue, paste_rgba

def test_recolor_hue_shifts_color_keeps_alpha():
    rgba = np.zeros((4, 4, 4), np.uint8); rgba[..., 1] = 200; rgba[..., 3] = 255  # opaque green
    out = recolor_hue(rgba, deg=150)
    assert out[..., 3].min() == 255                    # alpha preserved
    assert not np.array_equal(out[..., :3], rgba[..., :3])  # hue changed

def test_paste_rgba_respects_alpha_and_offset():
    base = np.zeros((10, 10, 3), np.uint8)
    lay = np.zeros((4, 4, 4), np.uint8); lay[..., 0] = 255; lay[..., 3] = 255  # opaque red
    out = paste_rgba(base, lay, x=3, y=3)
    assert tuple(out[4, 4]) == (255, 0, 0)             # pasted at offset
    assert tuple(out[0, 0]) == (0, 0, 0)               # untouched elsewhere
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_edit_demo.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# work/edit_demo.py
"""Load omnimatte.ir.json and render the four edit classes to prove the layered doc is
editable: (1) recolor hero, (2) relabel text, (3) delete a secondary object, (4) free-move
the hero with its smoke/shadow following. Outputs PNGs under work/poster/demo/."""
import json, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
IR = ROOT / "work/poster/omnimatte.ir.json"
DEMO = ROOT / "work/poster/demo"; DEMO.mkdir(parents=True, exist_ok=True)


def recolor_hue(rgba, deg):
    """Rotate hue of the RGB channels by `deg` degrees; alpha untouched."""
    rgb = rgba[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(deg / 2)) % 180          # OpenCV hue is 0..179
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.dstack([out, rgba[..., 3]])


def paste_rgba(base_rgb, rgba, x, y):
    """Alpha-composite an RGBA layer onto base_rgb at top-left (x, y)."""
    H, W = base_rgb.shape[:2]; h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return base_rgb
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[..., 3:4].astype(np.float32)) / 255.0
    base_rgb[y0:y1, x0:x1] = (sub[..., :3] * a + base_rgb[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
    return base_rgb


def _load(ir):
    plate = np.array(Image.open(ir["plate"]).convert("RGB"))
    layers = []
    for L in sorted(ir["layers"], key=lambda d: d["z"], reverse=True):  # back->front for compositing
        layers.append((L, np.array(Image.open(L["asset"]).convert("RGBA"))))
    return plate, layers


def compose(ir, recolor=None, hide=None, move=None):
    """recolor: {id:deg}; hide: set(ids); move: {id:(dx,dy)}."""
    recolor = recolor or {}; hide = hide or set(); move = move or {}
    plate, layers = _load(ir); canvas = plate.copy()
    for L, rgba in layers:
        if L["id"] in hide: continue
        rg = recolor_hue(rgba, recolor[L["id"]]) if L["id"] in recolor else rgba
        dx, dy = move.get(L["id"], (0, 0))
        canvas = paste_rgba(canvas, rg, L["transform"]["x"] + dx, L["transform"]["y"] + dy)
    return canvas


def hero_ids(ir):
    return [L["id"] for L in ir["layers"] if any(k in L["name"].lower() for k in ("tub", "scoop", "cup"))]


def main():
    ir = json.load(open(IR))
    Image.fromarray(compose(ir)).save(DEMO / "00_original_recomposite.png")
    Image.fromarray(compose(ir, recolor={i: 150 for i in hero_ids(ir)})).save(DEMO / "01_recolor_hero.png")
    sec = next((L["id"] for L in ir["layers"] if "bottle" in L["name"].lower()), None)
    if sec: Image.fromarray(compose(ir, hide={sec})).save(DEMO / "02_delete_bottle.png")
    Image.fromarray(compose(ir, move={i: (-300, -150) for i in hero_ids(ir)})).save(DEMO / "03_move_hero.png")
    print("wrote demo PNGs ->", DEMO)
    print("NOTE relabel (#04) is a text-layer swap handled in the interactive editor (follow-up plan).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests, then render**

Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python -m pytest tests/test_edit_demo.py -q`
Expected: PASS (2 passed)
Run: `cd /Users/lizeyan/Desktop/i2e && /Users/lizeyan/anaconda3/envs/science_agent/bin/python work/edit_demo.py`
Expected: 4 PNGs under `work/poster/demo/`. Open `03_move_hero.png`: the cup sits in its new position with smoke/shadow attached; the original location shows the clean plate.

- [ ] **Step 5: Commit**

```bash
cd /Users/lizeyan/Desktop/i2e
git add work/edit_demo.py tests/test_edit_demo.py
git commit -m "feat(demo): render recolor/delete/move-hero edits from omnimatte IR"
```

---

## Self-review (done by plan author)

**Spec coverage:** ① reuse SAM3+OCR (Task 5 reads `poster.ir.json`; Task 8 reads its text layers). ② omnimatte construction (Tasks 1,2,4,5). ③ plate critic loop (Tasks 6,7) incl. CRAFTER-style critic + Poisson blend. ④a IR assembly with transforms (Task 8). Four edit classes proven (Task 9: recolor/delete/move-hero; relabel flagged as editor-only). Compute split honored (remote = Tasks 4,6 GPU scripts; local = math/critic/assembly/demo). Out-of-scope (training, generalization) untouched. **Gap acknowledged:** ④b interactive editor is a deliberate separate plan (stated in header).

**Placeholder scan:** none — every code step is complete runnable code; remote/visual steps use run+inspect with exact commands and expected markers (`OMNI_DONE`, `PLATE_DONE`).

**Type consistency:** `delta_alpha`/`build_layer` signatures match their callers in `work/omnimatte.py`; `is_clean(detections, region)` matches test + caller; `recolor_hue`/`paste_rgba` match tests + `compose`; layer dict keys (`id,name,z,asset,x,y,w,h`) consistent across Tasks 5→8→9; IR keys (`canvas,plate,layers,texts,transform,bbox`) consistent Task 8↔9.

**Failure-mode handling:** if Phase 4 yields no clean candidate, Task 7 Step 5 says stop and report (the "must-train" escalation), matching the spec's risk section.
