#!/usr/bin/env python3
"""On-box best-of-N steered fill of ONE region for plate crafting.

Loads FLUX + OmniEraser once, fills test_inputs/plate_job.json's mask region N
times with different seeds, writing candidates to test_inputs/plate_cand/.

Run pattern:
  PYTHONPATH=/home/lzy/AAAI_2026/i2e/pylibs CUDA_VISIBLE_DEVICES=<gpu> \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -u plate_fill_remote.py
"""
import os, sys, json, time
import numpy as np
import cv2
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
        if os.path.isdir(os.path.join(p, "transformer")):
            return p
    raise SystemExit("no flux")


def dims(W, H, m):
    s = min(1.0, m / max(W, H))
    return max(16, int(W * s) // 16 * 16), max(16, int(H * s) // 16 * 16)


def main():
    job = json.load(open(os.path.join(ROOT, "test_inputs/plate_job.json")))
    img = Image.open(os.path.join(ROOT, job["image"])).convert("RGB")
    W, H = img.size
    nw, nh = dims(W, H, job.get("max_edge", 1024))

    cn = FluxControlNetModel.from_pretrained(
        os.path.join(ROOT, "models/alimama-cn-beta"), torch_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(
        flux_dir(), subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        flux_dir(), controlnet=cn, transformer=tr, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(
        os.path.join(ROOT, "models/omnieraser-lora"),
        weight_name="controlnet_flux_pytorch_lora_weights.safetensors")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    print("loaded", flush=True)

    image = img.resize((nw, nh))
    m = cv2.imread(os.path.join(ROOT, job["mask"]), cv2.IMREAD_GRAYSCALE)
    mk = Image.fromarray(
        m.astype(np.uint8)[..., None].repeat(3, -1)).convert("RGB").resize((nw, nh))

    out = os.path.join(ROOT, "test_inputs/plate_cand")
    os.makedirs(out, exist_ok=True)

    for s in job["seeds"]:
        g = torch.Generator("cuda").manual_seed(int(s))
        t = time.time()
        with torch.inference_mode():
            r = pipe(
                prompt=job.get("prompt", "empty dark smoky studio, wet black marble, deep shadow, no objects"),
                negative_prompt=job.get("neg", "cup,bowl,fruit,lime,food,object,plate,glass,vessel,ice cream"),
                height=nh, width=nw,
                control_image=image, control_mask=mk,
                num_inference_steps=28,
                true_guidance_scale=job.get("true_cfg", 3.5),
                guidance_scale=3.5,
                generator=g,
                controlnet_conditioning_scale=0.9,
            ).images[0]
        r.resize((W, H)).save(os.path.join(out, f"cand_{s}.png"))
        print(f"CAND {s} ({time.time()-t:.0f}s)", flush=True)

    print("PLATE_DONE", flush=True)


if __name__ == "__main__":
    main()
