#!/usr/bin/env python3
"""On-box iterative omnimatte removal: load OmniEraser once, peel objects
front-to-back (smaller area first). Each step removes ONE object from the
current scene, writing scene_0..scene_N. The consecutive-frame deltas become
RGBA omnimatte layers (object + its smoke/shadow in the alpha channel).

Run pattern:
  PYTHONPATH=/home/lzy/AAAI_2026/i2e/pylibs CUDA_VISIBLE_DEVICES=<gpu> \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -u omnimatte_remote.py

Reads test_inputs/omni_job.json = {"image":..,"masks":[..ordered..],"max_edge":1024}.
"""
import os, sys, json, time
import numpy as np
import cv2
from PIL import Image

ROOT = "/home/lzy/AAAI_2026/i2e"
sys.path.insert(0, os.path.join(ROOT, "Omnieraser", "ControlNet_version"))


def flux_dir():
    for c in ["models/black-forest-labs/FLUX___1-dev", "models/AI-ModelScope/FLUX___1-dev"]:
        p = os.path.join(ROOT, c)
        if os.path.isdir(os.path.join(p, "transformer")):
            return p
    raise SystemExit("FLUX dir not found")


def dims(W, H, m):
    s = min(1.0, m / max(W, H))
    return max(16, int(W * s) // 16 * 16), max(16, int(H * s) // 16 * 16)


def main():
    import torch
    from controlnet_flux import FluxControlNetModel
    from transformer_flux import FluxTransformer2DModel
    from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline

    spec = json.load(open(os.path.join(ROOT, "test_inputs/omni_job.json")))
    img = Image.open(os.path.join(ROOT, spec["image"])).convert("RGB")
    W, H = img.size
    nw, nh = dims(W, H, spec.get("max_edge", 1024))

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

    outdir = os.path.join(ROOT, "test_inputs/omni_out")
    os.makedirs(outdir, exist_ok=True)
    scene = img.resize((nw, nh))
    scene.resize((W, H)).save(os.path.join(outdir, "scene_0.png"))   # original

    for i, mrel in enumerate(spec["masks"], 1):
        m = cv2.imread(os.path.join(ROOT, mrel), cv2.IMREAD_GRAYSCALE)
        mk = Image.fromarray(
            m.astype(np.uint8)[..., None].repeat(3, -1)).convert("RGB").resize((nw, nh))
        g = torch.Generator("cuda").manual_seed(24)
        t = time.time()
        with torch.inference_mode():
            scene = pipe(
                prompt="There is nothing here.",
                negative_prompt="",
                height=nh, width=nw,
                control_image=scene, control_mask=mk,
                num_inference_steps=28,
                true_guidance_scale=1.0, guidance_scale=3.5,
                generator=g,
                controlnet_conditioning_scale=0.9,
            ).images[0]
        scene.resize((W, H)).save(os.path.join(outdir, f"scene_{i}.png"))
        print(f"PEELED {i}/{len(spec['masks'])} {mrel} ({time.time()-t:.0f}s)", flush=True)

    print("OMNI_DONE", flush=True)


if __name__ == "__main__":
    main()
