"""Offline test for inpaint + layered render (uses real OpenCV — installed, no
model download). Builds a bg+raster IR, reconstructs a clean background, then
layered-renders with the raster MOVED to prove the old spot fills (no ghost).

    python tests/inpaint_smoke.py
"""

from __future__ import annotations

import copy
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw  # noqa: E402

from extractor.assemble import assemble_ir, validate_ir  # noqa: E402
from inpaint.fill import OpenCVInpainter, reconstruct_background  # noqa: E402
from render.export import render  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        img = str(Path(d) / "card.png")
        im = Image.new("RGB", (1080, 1350), (18, 38, 30))
        ImageDraw.Draw(im).rounded_rectangle([660, 585, 960, 1120], radius=45, fill=(232, 240, 232))
        im.save(img)

        raw = [
            {"type": "background", "name": "bg", "bbox": {"x": 0, "y": 0, "w": 1080, "h": 1350},
             "confidence": 0.9, "text": None, "raster": None, "logo": None, "vector": None, "children": None},
            {"type": "raster", "name": "product", "bbox": {"x": 600, "y": 540, "w": 400, "h": 600},
             "confidence": 0.9, "text": None, "raster": {"kind": "product"}, "logo": None,
             "vector": None, "children": None},
        ]
        ir = assemble_ir(raw, image_path=img, generator="synthetic", provider_name="mock",
                         model_version="mock-0.1", method="mock:vlm-extract")

        # reconstruct clean background (remove the product) — real OpenCV inpaint
        bgp = str(Path(d) / "bg.png")
        reconstruct_background(ir, img, OpenCVInpainter(), bgp)
        bg_el = next(e for e in ir["elements"] if e["type"] == "background")
        assert Path(bgp).exists() and bg_el["background"]["asset_ref"] == bgp
        assert bg_el["extraction"]["method"].endswith("+opencv")
        clean = Image.open(bgp).convert("RGB")
        # the product region should now look like background (low variance), not the bright bottle
        patch = list(clean.crop((700, 700, 900, 900)).getdata())
        bright = sum(1 for p in patch if p[0] > 150) / len(patch)
        assert bright < 0.2, f"product not removed from background plate (bright={bright:.0%})"
        print(f"[inpaint] clean background reconstructed; product removed (bright pixels {bright:.0%})")

        # fake a cutout asset so the layered render can composite the product
        raster = next(e for e in ir["elements"] if e["type"] == "raster")
        cut_path = str(Path(d) / "raster-cut.png")
        raster["raster"]["asset_ref"] = cut_path   # inside tempdir — never touch the repo
        cut = Image.new("RGBA", (400, 600), (0, 0, 0, 0))
        ImageDraw.Draw(cut).rounded_rectangle([20, 20, 380, 580], radius=40, fill=(232, 240, 232, 255))
        cut.save(cut_path)

        # MOVE the product to the left, then layered-render
        edited = copy.deepcopy(ir)
        next(e for e in edited["elements"] if e["type"] == "raster")["bbox"]["x"] = 120
        png = render(edited, ir, img)
        out = Image.open(io.BytesIO(png)).convert("RGB")
        assert out.size == (1080, 1350)
        # old location (~x 600-960) should be clean bg now; new location (~x 120-520) should have product
        old_bright = sum(1 for p in out.crop((700, 700, 900, 900)).getdata() if p[0] > 150) / 40000
        new_bright = sum(1 for p in out.crop((200, 700, 400, 900)).getdata() if p[0] > 150) / 40000
        print(f"[layered] product moved: old spot bright={old_bright:.0%} (clean), new spot bright={new_bright:.0%} (product)")
        assert old_bright < 0.2 and new_bright > 0.6, (old_bright, new_bright)
        validate_ir(edited)

    print("\nINPAINT SMOKE OK — clean background + layered render: element moved, old spot filled (no ghost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
