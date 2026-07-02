"""CLI: a flat image -> a schema-valid IR (Node ②).

    python -m extractor.extract <image> -o out.ir.json [--provider mock|anthropic] [--generator jimeng]

`mock` needs no API key and proves the pipeline end-to-end. `anthropic` calls
Claude vision (requires ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import sys

from .assemble import assemble_ir
from .providers import get_provider


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract a flat image into IR v1.")
    ap.add_argument("image", help="path to the flat image (png/jpg/webp)")
    ap.add_argument("-o", "--out", required=True, help="output .ir.json path")
    ap.add_argument("--provider", default="mock", choices=["mock", "openai-compat", "anthropic"])
    ap.add_argument("--ocr", default="none", choices=["none", "auto", "rapid", "paddle"],
                    help="refine text boxes with a PP-OCR detector (precise geometry)")
    ap.add_argument("--assets", default="none", choices=["none", "auto", "rembg", "sam2"],
                    help="realize real cutout assets for raster elements (foreground layers)")
    ap.add_argument("--inpaint", default="none", choices=["none", "auto", "flat", "opencv", "lama"],
                    help="reconstruct a clean background (remove foreground, fill holes)")
    ap.add_argument("--fonts", default="none", choices=["none", "auto"],
                    help="match each text element's color + closest font/weight from the original")
    ap.add_argument("--generator", default="unknown", help="origin of the image, e.g. jimeng/midjourney")
    args = ap.parse_args(argv)

    try:
        provider = get_provider(args.provider)
        raw = provider.extract(args.image)
        doc = assemble_ir(raw, image_path=args.image, generator=args.generator,
                          provider_name=provider.name, model_version=provider.model_version,
                          method=f"{provider.name}:vlm-extract")
        ocr_lines = None
        if args.ocr != "none":
            # fuse AFTER assemble so the +ocr provenance attaches to extraction.method
            from ocr.detect import get_text_detector, refine_text_with_ocr
            ocr_lines = get_text_detector(args.ocr).detect(args.image)
            refine_text_with_ocr(doc["elements"], ocr_lines)
            W, H = doc["canvas"]["width"], doc["canvas"]["height"]
            for el in doc["elements"]:  # snapped boxes -> refresh normalized coords
                b = el["bbox"]
                el["nbox"] = {"x": b["x"] / W, "y": b["y"] / H, "w": b["w"] / W, "h": b["h"] / H}
            print(f"   OCR refined text boxes using {len(ocr_lines)} detected lines")
        if args.assets != "none":
            from segment.cutout import get_segmenter, realize_assets
            seg = get_segmenter(args.assets)
            adir = args.out + ".assets"
            k = realize_assets(doc, args.image, adir, seg)
            print(f"   realized {k} cutout asset(s) into {adir} via {seg.name}")
        if args.inpaint != "none":
            from inpaint.fill import get_inpainter, reconstruct_background
            inp = get_inpainter(args.inpaint)
            bgp = reconstruct_background(doc, args.image, inp, args.out + ".assets/_background.png")
            print(f"   reconstructed clean background -> {bgp} via {inp.name}")
        if args.fonts != "none":
            from PIL import Image
            from fonts.match import match_text_style
            _im = Image.open(args.image).convert("RGB")
            nf = sum(1 for el in doc["elements"] if el["type"] == "text" and match_text_style(_im, el))
            print(f"   font-matched {nf} text element(s) (color + closest font/weight)")
        # real needs_review from evidence (replaces the model's self-confidence)
        from verify.check import verify_ir
        flagged = verify_ir(doc, ocr_lines=ocr_lines)
        print(f"   verify: {flagged}/{len(doc['elements'])} element(s) flagged needs_review")
    except Exception as e:  # surface a clean message; the CLI is a thin shell
        print(f"extract failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    with open(args.out, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    n = len(doc["elements"])
    nr = sum(1 for el in doc["elements"] if el.get("needs_review"))
    print(f"OK  {args.image} -> {args.out}  ({n} elements, {nr} need review, provider={provider.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
