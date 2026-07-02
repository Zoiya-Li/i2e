"""CLI: image → diagram2ppt. Default = the decompose→process→integrate
pipeline; --legacy = the old one-shot iterative loop (kept for comparison).

New pipeline (default) — faithful AND editable, zero original pixels:
    python -m work.diagram2ppt.v2.run framework.png -o work/diagram2ppt/v2_out

Legacy one-shot loop (whole-image extraction + residual refine):
    python -m work.diagram2ppt.v2.run framework.png --legacy -o ...
    python -m work.diagram2ppt.v2.run --ir work/diagram2ppt/v2_out/diagram.ir.json

Needs I2E_VLM_* in .env (see vlm.py) unless rebuilding from an existing IR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="diagram image to extract")
    ap.add_argument("--ir", help="(legacy) skip extraction, build from this IR json")
    ap.add_argument("-o", "--out", default="work/diagram2ppt/v2_out",
                    help="output directory")
    ap.add_argument("--legacy", action="store_true",
                    help="use the OLD one-shot iterative loop instead of the "
                         "decompose→process→integrate pipeline")
    # legacy-only knobs (ignored by the new pipeline)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="(legacy) per-element residual demotion threshold")
    ap.add_argument("--ocr", default=None,
                    help="(legacy) 'remote' for RapidOCR on the A800 box, or a "
                         "path to a cached ocr_lines.json")
    ap.add_argument("--raw", action="store_true",
                    help="(legacy) skip the design post-process (extraction only)")
    ap.add_argument("--fidelity", default="all-native",
                    choices=["all-native", "hybrid"])
    args = ap.parse_args()

    out = Path(args.out)

    if not args.legacy:
        # ---- new pipeline: decompose → parallel handlers → all-native SVG ----
        if not args.image:
            ap.error("the new pipeline needs an image")
        from .pipeline import run
        from .vlm import VLMClient
        stats = run(args.image, VLMClient(), str(out))
        print(f"✓ {out/'diagram_v2.svg'}  {json.dumps(stats)}")
        return

    # ---- legacy: one-shot whole-image extraction → PPTX + SVG --------------
    from . import ir as ir_mod
    from .build_pptx import build_pptx
    from PIL import Image

    if args.ir:
        ir = ir_mod.load(args.ir)
    else:
        if not args.image:
            ap.error("give an image, or --ir to rebuild from a saved IR")
        from .loop import run_loop
        from .vlm import VLMClient
        vlm = VLMClient()
        ocr_lines = None
        if args.ocr == "remote":
            from .ocr_snap import fetch_ocr_lines
            ocr_lines = fetch_ocr_lines(args.image)
        elif args.ocr:
            ocr_lines = json.loads(Path(args.ocr).read_text())["lines"]
        ir = run_loop(args.image, vlm, str(out),
                      max_rounds=args.rounds,
                      residual_threshold=args.threshold,
                      ocr_lines=ocr_lines)
        if not args.raw:
            from .postprocess import postprocess
            postprocess(ir, Image.open(args.image).convert("RGB"), vlm,
                        fidelity=args.fidelity)
            ir_mod.save(ir, str(out / "diagram.final.ir.json"))

    pptx_path = out / "diagram_v2.pptx"
    counts = build_pptx(ir, str(pptx_path))

    try:
        from .svg_export import export_svg
        export_svg(ir, Image.open(ir["image"]["path"]).convert("RGB"),
                   str(out / "figure.svg"))
    except Exception as e:
        print(f"  (svg export skipped: {e})")
    final = ir["history"][-1] if ir.get("history") else {}
    print(f"✓ {pptx_path}  {json.dumps(counts)}")
    if final:
        print(f"  metrics: {json.dumps(final)}")


if __name__ == "__main__":
    main()
