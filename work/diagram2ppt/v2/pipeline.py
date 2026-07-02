"""End-to-end diagram2ppt v2 pipeline: decompose → process → integrate.

The three stages the user specified (task decomposition then continuous
optimization), replacing loop.py's one-shot whole-image extraction:

  1. decompose  (decompose.py)   — detection-only VLM: typed bboxes, no content.
                                   Two passes (structure / text-formula) merged.
  2. process    (handlers.py)    — each entity cropped to FULL resolution and
                                   routed to its best tool in parallel:
                                   text/formula/chart/icon → VLM, shape/arrow/
                                   surface/dotcloud → deterministic CV.
  3. integrate  (svg_export)     — z-ordered, de-duplicated, all-native SVG
                                   export (zero embedded original pixels).

Run:
    python -m work.diagram2ppt.v2.pipeline framework.png -o work/diagram2ppt/v2_out
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from .decompose import decompose, render_boxes
from .handlers import process_all
from .postprocess import (drop_hollow_group_shapes, repair_icons_by_context,
                          separate_overlapping_panels)
from .svg_export import export_svg
from .vlm import VLMClient


def run(image_path: str, vlm, out_dir: str = "work/diagram2ppt/v2_out",
        log=print) -> dict:
    """Run all three stages. Saves decompose.json, decompose_boxes.png,
    processed.json, and the final all-native SVG. Returns the SVG stats."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    original = Image.open(image_path).convert("RGB")
    w, h = original.size

    # Stage 1 — detect (type + bbox only).  Optional stronger model for detection.
    decompose_vlm = vlm
    decompose_model = os.environ.get("I2E_DECOMPOSE_MODEL")
    if decompose_model and decompose_model != vlm.model:
        decompose_vlm = VLMClient(model=decompose_model)
    entities = decompose(image_path, decompose_vlm, log=log)
    (out / "decompose.json").write_text(
        json.dumps(entities, indent=2, ensure_ascii=False))
    render_boxes(image_path, entities, str(out / "decompose_boxes.png"))

    # Stage 2 — per-type content, in parallel.
    process_all(entities, original, vlm, max_workers=6, log=log)
    (out / "processed.json").write_text(
        json.dumps(entities, indent=2, ensure_ascii=False, default=_jsonable))

    # Stage 3 — clean up detection artifacts before integrating.
    ir = {"image": {"width": w, "height": h}, "elements": entities}
    drop_hollow_group_shapes(ir, log=log)
    repair_icons_by_context(ir, log=log)
    separate_overlapping_panels(ir, log=log)

    # Stage 4 — integrate into one all-native SVG.
    stats = export_svg(ir, original, str(out / "diagram_v2.svg"), log=log)
    log(f"[pipeline] done → {out/'diagram_v2.svg'}")
    return stats


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="work/diagram2ppt/v2_out")
    args = ap.parse_args()
    run(args.image, VLMClient(), args.out)


if __name__ == "__main__":
    main()
