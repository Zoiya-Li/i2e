"""Standalone test: load decompose.json, run process_all on framework.png,
report per-type content fill-rate, and dump processed.json. Run from repo root:
    python -m work.diagram2ppt.v2._test_handlers
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .vlm import VLMClient
from .handlers import process_all

OUT = Path("work/diagram2ppt/v2_out")
IMG = "framework.png"


def main() -> None:
    entities = json.loads((OUT / "decompose.json").read_text())
    original = Image.open(IMG).convert("RGB")
    vlm = VLMClient()
    process_all(entities, original, vlm, max_workers=6)

    # fill-rate: per type, how many entities got non-empty content
    by_type = defaultdict(lambda: [0, 0])  # [filled, total]
    for e in entities:
        t = e.get("type")
        by_type[t][1] += 1
        if _filled(e):
            by_type[t][0] += 1

    print("\n=== handler fill-rate ===")
    for t in sorted(by_type):
        f, n = by_type[t]
        print(f"  {t:12s} {f:3d}/{n:<3d}  ({100*f/n:5.1f}%)")
    tot_f = sum(f for f, _ in by_type.values())
    tot_n = sum(n for _, n in by_type.values())
    print(f"  {'TOTAL':12s} {tot_f:3d}/{tot_n:<3d}  ({100*tot_f/tot_n:5.1f}%)")
    print(f"  VLM calls: {vlm.calls}")

    # show samples
    print("\n=== samples ===")
    for t in ("text", "formula", "chart", "icon", "shape", "container", "arrow"):
        ex = next((e for e in entities if e.get("type", "").startswith(t)
                   and _filled(e)), None)
        if ex:
            ex2 = {k: ex[k] for k in ex
                   if k not in ("bbox", "id", "z", "content")}
            ex2["bbox"] = ex["bbox"]
            print(f"  [{t}] {json.dumps(ex2, ensure_ascii=False)[:200]}")

    # debug: raw VLM responses for charts that failed to parse categories
    print("\n=== chart parse failures (raw) ===")
    for e in entities:
        if e.get("type") == "chart" and e.get("_raw"):
            print(f"  {e['id']} bbox={e['bbox']} raw={e['_raw']!r}")

    (OUT / "processed.json").write_text(
        json.dumps(entities, indent=2, ensure_ascii=False, default=_jsonable))


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _filled(e) -> bool:
    t = e.get("type")
    if t == "text":
        return bool(e.get("text", "").strip())
    if t == "formula":
        return bool(e.get("text", "").strip())
    if t == "chart":
        return bool(e.get("chart", {}).get("categories"))
    if t == "icon":
        return bool(e.get("icon", {}).get("kind", "other") != "other") \
            or bool(e.get("icon"))
    if t in ("surface", "dotcloud"):
        return bool(e.get("dots"))
    if t == "arrow":
        return bool(e.get("points"))
    if t in ("shape", "container", "rect", "rounded_rect", "oval"):
        return bool(e.get("fill") is not None or e.get("border_color"))
    return e.get("content") is not None


if __name__ == "__main__":
    main()
