"""Component IR — promote strategy-plan regions into stable, first-class
components (P2 of the Decompiler plan).

A ``Component`` groups the IR elements of one semantic region (a card, a chart,
a surface, ...) so the pipeline can reason and audit at region granularity
instead of re-inferring regions every round. This module derives components
from a *completed* run's ``ir_final.json`` + ``strategy_plan*.json`` + source
image, and writes ``components.json`` plus per-component crops and sub-IR.

It is offline and deterministic. ``local_visual_delta`` (which needs a per-
component render) is left as an explicit Target hook (``None``), to be filled
once the component-level render/diff loop exists.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from . import fallback as _fallback

# Component lifecycle, in progress order.
COMPONENT_STATES = ["planned", "generated", "rendered", "audited", "accepted", "fallback"]

SCHEMA_VERSION = "components-v1"


def _area(bbox: Any) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x0, y0, x1, y1 = bbox[:4]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _elements_by_id(ir: dict) -> dict:
    return {e.get("id"): e for e in (ir.get("elements") or []) if e.get("id")}


def _component_metrics(elems: list, comp_bbox: Any) -> dict:
    n = len(elems)
    fb = [e for e in elems if _fallback.is_fallback(e)]
    native_ratio = round((n - len(fb)) / n, 4) if n else 0.0
    comp_area = _area(comp_bbox)
    fb_area = sum(_area(e.get("bbox")) for e in fb)
    fb_ratio = min(1.0, round(fb_area / comp_area, 4)) if comp_area else 0.0
    return {
        "element_count": n,
        "native_element_ratio": native_ratio,
        "fallback_count": len(fb),
        "fallback_area_ratio": fb_ratio,
        "editability_score": round(max(0.0, 1.0 - fb_ratio), 4),
        "local_visual_delta": None,  # Target: needs per-component render/diff
    }


def _component_status(elems: list, ir_status: Optional[str], comp_defects: list) -> str:
    if not elems:
        return "planned"
    if all(_fallback.is_fallback(e) for e in elems):
        return "fallback"
    if comp_defects:
        return "audited"
    if ir_status == "accepted":
        return "accepted"
    return "rendered"


def build_components(ir: dict, strategy_plan: dict) -> list:
    """Derive Component objects from strategy-plan regions + a (final) IR."""
    by_id = _elements_by_id(ir)
    canvas = ir.get("canvas") or {}
    ir_status = ir.get("status")
    defects_by_el: dict = {}
    for d in (ir.get("defects") or []):
        if d.get("status") == "skipped":
            continue
        eid = d.get("element_id")
        if eid:
            defects_by_el.setdefault(eid, []).append(d)

    comps = []
    for i, region in enumerate(strategy_plan.get("regions") or []):
        eids = region.get("element_ids") or []
        elems = [by_id[e] for e in eids if e in by_id]
        comp_defects = [d for e in eids for d in defects_by_el.get(e, [])]
        region_id = region.get("id") or f"{region.get('kind', 'region')}_{i:02d}"
        comp_id = "comp_" + "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in str(region_id))
        comps.append({
            "id": comp_id,
            "kind": region.get("kind"),
            "bbox": region.get("bbox"),
            "canvas": {"width_px": canvas.get("width_px"), "height_px": canvas.get("height_px")},
            "element_ids": eids,
            "status": _component_status(elems, ir_status, comp_defects),
            "metrics": _component_metrics(elems, region.get("bbox")),
            "defect_count": len(comp_defects),
            "provenance": {
                "region_id": region.get("id"),
                "preferred_agent": region.get("preferred_agent"),
                "primary_method": region.get("primary_method"),
                "reason": region.get("reason"),
            },
        })
    return comps


def _crop(source_img, bbox):
    if not bbox or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in bbox[:4])
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return None
    return source_img.crop((x0, y0, x1, y1))


def write_component_artifacts(components: list, ir: dict, source_image_path,
                              out_dir) -> dict:
    """Write per-component crop + sub-IR and a top-level ``components.json``."""
    out_dir = Path(out_dir)
    comp_root = out_dir / "components"
    comp_root.mkdir(parents=True, exist_ok=True)
    by_id = _elements_by_id(ir)

    source_img = None
    sp = Path(source_image_path) if source_image_path else None
    if sp and sp.exists():
        from PIL import Image
        source_img = Image.open(sp).convert("RGB")

    for comp in components:
        cdir = comp_root / comp["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        sub_ir = {
            "canvas": ir.get("canvas"),
            "elements": [by_id[e] for e in comp["element_ids"] if e in by_id],
        }
        (cdir / "component_ir.json").write_text(
            json.dumps(sub_ir, indent=2, ensure_ascii=False, default=str))
        crop_path = None
        if source_img is not None:
            crop = _crop(source_img, comp.get("bbox"))
            if crop is not None:
                crop_path = cdir / "component_crop.png"
                crop.save(crop_path)
        comp["artifacts"] = {
            "component_ir": str(cdir / "component_ir.json"),
            "component_crop": str(crop_path) if crop_path else None,
            # Target hooks — produced by the future component render/diff loop:
            "component_preview_svg": None,
            "component_preview_png": None,
            "component_diff_png": None,
            "component_audit": None,
        }

    index = {
        "schema": SCHEMA_VERSION,
        "count": len(components),
        "states": COMPONENT_STATES,
        "status_summary": dict(sorted(Counter(c["status"] for c in components).items())),
        "components": components,
    }
    (out_dir / "components.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, default=str))
    return index


def _load_run(run_dir: Path):
    ir = json.loads((run_dir / "ir_final.json").read_text())
    sp_path = run_dir / "strategy_plan_processed.json"
    if not sp_path.exists():
        sp_path = run_dir / "strategy_plan.json"
    sp = json.loads(sp_path.read_text()) if sp_path.exists() else {"regions": []}
    return ir, sp


def _source_path(ir: dict, run_dir: Path):
    src = (ir.get("source") or {}).get("path") or ""
    if not src:
        return None
    p = Path(src)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (Path.cwd(), Path(__file__).resolve().parents[3], run_dir):
        if (base / p).exists():
            return base / p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Promote strategy regions into Component IR artifacts.")
    ap.add_argument("run_dir", help="v3 output dir with ir_final.json + strategy_plan*.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    ir, sp = _load_run(run_dir)
    comps = build_components(ir, sp)
    index = write_component_artifacts(comps, ir, _source_path(ir, run_dir), run_dir)
    print(f"components: {index['count']} → {run_dir / 'components.json'}")
    for state, n in index["status_summary"].items():
        print(f"  {state}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
