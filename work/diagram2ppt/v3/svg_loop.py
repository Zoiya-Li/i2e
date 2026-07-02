"""SVG canonical loop (P3): IR → SVG → PNG preview → diff.

SVG is the canonical *debug / preview* renderer for v3: its coordinates track
image pixels, it is deterministic, and (unlike PPTX) it needs no Office install.
It does NOT replace PPTX as the delivery format — it is the fast inner loop for
whole-slide / component visual diffing and the first layer of cross-format
lowering.

Rasterization uses ``rsvg-convert`` when available; without it the SVG is still
produced and the raster/diff steps are skipped (reported, never failed), so the
loop degrades gracefully on machines with no rasterizer.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import builder as _builder


def has_rasterizer() -> bool:
    return shutil.which("rsvg-convert") is not None


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _svg_safe(elements: list) -> list:
    """Coerce v3 element types the v2 exporter cannot handle into placeholders.

    The v2 ``export_svg`` predates v3's chart/data shapes; a v3 ``chart`` element
    carries nested series it mis-parses. For a *preview* renderer, showing the
    region as a labelled box is an acceptable, deterministic degrade.
    """
    safe = []
    for el in elements:
        if el.get("type") == "chart":
            safe.append({
                "id": el.get("id"),
                "type": "rounded_rect",
                "bbox": el.get("bbox"),
                "z": el.get("z", 0),
                "fill": el.get("fill") or "#eef1f5",
                "border_color": "#8a94a6",
            })
        else:
            safe.append(el)
    return safe


def _minimal_svg(v3_ir: dict, out_path) -> dict:
    """Fully v3-aware fallback: draw every element's bbox + text label."""
    canvas = v3_ir.get("canvas") or {}
    w = int(canvas.get("width_px") or 0) or 1
    h = int(canvas.get("height_px") or 0) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}">']
    for el in sorted(v3_ir.get("elements") or [], key=lambda e: e.get("z", 0)):
        bb = el.get("bbox")
        if not bb or len(bb) < 4:
            continue
        x0, y0, x1, y1 = bb[:4]
        fill = el.get("fill") or "none"
        parts.append(f'<rect x="{x0}" y="{y0}" width="{max(0, x1 - x0)}" '
                     f'height="{max(0, y1 - y0)}" fill="{fill}" stroke="#888" stroke-width="1"/>')
        txt = el.get("text")
        content = txt.get("content") if isinstance(txt, dict) else (txt if isinstance(txt, str) else None)
        if content:
            parts.append(f'<text x="{x0 + 2}" y="{y0 + 14}" font-size="12" '
                         f'fill="#111">{_xml_escape(str(content)[:40])}</text>')
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts))
    return {"svg": str(out_path), "stats": {"minimal": True,
                                            "elements": len(v3_ir.get("elements") or [])}}


def svg_from_v3_ir(v3_ir: dict, original, out_path) -> dict:
    """Export a v3 IR to SVG, reusing the v2 exporter with a robust fallback."""
    from work.diagram2ppt.v2.svg_export import export_svg

    v2_ir = _builder._to_v2_ir(v3_ir)
    # export_svg mutates elements (dedup); copy so we never corrupt the caller's IR.
    v2_ir["elements"] = _svg_safe(copy.deepcopy(v2_ir["elements"]))
    try:
        stats = export_svg(v2_ir, original, str(out_path), log=lambda *a: None)
        return {"svg": str(out_path), "stats": stats, "renderer": "v2_export_svg"}
    except Exception as exc:  # noqa: BLE001 - preview must never crash the loop
        res = _minimal_svg(v3_ir, out_path)
        res["stats"]["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        res["renderer"] = "minimal"
        return res


def rasterize_svg(svg_path, png_path, width: Optional[int] = None) -> Optional[str]:
    """Rasterize an SVG to PNG via rsvg-convert; None if unavailable/failed."""
    if not has_rasterizer():
        return None
    cmd = ["rsvg-convert", "-o", str(png_path)]
    if width:
        cmd += ["-w", str(int(width))]
    cmd.append(str(svg_path))
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception:
        return None
    return str(png_path) if Path(png_path).exists() else None


def diff_png(render_png, source_png) -> Optional[float]:
    """Normalized mean absolute pixel difference in [0, 1] (0 == identical)."""
    from PIL import Image, ImageChops

    try:
        a = Image.open(render_png).convert("RGB")
        b = Image.open(source_png).convert("RGB")
    except Exception:
        return None
    if a.size != b.size:
        a = a.resize(b.size)
    hist = ImageChops.difference(a, b).convert("L").histogram()
    total = sum(hist)
    if not total:
        return 0.0
    weighted = sum(i * c for i, c in enumerate(hist))
    return round(weighted / (255.0 * total), 4)


def run_svg_loop(run_dir) -> dict:
    """Run IR→SVG→PNG→diff for a completed v3 run dir; write svg_loop.json."""
    from PIL import Image

    from .components import _source_path

    run_dir = Path(run_dir)
    v3_ir = json.loads((run_dir / "ir_final.json").read_text())
    canvas = v3_ir.get("canvas") or {}
    w = int(canvas.get("width_px") or 0) or 1
    h = int(canvas.get("height_px") or 0) or 1

    src = _source_path(v3_ir, run_dir)
    original = (Image.open(src).convert("RGB")
                if src and Path(src).exists() else Image.new("RGB", (w, h), "white"))

    svg_path = run_dir / "diagram_v3.svg"
    result = svg_from_v3_ir(v3_ir, original, svg_path)

    png_path = run_dir / "diagram_v3.svg.png"
    raster = rasterize_svg(svg_path, png_path, width=w)
    visual_delta = diff_png(raster, src) if (raster and src) else None

    payload = {
        "schema": "svg-loop-v1",
        "svg": result["svg"],
        "svg_stats": result["stats"],
        "raster_png": raster,
        "rasterizer_available": has_rasterizer(),
        "visual_delta_vs_source": visual_delta,
    }
    (run_dir / "svg_loop.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SVG canonical loop: v3 IR → SVG → PNG → diff vs source.")
    ap.add_argument("run_dir", help="v3 output dir with ir_final.json")
    args = ap.parse_args()
    p = run_svg_loop(args.run_dir)
    print(f"svg: {p['svg']}  stats={p['svg_stats']}")
    print(f"rasterizer_available={p['rasterizer_available']}  raster={p['raster_png']}")
    print(f"visual_delta_vs_source={p['visual_delta_vs_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
