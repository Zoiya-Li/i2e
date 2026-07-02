"""Diagram IR for v2 — pixel-bbox elements with iteration state.

Element types:
    shapes      rect | rounded_rect | oval | diamond | hexagon | parallelogram
    text        free-standing text block (becomes a PPT text box)
    arrow|line  connector; either from_id/to_id (attached) or points (free)
    raster_crop pixel-faithful crop of the ORIGINAL image (the fidelity
                fallback — by construction it re-composites perfectly)

Iteration state per element:
    status      "native" | "demoted"
    tries       refine attempts consumed
    residual    last render-diff residual (None until first scored)

bbox is [x0, y0, x1, y1] in ORIGINAL-image pixels. The VLM boundary speaks
fractions; convert at the edge (see from_vlm_elements).
"""
from __future__ import annotations

import json
from pathlib import Path

SHAPE_TYPES = {"rect", "rounded_rect", "oval", "diamond", "hexagon", "parallelogram"}
CONNECTOR_TYPES = {"arrow", "line"}
# formula/chart/icon/dotcloud are assigned by expert post-passes, never by the VLM
ALL_TYPES = SHAPE_TYPES | CONNECTOR_TYPES | {
    "text", "raster_crop", "formula", "chart", "icon", "dotcloud"}

VERSION = "d2p-2"


def new_ir(image_path: str, width: int, height: int) -> dict:
    return {
        "version": VERSION,
        "image": {"path": str(image_path), "width": int(width), "height": int(height)},
        "elements": [],
        "history": [],   # one entry per loop round: metrics snapshot
    }


def clamp_bbox(b: list, w: int, h: int) -> list:
    x0, y0, x1, y1 = (float(v) for v in b)
    x0, x1 = sorted((max(0.0, min(x0, w)), max(0.0, min(x1, w))))
    y0, y1 = sorted((max(0.0, min(y0, h)), max(0.0, min(y1, h))))
    if x1 - x0 < 2:
        x1 = min(w, x0 + 2)
    if y1 - y0 < 2:
        y1 = min(h, y0 + 2)
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def from_vlm_elements(raw: list[dict], w: int, h: int,
                      id_prefix: str = "el") -> list[dict]:
    """Normalize VLM fraction-coordinate elements into IR elements (px bbox)."""
    out: list[dict] = []
    n = 0
    for el in raw:
        t = str(el.get("type", "rect")).strip().lower()
        if t in ("circle",):
            t = "oval"
        if t in ("raster", "image", "photo", "plot", "chart", "figure", "icon"):
            t = "raster_crop"
        if t not in ALL_TYPES:
            t = "rect"
        n += 1
        e: dict = {
            "id": str(el.get("id") or f"{id_prefix}-{n}"),
            "type": t,
            "status": "native" if t != "raster_crop" else "demoted",
            "tries": 0,
            "residual": None,
            "z": int(el.get("z", n)),
            "ext": {},
        }
        if t == "raster_crop":
            e["ext"]["original_type"] = str(el.get("type", "raster"))

        if t in CONNECTOR_TYPES:
            e["from_id"] = el.get("from_id") or el.get("from") or ""
            e["to_id"] = el.get("to_id") or el.get("to") or ""
            e["color"] = el.get("color") or "#333333"
            e["label"] = el.get("text") or el.get("label") or ""
            pts = el.get("points")
            if pts and len(pts) == 4:
                e["points"] = [pts[0] * w, pts[1] * h, pts[2] * w, pts[3] * h]
        else:
            x = float(el.get("x", 0.0)); y = float(el.get("y", 0.0))
            bw = float(el.get("width", 0.1)); bh = float(el.get("height", 0.05))
            e["bbox"] = clamp_bbox([x * w, y * h, (x + bw) * w, (y + bh) * h], w, h)
            e["text"] = el.get("text", "") or ""
            e["fill"] = el.get("fill", "") or ""
            e["border_color"] = el.get("border_color", "") or ""
            e["text_color"] = el.get("text_color", "") or ""
            e["bold"] = bool(el.get("bold", False))
            fs = el.get("font_size")
            e["font_size"] = float(fs) * h if fs and float(fs) <= 1.0 else (float(fs) if fs else None)
        out.append(e)
    return out


def demote(el: dict) -> None:
    """Fidelity fallback: keep the element, ship it as a faithful crop."""
    el["ext"]["original_type"] = el.get("ext", {}).get("original_type") or el["type"]
    el["type"] = "raster_crop"
    el["status"] = "demoted"
    el["residual"] = 0.0   # by construction: crop of the original


def metrics(ir: dict) -> dict:
    els = ir["elements"]
    boxed = [e for e in els if "bbox" in e]
    native = [e for e in els if e["status"] == "native"]
    native_boxed = [e for e in boxed if e["status"] == "native"]

    def area(e):
        x0, y0, x1, y1 = e["bbox"]
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    tot_area = sum(area(e) for e in boxed) or 1.0
    scored = [e["residual"] for e in native if e.get("residual") is not None]
    return {
        "elements": len(els),
        "native_count": len(native),
        "demoted_count": len(els) - len(native),
        "native_fraction_count": round(len(native) / len(els), 4) if els else 0.0,
        "native_fraction_area": round(sum(area(e) for e in native_boxed) / tot_area, 4),
        "mean_native_residual": round(sum(scored) / len(scored), 4) if scored else None,
    }


def save(ir: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ir, ensure_ascii=False, indent=2))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())
