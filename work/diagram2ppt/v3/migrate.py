"""Migrate v2 detection/processing output into v3 Global Native IR.

The v2 pipeline emits a list of entities with types such as:
  shape, container, text, formula, chart, icon, arrow, line, dotcloud, surface,
  raster_crop

v3 IR rejects raster_crop. Any v2 entity that cannot be represented natively is
reported as a pending defect and skipped from the build.
"""
from __future__ import annotations

from . import ir as IR


V2_TO_V3_TYPE = {
    "shape": "rect",
    "container": "rect",
    "rect": "rect",
    "rounded_rect": "rounded_rect",
    "oval": "oval",
    "diamond": "diamond",
    "hexagon": "hexagon",
    "parallelogram": "parallelogram",
    "text": "text",
    "formula": "formula",
    "chart": "chart",
    "icon": "icon",
    "arrow": "arrow",
    "line": "line",
    "dotcloud": "dotcloud",
    "surface": "surface",
    "raster_crop": None,  # rejected
}


def from_v2_entities(entities: list[dict], image_path: str, width: int,
                     height: int, round: int = 0) -> dict:
    """Convert a v2 entity list into a v3 IR blackboard.

    Args:
        entities: v2-style entity list from decompose/process_all.
        image_path: original image path.
        width: original image width in pixels.
        height: original image height in pixels.
        round: initial planner round.

    Returns:
        Populated v3 IR dict.
    """
    out = IR.new_ir(
        source={"type": "raster", "path": image_path,
                "width": width, "height": height},
        width=width,
        height=height,
    )
    out["round"] = round
    out["status"] = "extracting"

    for idx, e in enumerate(entities):
        v2_type = e.get("type", "shape")
        v3_type = V2_TO_V3_TYPE.get(v2_type)

        if v3_type is None:
            out["defects"].append({
                "id": f"defect_unsupported_{idx}",
                "type": "unsupported_element",
                "element_id": e.get("id", f"v2-{idx}"),
                "bbox": e.get("bbox", [0, 0, width, height]),
                "severity": 0.9,
                "reason": f"v2 type {v2_type!r} has no native v3 representation",
                "suggested_agent": "LayoutAgent" if v2_type == "raster_crop" else "HumanCheckpoint",
            })
            continue

        bbox = e.get("bbox")
        if not bbox and v3_type not in IR.CONNECTOR_TYPES:
            # skip shape-like elements without geometry
            out["defects"].append({
                "id": f"defect_no_bbox_{idx}",
                "type": "missing_geometry",
                "element_id": e.get("id", f"v2-{idx}"),
                "bbox": [0, 0, width, height],
                "severity": 0.6,
                "reason": "element has no bbox",
                "suggested_agent": "LayoutAgent",
            })
            continue

        if v3_type == "rect" and v2_type == "shape":
            # preserve rounded_rect if v2 had that info
            shape_sub = e.get("shape_type") or e.get("ext", {}).get("shape_type")
            if shape_sub in IR.EDITABLE_SHAPE_TYPES:
                v3_type = shape_sub

        el = IR.element(
            id=str(e.get("id", f"v2-{idx}")),
            type=v3_type,
            bbox=bbox or [0, 0, width, height],
            provenance=IR.provenance("MigrateAgent", "from_v2_entities", round=round),
            confidence=float(e.get("confidence", 0.0) or 0.0),
            status="native" if v3_type != "group" else "pending",
        )

        # copy over text, style, and ext fields
        el["text"] = e.get("text", "")
        el["fill"] = e.get("fill", "") or ""
        el["border_color"] = e.get("border_color", "") or ""
        el["text_color"] = e.get("text_color", "") or ""
        el["font_size"] = e.get("font_size")
        el["bold"] = bool(e.get("bold", False))
        el["z"] = int(e.get("z", idx))
        if "corner" in e:
            el["corner"] = e["corner"]
        if "rotation" in e:
            el["rotation"] = e["rotation"]

        # connector fields
        if v3_type in IR.CONNECTOR_TYPES:
            el["from_id"] = e.get("from_id") or e.get("from") or ""
            el["to_id"] = e.get("to_id") or e.get("to") or ""
            el["color"] = e.get("color", "#333333")
            el["label"] = e.get("label", "") or e.get("text", "")
            pts = e.get("points")
            if pts and len(pts) == 4:
                el["points"] = pts

        # formula / chart / icon ext
        ext = dict(e.get("ext", {}))
        if v3_type == "formula":
            ext["latex"] = e.get("latex") or e.get("text", "")
        if v3_type == "chart":
            # Preserve the full chart spec used by the v2 builder.
            chart_spec = e.get("chart")
            if chart_spec:
                el["chart"] = dict(chart_spec)
                ext["chart"] = dict(chart_spec)
            for k in ("chart_type", "categories", "series", "points", "trend"):
                if k in e:
                    ext[k] = e[k]
        if v3_type == "icon":
            icon_payload = e.get("icon")
            if icon_payload:
                el["icon"] = dict(icon_payload)
                ext["icon"] = dict(icon_payload)
        if v3_type in ("surface", "dotcloud"):
            # These are still fully native in v3, but they need their vector
            # payload.  Without it the builder can only draw an empty rectangle,
            # which is exactly the failure mode on framework.png.
            for k in (
                "dots",
                "streamlines",
                "wave_bands",
                "silhouette",
                "surface_layers",
                "style",
                "curves",
                "trend",
            ):
                if k in e:
                    el[k] = e[k]
                    ext[k] = e[k]
        el["ext"].update(ext)

        out["elements"].append(el)

    out["metrics"] = IR.metrics(out)
    return out


def to_v2_compatible(ir: dict) -> dict:
    """Produce a v2-compatible dict for reuse of v2 rendering utilities."""
    return {
        "version": "d2p-2",
        "image": {
            "path": ir["source"].get("path", ""),
            "width": ir["canvas"]["width_px"],
            "height": ir["canvas"]["height_px"],
        },
        "elements": ir["elements"],
    }
