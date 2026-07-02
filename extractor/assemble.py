"""Assemble raw VLM elements into the canonical IR (ir/ir-v1.schema.json) and
validate. This is the provider-independent spine — the single source of truth.

Asset files (cutouts, inpainted background, traced SVGs) are produced by later
pipeline stages; here we emit the placeholder paths where they WILL be written,
so the IR is complete and schema-valid from extraction onward.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "ir" / "ir-v1.schema.json"
_HEX = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_VALID_ALIGN = {"left", "center", "right", "justify"}
CONF_THRESHOLD = 0.75  # below this -> needs_review (surfaced to the human first)

_EDITABLE = {
    "text": ["text.content", "text.color", "text.font_family"],
    "raster": ["bbox"],
    "logo": ["logo.matched_asset_id", "bbox"],
    "vector": ["bbox"],
    "background": [],
    "group": [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canvas_size(image_path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as im:
        return im.width, im.height


def assemble_ir(raw_elements: list[dict], *, image_path: str, generator: str,
                provider_name: str, model_version: str, method: str) -> dict:
    cw, ch = _canvas_size(image_path)
    doc_id = "img_" + Path(image_path).stem
    now = _now()

    # Stable, unique ids per type (e.g. text-1, logo-1). Also lets group children
    # (given by name) be resolved to ids.
    name_to_id: dict[str, str] = {}
    counts: dict[str, int] = {}
    prepared = []
    for raw in raw_elements:
        t = raw["type"]
        counts[t] = counts.get(t, 0) + 1
        eid = f"{t}-{counts[t]}"
        name_to_id.setdefault(raw.get("name", eid), eid)
        prepared.append((eid, raw))

    elements = []
    background_id = None
    for z, (eid, raw) in enumerate(prepared):
        t = raw["type"]
        bbox = {k: float(raw["bbox"][k]) for k in ("x", "y", "w", "h")}
        conf = float(raw.get("confidence", 0.0))
        asset = f"assets/layers/{doc_id}/{eid}"

        el = {
            "id": eid,
            "type": t,
            "name": raw.get("name", eid),
            "z": z,
            "bbox": bbox,
            "nbox": {"x": bbox["x"] / cw, "y": bbox["y"] / ch,
                     "w": bbox["w"] / cw, "h": bbox["h"] / ch},
            "extraction": {"confidence": conf, "model": provider_name,
                           "model_version": model_version, "method": method,
                           "extracted_at": now},
            "needs_review": conf < CONF_THRESHOLD,
            "editable": _EDITABLE.get(t, []),
        }

        if t == "text":
            src = raw.get("text") or {}
            payload = {"content": src.get("content", "")}
            if src.get("lang"):
                payload["lang"] = src["lang"]
            if src.get("font_family"):
                payload["font_family"] = src["font_family"]
            if isinstance(src.get("font_size_px"), (int, float)):
                payload["font_size_px"] = float(src["font_size_px"])
            if isinstance(src.get("color"), str) and _HEX.match(src["color"]):
                payload["color"] = src["color"].upper()
            if src.get("align") in _VALID_ALIGN:
                payload["align"] = src["align"]
            payload["baked_region_ref"] = asset + "_text.png"
            el["text"] = payload

        elif t == "background":
            background_id = eid
            el["background"] = {"asset_ref": asset + "_bg.png",
                                "inpaint": {"model": provider_name,
                                            "model_version": model_version,
                                            "confidence": conf}}
        elif t == "raster":
            kind = (raw.get("raster") or {}).get("kind") or "foreground"
            el["raster"] = {"asset_ref": asset + ".png", "mask_ref": asset + "_mask.png", "kind": kind}

        elif t == "logo":
            el["logo"] = {"matched_asset_id": None, "match_confidence": conf,
                          "vector_ref": None, "raster_ref": asset + ".png"}

        elif t == "vector":
            el["vector"] = {"svg_ref": asset + ".svg", "svg_path": None, "fill": None, "stroke": None}

        elif t == "group":
            children = [name_to_id[n] for n in (raw.get("children") or []) if n in name_to_id]
            el["children"] = children

        elements.append(el)

    doc = {
        "ir_version": "1.0",
        "id": doc_id,
        "source": {"generator": generator, "original_image_ref": image_path,
                   "width": cw, "height": ch, "ingested_at": now},
        "canvas": {"width": cw, "height": ch, "unit": "px", "background_element_id": background_id},
        "brand_ref": None,
        "elements": elements,
        "corrections": [],
        "created_at": now,
        "updated_at": now,
    }
    validate_ir(doc)
    return doc


def validate_ir(doc: dict) -> None:
    """Raise jsonschema.ValidationError if `doc` is not a valid IR v1 document."""
    schema = json.loads(_SCHEMA_PATH.read_text())
    Draft202012Validator(schema).validate(doc)
