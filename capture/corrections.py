"""Node ⑤ — capture (predicted -> corrected) pairs from a user's edits.

The moat mechanism: the user edits the extracted IR to ship their asset; we diff
the shipped IR against what Node ② predicted and record each changed field as a
Correction. No extra labeling — the edits the user makes anyway become training
data. Each Correction is routed to a sub-model by `field_path`/`kind`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

# Which extraction sub-model each kind of correction trains (drives the flywheel).
_FIELD_KIND = {
    "text.content": "text_content",
    "text.font_family": "font",
    "text.font_size_px": "font_size",
    "text.color": "color",
    "bbox": "geometry",
    "type": "type_reclassify",
    "z": "zorder",
    "logo.matched_asset_id": "brand_apply",
    "logo.raster_ref": "brand_apply",
    "raster.asset_ref": "asset_replace",
    "raster.mask_ref": "mask_refine",
    "ext.cutout": "inpaint_redo",     # re-run amodal completion / inpaint of the layer
}
# Fields compared element-vs-element when diffing. Order is the report order. Covers every
# field the editor can change (font size, asset/mask swap, re-inpainted cutout), not just a few.
_TRACKED = ["type", "z", "bbox", "text.content", "text.font_family", "text.font_size_px",
            "text.color", "logo.matched_asset_id", "logo.raster_ref",
            "raster.asset_ref", "raster.mask_ref", "ext.cutout"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(el: dict, path: str):
    cur = el
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _by_id(ir: dict) -> dict[str, dict]:
    return {el["id"]: el for el in ir["elements"]}


def capture_diff(original_ir: dict, edited_ir: dict, *, session_id: str | None = None,
                 user_hash: str | None = None) -> list[dict]:
    """Return Correction records for every field the user changed from Node ②'s
    prediction. Handles per-field edits, element add, and element delete."""
    image_id = original_ir["id"]
    orig, edited = _by_id(original_ir), _by_id(edited_ir)
    out: list[dict] = []

    def make(element_id, field_path, kind, predicted, corrected, conf, model_version):
        out.append({
            "id": "corr-" + uuid.uuid4().hex[:10],
            "image_id": image_id,
            "element_id": element_id,
            "field_path": field_path,
            "kind": kind,
            "predicted": predicted,
            "corrected": corrected,
            "confidence_at_prediction": conf,
            "model_version": model_version,
            "session_id": session_id,
            "user_hash": user_hash,
            "corrected_at": _now(),
        })

    # changed fields on elements present in both
    for eid in orig.keys() & edited.keys():
        oe, ee = orig[eid], edited[eid]
        mv = (oe.get("extraction") or {}).get("model_version", "unknown")
        conf = (oe.get("extraction") or {}).get("confidence")
        for path in _TRACKED:
            ov, ev = _get(oe, path), _get(ee, path)
            if ov != ev:
                make(eid, path, _FIELD_KIND[path], ov, ev, conf, mv)

    # deletions (predicted but removed by the user)
    for eid in orig.keys() - edited.keys():
        mv = (orig[eid].get("extraction") or {}).get("model_version", "unknown")
        conf = (orig[eid].get("extraction") or {}).get("confidence")
        make(eid, "element", "element_delete", orig[eid].get("type"), None, conf, mv)

    # additions (user added something Node ② missed) — distinguish a layer split from a
    # fresh add so the splitter sub-model gets its own signal
    for eid in edited.keys() - orig.keys():
        ee = edited[eid]; ext = ee.get("ext") or {}
        kind = "layer_split" if (ext.get("split_from") or ext.get("split_parent")) else "element_add"
        make(eid, "element", kind, None, ee.get("type"), None, "human")

    return out


def append_corrections(ir: dict, corrections: list[dict]) -> dict:
    """Append corrections to an IR document's log and bump updated_at (in place)."""
    ir.setdefault("corrections", []).extend(corrections)
    ir["updated_at"] = _now()
    return ir
