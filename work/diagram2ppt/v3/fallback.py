"""Fallback records and audit (§9 of the Decompiler plan).

Product rule: fallback (raster crops for content that could not be made native)
is allowed, but it must be **local, explicit, and tracked** — never a silent
full-page raster. Each fallback element should carry:

    editable=False, reason, source_bbox, confidence, future_replacement

This module detects fallback elements in an IR and audits them against that
rule. It is deterministic and offline: it reads the IR dict only.

Notes grounded in the current IR shape:
- v2 hybrid marks a raster fallback with ``type == "raster_crop"`` (and an
  ``ext.fidelity`` block); those elements do NOT yet carry the §9 metadata, so
  the audit reports them as ``undocumented`` — real, actionable signal.
- ``ext.forced`` means "forced to a NATIVE shape despite low confidence" — the
  opposite of a fallback — and is deliberately NOT treated as one.
"""
from __future__ import annotations

from typing import Any

# Element ``type`` values that denote a raster / non-native fallback.
FALLBACK_TYPES = {
    "raster_crop", "raster", "raster_fallback", "picture", "image", "faithful_crop",
}

# §9 metadata a well-formed fallback record must carry.
REQUIRED_FIELDS = ("reason", "future_replacement")


def is_fallback(el: dict[str, Any]) -> bool:
    if el.get("editable") is False:
        return True
    return str(el.get("type")) in FALLBACK_TYPES


def fallback_record(el: dict[str, Any]) -> dict[str, Any]:
    """Normalize a fallback element into the §9 record shape."""
    ext = el.get("ext") or {}
    fb = ext.get("fallback") or {}
    conf = el.get("confidence")
    if conf is None:
        conf = ext.get("evidence_confidence")
    return {
        "id": el.get("id"),
        "type": el.get("type"),
        "editable": el.get("editable", False),
        "reason": fb.get("reason") or ext.get("reason"),
        "source_bbox": fb.get("source_bbox") or el.get("bbox"),
        "confidence": conf,
        "future_replacement": fb.get("future_replacement") or ext.get("future_replacement"),
    }


def _area(bbox: Any) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x0, y0, x1, y1 = bbox[:4]
    return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))


def _canvas_area(ir: dict[str, Any]) -> float:
    canvas = ir.get("canvas") or {}
    image = ir.get("image") or {}
    w = canvas.get("width_px") or image.get("width") or ir.get("width") or 0
    h = canvas.get("height_px") or image.get("height") or ir.get("height") or 0
    return float(w) * float(h)


def audit_fallbacks(ir: dict[str, Any], full_page_threshold: float = 0.6) -> dict[str, Any]:
    """Audit every fallback element in ``ir`` against the §9 rule.

    Returns a summary with the normalized records and any violations:
    - ``undocumented``: missing a required §9 field (reason / future_replacement)
    - ``full_page``: a single fallback covers > ``full_page_threshold`` of the canvas
    """
    elements = ir.get("elements") or []
    canvas_area = _canvas_area(ir)
    records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    fallback_area = 0.0

    for el in elements:
        if not is_fallback(el):
            continue
        rec = fallback_record(el)
        records.append(rec)
        area = _area(el.get("bbox"))
        fallback_area += area

        missing = [f for f in REQUIRED_FIELDS if not rec.get(f)]
        if missing:
            violations.append({
                "id": rec["id"], "kind": "undocumented", "missing": missing,
            })
        if canvas_area and area / canvas_area > full_page_threshold:
            violations.append({
                "id": rec["id"], "kind": "full_page",
                "area_ratio": round(area / canvas_area, 4),
            })

    return {
        "fallback_count": len(records),
        "fallback_area_ratio": round(fallback_area / canvas_area, 4) if canvas_area else 0.0,
        "records": records,
        "violations": violations,
        "compliant": not violations,
    }
