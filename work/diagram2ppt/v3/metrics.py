"""Multi-dimensional, offline-computable quality metrics (§8 of the plan).

The v3 north star is not one number. This module assembles the offline-derivable
part of the §8 metric set for a decompiled IR (and, when present, its rendered
deck):

    native_element_ratio    fraction of IR elements that are native (not fallback)
    fallback_area_ratio     canvas area covered by raster fallback
    editability_score       1 - fallback_area_ratio (area you can still edit natively)
    object_coverage         read through from ir['metrics'].coverage_explained

Scores that require comparing against the source image (visual_delta,
text_accuracy, connector_accuracy, ...) are produced by the live verifier during
a run; we pass them through from ``ir['metrics']`` rather than recomputing them,
so this module stays deterministic and network-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import fallback as _fallback

# verifier-produced scores we surface as-is (computed during a live run).
_PASSTHROUGH = (
    "visual_delta", "coverage_explained", "text_accuracy", "connector_accuracy",
    "critical_defect_count", "typography_role_fraction",
)


def native_element_ratio(ir: dict[str, Any]) -> float:
    elements = ir.get("elements") or []
    if not elements:
        return 0.0
    native = sum(1 for el in elements if not _fallback.is_fallback(el))
    return round(native / len(elements), 4)


def ir_metrics(ir: dict[str, Any]) -> dict[str, Any]:
    """Compute the offline metric set for an IR dict."""
    audit = _fallback.audit_fallbacks(ir)
    fallback_area = audit["fallback_area_ratio"]
    base = ir.get("metrics") or {}
    out: dict[str, Any] = {
        "element_count": len(ir.get("elements") or []),
        "native_element_ratio": native_element_ratio(ir),
        "fallback_count": audit["fallback_count"],
        "fallback_area_ratio": fallback_area,
        "editability_score": round(max(0.0, 1.0 - fallback_area), 4),
        "fallback_compliant": audit["compliant"],
    }
    for key in _PASSTHROUGH:
        if key in base:
            out[key] = base[key]
    return out


def deck_metrics(pptx_path: str | Path, ir: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine rendered-deck structure (native_object_ratio, pictures, OMML) with
    IR-level metrics. ``pptx_path`` may be absent — structure is then omitted."""
    from . import pptx_stats

    out: dict[str, Any] = {}
    path = Path(pptx_path)
    if path.exists():
        out["structure"] = pptx_stats.pptx_structure(path)
    if ir is not None:
        out["ir"] = ir_metrics(ir)
    return out
