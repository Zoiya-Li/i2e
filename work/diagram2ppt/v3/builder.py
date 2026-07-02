"""v3 PPTX builder: compiles Global Native IR into a native editable .pptx.

The builder enforces the absolute v3 invariant: no raster crops, no screenshots,
no opaque raster layers. If the IR contains any non-native or unrepresentable
element, building is blocked and a structured error is returned so the Planner
can schedule repairs.
"""
from __future__ import annotations

import os
from pathlib import Path

from work.diagram2ppt.v2 import build_pptx as v2_builder
from . import fallback as _fallback
from . import ir as IR

# Build profiles (P4):
#   all_native       — research constraint: zero raster; block every non-native
#                      element so the pipeline is forced to raise native coverage.
#   product_delivery — allow *documented, local* raster fallback (§9), but still
#                      reject undocumented and full-page fallback, so a usable
#                      deck never silently regresses to a full-page screenshot.
PROFILE_ALL_NATIVE = "all_native"
PROFILE_PRODUCT = "product_delivery"
PROFILES = (PROFILE_ALL_NATIVE, PROFILE_PRODUCT)

_FULL_PAGE_FALLBACK_RATIO = 0.6


class UneditableElementError(Exception):
    """Raised when the IR contains an element that cannot be rendered natively."""

    def __init__(self, element_id: str, element_type: str, reason: str) -> None:
        self.element_id = element_id
        self.element_type = element_type
        self.reason = reason
        super().__init__(f"Uneditable element {element_id} ({element_type}): {reason}")


class BuildBlockedError(Exception):
    """Raised when the builder refuses to produce a PPTX due to policy violations."""

    def __init__(self, reasons: list[dict]) -> None:
        self.reasons = reasons
        msg = "Build blocked:\n" + "\n".join(
            f"  - {r['element_id']} ({r['element_type']}): {r['reason']}"
            for r in reasons
        )
        super().__init__(msg)


def _area(bbox) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x0, y0, x1, y1 = bbox[:4]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _resolve_profile(profile: str | None) -> str:
    if profile is None:
        profile = os.environ.get("I2E_BUILD_PROFILE", PROFILE_ALL_NATIVE)
    return profile if profile in PROFILES else PROFILE_ALL_NATIVE


def validate_buildable(ir: dict, profile: str | None = None) -> list[dict]:
    """Return a list of blockers; empty means the IR can be built under ``profile``."""
    profile = _resolve_profile(profile)
    canvas = ir.get("canvas") or {}
    canvas_area = float(canvas.get("width_px") or 0) * float(canvas.get("height_px") or 0)
    blockers: list[dict] = []
    for el in ir.get("elements", []):
        t = el.get("type")
        if t == "group":
            blockers.append({
                "element_id": el.get("id", "?"),
                "element_type": t,
                "reason": "group elements are not yet rendered by the native builder",
            })
            continue
        if t in IR.NATIVE_ELEMENT_TYPES:
            continue
        # non-native element
        if profile == PROFILE_PRODUCT and _fallback.is_fallback(el):
            rec = _fallback.fallback_record(el)
            missing = [f for f in _fallback.REQUIRED_FIELDS if not rec.get(f)]
            if missing:
                blockers.append({
                    "element_id": el.get("id", "?"),
                    "element_type": t,
                    "reason": f"undocumented fallback (missing {', '.join(missing)}); "
                              "§9 requires reason + future_replacement",
                })
            elif canvas_area and _area(el.get("bbox")) / canvas_area > _FULL_PAGE_FALLBACK_RATIO:
                blockers.append({
                    "element_id": el.get("id", "?"),
                    "element_type": t,
                    "reason": f"full-page fallback (> {_FULL_PAGE_FALLBACK_RATIO:.0%} of canvas) not allowed",
                })
            # else: documented local fallback — allowed under product_delivery
            continue
        blockers.append({
            "element_id": el.get("id", "?"),
            "element_type": t,
            "reason": f"non-native element type {t!r}",
        })
    return blockers


def build_pptx(ir: dict, output_path: str, profile: str | None = None) -> dict:
    """Build a PPTX from v3 IR under a build profile (default ``all_native``).

    Args:
        ir: v3 Global Native IR blackboard.
        output_path: destination .pptx path.
        profile: ``all_native`` (default) or ``product_delivery``; falls back to
            the ``I2E_BUILD_PROFILE`` env var when ``None``.

    Returns:
        Build statistics dict.

    Raises:
        BuildBlockedError: if the IR violates the profile.
    """
    profile = _resolve_profile(profile)
    blockers = validate_buildable(ir, profile)
    if blockers:
        raise BuildBlockedError(blockers)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    v2_ir = _to_v2_ir(ir)
    stats = v2_builder.build_pptx(v2_ir, str(out))

    # all_native forbids any raster; product_delivery permits documented ones
    # (already validated above), so only guard the all_native invariant here.
    if profile == PROFILE_ALL_NATIVE and stats.get("pictures", 0) > 0:
        raise BuildBlockedError([{
            "element_id": "unknown",
            "element_type": "raster_crop",
            "reason": "v2 builder embedded pictures; all_native profile forbids raster fallback",
        }])

    return {
        **stats,
        "output_path": str(out),
        "profile": profile,
        "native": profile == PROFILE_ALL_NATIVE or stats.get("pictures", 0) == 0,
        "pictures": stats.get("pictures", 0),
    }


def _to_v2_ir(ir: dict) -> dict:
    """Convert v3 IR to the shape expected by work.diagram2ppt.v2.build_pptx."""
    source = ir.get("source", {})
    return {
        "version": "d2p-2",
        "image": {
            "path": source.get("path", ""),
            "width": ir["canvas"]["width_px"],
            "height": ir["canvas"]["height_px"],
        },
        "elements": ir["elements"],
    }
