"""v3 PPTX builder: compiles Global Native IR into a native editable .pptx.

The builder enforces the absolute v3 invariant: no raster crops, no screenshots,
no opaque raster layers. If the IR contains any non-native or unrepresentable
element, building is blocked and a structured error is returned so the Planner
can schedule repairs.
"""
from __future__ import annotations

from pathlib import Path

from work.diagram2ppt.v2 import build_pptx as v2_builder
from . import ir as IR


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


def validate_buildable(ir: dict) -> list[dict]:
    """Return a list of blockers; empty means the IR can be built natively."""
    blockers: list[dict] = []
    for el in ir.get("elements", []):
        t = el.get("type")
        if t not in IR.NATIVE_ELEMENT_TYPES:
            blockers.append({
                "element_id": el.get("id", "?"),
                "element_type": t,
                "reason": f"non-native element type {t!r}",
            })
        if t == "group":
            blockers.append({
                "element_id": el.get("id", "?"),
                "element_type": t,
                "reason": "group elements are not yet rendered by the native builder",
            })
    return blockers


def build_pptx(ir: dict, output_path: str) -> dict:
    """Build a native-only PPTX from v3 IR.

    Args:
        ir: v3 Global Native IR blackboard.
        output_path: destination .pptx path.

    Returns:
        Build statistics dict.

    Raises:
        BuildBlockedError: if the IR contains non-native elements.
    """
    blockers = validate_buildable(ir)
    if blockers:
        raise BuildBlockedError(blockers)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert to v2-compatible IR and reuse the v2 native builder.
    # The validation above guarantees no raster_crop enters the v2 builder.
    v2_ir = _to_v2_ir(ir)
    stats = v2_builder.build_pptx(v2_ir, str(out))

    # Post-check: ensure the v2 builder did not silently embed any pictures.
    # (A picture would only come from a raster_crop element.)
    if stats.get("pictures", 0) > 0:
        raise BuildBlockedError([{
            "element_id": "unknown",
            "element_type": "raster_crop",
            "reason": "v2 builder embedded pictures; v3 forbids raster fallback",
        }])

    return {
        **stats,
        "output_path": str(out),
        "native": True,
        "pictures": 0,
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
