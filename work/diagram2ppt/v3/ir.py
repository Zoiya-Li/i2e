"""Global Native IR Blackboard for diagram2ppt v3.

This IR is the single source of truth for the Planner and all specialist agents.
All agents read from and write to this structure; the Planner serializes it after
every round to support rollback.

Absolute invariants:
  - No element type may imply a raster fallback.
  - Every element must carry provenance.
  - The IR tracks defects, repair history, and acceptance/rollback decisions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VERSION = "d2p-3"

NATIVE_ELEMENT_TYPES = {
    "text",
    "formula",
    "chart",
    "icon",
    "rect",
    "rounded_rect",
    "oval",
    "diamond",
    "hexagon",
    "parallelogram",
    "arrow",
    "line",
    "freeform",
    "dotcloud",
    "surface",
    "group",
}

EDITABLE_SHAPE_TYPES = {
    "rect", "rounded_rect", "oval", "diamond", "hexagon", "parallelogram"
}

CONNECTOR_TYPES = {"arrow", "line"}


def new_ir(source: dict | str | None = None, width: int = 0, height: int = 0) -> dict:
    """Create a fresh v3 IR blackboard.

    Args:
        source: either a dict from `source_parser`, a path string, or None.
        width: canvas width in pixels.
        height: canvas height in pixels.
    """
    if source is None:
        source_info = {"type": "unknown"}
    elif isinstance(source, str):
        source_info = {"type": "unknown", "path": source}
    else:
        source_info = dict(source)

    return {
        "version": VERSION,
        "round": 0,
        "goal": "reconstruct fully native editable PPTX",
        "status": "planning",  # planning | extracting | building | verifying | accepted | failed
        "source": source_info,
        "canvas": {
            "width_px": int(width),
            "height_px": int(height),
            "slide_width_in": 13.333,
            "slide_height_in": round(13.333 * height / width, 3) if width else 0.0,
        },
        "elements": [],
        "groups": [],
        "constraints": [],
        "relations": [],
        "styles": {},
        "defects": [],
        "history": [],
        "patches": [],
        "metrics": {},
        "checkpoints": [],
    }


def validate(ir: dict, raise_on_error: bool = True) -> list[str]:
    """Validate IR invariants. Returns list of violation messages."""
    errors: list[str] = []
    for el in ir.get("elements", []):
        t = el.get("type")
        if t not in NATIVE_ELEMENT_TYPES:
            errors.append(f"element {el.get('id')} has non-native type {t!r}")
        if not el.get("provenance"):
            errors.append(f"element {el.get('id')} missing provenance")
        if t in ("text", "formula") and not el.get("text"):
            # text/formula may be pending; warn only if status is native
            if el.get("status") == "native":
                errors.append(f"element {el.get('id')} is native {t} but has no text")
    if raise_on_error and errors:
        raise ValueError("IR validation failed:\n" + "\n".join(errors))
    return errors


def element(
    id: str,
    type: str,
    bbox: list[float],
    provenance: dict,
    **kwargs: Any,
) -> dict:
    """Factory for a single IR element with required fields."""
    if type not in NATIVE_ELEMENT_TYPES:
        raise ValueError(f"non-native element type: {type!r}")
    el: dict = {
        "id": id,
        "type": type,
        "status": kwargs.get("status", "native"),
        "bbox": [float(v) for v in bbox],
        "confidence": float(kwargs.get("confidence", 0.0)),
        "provenance": provenance,
        "repair_history": [],
        "defects": [],
        "ext": dict(kwargs.get("ext", {})),
    }
    for key, val in kwargs.items():
        if key not in el:
            el[key] = val
    return el


def provenance(agent: str, action: str, round: int = 0, **extra: Any) -> dict:
    """Standard provenance record for an IR mutation."""
    return {"agent": agent, "action": action, "round": round, **extra}


def add_element(ir: dict, el: dict) -> None:
    """Add an element to the IR and validate it is native."""
    if el.get("type") not in NATIVE_ELEMENT_TYPES:
        raise ValueError(f"cannot add non-native element {el.get('id')}: {el.get('type')!r}")
    ir["elements"].append(el)


def remove_element(ir: dict, element_id: str) -> dict | None:
    """Remove element by id; return it or None."""
    for i, el in enumerate(ir["elements"]):
        if el.get("id") == element_id:
            return ir["elements"].pop(i)
    return None


def get_element(ir: dict, element_id: str) -> dict | None:
    """Get element by id."""
    for el in ir["elements"]:
        if el.get("id") == element_id:
            return el
    return None


def snapshot(ir: dict) -> dict:
    """Deep copy the IR for rollback."""
    return json.loads(json.dumps(ir, default=str))


def restore(ir: dict, backup: dict) -> dict:
    """Replace IR contents with backup."""
    ir.clear()
    ir.update(backup)
    return ir


def record_patch(
    ir: dict,
    patch_id: str,
    agent: str,
    changed: list[str],
    expected_fixes: list[str],
    metrics_before: dict,
    metrics_after: dict,
    decision: str,
    reason: str = "",
) -> dict:
    """Record a patch transaction in the IR history."""
    patch = {
        "patch_id": patch_id,
        "round": ir.get("round", 0),
        "agent": agent,
        "changed": changed,
        "expected_fixes": expected_fixes,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "decision": decision,
        "reason": reason,
    }
    ir["patches"].append(patch)
    ir["history"].append({
        "round": ir.get("round", 0),
        "status": ir.get("status"),
        "metrics": metrics_after,
        "patch": patch_id,
    })
    return patch


def metrics(ir: dict) -> dict:
    """Compute global native-fidelity metrics from current IR."""
    els = ir.get("elements", [])
    native = [e for e in els if e.get("status") == "native"]
    boxed = [e for e in els if "bbox" in e]

    def area(e: dict) -> float:
        x0, y0, x1, y1 = e["bbox"]
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    total_area = sum(area(e) for e in boxed) or 1.0
    native_area = sum(area(e) for e in native if "bbox" in e)

    return {
        "elements": len(els),
        "native_count": len(native),
        "native_fraction_count": round(len(native) / len(els), 4) if els else 0.0,
        "native_fraction_area": round(native_area / total_area, 4),
        "defect_count": len(ir.get("defects", [])),
        "critical_defect_count": sum(
            1 for d in ir.get("defects", []) if d.get("severity", 0) >= 0.7
        ),
    }


def save(ir: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ir, ensure_ascii=False, indent=2, default=str))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())
