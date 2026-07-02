"""CrossPanelBridgeAgent: native semantic connector between panels."""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class CrossPanelBridgeAgent(Agent):
    """Build broad arrows that express relationships between major panels."""

    name = "CrossPanelBridgeAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if not _is_bridge_task(task):
            return []
        changed = set(_remove_competing_fragments(ir, task))
        bridge = _bridge_element(ir, task)
        existing = IR.get_element(ir, bridge["id"])
        if existing:
            existing.clear()
            existing.update(bridge)
        else:
            elements = ir.setdefault("elements", [])
            elements.append(bridge)
        changed.add(bridge["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "native_cross_panel_bridge",
            "round": ir.get("round", 0),
            "changed": sorted(changed),
        })
        return sorted(changed)


def _is_bridge_task(task: dict) -> bool:
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "kind", "region_id", "locked_method", "objective",
    )).lower()
    return "cross_panel_bridge" in text or "cross-panel" in text


def _bridge_element(ir: dict, task: dict) -> dict:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    h = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1414)
    bbox = task.get("bbox")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    else:
        x0, y0, x1, y1 = w * 0.425, h * 0.365, w * 0.505, h * 0.520
    sx = max(x0 + (x1 - x0) * 0.28, w * 0.448)
    ex = min(x0 + (x1 - x0) * 0.96, w * 0.506)
    cy = y0 + (y1 - y0) * 0.57
    thickness = max(34.0, min(54.0, (y1 - y0) * 0.30))
    return IR.element(
        id="cross_panel_bridge_problem_to_solution",
        type="icon",
        bbox=[sx, cy - thickness * 0.56, ex, cy + thickness * 0.56],
        provenance=IR.provenance(
            "CrossPanelBridgeAgent",
            "problem_to_solution_bridge",
            ir.get("round", 0),
        ),
        confidence=0.86,
        points=[sx, cy, ex, cy],
        color="#2b7fb6",
        thickness=thickness,
        icon={"kind": "arrow", "color": "#2b7fb6"},
        z=1.6,
        ext={
            "component": "cross_panel_bridge",
            "component_role": "problem_to_solution",
            "strategy": {
                "primary_method": "cross_panel_bridge",
                "preferred_agent": "CrossPanelBridgeAgent",
            },
            "text_contract": {
                "typography_locked": True,
            },
        },
    )


def _remove_competing_fragments(ir: dict, task: dict) -> list[str]:
    bbox = task.get("bbox")
    if not bbox or len(bbox) != 4:
        return []
    x0, y0, x1, y1 = [float(v) for v in bbox]
    region = [x0 - 30.0, y0 - 30.0, x1 + 30.0, y1 + 30.0]
    keep: list[dict] = []
    removed: list[str] = []
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        if eid == "cross_panel_bridge_problem_to_solution":
            keep.append(el)
            continue
        if el.get("type") not in {"arrow", "line", "path", "polygon", "freeform"}:
            keep.append(el)
            continue
        eb = el.get("bbox")
        if not eb or _bbox_overlap_fraction(eb, region) < 0.20:
            keep.append(el)
            continue
        ext = el.get("ext") or {}
        if ext.get("component") in {"pipeline_context", "auditor_card", "action_card"}:
            keep.append(el)
            continue
        removed.append(eid)
    if removed:
        ir["elements"] = keep
    return removed


def _bbox_overlap_fraction(a: list | tuple | None, b: list | tuple | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / area
