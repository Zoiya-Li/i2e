"""Connector Agent: repairs arrows and lines.

Uses deterministic CV from the v2 toolbox to re-detect connector endpoints
and snap them to the nearest shape perimeters.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class ConnectorAgent(Agent):
    """Specialist agent for arrow/line connector repair."""

    name = "ConnectorAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el and el.get("type") in ("arrow", "line"):
                # For explicit high_residual defects, force a re-detection even
                # if the ink fraction looks acceptable.
                force = defect.get("type") == "high_residual"
                return self._repair_connector(ir, original, el, force=force)

        # Proactive scan: try to fix any connector whose rendered segment has
        # poor ink support.
        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") in ("arrow", "line"):
                changed.extend(self._repair_connector(ir, original, el))
        return changed

    def _repair_connector(self, ir: dict, original: Image.Image,
                          el: dict, force: bool = False) -> list[str]:
        try:
            from work.diagram2ppt.v2 import vectorize as V
            from work.diagram2ppt.v2.diff import connector_ink_fraction
            from work.diagram2ppt.v2.handlers import _snap, _skeleton_arrow
            from work.diagram2ppt.v2.render import _center, _edge_point
        except Exception:
            return []

        bbox = el.get("bbox")
        if not bbox:
            return []

        elements = ir.get("elements", [])
        shape_map = {e["id"]: e for e in elements if "bbox" in e}

        # Current rendered segment for ink-fraction check.
        pts = el.get("points")
        if pts:
            start, end = (pts[0], pts[1]), (pts[2], pts[3])
        elif el.get("from_id") and el.get("to_id"):
            src = shape_map.get(el["from_id"])
            dst = shape_map.get(el["to_id"])
            if src and dst:
                start = _edge_point(src, _center(dst))
                end = _edge_point(dst, _center(src))
            else:
                return []
        else:
            return []

        ink = connector_ink_fraction(original, start, end)
        if ink >= 0.35 and not force:
            # Already well supported; nothing to repair.
            return []

        # Re-detect from the original crop.
        x0, y0, x1, y1 = bbox
        crop = original.crop((max(0, int(x0)), max(0, int(y0)),
                              min(original.width, int(x1)),
                              min(original.height, int(y1))))
        if crop.width < 8 or crop.height < 8:
            IR.remove_element(ir, el["id"])
            return []

        detection = self._detect_connector(original, bbox)
        if not detection:
            IR.remove_element(ir, el["id"])
            return []

        points = detection["points"]
        color = detection.get("color", el.get("color", "#333333"))
        thickness = detection.get("thickness", el.get("thickness", 2))

        # Re-snap to nearby shapes.
        from_id, to_id, from_pt, to_pt = _snap(points, elements, radius=80)
        if from_pt and to_pt:
            points = [from_pt[0], from_pt[1], to_pt[0], to_pt[1]]

        # Update element.
        el["points"] = [float(v) for v in points]
        el["color"] = color
        el["thickness"] = int(thickness)
        if from_id:
            el["from_id"] = from_id
        if to_id:
            el["to_id"] = to_id
        if thickness >= 6:
            el["type"] = "arrow"

        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "connector_redetect",
            "round": ir.get("round", 0),
            "ink_before": round(ink, 3),
        })
        return [el["id"]]

    @staticmethod
    def _detect_connector(original: Image.Image, bbox: list[float]) -> dict | None:
        """Re-detect a connector in the given bbox.

        Tries fat-arrow detection first; falls back to skeleton extraction.
        """
        from work.diagram2ppt.v2 import vectorize as V
        from work.diagram2ppt.v2.handlers import _skeleton_arrow

        # Try saturated straight strokes (bold arrows).
        try:
            arrows = V.detect_fat_arrows(original, bbox, max_n=1)
            if arrows:
                return arrows[0]
        except Exception:
            pass

        # Fallback: extreme ink points as endpoints.
        x0, y0, x1, y1 = bbox
        crop = original.crop((max(0, int(x0)), max(0, int(y0)),
                              min(original.width, int(x1)),
                              min(original.height, int(y1))))
        try:
            pts, color, thickness = _skeleton_arrow(crop, [x0, y0, x1, y1])
            if pts:
                return {"points": pts, "color": color, "thickness": thickness}
        except Exception:
            pass
        return None
