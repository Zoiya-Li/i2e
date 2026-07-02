"""Vectorize Agent: refine native freeform residual paths.

Verifier routes high-residual native freeforms to VectorizeAgent.  The agent
re-traces the corresponding original-image crop into editable PowerPoint
freeform paths; it never inserts raster crops.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class VectorizeAgent(Agent):
    """Specialist agent for residual freeform path quality."""

    name = "VectorizeAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, str(defect["element_id"]))
            if el and el.get("type") == "freeform":
                return self._repair_freeform(ir, original, el)

        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") == "freeform" and float(el.get("residual") or 0) >= 0.55:
                changed.extend(self._repair_freeform(ir, original, el))
        return changed

    def _repair_freeform(self, ir: dict, original: Image.Image,
                         el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []
        x0, y0, x1, y1 = [float(v) for v in bbox]
        crop_box = (
            max(0, int(round(x0))),
            max(0, int(round(y0))),
            min(original.width, int(round(x1))),
            min(original.height, int(round(y1))),
        )
        crop = original.crop(crop_box)
        if crop.width < 4 or crop.height < 4:
            return []

        try:
            from work.diagram2ppt.v2.native_trace import extract_paths
        except Exception:
            return []

        pale = bool((el.get("ext") or {}).get("pale_trace"))
        old_paths = el.get("paths") or []
        area_hint = float((el.get("ext") or {}).get("residual_area") or 0)
        dense = area_hint > 3000 or crop.width * crop.height > 5000
        paths = extract_paths(
            crop,
            max_paths=120 if dense else 48,
            min_area=max(5.0, crop.width * crop.height * (0.00025 if pale else 0.00055)),
            epsilon_frac=0.0045 if dense else 0.008,
            pale=pale,
        )
        if not paths:
            return []
        for path in paths:
            path["source"] = "vectorize_agent"
            if pale:
                path["closed"] = False
                path["fill"] = None
                path["line_width"] = max(0.3, float(path.get("line_width", 0.35)))
                path["alpha"] = min(int(path.get("alpha", 35)), 46)

        if _path_signature(paths) == _path_signature(old_paths):
            return []

        el["paths"] = paths
        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "retrace_freeform_paths",
            "round": ir.get("round", 0),
            "pale": pale,
        })
        return [str(el["id"])]


def _path_signature(paths: list[dict]) -> tuple:
    sig = []
    for path in paths:
        pts = path.get("points") or []
        sig.append((
            len(pts),
            path.get("fill"),
            path.get("line"),
            int(path.get("alpha", 100)),
            round(float(path.get("area", 0)), 1),
        ))
    return tuple(sig)
