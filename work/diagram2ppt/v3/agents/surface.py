"""Surface Agent: repairs native painterly surfaces and dot clouds.

No screenshot fallback is allowed.  Large manifold art and scatter thumbnails
must be represented as editable PowerPoint primitives: freeform bands,
streamlines, and oval dots.  This agent refreshes that vector payload from the
original crop when the Verifier reports a surface/dotcloud residual.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class SurfaceAgent(Agent):
    """Specialist agent for surface/dotcloud vector payload repair."""

    name = "SurfaceAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el and el.get("type") in ("surface", "dotcloud"):
                return self._repair_surface(ir, original, el)

        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") in ("surface", "dotcloud"):
                changed.extend(self._repair_surface(ir, original, el))
        return changed

    def _repair_surface(self, ir: dict, original: Image.Image,
                        el: dict) -> list[str]:
        try:
            from work.diagram2ppt.v2 import vectorize as V
            from work.diagram2ppt.v2.handlers import _local_excludes
            from work.diagram2ppt.v2.native_trace import extract_paths
        except Exception:
            return []

        bbox = el.get("bbox")
        if not bbox:
            return []
        x0, y0, x1, y1 = bbox
        crop = original.crop((max(0, int(x0)), max(0, int(y0)),
                              min(original.width, int(x1)),
                              min(original.height, int(y1))))
        if crop.width < 8 or crop.height < 8:
            return []

        before = _payload_signature(el)
        excludes = _local_excludes(el, ir.get("elements", []), (x0, y0, x1, y1))

        if el.get("type") == "surface":
            wb = V.extract_wave_bands(crop, exclude=excludes)
            if wb and len(wb.get("curves", [])) >= 2:
                el["wave_bands"] = wb
                el["streamlines"] = V._synth_flow_lines(wb)
            dots = V.extract_dots(crop, exclude=excludes, round_only=True,
                                  ink_threshold=130, max_dots=300)
            paths = extract_paths(crop, exclude=excludes, max_paths=80,
                                  min_area=max(18.0, crop.width * crop.height * 0.00035),
                                  epsilon_frac=0.01, pale=True)
        else:
            dots = V.extract_dots(crop, exclude=excludes, round_only=True,
                                  ink_threshold=130, max_dots=300)
            paths = extract_paths(crop, exclude=excludes, max_paths=48,
                                  min_area=max(10.0, crop.width * crop.height * 0.0008),
                                  epsilon_frac=0.014, pale=False)

        if dots:
            el["dots"] = dots
        if paths:
            el["paths"] = paths

        el.setdefault("ext", {}).update({
            k: el[k] for k in ("dots", "streamlines", "wave_bands", "paths")
            if k in el
        })
        after = _payload_signature(el)
        if after == before:
            return []
        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "surface_vector_refresh",
            "round": ir.get("round", 0),
        })
        return [el["id"]]


def _payload_signature(el: dict) -> tuple:
    return (
        el.get("type"),
        len(el.get("dots") or []),
        len(el.get("streamlines") or []),
        len((el.get("wave_bands") or {}).get("curves") or []),
        len(el.get("paths") or []),
    )
