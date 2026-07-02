"""Formula Agent: re-recognizes math expressions with a local OCR model.

The v2 pipeline delegates formula recognition to a VLM, which frequently
mistranscribes Greek letters and math symbols.  FormulaAgent repairs those
elements using the locally cached pix2text-mfr ONNX model.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent
from work.diagram2ppt.v3.providers.local import LocalModelProvider


class FormulaAgent(Agent):
    """Specialist agent for math-formula recognition and repair."""

    name = "FormulaAgent"

    def __init__(self) -> None:
        self.provider = LocalModelProvider(model_name="pix2tex")

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el and el.get("type") == "formula":
                return self._repair_formula(ir, original, el)

        # Proactive pass: re-recognize every formula element.
        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") == "formula":
                changed.extend(self._repair_formula(ir, original, el))
        return changed

    def _repair_formula(self, ir: dict, original: Image.Image,
                        el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []

        crop = _padded_crop(original, bbox, pad=0.08)
        if crop.width < 8 or crop.height < 8:
            return []

        try:
            latex = self.provider.ocr(crop)
        except Exception:
            return []

        latex = (latex or "").strip()
        if not latex:
            return []

        old_text = el.get("text", "")
        old_latex = el.get("ext", {}).get("latex", "")
        if latex == old_text or latex == old_latex:
            return []

        el["text"] = latex
        el.setdefault("ext", {})["latex"] = latex
        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "formula_reread",
            "round": ir.get("round", 0),
            "provider": self.provider.name,
            "model": self.provider.model_name,
        })
        return [el["id"]]


def _padded_crop(image: Image.Image, bbox: list[float],
                 pad: float = 0.08) -> Image.Image:
    """Crop a formula region with a small context padding."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    dx, dy = w * pad, h * pad
    left = max(0, int(x0 - dx))
    top = max(0, int(y0 - dy))
    right = min(image.width, int(x1 + dx))
    bottom = min(image.height, int(y1 + dy))
    return image.crop((left, top, right, bottom))
