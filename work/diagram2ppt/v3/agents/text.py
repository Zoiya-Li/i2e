"""Text Agent: detects and repairs text elements.

Phase 1 responsibilities:
  - Re-read high-residual text boxes with OCR.
  - Adjust font size / bbox based on rendered diff.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent
from work.diagram2ppt.v3.providers import get_provider


class TextAgent(Agent):
    """Specialist agent for text extraction and repair."""

    name = "TextAgent"

    def __init__(self) -> None:
        self.provider = get_provider("ocr")

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            target_id = defect["element_id"]
            el = IR.get_element(ir, target_id)
            if el and el.get("type") == "text":
                return self._repair_text(ir, original, el)

        # No specific defect: re-run OCR on all text elements without content.
        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") == "text" and not el.get("text"):
                changed.extend(self._repair_text(ir, original, el))
        return changed

    def _repair_text(self, ir: dict, original: Image.Image,
                     el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []
        x0, y0, x1, y1 = bbox
        crop = original.crop((max(0, int(x0)), max(0, int(y0)),
                              min(original.width, int(x1)),
                              min(original.height, int(y1))))
        try:
            new_text = self.provider.ocr(crop)
        except Exception:
            return []
        new_text = (new_text or "").strip()
        if new_text and new_text != el.get("text"):
            el["text"] = new_text
            el.setdefault("repair_history", []).append({
                "agent": self.name,
                "action": "ocr_reread",
                "round": ir.get("round", 0),
                "provider": self.provider.name,
            })
            return [el["id"]]
        return []
