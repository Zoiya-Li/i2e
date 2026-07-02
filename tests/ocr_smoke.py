"""Offline test for the OCR fusion logic (no OCR engine needed). Verifies that
loose VLM text boxes get snapped to tight OCR boxes via content matching — even
when the OCR lines are offset and in a different order from the VLM elements.

    python tests/ocr_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.detect import refine_text_with_ocr  # noqa: E402


def main() -> int:
    # VLM extraction: accurate content, LOOSE/offset boxes (the real Qwen failure mode)
    elements = [
        {"type": "background", "bbox": {"x": 0, "y": 0, "w": 1080, "h": 1350}},
        {"type": "text", "text": {"content": "夏日新品"}, "bbox": {"x": 40, "y": 150, "w": 480, "h": 100},
         "extraction": {"method": "vlm"}},
        {"type": "text", "text": {"content": "清凉上市"}, "bbox": {"x": 40, "y": 300, "w": 480, "h": 100},
         "extraction": {"method": "vlm"}},
    ]
    # OCR detector: TIGHT boxes, accurate, in a DIFFERENT order
    ocr_lines = [
        {"content": "清凉上市", "bbox": {"x": 60, "y": 380, "w": 480, "h": 120}, "confidence": 0.99},
        {"content": "夏日新品", "bbox": {"x": 60, "y": 230, "w": 480, "h": 120}, "confidence": 0.99},
    ]

    refine_text_with_ocr(elements, ocr_lines)

    head = next(e for e in elements if e.get("text", {}).get("content") == "夏日新品")
    sub = next(e for e in elements if e.get("text", {}).get("content") == "清凉上市")
    assert head["bbox"] == {"x": 60, "y": 230, "w": 480, "h": 120}, head["bbox"]
    assert sub["bbox"] == {"x": 60, "y": 380, "w": 480, "h": 120}, sub["bbox"]
    assert head["extraction"]["method"] == "vlm+ocr"
    print("[OCR] content-matched both lines despite offset + reordering; boxes snapped to OCR geometry")
    print("[✓] '夏日新品' box -> (60,230,480,120) | '清凉上市' box -> (60,380,480,120) | method=vlm+ocr")
    print("\nOCR FUSION OK — VLM content + OCR geometry (the hybrid, fused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
