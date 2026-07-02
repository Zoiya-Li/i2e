"""Offline test for evidence-based needs_review (no engine needed). Builds an IR
with known defects + OCR lines, and asserts the verifier flags the right ones
with the right reasons.

    python tests/verify_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify.check import verify_ir  # noqa: E402


def main() -> int:
    ir = {
        "canvas": {"width": 1080, "height": 1350},
        "elements": [
            # background that does NOT cover the canvas -> flag (the real Qwen defect)
            {"id": "background-1", "type": "background", "bbox": {"x": 0, "y": 0, "w": 1000, "h": 1333},
             "needs_review": False},
            # text with matching OCR line AND aligned box -> clean
            {"id": "text-1", "type": "text", "text": {"content": "夏日新品"},
             "bbox": {"x": 60, "y": 230, "w": 480, "h": 120}, "needs_review": False},
            # text matching OCR by content but box OFFSET (low IoU) -> box_offset
            {"id": "text-2", "type": "text", "text": {"content": "清凉上市"},
             "bbox": {"x": 60, "y": 60, "w": 480, "h": 120}, "needs_review": False},
            # text with no OCR match -> text_unverified
            {"id": "text-3", "type": "text", "text": {"content": "联系我们"},
             "bbox": {"x": 60, "y": 900, "w": 300, "h": 60}, "needs_review": False},
        ],
    }
    ocr_lines = [
        {"content": "夏日新品", "bbox": {"x": 62, "y": 233, "w": 472, "h": 123}, "confidence": 0.99},
        {"content": "清凉上市", "bbox": {"x": 60, "y": 380, "w": 476, "h": 125}, "confidence": 0.99},
    ]

    flagged = verify_ir(ir, ocr_lines=ocr_lines)
    by = {e["id"]: e for e in ir["elements"]}

    def reasons(eid):
        return (by[eid].get("ext") or {}).get("review", [])

    assert reasons("background-1") == ["background_not_full_canvas"], reasons("background-1")
    assert by["text-1"]["needs_review"] is False and reasons("text-1") == [], reasons("text-1")
    assert reasons("text-2") == ["box_offset"], reasons("text-2")
    assert reasons("text-3") == ["text_unverified"], reasons("text-3")
    assert flagged == 3, flagged

    print("[verify] flagged 3/4:")
    for eid in ("background-1", "text-2", "text-3"):
        print(f"    {eid}: {reasons(eid)}")
    print("[✓] aligned text (text-1) correctly NOT flagged; reasons explain each flag")
    print("\nVERIFY SMOKE OK — needs_review now from evidence, not self-reported confidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
