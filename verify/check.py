"""Compute needs_review from evidence. Sets el.needs_review + el.ext.review
(list of reason codes). Returns the number of flagged elements.

Signals (all render-independent):
- degenerate_bbox / out_of_canvas      — geometry sanity (any element)
- background_not_full_canvas           — background should cover the canvas
- text_unverified                      — no OCR line matches this text (wrong/hallucinated/mislocated)
- box_offset                           — text matches an OCR line by content, but the boxes don't overlap (VLM geometry off)
- low_ocr_confidence                   — matched OCR line is itself uncertain
"""

from __future__ import annotations

import difflib


def _norm(s: str | None) -> str:
    return "".join((s or "").split())


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def verify_ir(ir: dict, ocr_lines: list[dict] | None = None, *,
              sim_threshold: float = 0.6, ocr_conf: float = 0.6,
              iou_min: float = 0.3, margin: int = 4) -> int:
    W, H = ir["canvas"]["width"], ir["canvas"]["height"]
    flagged = 0
    for el in ir["elements"]:
        reasons: list[str] = []
        b = el["bbox"]

        if b["w"] <= 1 or b["h"] <= 1:
            reasons.append("degenerate_bbox")
        if b["x"] < -margin or b["y"] < -margin or b["x"] + b["w"] > W + margin or b["y"] + b["h"] > H + margin:
            reasons.append("out_of_canvas")
        if el["type"] == "background" and (b["w"] < 0.98 * W or b["h"] < 0.98 * H):
            reasons.append("background_not_full_canvas")

        if el["type"] == "text" and ocr_lines is not None:
            target = _norm((el.get("text") or {}).get("content"))
            best, best_sim = None, 0.0
            for line in ocr_lines:
                ot = _norm(line.get("content"))
                s = 1.0 if (target and ot and target == ot) else difflib.SequenceMatcher(None, target, ot).ratio()
                if s > best_sim:
                    best_sim, best = s, line
            if best_sim < sim_threshold:
                reasons.append("text_unverified")
            else:
                if best.get("confidence", 1.0) < ocr_conf:
                    reasons.append("low_ocr_confidence")
                if _iou(b, best["bbox"]) < iou_min:
                    reasons.append("box_offset")

        el["needs_review"] = bool(reasons)
        if reasons:
            el.setdefault("ext", {})["review"] = reasons
            flagged += 1
        elif isinstance(el.get("ext"), dict):
            el["ext"].pop("review", None)
    return flagged
