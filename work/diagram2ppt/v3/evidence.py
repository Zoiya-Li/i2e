"""Perception evidence blackboard for planner-led reconstruction.

Perception agents submit observations here; they do not overwrite each other.
The planner fuses evidence into v2-compatible entities only after it has the
multi-source blackboard.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


SOURCE_PRIORITY = {
    "cv_geometry": 90,
    "vlm_structure": 70,
    "vlm_text": 80,
    "ocr_text": 55,
    "legacy_decompose": 20,
}


def make(
    *,
    source: str,
    typ: str,
    bbox: list[float],
    confidence: float = 0.5,
    text: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "id": _evidence_id(source, typ, bbox, text),
        "source": source,
        "type": typ,
        "bbox": [float(v) for v in bbox[:4]],
        "confidence": float(confidence),
        "text": text,
        "payload": payload or {},
    }
    return item


def fuse_to_entities(
    evidence: list[dict[str, Any]],
    width: int,
    height: int,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Fuse multi-source evidence into v2-compatible candidate entities."""
    cleaned = [_clean_item(e, width, height) for e in evidence]
    cleaned = [e for e in cleaned if e]
    cleaned.sort(key=_rank_evidence)

    entities: list[dict[str, Any]] = []
    claims: dict[int, list[dict[str, Any]]] = {}
    for ev in cleaned:
        match_i = _find_merge_target(ev, entities)
        if match_i is None:
            ent = {
                "type": ev["type"],
                "bbox": [round(v) for v in ev["bbox"]],
                "content": None,
                "ext": {
                    "evidence_sources": [ev["source"]],
                    "evidence_ids": [ev["id"]],
                    "evidence_confidence": ev["confidence"],
                },
            }
            if ev.get("text") and ev["type"] in {"text", "formula"}:
                ent["text"] = ev["text"]
            entities.append(ent)
            claims[len(entities) - 1] = [ev]
            continue

        ent = entities[match_i]
        claims.setdefault(match_i, []).append(ev)
        ent.setdefault("ext", {}).setdefault("evidence_sources", [])
        ent["ext"].setdefault("evidence_ids", [])
        if ev["source"] not in ent["ext"]["evidence_sources"]:
            ent["ext"]["evidence_sources"].append(ev["source"])
        ent["ext"]["evidence_ids"].append(ev["id"])
        ent["ext"]["evidence_confidence"] = max(
            float(ent["ext"].get("evidence_confidence", 0.0)),
            ev["confidence"],
        )
        if ev.get("text") and ev["type"] in {"text", "formula"}:
            current = str(ent.get("text") or "")
            if _text_evidence_better(ev, current):
                ent["text"] = ev["text"]
                ent["bbox"] = [round(v) for v in ev["bbox"]]

    for idx, ent in enumerate(entities):
        ent.setdefault("ext", {})["evidence"] = _summarize_claims(claims.get(idx, []))

    by = {}
    for ent in entities:
        by[ent["type"]] = by.get(ent["type"], 0) + 1
    log(f"[PerceptionBlackboard] fused {len(evidence)} evidence -> "
        f"{len(entities)} entities {by}")
    return entities


def _clean_item(e: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    typ = str(e.get("type") or "")
    bbox = e.get("bbox")
    if typ not in {
        "container", "shape", "surface", "dotcloud", "chart",
        "arrow", "icon", "text", "formula",
    }:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    if (x1 - x0) * (y1 - y0) < 9:
        return None
    out = dict(e)
    out["bbox"] = [x0, y0, x1, y1]
    out["confidence"] = float(out.get("confidence", 0.5))
    return out


def _rank_evidence(e: dict[str, Any]) -> tuple:
    area = _area(e["bbox"])
    source_rank = -SOURCE_PRIORITY.get(str(e.get("source") or ""), 0)
    conf_rank = -float(e.get("confidence", 0.0))
    type_rank = {
        "container": 0, "chart": 1, "surface": 2, "dotcloud": 3,
        "shape": 4, "arrow": 5, "icon": 6, "formula": 7, "text": 8,
    }.get(e["type"], 9)
    return (type_rank, source_rank, conf_rank, -area)


def _find_merge_target(ev: dict[str, Any],
                       entities: list[dict[str, Any]]) -> int | None:
    best_i = None
    best_score = 0.0
    for i, ent in enumerate(entities):
        if ent.get("type") != ev["type"]:
            continue
        overlap = _iou(ev["bbox"], ent["bbox"])
        inside = max(
            _inside_frac(ev["bbox"], ent["bbox"]),
            _inside_frac(ent["bbox"], ev["bbox"]),
        )
        threshold = 0.25 if ev["type"] in {"text", "formula"} else 0.62
        score = max(overlap, inside * 0.65)
        if score >= threshold and score > best_score:
            best_i = i
            best_score = score
    return best_i


def _text_evidence_better(ev: dict[str, Any], current: str) -> bool:
    text = str(ev.get("text") or "").strip()
    if not text:
        return False
    if not current.strip():
        return True
    cur_alpha = sum(ch.isalpha() for ch in current)
    new_alpha = sum(ch.isalpha() for ch in text)
    if ev.get("source") == "vlm_text" and new_alpha >= max(2, cur_alpha):
        return True
    if ev.get("source") == "ocr_text" and len(current) <= 2 and len(text) > 2:
        return True
    return False


def _summarize_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.get("id"),
            "source": c.get("source"),
            "confidence": c.get("confidence"),
            "bbox": [round(v, 2) for v in c.get("bbox", [])],
        }
        for c in claims[:8]
    ]


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _inside_frac(inner: list[float], outer: list[float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1.0, _area(inner))


def _evidence_id(source: str, typ: str, bbox: list[float],
                 text: str | None = None) -> str:
    raw = json.dumps(
        {"s": source, "t": typ, "b": [round(float(v), 2) for v in bbox[:4]],
         "text": text or ""},
        sort_keys=True,
        ensure_ascii=False,
    )
    return "ev_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
