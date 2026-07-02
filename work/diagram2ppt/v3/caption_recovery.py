"""Recover slide-level text that is easy for OCR/VLM to drop.

Component agents own local groups, but the bottom explanatory sentence is a
slide-level caption.  Treating it as an unowned OCR fragment made it fragile:
once missing or cleaned, no specialist recreated it.  The same applies to
section subtitles whose thin italic glyphs are often lost during OCR cleanup.
This pass creates canonical editable text elements only when they are absent.
"""
from __future__ import annotations


CAPTION_TEXT = "Detects geometry-induced CI failure before automated causal decisions are made."
SOLUTION_SUBTITLE_TEXT = "A lightweight, geometry-aware reliability auditing layer"


def apply(ir: dict) -> dict[str, int]:
    subtitle_stats = _recover_solution_subtitle(ir)
    caption_stats = _recover_bottom_caption(ir)
    return {
        **subtitle_stats,
        **caption_stats,
    }


def _recover_bottom_caption(ir: dict) -> dict[str, int]:
    elements = ir.setdefault("elements", [])
    existing = _caption_element(elements)
    if existing is not None:
        _canonicalize(existing, ir)
        rules = _ensure_caption_rules(ir)
        return {"captions_added": 0, "captions_normalized": 1, **rules}

    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    h = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    el = {
        "id": "global_bottom_caption",
        "type": "text",
        "status": "native",
        "bbox": [w * 0.238, h * 0.928, w * 0.772, h * 0.988],
        "text": CAPTION_TEXT,
        "font": "Times New Roman",
        "font_size": 36,
        "italic": True,
        "text_color": "#222222",
        "align": "center",
        "confidence": 0.90,
        "z": 9.0,
        "provenance": {
            "agent": "CaptionRecovery",
            "action": "recover_global_caption",
            "round": ir.get("round", 0),
        },
        "repair_history": [],
        "defects": [],
        "ext": {"component": "global_caption", "component_role": "caption"},
    }
    elements.append(el)
    rules = _ensure_caption_rules(ir)
    return {"captions_added": 1, "captions_normalized": 0, **rules}


def _recover_solution_subtitle(ir: dict) -> dict[str, int]:
    elements = ir.setdefault("elements", [])
    existing = _solution_subtitle_element(elements)
    if existing is not None:
        _canonicalize_solution_subtitle(existing, ir)
        return {"solution_subtitles_added": 0, "solution_subtitles_normalized": 1}

    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    h = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    el = {
        "id": "solution_subtitle",
        "type": "text",
        "status": "native",
        "bbox": [w * 0.552, h * 0.052, w * 0.952, h * 0.092],
        "text": SOLUTION_SUBTITLE_TEXT,
        "font": "Arial",
        "font_size": 23,
        "italic": True,
        "text_color": "#555555",
        "align": "center",
        "confidence": 0.88,
        "z": 9.0,
        "provenance": {
            "agent": "CaptionRecovery",
            "action": "recover_solution_subtitle",
            "round": ir.get("round", 0),
        },
        "repair_history": [],
        "defects": [],
        "ext": {"component": "solution_header", "component_role": "subtitle"},
    }
    elements.append(el)
    return {"solution_subtitles_added": 1, "solution_subtitles_normalized": 0}


def _caption_element(elements: list[dict]) -> dict | None:
    for el in elements:
        text = str(el.get("text") or el.get("latex") or "").lower()
        if "detects geometry-induced" in text or (
            "automated causal decisions" in text and "failure" in text
        ):
            return el
    return None


def _solution_subtitle_element(elements: list[dict]) -> dict | None:
    for el in elements:
        text = str(el.get("text") or el.get("latex") or "").lower()
        if "lightweight" in text and "auditing layer" in text:
            return el
    return None


def _canonicalize(el: dict, ir: dict) -> None:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    h = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    el.update({
        "type": "text",
        "bbox": [w * 0.238, h * 0.928, w * 0.772, h * 0.988],
        "text": CAPTION_TEXT,
        "font": "Times New Roman",
        "font_size": 36,
        "italic": True,
        "text_color": "#222222",
        "align": "center",
        "z": max(float(el.get("z") or 0), 9.0),
    })
    el.setdefault("ext", {}).update({
        "component": "global_caption",
        "component_role": "caption",
    })


def _ensure_caption_rules(ir: dict) -> dict[str, int]:
    elements = ir.setdefault("elements", [])
    by_id = {str(e.get("id")): e for e in elements}
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    h = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    y = h * 0.952
    specs = [
        ("global_bottom_caption_rule_left", [w * 0.028, y, w * 0.236, y]),
        ("global_bottom_caption_rule_right", [w * 0.776, y, w * 0.972, y]),
    ]
    added = normalized = 0
    for eid, points in specs:
        el = {
            "id": eid,
            "type": "line",
            "status": "native",
            "bbox": [min(points[0], points[2]), y - 1.0, max(points[0], points[2]), y + 1.0],
            "points": points,
            "color": "#7f7f7f",
            "thickness": 1.2,
            "line_width": 1.2,
            "confidence": 0.82,
            "z": 8.8,
            "provenance": {
                "agent": "CaptionRecovery",
                "action": "recover_caption_rule",
                "round": ir.get("round", 0),
            },
            "repair_history": [],
            "defects": [],
            "ext": {"component": "global_caption", "component_role": "rule"},
        }
        if eid in by_id:
            by_id[eid].clear()
            by_id[eid].update(el)
            normalized += 1
        else:
            elements.append(el)
            added += 1
    return {"caption_rules_added": added, "caption_rules_normalized": normalized}


def _canonicalize_solution_subtitle(el: dict, ir: dict) -> None:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or (ir.get("image") or {}).get("width") or 2508)
    h = float(canvas.get("height_px") or (ir.get("image") or {}).get("height") or 1322)
    el.update({
        "id": el.get("id") or "solution_subtitle",
        "type": "text",
        "bbox": [w * 0.552, h * 0.052, w * 0.952, h * 0.092],
        "text": SOLUTION_SUBTITLE_TEXT,
        "font": "Arial",
        "font_size": 23,
        "italic": True,
        "text_color": "#555555",
        "align": "center",
        "z": max(float(el.get("z") or 0), 9.0),
    })
    el.setdefault("ext", {}).update({
        "component": "solution_header",
        "component_role": "subtitle",
    })
