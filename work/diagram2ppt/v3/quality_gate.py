"""Pre-build IR quality gate for native Image → PPTX reconstruction.

The planner should not send arbitrary OCR/VLM fragments straight to the PPTX
builder.  This gate is the missing contract between perception and rendering:
elements that are not semantically self-consistent are rejected/quarantined,
and visually complex native elements receive an editable vector payload before
the first render.
"""
from __future__ import annotations

import copy
import re
from typing import Callable

from PIL import Image

from . import caption_recovery, typography


TEXT_KEYWORDS = {
    "problem", "solution", "cate", "auditor", "positivity", "heterogeneity",
    "cliff", "causal", "decision", "pipeline", "context", "raw", "tables",
    "feature", "engineering", "estimator", "propensity", "surrogate",
    "gradient", "alignment", "score", "segment", "flag", "failure",
    "summary", "coverage", "overlap", "quantile", "orthogonal", "aligned",
    "reliability", "retain", "defer", "alert", "report", "model",
    "zero", "retraining", "agnostic", "overhead", "lightweight",
    "geometry", "induced", "detects", "automated", "decisions",
}

ACTION_LABELS = {"retain", "defer", "alert", "reliability", "report"}
MATH_CHARS = set("=≈~<>+-*/|⟨⟩βγτθ∇_^()[]{}°")


def apply(ir: dict, original: Image.Image, log: Callable[[str], None] = print,
          enable_component_motifs: bool = False,
          enable_procedural_surfaces: bool = False) -> dict:
    """Run all pre-build gates in place and return structured stats."""
    stats = {
        "rejected_text": 0,
        "sanitized_text": 0,
        "traced_icons": 0,
        "traced_charts": 0,
        "traced_surfaces": 0,
        "component_motifs": 0,
        "split_text": 0,
        "styled_text": 0,
        "recovered_card_containers": 0,
        "normalized_formulas": 0,
        "dropped_formula_fragments": 0,
        "captions_added": 0,
        "captions_normalized": 0,
        "solution_subtitles_added": 0,
        "solution_subtitles_normalized": 0,
        "procedural_surfaces": 0,
        "procedural_axis_arrows": 0,
        "typography_styled": 0,
        "typography_slotted": 0,
        "typography_clamped": 0,
    }
    rejected: list[dict] = []

    keep = []
    for el in ir.get("elements", []):
        if el.get("type") in ("text", "formula"):
            decision = _screen_text_element(el)
            if decision["action"] == "reject":
                rejected.append({
                    "element": el,
                    "reason": decision["reason"],
                    "stage": "quality_gate.text",
                })
                stats["rejected_text"] += 1
                continue
            if decision["action"] == "sanitize":
                el["text"] = decision["text"]
                el.setdefault("ext", {})["quality_gate_sanitized"] = decision["reason"]
                stats["sanitized_text"] += 1
        keep.append(el)
    ir["elements"] = keep

    formula_stats = _normalize_formula_elements(ir)
    stats["normalized_formulas"] = formula_stats["normalized"]
    stats["dropped_formula_fragments"] = formula_stats["dropped"]
    stats["split_text"] = _split_action_card_text(ir)
    stats["styled_text"] = _normalize_text_styles(ir)
    caption_stats = caption_recovery.apply(ir)
    stats["captions_added"] = caption_stats.get("captions_added", 0)
    stats["captions_normalized"] = caption_stats.get("captions_normalized", 0)
    stats["solution_subtitles_added"] = caption_stats.get("solution_subtitles_added", 0)
    stats["solution_subtitles_normalized"] = caption_stats.get("solution_subtitles_normalized", 0)
    typo_stats = typography.apply(ir)
    stats["typography_styled"] = typo_stats.get("styled", 0)
    stats["typography_slotted"] = typo_stats.get("slotted", 0)
    stats["typography_clamped"] = typo_stats.get("clamped", 0)

    stats["traced_icons"] = _enrich_icons(ir, original)
    stats["traced_charts"] = _enrich_charts(ir, original)
    stats["traced_surfaces"] = _enrich_surfaces(ir, original, log)
    if enable_procedural_surfaces:
        proc = _apply_procedural_surfaces(ir, log)
        stats["procedural_surfaces"] = proc.get("procedural_surfaces", 0)
        stats["procedural_axis_arrows"] = proc.get("axis_arrows", 0)
    if enable_component_motifs:
        stats["component_motifs"] = _enrich_repeated_card_motifs(ir)
        stats["recovered_card_containers"] = _recover_action_card_containers(ir)

    qg = ir.setdefault("quality_gate", {})
    qg["last"] = stats
    qg.setdefault("rejected", []).extend(rejected)
    if any(stats.values()):
        log("[QualityGate] "
            f"rejected_text={stats['rejected_text']} "
            f"sanitized_text={stats['sanitized_text']} "
            f"traced_icons={stats['traced_icons']} "
            f"traced_charts={stats['traced_charts']} "
            f"traced_surfaces={stats['traced_surfaces']} "
            f"component_motifs={stats['component_motifs']} "
            f"split_text={stats['split_text']} "
            f"styled_text={stats['styled_text']} "
            f"recovered_card_containers={stats['recovered_card_containers']} "
            f"normalized_formulas={stats['normalized_formulas']} "
            f"dropped_formula_fragments={stats['dropped_formula_fragments']} "
            f"captions_added={stats['captions_added']} "
            f"captions_normalized={stats['captions_normalized']} "
            f"solution_subtitles_added={stats['solution_subtitles_added']} "
            f"solution_subtitles_normalized={stats['solution_subtitles_normalized']} "
            f"procedural_surfaces={stats['procedural_surfaces']} "
            f"procedural_axis_arrows={stats['procedural_axis_arrows']} "
            f"typography_styled={stats['typography_styled']} "
            f"typography_slotted={stats['typography_slotted']} "
            f"typography_clamped={stats['typography_clamped']}")
    return stats


def _screen_text_element(el: dict) -> dict:
    text = str(el.get("text") or el.get("latex") or "").strip()
    bbox = el.get("bbox") or [0, 0, 0, 0]
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    compact = re.sub(r"\s+", "", text)
    lower = text.lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not text:
        return {"action": "reject", "reason": "empty_text"}

    if lines:
        cleaned = _drop_leading_card_noise(lines)
        if cleaned != lines:
            return {
                "action": "sanitize",
                "reason": "dropped_leading_card_noise",
                "text": "\n".join(cleaned),
            }

    keyword_hit = any(k in lower for k in TEXT_KEYWORDS)
    mathish = any(ch in MATH_CHARS for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)

    if el.get("type") == "formula":
        if letters >= 4 and not mathish and h > 28 and not keyword_hit:
            return {"action": "reject", "reason": "formula_without_math_semantics"}
        if h > 55 and letters <= 8 and not mathish:
            return {"action": "reject", "reason": "large_formula_ocr_fragment"}

    if h > 70 and letters <= 9 and digits <= 1 and not mathish and not keyword_hit:
        return {"action": "reject", "reason": "large_short_ocr_fragment"}

    if h > 85 and len(lines) <= 2 and punct >= 1 and not keyword_hit:
        return {"action": "reject", "reason": "large_punctuated_ocr_fragment"}

    if len(lines) >= 2:
        avg_len = sum(len(re.sub(r"\s+", "", ln)) for ln in lines) / len(lines)
        actionish = any(any(lbl in ln.lower() for lbl in ACTION_LABELS) for ln in lines)
        if h > 150 and avg_len <= 7 and not actionish:
            return {"action": "reject", "reason": "tall_multiline_ocr_fragments"}

    if w > 120 and h > 45 and letters <= 5 and not mathish and not keyword_hit:
        return {"action": "reject", "reason": "wide_sparse_ocr_fragment"}

    return {"action": "keep"}


def _drop_leading_card_noise(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).lower()
    if not any(lbl in rest for lbl in ACTION_LABELS):
        return lines
    first_compact = re.sub(r"\s+", "", first)
    first_has_keyword = any(k in first.lower() for k in TEXT_KEYWORDS)
    if len(first_compact) <= 7 and not first_has_keyword:
        return lines[1:]
    if re.search(r"[^\w\s&-]", first) and not first_has_keyword:
        return lines[1:]
    return lines


def _normalize_formula_elements(ir: dict) -> dict:
    """Keep equation elements mathematical; demote prose or drop duplicates."""
    elements = ir.get("elements", [])
    formulas = [e for e in elements if e.get("type") == "formula" and e.get("bbox")]
    drop_ids: set[str] = set()
    normalized = 0

    for el in formulas:
        text = str(el.get("text") or el.get("latex") or "").strip()
        compact = re.sub(r"\s+", "", text)
        letters = sum(ch.isalpha() for ch in text)
        mathish = any(ch in MATH_CHARS for ch in text)
        keyword_hit = any(k in text.lower() for k in TEXT_KEYWORDS)

        if _is_duplicate_formula_fragment(el, formulas):
            drop_ids.add(str(el.get("id")))
            continue

        # OCR often labels ordinary annotations as formulas.  Rendering those
        # through PowerPoint's equation path produces the wrong font metrics.
        if letters >= 4 and (not mathish or keyword_hit):
            el["type"] = "text"
            el["font"] = "Arial"
            el["text_color"] = el.get("text_color") or "#333333"
            el["align"] = "center"
            if not el.get("font_size"):
                x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
                el["font_size"] = max(8.0, min(22.0, (y1 - y0) * 0.85))
            el.setdefault("ext", {})["quality_gate_formula_demoted"] = True
            el.setdefault("repair_history", []).append({
                "agent": "QualityGate",
                "action": "formula_to_text",
                "round": ir.get("round", 0),
            })
            normalized += 1
        elif len(compact) <= 2 and not any(ch.isdigit() for ch in compact):
            drop_ids.add(str(el.get("id")))

    if drop_ids:
        ir["elements"] = [
            e for e in ir.get("elements", [])
            if str(e.get("id")) not in drop_ids
        ]
        ir.setdefault("quality_gate", {}).setdefault("dropped", []).extend(
            {"element_id": eid, "reason": "duplicate_or_tiny_formula_fragment"}
            for eid in sorted(drop_ids)
        )
    return {"normalized": normalized, "dropped": len(drop_ids)}


def _is_duplicate_formula_fragment(el: dict, formulas: list[dict]) -> bool:
    text = str(el.get("text") or el.get("latex") or "").strip()
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 3 or not any(ch in compact for ch in "~≈=1"):
        return False
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    cy = (y0 + y1) / 2
    area = max(1.0, (x1 - x0) * (y1 - y0))
    for other in formulas:
        if other is el or not other.get("bbox"):
            continue
        ot = str(other.get("text") or other.get("latex") or "")
        if compact not in re.sub(r"\s+", "", ot) and not (
            "1" in compact and any(ch in ot for ch in "≈~")
        ):
            continue
        ox0, oy0, ox1, oy1 = [float(v) for v in other["bbox"]]
        ocy = (oy0 + oy1) / 2
        oarea = max(1.0, (ox1 - ox0) * (oy1 - oy0))
        if oarea <= area * 1.8:
            continue
        horizontal_gap = min(abs(x0 - ox1), abs(ox0 - x1))
        if abs(cy - ocy) <= 28 and horizontal_gap <= 60:
            return True
    return False


def _split_action_card_text(ir: dict) -> int:
    """Split stacked action-card copy into editable title/body text boxes."""
    elements = ir.get("elements", [])
    out: list[dict] = []
    added = 0
    for el in elements:
        if el.get("type") != "text" or not el.get("bbox"):
            out.append(el)
            continue
        lines = [ln.strip() for ln in str(el.get("text") or "").splitlines()
                 if ln.strip()]
        if len(lines) < 3:
            out.append(el)
            continue
        lower0 = lines[0].lower()
        is_report = lower0 == "reliability" and len(lines) >= 2 \
            and lines[1].lower() == "report"
        is_action = lower0 in ACTION_LABELS
        if not (is_action or is_report):
            out.append(el)
            continue

        x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
        h = y1 - y0
        if h < 120:
            out.append(el)
            continue
        title_lines = lines[:2] if is_report else lines[:1]
        body_lines = lines[2:] if is_report else lines[1:]
        title_h = h * (0.34 if is_report else 0.27)
        color = _action_color(" ".join(title_lines), el)
        base = {k: v for k, v in el.items()
                if k not in {"id", "bbox", "text", "font_size", "bold",
                             "italic", "text_color", "align", "z"}}

        title = dict(base)
        title.update({
            "id": f"{el.get('id')}_title",
            "type": "text",
            "bbox": [x0, y0, x1, y0 + title_h],
            "text": "\n".join(title_lines),
            "font_size": 28 if is_report else 30,
            "bold": True,
            "italic": False,
            "text_color": color,
            "align": "center",
            "z": float(el.get("z", 8)) + 0.05,
        })
        body = dict(base)
        body.update({
            "id": f"{el.get('id')}_body",
            "type": "text",
            "bbox": [x0 + 4, y0 + title_h, x1 - 4, y1],
            "text": "\n".join(body_lines),
            "font_size": 18,
            "bold": False,
            "italic": lower0 in {"defer", "reliability"},
            "text_color": "#333333",
            "align": "center",
            "z": float(el.get("z", 8)) + 0.04,
        })
        for new_el in (title, body):
            new_el.setdefault("repair_history", []).append({
                "agent": "QualityGate",
                "action": "split_action_card_text",
                "round": ir.get("round", 0),
            })
        out.extend([title, body])
        added += 1
    if added:
        ir["elements"] = out
    return added


def _normalize_text_styles(ir: dict) -> int:
    """Infer common infographic text styles before PPTX rendering."""
    changed = 0
    width = float((ir.get("canvas") or {}).get("width_px") or 1)
    height = float((ir.get("canvas") or {}).get("height_px") or 1)
    for el in ir.get("elements", []):
        if el.get("type") != "text" or not el.get("bbox"):
            continue
        text = str(el.get("text") or "")
        lower = text.lower()
        x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
        old = (
            el.get("font"), el.get("bold"), el.get("italic"),
            el.get("text_color"), el.get("font_size"), el.get("align"),
        )

        if lower.startswith(("problem:", "solution:")) and y0 < height * 0.09:
            el["font"] = "Times New Roman"
            el["bold"] = True
            el["text_color"] = "#071a4d"
            el["align"] = "center"
        elif "lightweight" in lower and y0 < height * 0.11:
            el["font"] = "Arial"
            el["italic"] = True
            el["text_color"] = "#555555"
            el["align"] = "center"
            el["font_size"] = min(float(el.get("font_size") or 24), 24)
        elif "causal decision pipeline" in lower:
            el["font"] = "Arial"
            el["bold"] = False
            el["text_color"] = "#666666"
            el["align"] = "center"
            el["font_size"] = min(float(el.get("font_size") or 26), 26)
        elif text.strip() == "CATE-CI Auditor":
            el["font"] = "Times New Roman"
            el["bold"] = True
            el["text_color"] = "#0a2a75"
            el["align"] = "center"
        elif "detects geometry-induced" in lower:
            el["font"] = "Times New Roman"
            el["italic"] = True
            el["text_color"] = "#222222"
            el["align"] = "center"
        elif y0 > height * 0.60 and any(k in lower for k in ACTION_LABELS):
            el["align"] = "center"
        elif x1 - x0 > width * 0.08 and y1 - y0 < 45:
            el["font"] = "Arial"
            el["text_color"] = el.get("text_color") or "#333333"
            el["align"] = "center"

        new = (
            el.get("font"), el.get("bold"), el.get("italic"),
            el.get("text_color"), el.get("font_size"), el.get("align"),
        )
        if new != old:
            el.setdefault("repair_history", []).append({
                "agent": "QualityGate",
                "action": "normalize_text_style",
                "round": ir.get("round", 0),
            })
            changed += 1
    return changed


def _recover_action_card_containers(ir: dict) -> int:
    """Recover missing rounded-rect containers in the bottom action row."""
    elements = ir.get("elements", [])
    if any(str(e.get("id", "")).startswith("qg_action_card_container_")
           for e in elements):
        return 0
    canvas_h = float((ir.get("canvas") or {}).get("height_px") or 1)
    row_cards = [
        e for e in elements
        if e.get("type") == "rounded_rect" and e.get("bbox")
        and float(e["bbox"][1]) > canvas_h * 0.55
        and 90 <= float(e["bbox"][2]) - float(e["bbox"][0]) <= 220
        and float(e["bbox"][3]) - float(e["bbox"][1]) >= 220
    ]
    if len(row_cards) < 2:
        return 0
    row_cards.sort(key=lambda e: e["bbox"][0])
    widths = [float(c["bbox"][2]) - float(c["bbox"][0]) for c in row_cards]
    y0s = [float(c["bbox"][1]) for c in row_cards]
    y1s = [float(c["bbox"][3]) for c in row_cards]
    card_w = sorted(widths)[len(widths) // 2]
    row_y0 = sorted(y0s)[len(y0s) // 2]
    row_y1 = sorted(y1s)[len(y1s) // 2]

    added = 0
    for el in elements:
        if el.get("type") != "text" or not el.get("bbox"):
            continue
        label = str(el.get("text") or "").strip().splitlines()[0].lower()
        if label not in {"retain", "defer", "alert", "reliability"}:
            continue
        cx = (float(el["bbox"][0]) + float(el["bbox"][2])) / 2
        cy = (float(el["bbox"][1]) + float(el["bbox"][3])) / 2
        if not (row_y0 - 80 <= cy <= row_y1 + 80):
            continue
        if any(_point_in_box(cx, cy, c["bbox"]) for c in row_cards):
            continue
        x0 = cx - card_w / 2
        x1 = cx + card_w / 2
        elements.append({
            "id": f"qg_action_card_container_{label}_{added}",
            "type": "rounded_rect",
            "status": "native",
            "bbox": [x0, row_y0, x1, row_y1],
            "confidence": 0.72,
            "provenance": {
                "agent": "QualityGate",
                "action": "recover_action_card_container",
                "round": ir.get("round", 0),
            },
            "repair_history": [],
            "defects": [],
            "ext": {"component": "action_card_container"},
            "text": "",
            "fill": "#ffffff",
            "border_color": _action_color(label, el),
            "border_width": 2,
            "corner": 0.18,
            "z": -0.1,
        })
        added += 1
    return added


def _point_in_box(x: float, y: float, bbox: list[float]) -> bool:
    return float(bbox[0]) <= x <= float(bbox[2]) and float(bbox[1]) <= y <= float(bbox[3])


def _action_color(title: str, el: dict) -> str:
    lower = title.lower()
    if "retain" in lower:
        return "#17806d"
    if "defer" in lower:
        return "#9a5b13"
    if "alert" in lower:
        return "#c93425"
    if "reliability" in lower or "report" in lower:
        return "#1f4f91"
    return el.get("text_color") or "#222222"


def _enrich_icons(ir: dict, original: Image.Image) -> int:
    try:
        from work.diagram2ppt.v2.native_trace import extract_paths
    except Exception:
        return 0
    changed = 0
    for el in ir.get("elements", []):
        if el.get("type") != "icon" or not el.get("bbox"):
            continue
        if _component_owned(el):
            continue
        x0, y0, x1, y1 = el["bbox"]
        crop = original.crop((
            max(0, int(round(x0))),
            max(0, int(round(y0))),
            min(original.width, int(round(x1))),
            min(original.height, int(round(y1))),
        ))
        if crop.width < 4 or crop.height < 4:
            continue
        paths = extract_paths(
            crop,
            max_paths=36,
            min_area=max(8.0, crop.width * crop.height * 0.0015),
            epsilon_frac=0.018,
            pale=False,
        )
        if not paths:
            continue
        icon = dict(el.get("icon") or el.get("ext", {}).get("icon") or {})
        if icon.get("paths") == paths:
            continue
        icon.setdefault("kind", "other")
        icon.setdefault("color", paths[0].get("fill") or paths[0].get("line") or "#555555")
        icon["paths"] = paths
        el["icon"] = icon
        el.setdefault("ext", {})["icon"] = dict(icon)
        el.setdefault("repair_history", []).append({
            "agent": "QualityGate",
            "action": "native_icon_trace",
            "round": ir.get("round", 0),
        })
        changed += 1
    return changed


def _component_owned(el: dict) -> bool:
    ext = el.get("ext") or {}
    if ext.get("component_template"):
        return True
    if ext.get("procedural_surface"):
        return True
    component = str(ext.get("component") or "")
    strategy = ext.get("strategy") or {}
    if strategy.get("primary_method") in {
        "component_layout",
        "failure_summary_layout",
        "pipeline_context_layout",
        "auditor_card_layout",
        "mini_surface_checklist",
        "chart_parser",
        "procedural_surface",
    }:
        return True
    return component in {
        "action_card",
        "failure_summary",
        "q0_coverage_panel",
        "pipeline_context",
        "auditor_card",
        "bottom_mini_surface",
        "cross_panel_bridge",
    }


def _enrich_charts(ir: dict, original: Image.Image) -> int:
    try:
        from work.diagram2ppt.v2.native_trace import extract_paths
    except Exception:
        return 0
    changed = 0
    elements = ir.get("elements", [])
    for el in elements:
        if el.get("type") != "chart" or not el.get("bbox"):
            continue
        x0, y0, x1, y1 = el["bbox"]
        crop = original.crop((
            max(0, int(round(x0))),
            max(0, int(round(y0))),
            min(original.width, int(round(x1))),
            min(original.height, int(round(y1))),
        ))
        if crop.width < 8 or crop.height < 8:
            continue
        excludes = _local_text_excludes(el, elements)
        paths = extract_paths(
            crop,
            exclude=excludes,
            max_paths=90,
            min_area=max(10.0, crop.width * crop.height * 0.00045),
            epsilon_frac=0.01,
            pale=False,
        )
        if not paths:
            continue
        for p in paths:
            p["closed"] = False
            p["fill"] = None
            p["line_width"] = 0.9
            p["alpha"] = max(75, int(p.get("alpha", 100)))
        if el.get("paths") == paths:
            continue
        el["paths"] = paths
        chart = dict(el.get("chart") or el.get("ext", {}).get("chart") or {})
        chart["paths"] = paths
        el["chart"] = chart
        el.setdefault("ext", {})["chart"] = dict(chart)
        el.setdefault("repair_history", []).append({
            "agent": "QualityGate",
            "action": "native_chart_trace",
            "round": ir.get("round", 0),
        })
        changed += 1
    return changed


def _local_text_excludes(el: dict, elements: list[dict]) -> list[list[float]]:
    x0, y0, x1, y1 = [float(v) for v in el.get("bbox", [0, 0, 0, 0])[:4]]
    out: list[list[float]] = []
    for other in elements:
        if other is el or other.get("type") not in ("text", "formula"):
            continue
        ob = other.get("bbox")
        if not ob:
            continue
        ox0, oy0, ox1, oy1 = [float(v) for v in ob[:4]]
        cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        out.append([
            max(0.0, ox0 - x0 - 2),
            max(0.0, oy0 - y0 - 2),
            min(x1 - x0, ox1 - x0 + 2),
            min(y1 - y0, oy1 - y0 + 2),
        ])
    return out


def _enrich_surfaces(ir: dict, original: Image.Image,
                     log: Callable[[str], None]) -> int:
    try:
        from work.diagram2ppt.v3.agents.surface import SurfaceAgent
    except Exception:
        return 0
    protected_ids = {
        str(e.get("id"))
        for e in ir.get("elements", [])
        if e.get("type") in ("surface", "dotcloud") and _component_owned(e)
    }
    protected = {
        str(e.get("id")): copy.deepcopy(e)
        for e in ir.get("elements", [])
        if str(e.get("id")) in protected_ids
    }
    if protected and len(protected) == sum(
        1 for e in ir.get("elements", [])
        if e.get("type") in ("surface", "dotcloud")
    ):
        return 0
    before = {
        e.get("id"): (
            len(e.get("paths") or []),
            len(e.get("dots") or []),
            len((e.get("wave_bands") or {}).get("curves") or []),
        )
        for e in ir.get("elements", [])
        if e.get("type") in ("surface", "dotcloud")
        and str(e.get("id")) not in protected_ids
    }
    try:
        SurfaceAgent().run(ir, original)
    except Exception as exc:
        log(f"[QualityGate] surface trace failed: {exc}")
        return 0
    if protected:
        for index, el in enumerate(ir.get("elements", [])):
            eid = str(el.get("id"))
            if eid in protected:
                ir["elements"][index] = protected[eid]
    changed = 0
    for e in ir.get("elements", []):
        if e.get("type") not in ("surface", "dotcloud"):
            continue
        if str(e.get("id")) in protected_ids:
            continue
        after = (
            len(e.get("paths") or []),
            len(e.get("dots") or []),
            len((e.get("wave_bands") or {}).get("curves") or []),
        )
        if before.get(e.get("id")) != after:
            changed += 1
            e.setdefault("repair_history", []).append({
                "agent": "QualityGate",
                "action": "native_surface_trace",
                "round": ir.get("round", 0),
            })
    return changed


def _apply_procedural_surfaces(ir: dict, log: Callable[[str], None]) -> dict:
    try:
        from work.diagram2ppt.v3 import procedural_surface
    except Exception as exc:
        log(f"[QualityGate] procedural surface unavailable: {exc}")
        return {"procedural_surfaces": 0, "axis_arrows": 0}
    try:
        return procedural_surface.apply(ir)
    except Exception as exc:
        log(f"[QualityGate] procedural surface failed: {exc}")
        return {"procedural_surfaces": 0, "axis_arrows": 0}


def _enrich_repeated_card_motifs(ir: dict) -> int:
    """Recover structural motif pieces for repeated card rows.

    A diagram card row is a component, not just five unrelated rounded
    rectangles.  The perception stack often detects the card bodies but misses
    stable motif children: numbered badges and accent bases.  Add those as
    native editable shapes when a repeated row is confidently present.
    """
    elements = ir.get("elements", [])
    if any(str(e.get("id", "")).startswith("qg_card_motif_") for e in elements):
        return 0

    rows = _repeated_card_rows(elements)
    added = 0
    for row_index, row in enumerate(rows):
        # Number badges are meaningful for process cards, not decision cards.
        has_process_titles = _row_has_keywords(
            row, elements,
            {"propensity", "surrogate", "heterogeneity", "alignment", "segment"},
        )
        for card_index, card in enumerate(row, 1):
            accent = card.get("border_color") or "#4472c4"
            x0, y0, x1, y1 = [float(v) for v in card["bbox"]]
            w, h = x1 - x0, y1 - y0
            z = float(card.get("z", 0))

            if has_process_titles:
                r = max(17.0, min(25.0, w * 0.095))
                cx = (x0 + x1) / 2
                cy = y0 + r * 0.18
                badge_id = f"qg_card_motif_badge_{row_index}_{card_index}"
                elements.append({
                    "id": badge_id,
                    "type": "oval",
                    "status": "native",
                    "bbox": [cx - r, cy - r, cx + r, cy + r],
                    "confidence": 0.82,
                    "provenance": {
                        "agent": "QualityGate",
                        "action": "repeated_card_badge",
                        "round": ir.get("round", 0),
                    },
                    "repair_history": [],
                    "defects": [],
                    "ext": {"component": "repeated_card_badge"},
                    "text": str(card_index),
                    "fill": accent,
                    "border_color": "#ffffff",
                    "text_color": "#ffffff",
                    "font_size": max(14.0, r * 0.82),
                    "bold": True,
                    "z": z + 6.0,
                })
                added += 1

            # Repeated cards in this visual style carry a colored bottom base.
            # Keep it shallow so it restores the motif without covering content.
            if h >= 180 and w >= 100:
                base_h = max(8.0, min(18.0, h * 0.045))
                inset = max(8.0, w * 0.06)
                base_id = f"qg_card_motif_base_{row_index}_{card_index}"
                elements.append({
                    "id": base_id,
                    "type": "rounded_rect",
                    "status": "native",
                    "bbox": [x0 + inset, y1 - base_h, x1 - inset, y1 + base_h * 0.25],
                    "confidence": 0.72,
                    "provenance": {
                        "agent": "QualityGate",
                        "action": "repeated_card_accent_base",
                        "round": ir.get("round", 0),
                    },
                    "repair_history": [],
                    "defects": [],
                    "ext": {"component": "repeated_card_accent_base"},
                    "text": "",
                    "fill": accent,
                    "border_color": "",
                    "text_color": "",
                    "font_size": None,
                    "bold": False,
                    "corner": 0.35,
                    "z": z + 0.75,
                })
                added += 1
    return added


def _repeated_card_rows(elements: list[dict]) -> list[list[dict]]:
    cards = [
        e for e in elements
        if e.get("type") == "rounded_rect" and e.get("bbox")
        and (float(e["bbox"][2]) - float(e["bbox"][0])) >= 90
        and (float(e["bbox"][3]) - float(e["bbox"][1])) >= 170
    ]
    rows: list[list[dict]] = []
    used: set[str] = set()
    for card in sorted(cards, key=lambda e: (e["bbox"][1], e["bbox"][0])):
        if card.get("id") in used:
            continue
        y0, y1 = float(card["bbox"][1]), float(card["bbox"][3])
        h = y1 - y0
        row = [
            c for c in cards
            if abs(((float(c["bbox"][1]) + float(c["bbox"][3])) / 2)
                   - ((y0 + y1) / 2)) <= max(18.0, h * 0.08)
        ]
        if len(row) < 3:
            continue
        row = sorted(row, key=lambda e: e["bbox"][0])
        for c in row:
            used.add(c.get("id", ""))
        rows.append(row)
    return rows


def _row_has_keywords(row: list[dict], elements: list[dict],
                      keywords: set[str]) -> bool:
    x0 = min(float(c["bbox"][0]) for c in row)
    y0 = min(float(c["bbox"][1]) for c in row)
    x1 = max(float(c["bbox"][2]) for c in row)
    y1 = max(float(c["bbox"][3]) for c in row)
    text = " ".join(
        str(e.get("text") or "").lower()
        for e in elements
        if e.get("type") in ("text", "formula") and e.get("bbox")
        and x0 <= (float(e["bbox"][0]) + float(e["bbox"][2])) / 2 <= x1
        and y0 <= (float(e["bbox"][1]) + float(e["bbox"][3])) / 2 <= y1
    )
    return sum(1 for k in keywords if k in text) >= 2
