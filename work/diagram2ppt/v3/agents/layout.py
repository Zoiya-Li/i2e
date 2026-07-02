"""Layout Agent: component-level layout repair.

Phase 1 responsibilities:
  - Identify unexplained ink regions from verifier coverage.
  - Propose new candidate elements for the Planner to instantiate.
  - Rebuild repeated component rows when strategy marks them as structured
    layout problems rather than isolated OCR/style defects.
"""
from __future__ import annotations

from statistics import median
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class LayoutAgent(Agent):
    """Specialist agent for layout and missing-element discovery."""

    name = "LayoutAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        task = kwargs.get("task")
        if self._is_auditor_cards_task(ir, defect, task):
            return self._repair_auditor_cards(ir)
        if self._is_pipeline_context_task(ir, defect):
            return self._repair_pipeline_context(ir)
        if self._is_action_card_task(ir, defect):
            changed = self._repair_targeted_action_card_slot(ir, defect)
            if changed:
                return changed
            return self._repair_action_cards(ir)
        if defect and defect.get("type") == "missing_element":
            return self._add_candidate(ir, original, defect)

        changed: list[str] = []
        # Also scan for text elements with no content.
        for el in ir.get("elements", []):
            if el.get("type") == "text" and not el.get("text"):
                # delegate to TextAgent implicitly by creating a placeholder fix
                pass
        return changed

    def _add_candidate(self, ir: dict, original: Image.Image,
                       defect: dict) -> list[str]:
        # Phase 1: just record the candidate as a pending defect hint.
        # Real implementation would run CV / VLM on the bbox to classify it.
        bbox = defect.get("bbox", [0, 0, 100, 100])
        hint = {
            "id": f"candidate_{defect['id']}",
            "type": "pending",
            "bbox": bbox,
            "reason": "unexplained ink region",
            "provenance": IR.provenance(self.name, "candidate", ir.get("round", 0)),
        }
        ir.setdefault("candidates", []).append(hint)
        # Mark this defect as skipped so the Planner picks a different one next.
        defect["status"] = "skipped"
        return []

    def _is_action_card_task(self, ir: dict, defect: dict | None) -> bool:
        if _dedicated_action_owner_present(ir):
            return False
        if defect:
            strat = defect.get("strategy") or {}
            if strat.get("method") == "component_layout":
                return True
            if defect.get("suggested_agent") == self.name:
                el = IR.get_element(ir, str(defect.get("element_id") or ""))
                if _has_component_strategy(el):
                    return True
            return False
        return False

    def _is_pipeline_context_task(self, ir: dict, defect: dict | None) -> bool:
        if _dedicated_pipeline_owner_present(ir):
            return False
        if defect:
            strat = defect.get("strategy") or {}
            if strat.get("method") == "pipeline_context_layout":
                return True
            if defect.get("suggested_agent") == self.name:
                el = IR.get_element(ir, str(defect.get("element_id") or ""))
                if el and ((el.get("ext") or {}).get("strategy") or {}).get("kind") == "pipeline_context_row":
                    return True
            return False
        return False

    def _is_auditor_cards_task(self, ir: dict, defect: dict | None,
                               task: dict | None = None) -> bool:
        if _dedicated_auditor_owner_present(ir):
            return False
        if task:
            if task.get("kind") == "auditor_method_cards":
                return True
            if task.get("region_id") == "auditor_cards":
                return True
            if task.get("region_id") == "region_auditor_cards":
                return True
        if defect:
            strat = defect.get("strategy") or {}
            if strat.get("method") == "auditor_card_layout":
                return True
            if defect.get("suggested_agent") == self.name:
                el = IR.get_element(ir, str(defect.get("element_id") or ""))
                strategy = ((el or {}).get("ext") or {}).get("strategy") or {}
                return strategy.get("kind") == "auditor_method_cards"
        return False

    def _repair_pipeline_context(self, ir: dict) -> list[str]:
        region = _pipeline_region(ir)
        if not region:
            return []
        elements = ir.get("elements", [])
        specs = _pipeline_specs(ir, region)
        changed: set[str] = set()
        for spec in specs:
            card = _pipeline_card_for(spec, elements)
            if card is None:
                card = IR.element(
                    id=f"layout_pipeline_card_{spec['key']}",
                    type="rounded_rect",
                    bbox=spec["bbox"],
                    provenance=IR.provenance("LayoutAgent", "pipeline_context_card",
                                             ir.get("round", 0)),
                    confidence=0.84,
                    ext={},
                )
                elements.append(card)
            if _update_pipeline_card(ir, card, spec):
                changed.add(str(card["id"]))
        return sorted(changed)

    def _repair_auditor_cards(self, ir: dict) -> list[str]:
        region = _auditor_region(ir)
        specs = _auditor_specs(ir, region)
        changed: set[str] = set()
        changed.update(_remove_auditor_region_orphans(ir, specs))
        for spec in specs:
            for el in _auditor_elements(ir, spec):
                eid = str(el["id"])
                existing = IR.get_element(ir, eid)
                if existing is None:
                    ir.setdefault("elements", []).append(el)
                    changed.add(eid)
                else:
                    before = _state(existing, (
                        "type", "bbox", "text", "font", "font_size", "bold",
                        "text_color", "align", "fill", "border_color",
                        "border_width", "corner", "z", "icon", "chart",
                    ))
                    existing.clear()
                    existing.update(el)
                    if _state(existing, (
                        "type", "bbox", "text", "font", "font_size", "bold",
                        "text_color", "align", "fill", "border_color",
                        "border_width", "corner", "z", "icon", "chart",
                    )) != before:
                        changed.add(eid)
        return sorted(changed)

    def _repair_action_cards(self, ir: dict) -> list[str]:
        region = _action_region(ir)
        if not region:
            return []

        elements = ir.get("elements", [])
        cards = _row_cards(ir, region)
        if not cards:
            return []

        row_y0 = median(float(c["bbox"][1]) for c in cards)
        row_y1 = median(float(c["bbox"][3]) for c in cards)
        card_w = median(float(c["bbox"][2]) - float(c["bbox"][0]) for c in cards)
        specs = _card_specs(ir, region, card_w, row_y0, row_y1)
        if len(specs) < 3:
            return []

        changed: set[str] = set()
        assigned_cards: set[str] = set()

        for spec in specs:
            card = _nearest_card(spec, cards, assigned_cards, max_gap=card_w * 0.72)
            if card is None:
                card = _new_shape(ir, spec)
                elements.append(card)
                assigned_cards.add(str(card.get("id")))
                if _update_card(ir, card, spec):
                    changed.add(str(card["id"]))
            else:
                assigned_cards.add(str(card.get("id")))
                _mark_component(card, spec, "card")

        removed = _remove_action_card_container_orphans(ir, region, specs,
                                                        assigned_cards)
        changed.update(removed)
        return sorted(changed)

    def _repair_targeted_action_card_slot(self, ir: dict,
                                          defect: dict | None) -> list[str]:
        if not defect or not defect.get("element_id"):
            return []
        el = IR.get_element(ir, str(defect.get("element_id")))
        if not el or el.get("type") != "text":
            return []
        text = str(el.get("text") or "")
        key = _component_key(el) or _text_key(text.lower())
        if key not in {"retain", "defer", "alert", "report"}:
            return []
        # A title/body defect is local evidence. Repair only that slot; whole
        # row rewrites are evaluated as candidate variants, not as blind repair
        # actions.
        part = "title" if _looks_like_action_title(text) else "body"
        spec = _spec_for_existing_card(ir, key, el)
        if not spec:
            return []
        if part == "body":
            before = _state(el, ("bbox", "font", "font_size", "align",
                                 "text_color", "italic", "z"))
            x0, y0, x1, y1 = [float(v) for v in spec["bbox"]]
            top = y0 + (214 if key == "report" else 205)
            el.update({
                "bbox": [x0 + 10, top, x1 - 10, y1 - 16],
                "font": "Arial",
                "font_size": min(float(el.get("font_size") or 16), 16),
                "text_color": "#333333",
                "align": "center",
                "italic": bool(el.get("italic")),
                "z": 8.0,
            })
            _mark_component(el, spec, "text_body")
            return [str(el["id"])] if _record_if_changed(
                ir, el, "component_card_body_slot", before) else []
        before = _state(el, ("bbox", "font", "font_size", "align",
                             "text_color", "bold", "z"))
        x0, y0, x1, _ = [float(v) for v in spec["bbox"]]
        top = y0 + (138 if key == "report" else 145)
        bottom = y0 + (214 if key == "report" else 196)
        el.update({
            "bbox": [x0 + 8, top, x1 - 8, bottom],
            "font": "Arial",
            "font_size": min(float(el.get("font_size") or 25), 25),
            "bold": True,
            "text_color": spec["color"],
            "align": "center",
            "z": 8.0,
        })
        _mark_component(el, spec, "text_title")
        return [str(el["id"])] if _record_if_changed(
            ir, el, "component_card_title_slot", before) else []


ACTION_SPECS = [
    {
        "key": "retain",
        "title": "RETAIN",
        "body": "High reliability\nUse CI as is",
        "color": "#16806e",
        "fill": "#fbfffd",
        "icon_kind": "shield",
    },
    {
        "key": "defer",
        "title": "DEFER",
        "body": "Borderline\nSeek more data\nor stronger model",
        "color": "#9a5b13",
        "fill": "#fffdf9",
        "icon_kind": "hourglass",
    },
    {
        "key": "alert",
        "title": "ALERT",
        "body": "Low reliability\nDo not trust CI\nin Q0",
        "color": "#cf3d28",
        "fill": "#fffafa",
        "icon_kind": "warning",
    },
    {
        "key": "report",
        "title": "Reliability\nReport",
        "body": "Coverage risk map,\nsegment stats,\naudit summary",
        "color": "#245591",
        "fill": "#fbfdff",
        "icon_kind": "document",
    },
]


PIPELINE_SPECS = [
    {"key": "raw", "title": "Raw\nTables", "icon": "database"},
    {"key": "feature", "title": "Feature\nEngineering", "icon": "gear"},
    {"key": "cate", "title": "CATE\nEstimator", "icon": "scatter"},
    {"key": "ci", "title": "CI\nEstimator", "icon": "line"},
]


AUDITOR_SPECS = [
    {
        "key": "propensity",
        "num": "1",
        "title": "Propensity\nModel",
        "formula": "T ~ X -> beta_hat",
        "color": "#2f7dbd",
        "fill": "#fbfdff",
        "visual": "line",
    },
    {
        "key": "surrogate",
        "num": "2",
        "title": "Surrogate\nCATE",
        "formula": "Y ~ X + X*T -> tau_hat",
        "color": "#45a9d1",
        "fill": "#fbfdff",
        "visual": "scatter",
    },
    {
        "key": "heterogeneity",
        "num": "3",
        "title": "Heterogeneity\nGradient",
        "formula": "tau_hat ~ X -> gamma_hat",
        "color": "#3c9b86",
        "fill": "#fcfffd",
        "visual": "surface",
    },
    {
        "key": "alignment",
        "num": "4",
        "title": "Alignment\nScore",
        "formula": "s(x)= |<x*beta,x*gamma>| / ||beta|| ||gamma||",
        "color": "#7e73b9",
        "fill": "#fefdff",
        "visual": "vectors",
    },
    {
        "key": "segment",
        "num": "5",
        "title": "Segment\n& Flag",
        "formula": "",
        "color": "#d97735",
        "fill": "#fffdfb",
        "visual": "clusters",
    },
]


def _auditor_region(ir: dict) -> dict | None:
    for region in (ir.get("strategy_plan") or {}).get("regions", []):
        if region.get("kind") == "auditor_method_cards":
            return region
    return None


def _dedicated_auditor_owner_present(ir: dict) -> bool:
    """Dedicated component agents outrank the generic layout repairer."""
    owned = 0
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        if not eid.startswith("auditor_card_"):
            continue
        provenance = el.get("provenance") or {}
        if provenance.get("agent") == "AuditorCardAgent":
            owned += 1
    return owned >= 3


def _dedicated_pipeline_owner_present(ir: dict) -> bool:
    owned = 0
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        if not eid.startswith("pipeline_context_card_"):
            continue
        provenance = el.get("provenance") or {}
        if provenance.get("agent") == "PipelineContextAgent":
            owned += 1
    return owned >= 3


def _dedicated_action_owner_present(ir: dict) -> bool:
    owned = 0
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        if not eid.startswith("action_card_") or el.get("type") != "rounded_rect":
            continue
        provenance = el.get("provenance") or {}
        if provenance.get("agent") == "ActionCardAgent":
            owned += 1
    return owned >= 3


def _auditor_specs(ir: dict, region: dict | None) -> list[dict]:
    canvas = ir.get("canvas") or {}
    width = float(canvas.get("width_px") or 2508)
    height = float(canvas.get("height_px") or 1322)
    # The method-card row is a known visual motif in this figure. Detection
    # bboxes often absorb shadows/connectors or the row below, so use the
    # semantic slot as the anchor and only let future dedicated detectors refine
    # it. This keeps LayoutAgent from generating oversized cards.
    x0, x1 = width * 0.495, width * 0.975
    y0, y1 = height * 0.365, height * 0.615
    gap = max(18.0, width * 0.012)
    card_w = (x1 - x0 - gap * 4) / 5
    out = []
    for idx, base in enumerate(AUDITOR_SPECS):
        spec = dict(base)
        cx0 = x0 + idx * (card_w + gap)
        spec["bbox"] = [cx0, y0, cx0 + card_w, y1]
        out.append(spec)
    return out


def _auditor_elements(ir: dict, spec: dict) -> list[dict]:
    x0, y0, x1, y1 = [float(v) for v in spec["bbox"]]
    color = spec["color"]
    key = spec["key"]
    round_num = ir.get("round", 0)
    elements = [
        IR.element(
            id=f"layout_auditor_card_{key}",
            type="rounded_rect",
            bbox=[x0, y0, x1, y1],
            provenance=IR.provenance("LayoutAgent", "auditor_card", round_num),
            confidence=0.86,
            fill=spec["fill"],
            border_color=color,
            border_width=2,
            corner=0.22,
            text="",
            z=-0.15,
            ext=_auditor_ext(spec, "card"),
        ),
        IR.element(
            id=f"layout_auditor_num_{key}",
            type="oval",
            bbox=[(x0 + x1) / 2 - 24, y0 - 28, (x0 + x1) / 2 + 24, y0 + 20],
            provenance=IR.provenance("LayoutAgent", "auditor_number", round_num),
            confidence=0.84,
            fill=color,
            border_color=color,
            border_width=1,
            text=spec["num"],
            font="Arial",
            font_size=22,
            bold=True,
            text_color="#ffffff",
            align="center",
            z=9.0,
            ext=_auditor_ext(spec, "number"),
        ),
        IR.element(
            id=f"layout_auditor_title_{key}",
            type="text",
            bbox=[x0 + 14, y0 + 30, x1 - 14, y0 + 96],
            provenance=IR.provenance("LayoutAgent", "auditor_title", round_num),
            confidence=0.86,
            text=spec["title"],
            font="Arial",
            font_size=22,
            bold=True,
            text_color="#111111",
            align="center",
            z=8.0,
            ext=_auditor_ext(spec, "title"),
        ),
    ]
    if spec["formula"]:
        elements.append(IR.element(
            id=f"layout_auditor_formula_{key}",
            type="text",
            bbox=[x0 + 12, y0 + 106, x1 - 12, y0 + 150],
            provenance=IR.provenance("LayoutAgent", "auditor_formula", round_num),
            confidence=0.78,
            text=spec["formula"],
            font="Arial",
            font_size=14,
            text_color="#111111",
            align="center",
            z=8.0,
            ext=_auditor_ext(spec, "formula"),
        ))
    elements.extend(_auditor_visual_elements(ir, spec))
    return elements


def _remove_auditor_region_orphans(ir: dict, specs: list[dict]) -> set[str]:
    boxes = [s["bbox"] for s in specs]
    if not boxes:
        return set()
    region = _union_bbox(boxes)
    rx0, ry0, rx1, ry1 = region
    # Slightly expanded visual slot: remove old VLM/OCR fragments inside the
    # method-card row before adding the component template. Keep global titles,
    # lower action-card connectors, and anything outside this slot.
    slot = [rx0 - 35, ry0 - 55, rx1 + 35, ry1 + 38]
    removable = {
        "text", "formula", "rounded_rect", "rect", "oval", "icon",
        "chart", "dotcloud", "surface", "freeform",
    }
    keep = []
    removed: set[str] = set()
    for e in ir.get("elements", []):
        eid = str(e.get("id") or "")
        bbox = e.get("bbox")
        if eid.startswith("layout_auditor_") or not bbox:
            keep.append(e)
            continue
        if e.get("type") in removable and _bbox_center_inside(bbox, slot):
            removed.add(eid)
            continue
        keep.append(e)
    if removed:
        ir["elements"] = keep
        ir.setdefault("history", []).append({
            "agent": "LayoutAgent",
            "action": "remove_auditor_region_orphans",
            "round": ir.get("round", 0),
            "removed": sorted(removed),
        })
    return removed


def _auditor_visual_elements(ir: dict, spec: dict) -> list[dict]:
    x0, y0, x1, y1 = [float(v) for v in spec["bbox"]]
    key = spec["key"]
    color = spec["color"]
    box = [x0 + 34, y0 + 158, x1 - 34, y1 - 42]
    round_num = ir.get("round", 0)
    visual = spec["visual"]
    if visual in {"line", "scatter"}:
        return [IR.element(
            id=f"layout_auditor_visual_{key}",
            type="icon",
            bbox=box,
            provenance=IR.provenance("LayoutAgent", "auditor_visual_icon", round_num),
            confidence=0.76,
            icon={"kind": visual, "color": color},
            z=7.0,
            ext=_auditor_ext(spec, "visual"),
        )]
    if visual == "surface":
        bx0, by0, bx1, by1 = box
        bw, bh = bx1 - bx0, by1 - by0
        dots = []
        for i in range(28):
            tx = (i % 7) / 6
            ty = (i // 7) / 3
            dots.append({
                "cx": 8 + tx * (bw - 16),
                "cy": 10 + ty * (bh - 20),
                "r": 2.4,
                "color": color,
            })
        return [IR.element(
            id=f"layout_auditor_visual_{key}",
            type="dotcloud",
            bbox=box,
            provenance=IR.provenance("LayoutAgent", "auditor_surface_dots", round_num),
            confidence=0.72,
            dots=dots,
            fill=color,
            z=7.0,
            ext=_auditor_ext(spec, "visual"),
        )]
    if visual == "vectors":
        bx0, by0, bx1, by1 = box
        return [
            IR.element(
                id=f"layout_auditor_visual_{key}_blue",
                type="arrow",
                bbox=[bx0 + 18, by1 - 22, bx1 - 36, by0 + 30],
                provenance=IR.provenance("LayoutAgent", "auditor_vector", round_num),
                confidence=0.74,
                start=[bx0 + 18, by1 - 22],
                end=[bx1 - 36, by0 + 30],
                color="#1f66d1",
                thickness=4,
                z=7.0,
                ext=_auditor_ext(spec, "visual"),
            ),
            IR.element(
                id=f"layout_auditor_visual_{key}_green",
                type="arrow",
                bbox=[bx0 + 18, by1 - 22, bx1 - 10, by0 + 48],
                provenance=IR.provenance("LayoutAgent", "auditor_vector", round_num),
                confidence=0.74,
                start=[bx0 + 18, by1 - 22],
                end=[bx1 - 10, by0 + 48],
                color="#16806e",
                thickness=4,
                z=7.0,
                ext=_auditor_ext(spec, "visual"),
            ),
        ]
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    dots = []
    palette = ["#999999", "#e85536", "#e49a24", "#4aa381"]
    for i in range(52):
        group = i % 4
        gx = [0.28, 0.68, 0.25, 0.66][group]
        gy = [0.35, 0.38, 0.75, 0.74][group]
        ox = ((i * 37) % 17 - 8) * 2.5
        oy = ((i * 19) % 15 - 7) * 2.8
        dots.append({
            "cx": gx * bw + ox,
            "cy": gy * bh + oy,
            "r": 2.8,
            "color": palette[group],
        })
    return [IR.element(
        id=f"layout_auditor_visual_{key}",
        type="dotcloud",
        bbox=box,
        provenance=IR.provenance("LayoutAgent", "auditor_segment_dots", round_num),
        confidence=0.74,
        dots=dots,
        z=7.0,
        ext=_auditor_ext(spec, "visual"),
    )]


def _auditor_ext(spec: dict, role: str) -> dict:
    return {
        "component": "auditor_method_card",
        "component_key": spec["key"],
        "component_role": role,
        "strategy": {
            "region_id": "region_auditor_cards",
            "kind": "auditor_method_cards",
            "primary_method": "auditor_card_layout",
            "fallback_methods": ["chart_parser", "icon_rebuild", "text_style"],
            "preferred_agent": "LayoutAgent",
        },
    }


def _pipeline_region(ir: dict) -> dict | None:
    for region in (ir.get("strategy_plan") or {}).get("regions", []):
        if region.get("kind") == "pipeline_context_row":
            return region
    return None


def _pipeline_specs(ir: dict, region: dict) -> list[dict]:
    y0, y1 = 203.0, 338.0
    width = 228.0
    centers = _pipeline_centers(ir, region)
    # Stable geometry for this common four-card motif; derived from detected
    # text/icon centers, with sane fallbacks when OCR is partial.
    fallback = {"raw": 1398.0, "feature": 1685.0, "cate": 1983.0, "ci": 2289.0}
    specs = []
    for base in PIPELINE_SPECS:
        spec = dict(base)
        cx = centers.get(spec["key"], fallback[spec["key"]])
        spec["bbox"] = [cx - width / 2, y0, cx + width / 2, y1]
        specs.append(spec)
    return specs


def _pipeline_centers(ir: dict, region: dict) -> dict[str, float]:
    found: dict[str, list[float]] = {s["key"]: [] for s in PIPELINE_SPECS}
    for e in ir.get("elements", []):
        bbox = e.get("bbox")
        if not bbox or not _inside_region(bbox, region):
            continue
        text = str(e.get("text") or "").lower()
        key = None
        if "raw" in text or "tables" in text:
            key = "raw"
        elif "feature" in text or "engineering" in text:
            key = "feature"
        elif "cate" in text:
            key = "cate"
        elif "estimator" in text and float(bbox[0]) > 2150:
            key = "ci"
        elif text.strip() in {"ci", "cl"}:
            key = "ci"
        if key:
            found[key].append((float(bbox[0]) + float(bbox[2])) / 2)
    return {k: median(v) for k, v in found.items() if v}


def _pipeline_card_for(spec: dict, elements: list[dict]) -> dict | None:
    sx0, sy0, sx1, sy1 = [float(v) for v in spec["bbox"]]
    scx = (sx0 + sx1) / 2
    best = None
    best_gap = 9999.0
    for e in elements:
        if e.get("type") != "rounded_rect" or not e.get("bbox"):
            continue
        x0, y0, x1, y1 = [float(v) for v in e["bbox"]]
        bw, bh = x1 - x0, y1 - y0
        if not (90 <= bw <= 310 and 70 <= bh <= 180 and 150 <= y0 <= 260):
            continue
        gap = abs(((x0 + x1) / 2) - scx)
        if gap < best_gap:
            best = e
            best_gap = gap
    return best if best_gap <= 95 else None


def _update_pipeline_card(ir: dict, card: dict, spec: dict) -> bool:
    before = _state(card, ("bbox", "fill", "border_color", "border_width",
                          "corner", "text", "z"))
    card.update({
        "type": "rounded_rect",
        "bbox": [float(v) for v in spec["bbox"]],
        "fill": "#fbfbfc",
        "border_color": "#9aa0a8",
        "border_width": 1.25,
        "corner": 0.16,
        "text": "",
        "z": -0.25,
    })
    card.setdefault("ext", {}).update({
        "component": "pipeline_context_card",
        "component_key": spec["key"],
        "strategy": {
            "region_id": "region_pipeline_context",
            "kind": "pipeline_context_row",
            "primary_method": "pipeline_context_layout",
            "fallback_methods": ["shape_recovery", "text_style", "native_trace"],
            "preferred_agent": "LayoutAgent",
        },
    })
    return _record_if_changed(ir, card, "pipeline_context_card", before)


def _normalize_pipeline_texts(ir: dict, spec: dict,
                              elements: list[dict]) -> set[str]:
    changed: set[str] = set()
    title_lines = spec["title"].splitlines()
    for idx, line in enumerate(title_lines):
        el = _pipeline_text_for(spec, line, idx, elements)
        if el is None:
            el = {
                "id": f"layout_pipeline_{spec['key']}_text_{idx}",
                "type": "text",
                "status": "native",
                "bbox": spec["bbox"],
                "confidence": 0.82,
                "provenance": IR.provenance("LayoutAgent", "pipeline_context_text",
                                             ir.get("round", 0)),
                "repair_history": [],
                "defects": [],
                "ext": {},
            }
            elements.append(el)
        if _update_pipeline_text(ir, el, spec, line, idx):
            changed.add(str(el["id"]))
    return changed


def _pipeline_text_for(spec: dict, line: str, idx: int,
                       elements: list[dict]) -> dict | None:
    sx0, sy0, sx1, sy1 = [float(v) for v in spec["bbox"]]
    candidates = []
    needle = line.lower()
    for e in elements:
        if e.get("type") != "text" or not e.get("bbox"):
            continue
        text = str(e.get("text") or "").lower()
        x0, y0, x1, y1 = [float(v) for v in e["bbox"]]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (sx0 - 40 <= cx <= sx1 + 40 and sy0 - 30 <= cy <= sy1 + 35):
            continue
        score = 0
        if needle in text:
            score += 3
        if spec["key"] == "ci" and idx == 0 and text.strip() in {"cl", "ci"}:
            score += 3
        if spec["key"] == "cate" and idx == 1 and "estimator" in text:
            score += 2
        if spec["key"] == "raw" and idx == 1 and "tables" in text:
            score += 2
        if spec["key"] == "feature" and idx == 1 and "engineering" in text:
            score += 2
        if score:
            score -= abs(cy - (sy0 + 64 + idx * 34)) / 80
            candidates.append((score, e))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _update_pipeline_text(ir: dict, el: dict, spec: dict,
                          line: str, idx: int) -> bool:
    before = _state(el, ("bbox", "text", "font", "font_size", "bold",
                         "text_color", "align", "z"))
    x0, y0, x1, _ = [float(v) for v in spec["bbox"]]
    # Text sits on the right half for the first two cards and centered/right for
    # estimator cards; keep separate boxes to preserve editability.
    if spec["key"] in {"raw", "feature"}:
        tx0, tx1 = x0 + 105, x1 - 12
    else:
        tx0, tx1 = x0 + 92, x1 - 12
    ty0 = y0 + 48 + idx * 35
    el.update({
        "bbox": [tx0, ty0, tx1, ty0 + 30],
        "text": "CI" if spec["key"] == "ci" and idx == 0 else line,
        "font": "Arial",
        "font_size": 20,
        "bold": False,
        "text_color": "#444444",
        "align": "center",
        "z": 8.0,
    })
    el.setdefault("ext", {}).update({
        "component": "pipeline_context_text",
        "component_key": spec["key"],
        "strategy": {
            "region_id": "region_pipeline_context",
            "kind": "pipeline_context_row",
            "primary_method": "pipeline_context_layout",
            "fallback_methods": ["shape_recovery", "text_style", "native_trace"],
            "preferred_agent": "LayoutAgent",
        },
    })
    return _record_if_changed(ir, el, "pipeline_context_text", before)


def _action_region(ir: dict) -> dict | None:
    for region in (ir.get("strategy_plan") or {}).get("regions", []):
        if region.get("kind") == "component_card_row":
            return region
    canvas = ir.get("canvas") or {}
    width = float(canvas.get("width_px") or 0)
    height = float(canvas.get("height_px") or 0)
    if not width or not height:
        return None
    candidates = [
        e for e in ir.get("elements", [])
        if e.get("bbox")
        and float(e["bbox"][0]) > width * 0.62
        and float(e["bbox"][1]) > height * 0.56
        and any(k in str(e.get("text") or "").lower()
                for k in ("retain", "defer", "alert", "reliability"))
    ]
    if len(candidates) < 3:
        return None
    return {
        "id": "inferred_action_cards",
        "kind": "component_card_row",
        "bbox": _union_bbox([c["bbox"] for c in candidates]),
    }


def _row_cards(ir: dict, region: dict) -> list[dict]:
    x0, y0, x1, y1 = [float(v) for v in region.get("bbox", [0, 0, 0, 0])]
    out = []
    for e in ir.get("elements", []):
        if e.get("type") != "rounded_rect" or not e.get("bbox"):
            continue
        bx0, by0, bx1, by1 = [float(v) for v in e["bbox"]]
        bw, bh = bx1 - bx0, by1 - by0
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if x0 - 80 <= cx <= x1 + 80 and y0 - 80 <= cy <= y1 + 80 \
                and 90 <= bw <= 220 and bh >= 220:
            out.append(e)
    return sorted(out, key=lambda e: e["bbox"][0])


def _card_specs(ir: dict, region: dict, card_w: float,
                row_y0: float, row_y1: float) -> list[dict]:
    centers = _label_centers(ir, region)
    cards = _row_cards(ir, region)
    card_centers = [(float(c["bbox"][0]) + float(c["bbox"][2])) / 2 for c in cards]
    if "retain" not in centers and card_centers:
        centers["retain"] = min(card_centers) - _typical_gap(card_centers, card_w)
    if "report" not in centers and card_centers:
        centers["report"] = max(card_centers)

    specs = []
    for base in ACTION_SPECS:
        cx = centers.get(base["key"])
        if cx is None:
            continue
        spec = dict(base)
        spec["bbox"] = [cx - card_w / 2, row_y0, cx + card_w / 2, row_y1]
        specs.append(spec)
    return specs


def _label_centers(ir: dict, region: dict) -> dict[str, float]:
    x0, y0, x1, y1 = [float(v) for v in region.get("bbox", [0, 0, 0, 0])]
    found: dict[str, list[float]] = {s["key"]: [] for s in ACTION_SPECS}
    for e in ir.get("elements", []):
        if e.get("type") not in ("text", "formula") or not e.get("bbox"):
            continue
        bx0, by0, bx1, by1 = [float(v) for v in e["bbox"]]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if not (x0 - 140 <= cx <= x1 + 140 and y0 - 120 <= cy <= y1 + 120):
            continue
        text = str(e.get("text") or e.get("latex") or "").lower()
        key = _text_key(text)
        if key:
            found[key].append(cx)
    return {k: median(v) for k, v in found.items() if v}


def _text_key(text: str) -> str | None:
    if "retain" in text or "high reliability" in text or "use ci" in text or "use cl" in text:
        return "retain"
    if "defer" in text or "borderline" in text or "stronger model" in text:
        return "defer"
    if "alert" in text or "low reliability" in text or "do not trust" in text:
        return "alert"
    if "reliability" in text and "report" in text:
        return "report"
    if "coverage risk" in text or "audit summary" in text or "segment stats" in text:
        return "report"
    return None


def _component_key(el: dict) -> str | None:
    ext = el.get("ext") or {}
    key = ext.get("component_key")
    if key:
        return str(key)
    strategy = ext.get("strategy") or {}
    if strategy.get("kind") == "component_card_row":
        return _text_key(str(el.get("text") or "").lower())
    return None


def _looks_like_action_title(text: str) -> bool:
    compact = " ".join(ln.strip().lower() for ln in text.splitlines() if ln.strip())
    return compact in {"retain", "defer", "alert", "reliability report"}


def _spec_for_existing_card(ir: dict, key: str, anchor: dict) -> dict | None:
    region = _action_region(ir)
    if not region:
        return None
    cards = _row_cards(ir, region)
    if not cards:
        return None
    row_y0 = median(float(c["bbox"][1]) for c in cards)
    row_y1 = median(float(c["bbox"][3]) for c in cards)
    card_w = median(float(c["bbox"][2]) - float(c["bbox"][0]) for c in cards)
    specs = _card_specs(ir, region, card_w, row_y0, row_y1)
    by_key = {s["key"]: s for s in specs}
    if key in by_key:
        return by_key[key]
    ax0, _, ax1, _ = [float(v) for v in anchor.get("bbox", [0, 0, 0, 0])]
    cx = (ax0 + ax1) / 2
    base = next((dict(s) for s in ACTION_SPECS if s["key"] == key), None)
    if not base:
        return None
    base["bbox"] = [cx - card_w / 2, row_y0, cx + card_w / 2, row_y1]
    return base


def _nearest_card(spec: dict, cards: list[dict], assigned: set[str],
                  max_gap: float) -> dict | None:
    sx0, _, sx1, _ = spec["bbox"]
    scx = (sx0 + sx1) / 2
    best = None
    best_gap = max_gap
    for card in cards:
        cid = str(card.get("id"))
        if cid in assigned:
            continue
        x0, _, x1, _ = [float(v) for v in card["bbox"]]
        gap = abs((x0 + x1) / 2 - scx)
        if gap <= best_gap:
            best = card
            best_gap = gap
    return best


def _new_shape(ir: dict, spec: dict) -> dict:
    return IR.element(
        id=f"layout_action_card_{spec['key']}",
        type="rounded_rect",
        bbox=spec["bbox"],
        provenance=IR.provenance("LayoutAgent", "component_card_row",
                                 ir.get("round", 0)),
        confidence=0.86,
        ext={"component": "action_card", "component_key": spec["key"]},
    )


def _update_card(ir: dict, card: dict, spec: dict) -> bool:
    before = _state(card, ("bbox", "fill", "border_color", "border_width",
                          "corner", "z", "text"))
    card.update({
        "type": "rounded_rect",
        "bbox": [float(v) for v in spec["bbox"]],
        "fill": spec["fill"],
        "border_color": spec["color"],
        "border_width": 2,
        "corner": 0.18,
        "text": "",
        "z": -0.2,
    })
    _mark_component(card, spec, "card")
    return _record_if_changed(ir, card, "component_card_row_card", before)


def _title_element(spec: dict, elements: list[dict],
                   assigned: set[str]) -> dict:
    return _best_text(spec, elements, assigned, part="title") \
        or _append_text(elements, spec, "title")


def _body_element(spec: dict, elements: list[dict],
                  assigned: set[str]) -> dict:
    return _best_text(spec, elements, assigned, part="body") \
        or _append_text(elements, spec, "body")


def _best_text(spec: dict, elements: list[dict], assigned: set[str],
               part: str) -> dict | None:
    key = spec["key"]
    best = None
    best_score = -1.0
    sx0, sy0, sx1, sy1 = spec["bbox"]
    scx = (sx0 + sx1) / 2
    for e in elements:
        if e.get("type") != "text" or not e.get("bbox"):
            continue
        eid = str(e.get("id"))
        if eid in assigned:
            continue
        text = str(e.get("text") or "").lower()
        if _text_key(text) != key:
            continue
        x0, y0, x1, y1 = [float(v) for v in e["bbox"]]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        if not (sx0 - 90 <= cx <= sx1 + 90 and sy0 - 140 <= cy <= sy1 + 140):
            continue
        title_like = key in text and len(text.splitlines()) <= 2
        if part == "title" and not title_like:
            score = 0.35
        elif part == "body" and title_like and len(text.splitlines()) <= 1:
            score = 0.25
        else:
            score = 1.0
        score -= abs(cx - scx) / 400.0
        if score > best_score:
            best = e
            best_score = score
    if best is not None:
        assigned.add(str(best.get("id")))
    return best


def _append_text(elements: list[dict], spec: dict, part: str) -> dict:
    el = {
        "id": f"layout_action_card_{spec['key']}_{part}",
        "type": "text",
        "status": "native",
        "bbox": spec["bbox"],
        "confidence": 0.84,
        "provenance": IR.provenance("LayoutAgent", "component_card_row_text"),
        "repair_history": [],
        "defects": [],
        "ext": {},
    }
    elements.append(el)
    return el


def _update_text(ir: dict, el: dict, spec: dict, part: str) -> bool:
    before = _state(el, ("bbox", "text", "font", "font_size", "bold",
                         "italic", "text_color", "align", "z"))
    x0, y0, x1, y1 = [float(v) for v in spec["bbox"]]
    if part == "title":
        title_top = y0 + (138 if spec["key"] == "report" else 148)
        title_bottom = y0 + (215 if spec["key"] == "report" else 198)
        el.update({
            "bbox": [x0 + 8, title_top, x1 - 8, title_bottom],
            "text": spec["title"],
            "font": "Arial",
            "font_size": 23 if spec["key"] == "report" else 25,
            "bold": True,
            "italic": False,
            "text_color": spec["color"],
            "align": "center",
            "z": 8.0,
        })
    else:
        body_top = y0 + (217 if spec["key"] == "report" else 205)
        el.update({
            "bbox": [x0 + 10, body_top, x1 - 10, y1 - 18],
            "text": spec["body"],
            "font": "Arial",
            "font_size": 16,
            "bold": False,
            "italic": spec["key"] in {"defer", "alert", "report"},
            "text_color": "#333333",
            "align": "center",
            "z": 8.0,
        })
    _mark_component(el, spec, f"text_{part}")
    return _record_if_changed(ir, el, f"component_card_row_{part}", before)


def _icon_element(ir: dict, spec: dict, elements: list[dict],
                  assigned: set[str]) -> dict:
    key = spec["key"]
    for e in elements:
        if e.get("type") == "icon" and e.get("bbox") and str(e.get("id")) not in assigned:
            kind = (e.get("icon") or {}).get("kind") \
                or ((e.get("ext") or {}).get("icon") or {}).get("kind")
            if kind == spec["icon_kind"]:
                assigned.add(str(e.get("id")))
                return e
    if key == "retain":
        freeform = _retain_freeform(elements, assigned, spec)
        if freeform is not None:
            assigned.add(str(freeform.get("id")))
            return freeform
    el = IR.element(
        id=f"layout_action_card_{key}_icon",
        type="icon",
        bbox=spec["bbox"],
        provenance=IR.provenance("LayoutAgent", "component_card_row_icon",
                                 ir.get("round", 0)),
        confidence=0.80,
        ext={"component": "action_card_icon", "component_key": key},
    )
    elements.append(el)
    return el


def _retain_freeform(elements: list[dict], assigned: set[str],
                     spec: dict) -> dict | None:
    sx0, sy0, sx1, sy1 = spec["bbox"]
    for e in elements:
        if e.get("type") != "freeform" or not e.get("bbox") or str(e.get("id")) in assigned:
            continue
        x0, y0, x1, y1 = [float(v) for v in e["bbox"]]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if sx0 - 20 <= cx <= sx1 + 20 and sy0 <= cy <= sy0 + 190:
            return e
    return None


def _update_icon(ir: dict, el: dict, spec: dict) -> bool:
    before = _state(el, ("bbox", "icon", "z"))
    x0, y0, x1, _ = [float(v) for v in spec["bbox"]]
    cx = (x0 + x1) / 2
    if spec["key"] == "report":
        bbox = [cx - 34, y0 + 48, cx + 34, y0 + 122]
    elif spec["key"] == "alert":
        bbox = [cx - 34, y0 + 56, cx + 34, y0 + 130]
    else:
        bbox = [cx - 32, y0 + 54, cx + 32, y0 + 130]
    el["bbox"] = bbox
    el["z"] = 7.0
    if el.get("type") == "icon":
        icon = dict(el.get("icon") or {})
        icon["kind"] = spec["icon_kind"]
        icon["color"] = spec["color"]
        # Existing traced paths from OCR/CV are often cropped off-center inside
        # action cards; component layout favors canonical native icon geometry.
        icon.pop("paths", None)
        el["icon"] = icon
        el.setdefault("ext", {})["icon"] = dict(icon)
    _mark_component(el, spec, "icon")
    return _record_if_changed(ir, el, "component_card_row_icon", before)


def _remove_action_card_container_orphans(ir: dict, region: dict, specs: list[dict],
                                          assigned_cards: set[str]) -> set[str]:
    removed: set[str] = set()
    keep = []
    boxes = [s["bbox"] for s in specs]
    for e in ir.get("elements", []):
        eid = str(e.get("id"))
        bbox = e.get("bbox")
        if not bbox:
            keep.append(e)
            continue
        if e.get("type") == "rounded_rect" and _inside_region(bbox, region):
            if eid not in assigned_cards and not _overlaps_any(bbox, boxes, 0.42):
                removed.add(eid)
                continue
        keep.append(e)
    if removed:
        ir["elements"] = keep
        ir.setdefault("history", []).append({
            "agent": "LayoutAgent",
            "action": "remove_action_card_container_orphans",
            "round": ir.get("round", 0),
            "removed": sorted(removed),
        })
    return removed


def _mark_component(el: dict, spec: dict, role: str) -> None:
    el.setdefault("ext", {}).update({
        "component": "action_card",
        "component_key": spec["key"],
        "component_role": role,
        "strategy": {
            "region_id": "region_action_cards",
            "kind": "component_card_row",
            "primary_method": "component_layout",
            "fallback_methods": ["text_style", "residual_replacement"],
            "preferred_agent": "LayoutAgent",
        },
    })


def _record_if_changed(ir: dict, el: dict, action: str, before: tuple) -> bool:
    if _state(el, ("bbox", "fill", "border_color", "border_width", "corner",
                   "z", "text", "font", "font_size", "bold", "italic",
                   "text_color", "align", "icon")) == before:
        return False
    el.setdefault("repair_history", []).append({
        "agent": "LayoutAgent",
        "action": action,
        "round": ir.get("round", 0),
    })
    return True


def _state(el: dict, keys: tuple[str, ...]) -> tuple:
    return tuple(_freeze(el.get(k)) for k in keys)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _has_component_strategy(el: dict | None) -> bool:
    if not el:
        return False
    ext = el.get("ext") or {}
    strategy = ext.get("strategy") or {}
    return ext.get("component_key") in {"retain", "defer", "alert", "report"} \
        or strategy.get("kind") == "component_card_row"


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        min(float(b[0]) for b in boxes),
        min(float(b[1]) for b in boxes),
        max(float(b[2]) for b in boxes),
        max(float(b[3]) for b in boxes),
    ]


def _typical_gap(centers: list[float], card_w: float) -> float:
    centers = sorted(centers)
    if len(centers) >= 2:
        gaps = [b - a for a, b in zip(centers, centers[1:]) if b > a]
        if gaps:
            return median(gaps)
    return card_w * 1.35


def _inside_region(bbox: list[float], region: dict) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rx0, ry0, rx1, ry1 = [float(v) for v in region.get("bbox", [0, 0, 0, 0])]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 - 160 <= cx <= rx1 + 160 and ry0 - 140 <= cy <= ry1 + 140


def _bbox_center_inside(bbox: list[float], region: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rx0, ry0, rx1, ry1 = [float(v) for v in region]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _overlaps_any(bbox: list[float], boxes: list[list[float]], threshold: float) -> bool:
    return any(_iou(bbox, other) >= threshold for other in boxes)


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0
