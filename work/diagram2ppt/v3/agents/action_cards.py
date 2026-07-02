"""ActionCardAgent: native rebuild for bottom decision/report cards."""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import component_templates, ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class ActionCardAgent(Agent):
    """Specialist for RETAIN / DEFER / ALERT / Reliability Report cards."""

    name = "ActionCardAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if not _is_action_task(task):
            return []
        cleanup_region = _region_bbox(ir, task)
        layout_region = _layout_region_bbox(ir, task) or cleanup_region
        changed = set(_remove_orphans(ir, cleanup_region))
        changed.update(_remove_existing_action_elements(ir))
        for el in _action_elements(ir, layout_region):
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.clear()
                existing.update(el)
            else:
                ir.setdefault("elements", []).append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "native_action_cards_transaction",
            "round": ir.get("round", 0),
            "changed": sorted(changed),
        })
        return sorted(changed)


SPECS = [
    ("retain", "RETAIN", "shield", "#16806e", "#fbfffd"),
    ("defer", "DEFER", "hourglass", "#9a5b13", "#fffdf9"),
    ("alert", "ALERT", "warning", "#cf3d28", "#fffafa"),
    ("report", "Reliability\nReport", "document", "#245591", "#fbfdff"),
]

BODY_LINES = {
    "retain": [
        ("High reliability", "body", False, "Arial", 20.8, "#16806e"),
        ("Use CI as is", "body_emphasis", True, "Arial", 20.2, "#333333"),
    ],
    "defer": [
        ("Borderline", "body", False, "Arial", 20.4, "#9a5b13"),
        ("Seek more data", "body", False, "Arial", 20.0, "#333333"),
        ("or stronger model", "body_emphasis", True, "Arial", 19.4, "#333333"),
    ],
    "alert": [
        ("Low reliability", "body", False, "Arial", 20.6, "#cf3d28"),
        ("Do not trust CI", "body_emphasis", True, "Arial", 19.8, "#333333"),
        ("in Q0", "body_math", True, "Times New Roman", 20.2, "#333333"),
    ],
    "report": [
        ("Coverage risk map,", "report_body_emphasis", True, "Arial", 19.2, "#245591"),
        ("segment stats,", "report_body_emphasis", True, "Arial", 19.2, "#245591"),
        ("audit summary", "report_body_emphasis", True, "Arial", 19.2, "#245591"),
    ],
}


def _is_action_task(task: dict) -> bool:
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "kind", "region_id", "locked_method", "objective",
    )).lower()
    return "action_cards" in text or "component_card_row" in text or "repeated action cards" in text


def _region_bbox(ir: dict, task: dict) -> list[float]:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    bbox = task.get("bbox")
    if bbox and len(bbox) == 4 and max(float(v) for v in bbox) > 0:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [min(x0 - 90, w * 0.640), y0 - 130, x1 + 45, y1 + 40]
    return [w * 0.640, h * 0.580, w * 0.995, h * 0.980]


def _layout_region_bbox(ir: dict, task: dict) -> list[float] | None:
    for region in (ir.get("strategy_plan") or {}).get("regions", []):
        if region.get("id") == "region_action_cards" and region.get("bbox"):
            bbox = [float(v) for v in region["bbox"][:4]]
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return bbox
    bbox = task.get("bbox")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 > x0 and y1 > y0:
            return [x0, y0, x1, y1]
    return None


def _action_elements(ir: dict, region: list[float] | None = None) -> list[dict]:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    r = ir.get("round", 0)
    # Measured from the source panel: the action cards occupy the far-right
    # bottom band, below the auditor row and to the right of the mini surface.
    if region and len(region) == 4:
        rx0, ry0, rx1, ry1 = [float(v) for v in region]
        rw, rh = max(1.0, rx1 - rx0), max(1.0, ry1 - ry0)
        # Region bbox includes routed connectors above the card tops.  Keep the
        # connectors in the owned region but place the actual cards in the lower
        # source-measured band.
        y0, y1 = ry0 + rh * 0.095, ry1
        x0, x1 = rx0, rx1
        total_gap = rw * 0.070
        gap = total_gap / 3
        weights = [0.218, 0.235, 0.231, 0.263]
        unit = (rw - total_gap) / sum(weights)
        card_boxes = []
        cursor = x0
        for weight in weights:
            width_i = unit * weight
            card_boxes.append((cursor, cursor + width_i))
            cursor += width_i + gap
    else:
        y0, y1 = h * 0.670, h * 0.891
        card_boxes = [
            (w * 0.702, w * 0.764),
            (w * 0.770, w * 0.837),
            (w * 0.843, w * 0.909),
            (w * 0.915, w * 0.990),
        ]
    elements: list[dict] = []
    for idx, (key, title, icon, color, fill) in enumerate(SPECS):
        x0, x1 = card_boxes[idx]
        cx = (x0 + x1) / 2
        width = x1 - x0
        height = y1 - y0
        elements.extend([
            IR.element(
                id=f"action_card_{key}",
                type="rounded_rect",
                bbox=[x0, y0, x1, y1],
                provenance=IR.provenance("ActionCardAgent", "action_card", r),
                confidence=0.90,
                fill=fill,
                border_color=color,
                border_width=2.0,
                corner=0.16,
                z=-0.10,
                ext=_ext("card", key),
            ),
            IR.element(
                id=f"action_card_icon_{key}",
                type="icon",
                bbox=[cx - width * 0.25, y0 + height * 0.075,
                      cx + width * 0.25, y0 + height * 0.325],
                provenance=IR.provenance("ActionCardAgent", "action_icon", r),
                confidence=0.86,
                icon={"kind": icon, "color": color, "variant": "solid"},
                z=8.0,
                ext=_ext("icon", key),
            ),
            IR.element(
                id=f"action_card_title_{key}",
                type="text",
                bbox=[x0 + width * 0.08, y0 + height * 0.345,
                      x1 - width * 0.08, y0 + height * 0.545],
                provenance=IR.provenance("ActionCardAgent", "action_title", r),
                confidence=0.88,
                text=title,
                font="Arial",
                font_size=26 if key != "report" else 24,
                bold=True,
                text_color=color,
                align="center",
                z=8.5,
                ext=_ext("report_title" if key == "report" else "title", key),
            ),
        ])
        elements.extend(_body_line_elements(ir, key, x0, y0, x1, y1))
        # Routed decision connectors from the auditor row into the cards.
        if key in {"defer", "alert", "report"}:
            source_shift = {"defer": -0.018, "alert": -0.030, "report": -0.046}[key]
            sx = cx + w * source_shift
            sy = h * 0.555
            route_y = h * 0.614
            ey = y0 - h * 0.014
            elements.extend([
                _connector_segment(ir, key, "stem", sx, sy, sx, route_y, color),
                _connector_segment(ir, key, "elbow", sx, route_y, cx, route_y, color),
                _connector_segment(ir, key, "", cx, route_y, cx, ey, color, arrow=True),
            ])
    return elements


def _body_line_elements(ir: dict, key: str, x0: float, y0: float,
                        x1: float, y1: float) -> list[dict]:
    width = x1 - x0
    height = y1 - y0
    r = ir.get("round", 0)
    lines = BODY_LINES[key]
    if key == "retain":
        centers = [0.655, 0.735]
    elif key == "report":
        centers = [0.625, 0.705, 0.785]
    else:
        centers = [0.615, 0.695, 0.775]
    out: list[dict] = []
    for idx, ((text, role, italic, font, size, color), cy_frac) in enumerate(zip(lines, centers)):
        line_h = height * (0.105 if key != "report" else 0.095)
        el = IR.element(
            id=f"action_card_body_{key}_{idx}",
            type="text",
            bbox=[
                x0 + width * 0.035,
                y0 + height * cy_frac - line_h / 2,
                x1 - width * 0.035,
                y0 + height * cy_frac + line_h / 2,
            ],
            provenance=IR.provenance("ActionCardAgent", "action_body_line", r),
            confidence=0.87,
            text=text,
            font=font,
            font_size=size,
            italic=italic,
            text_color=color,
            align="center",
            z=8.4,
            ext=_ext(role, key, font=font, size=size, color=color, italic=italic),
        )
        if role == "body_math" and text == "in Q0":
            el["runs"] = [
                {"text": "in ", "font": "Times New Roman", "italic": True, "color": color},
                {"text": "Q", "font": "Times New Roman", "italic": True, "color": color},
                {"text": "0", "font": "Times New Roman", "italic": True, "font_size": 16, "baseline": -25000, "color": color},
            ]
        out.append(el)
    return out


def _connector_segment(
    ir: dict,
    key: str,
    part: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    arrow: bool = False,
) -> dict:
    suffix = f"_{part}" if part else ""
    return IR.element(
        id=f"action_card_connector_{key}{suffix}",
        type="arrow" if arrow else "line",
        bbox=[min(x0, x1) - 5, min(y0, y1) - 5, max(x0, x1) + 5, max(y0, y1) + 5],
        provenance=IR.provenance("ActionCardAgent", "action_connector", ir.get("round", 0)),
        confidence=0.80,
        points=[x0, y0, x1, y1],
        color=color,
        thickness=3,
        line_width=3,
        z=5.5,
        ext=_ext("connector", key),
    )


def _remove_orphans(ir: dict, region: list[float]) -> set[str]:
    removable = {"text", "formula", "rounded_rect", "rect", "icon", "arrow", "line", "freeform", "dotcloud"}
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if (
            eid.startswith("action_card_")
            or _is_foreign_owner(eid)
            or not bbox
        ):
            keep.append(el)
            continue
        if el.get("type") in removable and _center_inside(bbox, region):
            removed.add(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _remove_existing_action_elements(ir: dict) -> set[str]:
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        if eid.startswith("action_card_"):
            removed.add(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _is_foreign_owner(eid: str) -> bool:
    return eid.startswith((
        "auditor_",
        "bottom_",
        "chart_q0_",
        "failure_summary_",
        "pipeline_context_",
        "proc_",
    ))


def _center_inside(bbox: list[float], region: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rx0, ry0, rx1, ry1 = [float(v) for v in region]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _ext(role: str, key: str, **overrides: Any) -> dict:
    return component_templates.component_ext(
        "action_card",
        role,
        key,
        region_id="region_action_cards",
        **overrides,
    )
