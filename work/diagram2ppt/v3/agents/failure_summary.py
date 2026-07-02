"""FailureSummaryAgent: native rebuild for the bottom-left summary panel."""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import component_templates, ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class FailureSummaryAgent(Agent):
    """Rebuild the failure summary panel as native text and symbols."""

    name = "FailureSummaryAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if not _is_failure_task(task):
            return []
        region = _region_bbox(ir, task)
        changed = set(_remove_orphans(ir, region))
        for el in _failure_elements(ir, region):
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.clear()
                existing.update(el)
            else:
                ir.setdefault("elements", []).append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "native_failure_summary_transaction",
            "round": ir.get("round", 0),
            "changed": sorted(changed),
        })
        return sorted(changed)


ROWS = [
    ("overlap", "Low overlap coincides with\nhigh effect variation", "warning_outline", "#ef4b36"),
    ("undercover", "Honest CI undercovers in", "warning", "#e95a42"),
    ("invisible", "Invisible to existing data /\nML monitoring tools", "eye_slash", "#4b718f"),
]


def _is_failure_task(task: dict) -> bool:
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "kind", "region_id", "locked_method", "objective",
    )).lower()
    return "failure_summary" in text or "failure summary" in text


def _region_bbox(ir: dict, task: dict) -> list[float]:
    visual = task.get("visual_defect") or {}
    bbox = visual.get("bbox") or task.get("bbox")
    if bbox and len(bbox) == 4 and max(float(v) for v in bbox) > 0:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        # The task bbox can be clipped upward by defect clustering.  The visual
        # review bbox describes the semantic panel and should anchor the native
        # component; keep the bottom loose so row text has the same breathing
        # room as the source infographic.
        return [x0 + 30, y0 + 6, x1 - 30, y1 - 64]
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    return [w * 0.018, h * 0.645, w * 0.170, h * 0.885]


def _failure_elements(ir: dict, region: list[float]) -> list[dict]:
    x0, y0, x1, y1 = region
    r = ir.get("round", 0)
    w, h = x1 - x0, y1 - y0
    panel = [x0, y0, x1, y1]
    elements: list[dict] = [
        IR.element(
            id="failure_summary_panel",
            type="rounded_rect",
            bbox=panel,
            provenance=IR.provenance("FailureSummaryAgent", "failure_panel", r),
            confidence=0.90,
            fill="#ffffff",
            border_color="#6a91c9",
            border_width=2.0,
            corner=0.08,
            z=-0.1,
            ext=_ext("panel"),
        ),
        IR.element(
            id="failure_summary_title",
            type="text",
            bbox=[x0 + w * 0.18, y0 + h * 0.030, x1 - w * 0.060, y0 + h * 0.205],
            provenance=IR.provenance("FailureSummaryAgent", "failure_title", r),
            confidence=0.88,
            text="Failure summary",
            font="Arial",
            font_size=31,
            bold=True,
            text_color="#18244f",
            align="left",
            z=7.0,
            ext=_ext("title"),
        ),
    ]
    for idx, (key, text, icon, color) in enumerate(ROWS):
        top = y0 + h * (0.245 + idx * 0.245)
        icon_box = [x0 + w * 0.050, top - h * 0.012, x0 + w * 0.165, top + h * 0.115]
        if key == "undercover":
            elements.append(IR.element(
                id=f"failure_summary_icon_{key}",
                type="oval",
                bbox=icon_box,
                provenance=IR.provenance("FailureSummaryAgent", "failure_icon", r),
                confidence=0.85,
                fill=color,
                border_color=color,
                border_width=1.0,
                text="!",
                font="Arial",
                font_size=22,
                bold=True,
                text_color="#ffffff",
                align="center",
                z=8.0,
                ext=_ext("icon", key),
            ))
        else:
            icon_kind = icon
            icon_spec = {"kind": icon_kind, "color": color}
            if icon == "warning_outline":
                icon_spec = {"kind": "warning", "color": color, "variant": "outline"}
            elements.append(IR.element(
                id=f"failure_summary_icon_{key}",
                type="icon",
                bbox=icon_box,
                provenance=IR.provenance("FailureSummaryAgent", "failure_icon", r),
                confidence=0.84,
                icon=icon_spec,
                z=8.0,
                ext=_ext("icon", key),
            ))
        text_box = [x0 + w * 0.205, top - h * 0.020, x1 - w * 0.045, top + h * 0.155]
        if key == "undercover":
            text_box = [x0 + w * 0.205, top - h * 0.020, x1 - w * 0.045, top + h * 0.155]
        text_el = IR.element(
            id=f"failure_summary_text_{key}",
            type="text",
            bbox=text_box,
            provenance=IR.provenance("FailureSummaryAgent", "failure_text", r),
            confidence=0.88,
            text=text,
            font="Arial",
            font_size=25 if key != "invisible" else 24,
            italic=(key == "invisible"),
            text_color="#111827",
            align="left",
            z=7.5,
            ext=_ext("text", key, size=25 if key != "invisible" else 24,
                     italic=(key == "invisible")),
        )
        if key == "undercover":
            text_el["text"] = "Honest CI undercovers in Q0"
            text_el["runs"] = [
                {"text": "Honest CI undercovers in ", "font": "Arial"},
                {"text": "Q", "font": "Times New Roman", "italic": True},
                {"text": "0", "font": "Times New Roman", "italic": True, "font_size": 15, "baseline": -25000},
            ]
        elements.append(text_el)
    return elements


def _remove_orphans(ir: dict, region: list[float]) -> set[str]:
    removable = {"text", "formula", "rounded_rect", "rect", "icon", "oval", "line", "freeform"}
    cleanup = [
        max(0.0, region[0] - 60.0),
        max(0.0, region[1] - 55.0),
        region[2] + 90.0,
        region[3] + 170.0,
    ]
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if (
            eid.startswith("failure_summary_")
            or eid.startswith(("auditor_", "action_card_", "bottom_", "chart_q0_", "pipeline_context_", "proc_"))
            or not bbox
        ):
            keep.append(el)
            continue
        if el.get("type") in removable and _center_inside(bbox, cleanup):
            removed.add(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _center_inside(bbox: list[float], region: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rx0, ry0, rx1, ry1 = [float(v) for v in region]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _ext(role: str, key: str = "", **overrides: Any) -> dict:
    return component_templates.component_ext(
        "failure_summary",
        role,
        key,
        region_id="failure_summary",
        **overrides,
    )
