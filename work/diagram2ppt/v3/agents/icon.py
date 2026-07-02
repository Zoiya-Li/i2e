"""Icon Agent: re-classifies icons using a VLM crop.

The v2 pipeline already has an icon prompt; IconAgent just plugs it into the
v3 repair loop so high-residual icons get a second chance.
"""
from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent
from work.diagram2ppt.v3.providers import get_provider

_ICON_PROMPT = (
    "This small image is an icon/pictogram from a technical diagram. "
    "Look at the SHAPE only. Identify its concrete visual TYPE from the list below (be specific):\n"
    "- database: stacked cylinders / disk-drive symbol\n"
    "- gear: cogwheel with teeth\n"
    "- scatter: cloud of dots, often with a trend/regression line\n"
    "- line: smooth curve / line-chart / bell curve / normal distribution\n"
    "- warning: triangle with exclamation mark\n"
    "- hourglass: sand timer\n"
    "- shield: shield shape, optionally with a checkmark inside\n"
    "- document: sheet of paper with lines or a small chart\n"
    "- check: tick / checkmark\n"
    "- cross: X / cancel mark\n"
    "- arrow: directional arrow head or block arrow\n"
    "- other: anything that does not match the above\n\n"
    "Examples:\n"
    '{"kind":"gear","color":"#6b7a8d","glyph":"⚙"}\n'
    '{"kind":"scatter","color":"#4472c4","glyph":"📊"}\n'
    '{"kind":"line","color":"#6b7a8d","glyph":"📈"}\n'
    '{"kind":"database","color":"#6b7a8d","glyph":"🗄"}\n\n'
    'Output STRICT JSON: {"kind": "...", "color": "#hex of the dominant color", '
    '"glyph": "single unicode character that best represents it"}. '
    'Use "scatter" for point-cloud icons, "line" for curve icons, "gear" for '
    'cog wheels. Output ONLY the JSON.'
)

_ICON_SECOND_PROMPT = (
    "This small image is an icon. Look carefully at its SHAPE. Choose the "
    "closest match from: database, gear, scatter, line, warning, hourglass, "
    "shield, document, check, cross, arrow. If it is genuinely none of these, "
    'say other. Output ONLY JSON: {\"kind\":\"...\",\"color\":\"#hex\"}'
)

_SUPPORTED_KINDS = {
    "database", "gear", "scatter", "line", "warning", "hourglass",
    "shield", "document", "check", "cross", "arrow", "other",
}


class IconAgent(Agent):
    """Specialist agent for icon classification and repair."""

    name = "IconAgent"

    def __init__(self) -> None:
        self.provider = get_provider("icon")

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        defect = kwargs.get("defect")
        if defect and defect.get("element_id"):
            el = IR.get_element(ir, defect["element_id"])
            if el and el.get("type") == "icon":
                return self._repair_icon(ir, original, el)

        changed: list[str] = []
        for el in ir.get("elements", []):
            if el.get("type") == "icon":
                changed.extend(self._repair_icon(ir, original, el))
        return changed

    def _repair_icon(self, ir: dict, original: Image.Image,
                     el: dict) -> list[str]:
        bbox = el.get("bbox")
        if not bbox:
            return []

        crop = _padded_crop(original, bbox, pad=0.15)
        exact_crop = _exact_crop(original, bbox)
        if crop.width < 8 or crop.height < 8 or exact_crop.width < 4 or exact_crop.height < 4:
            return []

        old_icon = el.get("icon") or el.get("ext", {}).get("icon") or {}
        paths = _trace_icon_paths(exact_crop)
        try:
            kind, color, glyph = self._classify(crop)
        except Exception:
            kind = str(old_icon.get("kind") or "other").lower()
            color = str(old_icon.get("color") or "#555555")
            glyph = str(old_icon.get("glyph") or "◆")
        if kind == "other" and not paths:
            return []
        old_kind = str(old_icon.get("kind") or old_icon.get("name") or "").lower()
        if old_kind and old_kind != "other" and kind != old_kind:
            kind = old_kind
        if kind == "other" and old_kind:
            kind = old_kind

        icon_payload = {
            "kind": kind,
            "color": color,
            "glyph": glyph,
        }
        if paths:
            icon_payload["paths"] = paths
        if icon_payload == {k: old_icon.get(k) for k in icon_payload}:
            return []

        el["icon"] = icon_payload
        el.setdefault("ext", {})["icon"] = dict(icon_payload)
        el.setdefault("repair_history", []).append({
            "agent": self.name,
            "action": "icon_reclassify",
            "round": ir.get("round", 0),
            "provider": self.provider.name,
        })
        return [el["id"]]

    def _classify(self, crop: Image.Image) -> tuple[str, str, str]:
        raw = self.provider.ask(crop, _ICON_PROMPT, temperature=0.0)
        kind, color, glyph = _parse_icon_json(raw)
        if kind == "other":
            raw2 = self.provider.ask(crop, _ICON_SECOND_PROMPT, temperature=0.0)
            kind2, color2, _ = _parse_icon_json(raw2)
            if kind2 in _SUPPORTED_KINDS and kind2 != "other":
                kind = kind2
                color = color2
        return kind, color, glyph


def _padded_crop(image: Image.Image, bbox: list[float],
                 pad: float = 0.15) -> Image.Image:
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    dx, dy = w * pad, h * pad
    left = max(0, int(x0 - dx))
    top = max(0, int(y0 - dy))
    right = min(image.width, int(x1 + dx))
    bottom = min(image.height, int(y1 + dy))
    return image.crop((left, top, right, bottom))


def _exact_crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    return image.crop((
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(image.width, int(round(x1))),
        min(image.height, int(round(y1))),
    ))


def _trace_icon_paths(crop: Image.Image) -> list[dict]:
    try:
        from work.diagram2ppt.v2.native_trace import extract_paths
    except Exception:
        return []
    area = max(1, crop.width * crop.height)
    min_area = max(8.0, area * 0.0015)
    return extract_paths(
        crop,
        max_paths=36,
        min_area=min_area,
        epsilon_frac=0.018,
        pale=False,
    )


def _parse_icon_json(raw: str) -> tuple[str, str, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return "other", "#555555", "◆"
    try:
        d = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return "other", "#555555", "◆"
    kind = str(d.get("kind", "other")).lower().strip()
    if kind not in _SUPPORTED_KINDS:
        kind = "other"
    color = str(d.get("color", "#555555")).strip()
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        color = "#555555"
    glyph = str(d.get("glyph", "◆"))[:2]
    return kind, color, glyph
