"""PipelineContextAgent: native rebuild for the top process row.

The top causal-decision pipeline is a known component row, not a generic OCR
cluster.  This agent replaces noisy fragments inside that semantic slot with
editable process cards, icons, labels, and connectors.
"""
from __future__ import annotations

import os
import math
import re
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent
from work.diagram2ppt.v3.providers import get_provider


_FLOW_SEMANTIC_PROMPT = """You are reconstructing an editable PPT diagram region.
Do not think aloud. Do not explain. Return only the final JSON object.

Read ONLY visible content inside this crop. Return STRICT JSON with this schema:
{
  "title": {"text": "...", "bbox": [x0,y0,x1,y1]} | null,
  "blocks": [
    {
      "label": "visible text, line breaks allowed",
      "bbox": [x0,y0,x1,y1],
      "role": "process|axis_label|thumbnail|chart|panel|annotation",
      "rotation": 0|90|-90,
      "fill": "#RRGGBB|null",
      "text_color": "#RRGGBB|null"
    }
  ],
  "connectors": [
    {"from": 0, "to": 1, "bbox": [x0,y0,x1,y1], "direction": "right|left|up|down"}
  ]
}

Coordinates are normalized to the crop in [0,1]. Do not invent missing words.
If text is unreadable, use an empty label but still return the block bbox.
Output JSON only."""


class PipelineContextAgent(Agent):
    """Specialist for the existing causal decision pipeline context row."""

    name = "PipelineContextAgent"

    def __init__(self) -> None:
        self.provider = get_provider("vlm")
        self.ocr_provider = get_provider("ocr")

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if not _is_pipeline_task(task):
            self.record_contract_result(ir, task, [], status="task_not_pipeline")
            return []
        region = _region_bbox(ir, task)
        source_elements = list(ir.get("elements", []))
        elements = (
            _generic_flow_elements(
                ir, task, region, source_elements, original,
                self.provider, self.ocr_provider)
            if _use_generic_flow(task)
            else _pipeline_elements(ir)
        )
        if not elements:
            self.record_contract_result(ir, task, [], status="no_flow_elements")
            return []
        changed = set(
            _remove_generic_flow_outputs(ir)
            if _use_generic_flow(task)
            else _remove_orphans(ir, region)
        )
        for el in elements:
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.clear()
                existing.update(el)
            else:
                ir.setdefault("elements", []).append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "native_pipeline_context_transaction",
            "round": ir.get("round", 0),
            "changed": sorted(changed),
        })
        out = sorted(changed)
        self.record_contract_result(ir, task, out)
        return out


def _is_pipeline_task(task: dict) -> bool:
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "kind", "region_id", "locked_method", "objective",
    )).lower()
    return "pipeline_context" in text or "process row" in text


def _use_generic_flow(task: dict) -> bool:
    region_id = str(task.get("region_id") or "").lower()
    kind = str(task.get("kind") or "").lower()
    objective = str(task.get("objective") or "").lower()
    if region_id == "pipeline_context":
        return False
    return (
        "flow_pipeline" in region_id
        or "architecture" in objective
        or kind == "pipeline_context_row"
    )


def _region_bbox(ir: dict, task: dict) -> list[float]:
    bbox = task.get("bbox")
    if bbox and len(bbox) == 4 and max(float(v) for v in bbox) > 0:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if _use_generic_flow(task):
            canvas = ir.get("canvas") or {}
            w = float(canvas.get("width_px") or max(x1, 1.0))
            h = float(canvas.get("height_px") or max(y1, 1.0))
            return [max(0.0, x0), max(0.0, y0), min(w, x1), min(h, y1)]
        return [x0 - 35, y0 - 55, x1 + 45, y1 + 40]
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    return [w * 0.50, h * 0.12, w * 0.98, h * 0.30]


def _pipeline_elements(ir: dict) -> list[dict]:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    r = ir.get("round", 0)
    y0, y1 = h * 0.145, h * 0.238
    card_w, card_h = w * 0.085, y1 - y0
    centers = [w * 0.560, w * 0.680, w * 0.800, w * 0.920]
    specs = [
        ("raw", "Raw\nTables", "database", "#8b8f96"),
        ("feature", "Feature\nEngineering", "gear", "#8b8f96"),
        ("cate", "CATE\nEstimator", "scatter", "#4f7db8"),
        ("ci", "CI\nEstimator", "bell", "#555555"),
    ]
    elements: list[dict] = [
        IR.element(
            id="pipeline_context_title",
            type="text",
            bbox=[w * 0.632, h * 0.112, w * 0.903, h * 0.148],
            provenance=IR.provenance("PipelineContextAgent", "pipeline_title", r),
            confidence=0.88,
            text="Existing causal decision pipeline (context)",
            font="Arial",
            font_size=25,
            text_color="#555555",
            align="center",
            z=7.0,
            ext=_ext("title"),
        )
    ]
    for idx, (key, title, icon, color) in enumerate(specs):
        cx = centers[idx]
        x0, x1 = cx - card_w / 2, cx + card_w / 2
        elements.extend([
            IR.element(
                id=f"pipeline_context_card_{key}",
                type="rounded_rect",
                bbox=[x0, y0, x1, y0 + card_h],
                provenance=IR.provenance("PipelineContextAgent", "pipeline_card", r),
                confidence=0.90,
                fill="#fbfbfc",
                border_color="#9fa5ad",
                border_width=1.12,
                corner=0.16,
                z=-0.15,
                ext=_ext("card", key),
            ),
            IR.element(
                id=f"pipeline_context_text_{key}",
                type="text",
                bbox=[x0 + card_w * 0.380, y0 + card_h * 0.235,
                      x1 - card_w * 0.045, y0 + card_h * 0.795],
                provenance=IR.provenance("PipelineContextAgent", "pipeline_text", r),
                confidence=0.88,
                text=title,
                font="Arial",
                font_size=19.8,
                text_color="#444444",
                align="center",
                z=8.0,
                ext=_ext("text", key),
            ),
        ])
        icon_box = [x0 + card_w * 0.115, y0 + card_h * 0.170,
                    x0 + card_w * 0.415, y0 + card_h * 0.780]
        elements.extend(_icon_elements(ir, key, icon_box, icon, color))
        if idx < len(specs) - 1:
            sx = centers[idx] + card_w / 2 + w * 0.006
            ex = centers[idx + 1] - card_w / 2 - w * 0.006
            cy = y0 + card_h * 0.52
            elements.append(IR.element(
                id=f"pipeline_context_arrow_{key}_{specs[idx + 1][0]}",
                type="arrow",
                bbox=[sx, cy - 12, ex, cy + 12],
                provenance=IR.provenance("PipelineContextAgent", "pipeline_arrow", r),
                confidence=0.84,
                points=[sx, cy, ex, cy],
                color="#8d8d8d",
                thickness=8,
                z=6.5,
                ext=_ext("arrow", key),
            ))
    elements.append(IR.element(
        id="pipeline_context_separator",
        type="line",
        bbox=[w * 0.500, h * 0.270, w * 0.985, h * 0.272],
        provenance=IR.provenance("PipelineContextAgent", "pipeline_separator", r),
        confidence=0.72,
        points=[w * 0.500, h * 0.271, w * 0.985, h * 0.271],
        color="#bfc2c7",
        thickness=1.0,
        dash=True,
        z=4.0,
        ext=_ext("separator"),
    ))
    return elements


def _generic_flow_elements(
    ir: dict,
    task: dict,
    region: list[float],
    source_elements: list[dict],
    original: Image.Image,
    provider: Any,
    ocr_provider: Any,
) -> list[dict]:
    """Rebuild a generic architecture/flow region from its own primitives."""
    r = ir.get("round", 0)
    cards = _flow_cards(source_elements, region)
    semantic = _read_flow_semantics(ir, original, region, cards, source_elements, provider)
    if not semantic.get("blocks"):
        semantic = _read_flow_ocr_semantics(ir, original, region, cards, ocr_provider)
    semantic_cards = _semantic_cards(semantic, region)
    cards = _merge_flow_cards(cards, semantic_cards)
    if not cards:
        return []
    elements: list[dict] = []
    for idx, card in enumerate(cards):
        bbox = [float(v) for v in card["bbox"][:4]]
        key = f"{idx:02d}"
        elements.append(IR.element(
            id=f"generic_flow_card_{key}",
            type="rounded_rect",
            bbox=bbox,
            provenance=IR.provenance("PipelineContextAgent", "generic_flow_card", r),
            confidence=float(card.get("confidence") or 0.78),
            fill=card.get("fill") or "#f8f9fb",
            border_color=card.get("border_color") or "#8f969e",
            border_width=float(card.get("border_width") or 1.0),
            corner=float(card.get("corner") or 0.10),
            z=float(card.get("z") or -0.05),
            ext=_generic_ext("card", key, task),
        ))
        labels = _texts_inside_card(source_elements, bbox)
        labels = labels or _semantic_labels_for_card(semantic, bbox, region)
        for j, label in enumerate(labels[:2]):
            elements.append(IR.element(
                id=f"generic_flow_text_{key}_{j}",
                type="text",
                bbox=[float(v) for v in label["bbox"][:4]],
                provenance=IR.provenance("PipelineContextAgent", "generic_flow_text", r),
                confidence=float(label.get("confidence") or 0.72),
                text=str(label.get("text") or ""),
                font=label.get("font") or "Arial",
                font_size=float(label.get("font_size") or _fit_label_font(label)),
                text_color=label.get("text_color") or "#333333",
                align=label.get("align") or "center",
                rotation=label.get("rotation"),
                z=float(label.get("z") or 6.0),
                ext=_generic_ext("text", key, task),
            ))
    elements.extend(_generic_flow_arrows(ir, cards, task))
    elements.extend(_semantic_connectors(ir, semantic, region, task))
    title = _generic_flow_title(source_elements, region) or _semantic_title(semantic, region, task, r)
    if title:
        elements.append(title)
    return elements


def _flow_cards(elements: list[dict], region: list[float]) -> list[dict]:
    candidates = []
    for el in elements:
        if el.get("type") not in {"rect", "rounded_rect"} or not el.get("bbox"):
            continue
        bbox = [float(v) for v in el["bbox"][:4]]
        if not _center_inside(bbox, region):
            continue
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < 12 or h < 28:
            continue
        if h > (region[3] - region[1]) * 0.72 and w > (region[2] - region[0]) * 0.55:
            continue
        candidates.append(el)
    candidates.sort(key=lambda e: (
        (float(e["bbox"][0]) + float(e["bbox"][2])) / 2.0,
        (float(e["bbox"][1]) + float(e["bbox"][3])) / 2.0,
    ))
    deduped: list[dict] = []
    for el in candidates:
        if any(_bbox_overlap_fraction(el.get("bbox"), old.get("bbox")) > 0.65 for old in deduped):
            continue
        deduped.append(el)
    return deduped[:12]


def _read_flow_semantics(
    ir: dict,
    original: Image.Image,
    region: list[float],
    cards: list[dict],
    source_elements: list[dict],
    provider: Any,
) -> dict:
    if os.environ.get("I2E_DISABLE_FLOW_REGION_VLM", "0") == "1":
        return {}
    if len(cards) < 3:
        return {}
    existing_labels = [
        el for el in source_elements
        if el.get("type") in {"text", "formula"}
        and el.get("bbox")
        and _center_inside(el["bbox"], region)
        and _usable_flow_label(el)
        and any(_center_inside(el["bbox"], c.get("bbox")) for c in cards if c.get("bbox"))
    ]
    if len(existing_labels) >= max(3, len(cards) // 2):
        return {}
    try:
        crop, crop_box = _crop_region(original, region, pad=0.02)
        raw = _ask_flow_json(provider, crop)
        semantic = _normalize_flow_semantics(raw, crop_box)
        ir.setdefault("agent_tool_calls", []).append({
            "agent": "PipelineContextAgent",
            "tool": getattr(provider, "name", "vlm"),
            "action": "read_flow_region_semantics",
            "region": region,
            "blocks": len(semantic.get("blocks") or []),
            "connectors": len(semantic.get("connectors") or []),
        })
        return semantic
    except Exception as exc:
        ir.setdefault("agent_tool_failures", []).append({
            "agent": "PipelineContextAgent",
            "tool": getattr(provider, "name", "vlm"),
            "action": "read_flow_region_semantics",
            "error": f"{type(exc).__name__}: {exc}",
            "region": region,
        })
        return {}


def _ask_flow_json(provider: Any, crop: Image.Image) -> dict:
    max_tokens = int(os.environ.get("I2E_FLOW_REGION_MAX_TOKENS", "4096"))
    base_url = str(getattr(provider, "base_url", "") or "").lower()
    old_timeout = getattr(provider, "timeout", None)
    try:
        if old_timeout is not None:
            provider.timeout = float(os.environ.get("I2E_FLOW_REGION_TIMEOUT", "18"))
        return _ask_flow_json_with_timeout(provider, crop, max_tokens, base_url)
    finally:
        if old_timeout is not None:
            provider.timeout = old_timeout


def _ask_flow_json_with_timeout(provider: Any, crop: Image.Image, max_tokens: int, base_url: str) -> dict:
    if "siliconflow.cn" not in base_url:
        return provider.ask_json(
            crop,
            _FLOW_SEMANTIC_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    old = os.environ.get("I2E_SILICONFLOW_SEND_THINKING_OPTIONS")
    first_error: Exception | None = None
    try:
        os.environ["I2E_SILICONFLOW_SEND_THINKING_OPTIONS"] = "1"
        return provider.ask_json(
            crop,
            _FLOW_SEMANTIC_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        first_error = exc
    finally:
        if old is None:
            os.environ.pop("I2E_SILICONFLOW_SEND_THINKING_OPTIONS", None)
        else:
            os.environ["I2E_SILICONFLOW_SEND_THINKING_OPTIONS"] = old

    try:
        return provider.ask_json(
            crop,
            _FLOW_SEMANTIC_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(
            f"strict_json_failed={type(first_error).__name__}: {first_error}; "
            f"fallback_failed={type(exc).__name__}: {exc}"
        ) from exc


def _read_flow_ocr_semantics(
    ir: dict,
    original: Image.Image,
    region: list[float],
    cards: list[dict],
    provider: Any,
) -> dict:
    if os.environ.get("I2E_DISABLE_FLOW_REGION_OCR", "0") == "1":
        return {}
    if not cards:
        return {}
    try:
        crop, _ = _crop_region(original, region, pad=0.0)
        old_timeout = getattr(provider, "timeout", None)
        try:
            if old_timeout is not None:
                provider.timeout = float(os.environ.get("I2E_FLOW_OCR_TIMEOUT", "12"))
            raw = provider.ocr(
                crop,
                prompt=(
                    "Read only the visible labels of process blocks/cards in this "
                    "diagram crop. Return one label per line. Do not explain."
                ),
            )
        finally:
            if old_timeout is not None:
                provider.timeout = old_timeout
        labels = _parse_ocr_labels(raw)
        min_labels = min(2, max(1, len(cards)))
        if len(labels) < min_labels:
            ir.setdefault("agent_tool_failures", []).append({
                "agent": "PipelineContextAgent",
                "tool": getattr(provider, "name", "ocr"),
                "action": "read_flow_region_ocr_labels",
                "error": f"inadequate_ocr_labels:{len(labels)}",
                "region": region,
            })
            return {}
        ordered = sorted(cards, key=lambda e: (
            (float(e["bbox"][0]) + float(e["bbox"][2])) / 2.0,
            (float(e["bbox"][1]) + float(e["bbox"][3])) / 2.0,
        ))
        blocks = []
        for card, label in zip(ordered, labels):
            bbox = [float(v) for v in card["bbox"][:4]]
            blocks.append({
                "bbox": bbox,
                "text": label,
                "role": "process",
                "rotation": 90 if (bbox[3] - bbox[1]) > (bbox[2] - bbox[0]) * 1.35 else 0,
            })
        ir.setdefault("agent_tool_calls", []).append({
            "agent": "PipelineContextAgent",
            "tool": getattr(provider, "name", "ocr"),
            "action": "read_flow_region_ocr_labels",
            "region": region,
            "labels": len(labels),
            "blocks": len(blocks),
        })
        return {"blocks": blocks, "connectors": []}
    except Exception as exc:
        ir.setdefault("agent_tool_failures", []).append({
            "agent": "PipelineContextAgent",
            "tool": getattr(provider, "name", "ocr"),
            "action": "read_flow_region_ocr_labels",
            "error": f"{type(exc).__name__}: {exc}",
            "region": region,
        })
        return {}


def _parse_ocr_labels(raw: Any) -> list[str]:
    labels: list[str] = []
    for line in str(raw or "").splitlines():
        text = line.strip().strip("-*•0123456789. ")
        text = re.sub(r"\s+", " ", text)
        if not text or len(text) < 3:
            continue
        lowered = text.lower()
        if lowered.startswith(("here are", "the labels", "visible labels")):
            continue
        if "return one label" in lowered or "do not explain" in lowered:
            continue
        letters = sum(ch.isalpha() for ch in text)
        if letters < 3:
            continue
        if text not in labels:
            labels.append(text[:80])
    return labels[:14]


def _crop_region(image: Image.Image, region: list[float], pad: float = 0.02) -> tuple[Image.Image, list[float]]:
    x0, y0, x1, y1 = [float(v) for v in region]
    w, h = x1 - x0, y1 - y0
    box = [
        max(0.0, x0 - w * pad),
        max(0.0, y0 - h * pad),
        min(float(image.width), x1 + w * pad),
        min(float(image.height), y1 + h * pad),
    ]
    left, top, right, bottom = [int(round(v)) for v in box]
    return image.crop((left, top, right, bottom)), [float(left), float(top), float(right), float(bottom)]


def _normalize_flow_semantics(raw: dict, crop_box: list[float]) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {"blocks": [], "connectors": []}
    title = raw.get("title")
    if isinstance(title, dict):
        tb = _denorm_bbox(title.get("bbox"), crop_box)
        text = _clean_label(title.get("text"))
        if tb and text:
            out["title"] = {"text": text, "bbox": tb}
    for block in raw.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        bbox = _denorm_bbox(block.get("bbox"), crop_box)
        if not bbox:
            continue
        label = _clean_label(block.get("label") or block.get("text"))
        role = str(block.get("role") or "process").lower()
        if role not in {"process", "axis_label", "thumbnail", "chart", "panel", "annotation"}:
            role = "process"
        item = {
            "bbox": bbox,
            "text": label,
            "role": role,
            "rotation": _normalize_rotation(block.get("rotation")),
        }
        fill = _hex_or_none(block.get("fill"))
        color = _hex_or_none(block.get("text_color"))
        if fill:
            item["fill"] = fill
        if color:
            item["text_color"] = color
        out["blocks"].append(item)
    for conn in raw.get("connectors") or []:
        if not isinstance(conn, dict):
            continue
        bbox = _denorm_bbox(conn.get("bbox"), crop_box)
        if not bbox:
            continue
        out["connectors"].append({
            "bbox": bbox,
            "direction": str(conn.get("direction") or "right").lower(),
        })
    return out


def _denorm_bbox(bbox: Any, crop_box: list[float]) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.5:
        return [x0 + crop_box[0], y0 + crop_box[1], x1 + crop_box[0], y1 + crop_box[1]]
    cw, ch = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    return [
        crop_box[0] + x0 * cw,
        crop_box[1] + y0 * ch,
        crop_box[0] + x1 * cw,
        crop_box[1] + y1 * ch,
    ]


def _semantic_cards(semantic: dict, region: list[float]) -> list[dict]:
    cards = []
    for idx, block in enumerate(semantic.get("blocks") or []):
        if block.get("role") not in {"process", "panel", "chart", "thumbnail"}:
            continue
        bbox = block.get("bbox")
        if not bbox or not _center_inside(bbox, region):
            continue
        w, h = float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])
        if w < 10 or h < 12:
            continue
        cards.append({
            "id": f"semantic_flow_card_{idx}",
            "type": "rounded_rect",
            "bbox": bbox,
            "fill": block.get("fill") or "#f8f9fb",
            "border_color": "#8f969e",
            "confidence": 0.70,
        })
    return cards


def _merge_flow_cards(cards: list[dict], semantic_cards: list[dict]) -> list[dict]:
    merged = list(cards)
    for card in semantic_cards:
        if any(_bbox_overlap_fraction(card.get("bbox"), old.get("bbox")) > 0.55 for old in merged):
            continue
        merged.append(card)
    merged.sort(key=lambda e: (
        (float(e["bbox"][0]) + float(e["bbox"][2])) / 2.0,
        (float(e["bbox"][1]) + float(e["bbox"][3])) / 2.0,
    ))
    return merged[:14]


def _semantic_labels_for_card(semantic: dict, card_bbox: list[float], region: list[float]) -> list[dict]:
    labels = []
    for block in semantic.get("blocks") or []:
        text = str(block.get("text") or "").strip()
        bbox = block.get("bbox")
        if not text or not bbox:
            continue
        if not (_center_inside(bbox, card_bbox) or _bbox_overlap_fraction(bbox, card_bbox) > 0.35):
            continue
        labels.append({
            "bbox": _label_bbox_inside_card(block, card_bbox),
            "text": text,
            "font": "Arial",
            "text_color": block.get("text_color") or "#222222",
            "align": "center",
            "rotation": block.get("rotation"),
            "confidence": 0.70,
        })
    labels.sort(key=lambda e: (float(e["bbox"][1]), float(e["bbox"][0])))
    return labels


def _label_bbox_inside_card(block: dict, card_bbox: list[float]) -> list[float]:
    bbox = [float(v) for v in (block.get("bbox") or card_bbox)[:4]]
    if _bbox_overlap_fraction(bbox, card_bbox) < 0.35:
        x0, y0, x1, y1 = [float(v) for v in card_bbox]
        pad_x = max(2.0, (x1 - x0) * 0.08)
        pad_y = max(2.0, (y1 - y0) * 0.08)
        return [x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y]
    return bbox


def _semantic_connectors(ir: dict, semantic: dict, region: list[float], task: dict) -> list[dict]:
    out = []
    r = ir.get("round", 0)
    for idx, conn in enumerate(semantic.get("connectors") or []):
        bbox = conn.get("bbox")
        if not bbox or not _center_inside(bbox, region):
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        direction = str(conn.get("direction") or "right").lower()
        if direction == "left":
            points = [x1, (y0 + y1) / 2.0, x0, (y0 + y1) / 2.0]
        elif direction == "up":
            points = [(x0 + x1) / 2.0, y1, (x0 + x1) / 2.0, y0]
        elif direction == "down":
            points = [(x0 + x1) / 2.0, y0, (x0 + x1) / 2.0, y1]
        else:
            points = [x0, (y0 + y1) / 2.0, x1, (y0 + y1) / 2.0]
        out.append(IR.element(
            id=f"generic_flow_semantic_arrow_{idx:02d}",
            type="arrow",
            bbox=[min(points[0], points[2]), min(points[1], points[3]), max(points[0], points[2]), max(points[1], points[3])],
            provenance=IR.provenance("PipelineContextAgent", "generic_flow_semantic_arrow", r),
            confidence=0.70,
            points=points,
            color="#8d8d8d",
            thickness=4.0,
            z=5.2,
            ext=_generic_ext("arrow", f"semantic_{idx:02d}", task),
        ))
    return out


def _semantic_title(semantic: dict, region: list[float], task: dict, round_num: int) -> dict | None:
    title = semantic.get("title")
    if not isinstance(title, dict) or not title.get("text") or not title.get("bbox"):
        return None
    bbox = [float(v) for v in title["bbox"][:4]]
    if not _center_inside(bbox, region):
        return None
    return IR.element(
        id="generic_flow_title",
        type="text",
        bbox=bbox,
        provenance=IR.provenance("PipelineContextAgent", "generic_flow_semantic_title", round_num),
        confidence=0.72,
        text=str(title.get("text") or ""),
        font="Arial",
        font_size=_fit_label_font(title),
        text_color="#111111",
        align="left",
        z=7.0,
        ext=_generic_ext("title", "title", task),
    )


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 80:
        text = text[:80].rstrip()
    return text


def _normalize_rotation(value: Any) -> int:
    try:
        rot = int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0
    if abs(rot) >= 45:
        return 90 if rot > 0 else -90
    return 0


def _hex_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.lower()
    return None


def _fit_label_font(label: dict) -> float:
    text = str(label.get("text") or "")
    bbox = label.get("bbox") or [0, 0, 80, 20]
    try:
        w = max(1.0, float(bbox[2]) - float(bbox[0]))
        h = max(1.0, float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError, IndexError):
        return 12.0
    chars = max(1, len(text.replace("\n", "")))
    by_width = w / max(4.0, chars * 0.58)
    by_height = h * 0.62
    return round(max(5.5, min(16.0, by_width, by_height)), 1)


def _texts_inside_card(elements: list[dict], card_bbox: list[float]) -> list[dict]:
    out = []
    for el in elements:
        if el.get("type") not in {"text", "formula"} or not el.get("bbox"):
            continue
        if _center_inside(el["bbox"], card_bbox) and _usable_flow_label(el):
            out.append(el)
    out.sort(key=lambda e: (float(e["bbox"][1]), float(e["bbox"][0])))
    return out


def _usable_flow_label(el: dict) -> bool:
    text = str(el.get("text") or el.get("latex") or "").strip()
    if len(text) < 3:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 3:
        return False
    if re.fullmatch(r"[-—_\\/\sA-Z]{1,8}", text):
        return False
    return True


def _generic_flow_arrows(ir: dict, cards: list[dict], task: dict) -> list[dict]:
    out = []
    r = ir.get("round", 0)
    ordered = sorted(cards, key=lambda e: (float(e["bbox"][0]) + float(e["bbox"][2])) / 2.0)
    for idx, (left, right) in enumerate(zip(ordered, ordered[1:])):
        lb = [float(v) for v in left["bbox"][:4]]
        rb = [float(v) for v in right["bbox"][:4]]
        ly = (lb[1] + lb[3]) / 2.0
        ry = (rb[1] + rb[3]) / 2.0
        if abs(ly - ry) > max(lb[3] - lb[1], rb[3] - rb[1]) * 0.55:
            continue
        sx, ex = lb[2] + 5.0, rb[0] - 5.0
        if ex <= sx + 8.0:
            continue
        cy = (ly + ry) / 2.0
        out.append(IR.element(
            id=f"generic_flow_arrow_{idx:02d}",
            type="arrow",
            bbox=[sx, cy - 8.0, ex, cy + 8.0],
            provenance=IR.provenance("PipelineContextAgent", "generic_flow_arrow", r),
            confidence=0.72,
            points=[sx, cy, ex, cy],
            color="#8d8d8d",
            thickness=5.0,
            z=5.0,
            ext=_generic_ext("arrow", f"{idx:02d}", task),
        ))
    return out


def _generic_flow_title(elements: list[dict], region: list[float]) -> dict | None:
    for el in elements:
        if el.get("type") != "text" or not el.get("bbox"):
            continue
        text = str(el.get("text") or "")
        if "architecture" not in text.lower() and "pipeline" not in text.lower():
            continue
        bbox = [float(v) for v in el["bbox"][:4]]
        if not _center_inside(bbox, region):
            continue
        out = dict(el)
        out["id"] = "generic_flow_title"
        out.setdefault("provenance", IR.provenance("PipelineContextAgent", "generic_flow_title"))
        out.setdefault("ext", {}).update(_generic_ext("title", "title", {}))
        return out
    return None


def _icon_elements(ir: dict, key: str, box: list[float], kind: str, color: str) -> list[dict]:
    if kind == "database":
        return _database_icon(ir, key, box, color)
    if kind == "gear":
        return _gear_icon(ir, key, box, color)
    if kind == "scatter":
        return _scatter_icon(ir, key, box, color)
    if kind == "bell":
        return _bell_icon(ir, key, box, color)
    return []


def _database_icon(ir: dict, key: str, box: list[float], color: str) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    els: list[dict] = []
    for i, frac in enumerate([0.18, 0.40, 0.62]):
        cy = y0 + h * frac
        els.append(IR.element(
            id=f"pipeline_context_icon_{key}_oval_{i}",
            type="oval",
            bbox=[x0 + w * 0.23, cy, x1 - w * 0.23, cy + h * 0.18],
            provenance=IR.provenance("PipelineContextAgent", "database_icon", r),
            confidence=0.82,
            fill="#f2f2f2",
            border_color=color,
            border_width=1.3,
            z=9.0,
            ext=_ext("icon", key),
        ))
        if i < 2:
            els.append(_line(ir, f"pipeline_context_icon_{key}_side_{i}a",
                             x0 + w * 0.23, cy + h * 0.09, x0 + w * 0.23, cy + h * 0.30,
                             color, 9.0, key))
            els.append(_line(ir, f"pipeline_context_icon_{key}_side_{i}b",
                             x1 - w * 0.23, cy + h * 0.09, x1 - w * 0.23, cy + h * 0.30,
                             color, 9.0, key))
    return els


def _gear_icon(ir: dict, key: str, box: list[float], color: str) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = ir.get("round", 0)
    els: list[dict] = []
    for i, (tx, ty) in enumerate([
        (0.50, 0.18), (0.70, 0.26), (0.82, 0.50), (0.70, 0.74),
        (0.50, 0.82), (0.30, 0.74), (0.18, 0.50), (0.30, 0.26),
    ]):
        els.append(IR.element(
            id=f"pipeline_context_icon_{key}_tooth_{i}",
            type="rect",
            bbox=[
                x0 + w * tx - w * 0.040,
                y0 + h * ty - h * 0.040,
                x0 + w * tx + w * 0.040,
                y0 + h * ty + h * 0.040,
            ],
            provenance=IR.provenance("PipelineContextAgent", "gear_tooth", r),
            confidence=0.78,
            fill="#ffffff",
            border_color=color,
            border_width=1.0,
            z=8.9,
            ext=_ext("icon", key),
        ))
    els.append(IR.element(
        id=f"pipeline_context_icon_{key}_ring",
        type="oval",
        bbox=[cx - w * 0.245, cy - h * 0.245, cx + w * 0.245, cy + h * 0.245],
        provenance=IR.provenance("PipelineContextAgent", "gear_icon", r),
        confidence=0.82,
        fill="#ffffff",
        border_color=color,
        border_width=1.8,
        z=9.1,
        ext=_ext("icon", key),
    ))
    els.append(IR.element(
        id=f"pipeline_context_icon_{key}_hub",
        type="oval",
        bbox=[cx - w * 0.070, cy - h * 0.070, cx + w * 0.070, cy + h * 0.070],
        provenance=IR.provenance("PipelineContextAgent", "gear_icon", r),
        confidence=0.82,
        fill="#ffffff",
        border_color=color,
        border_width=1.05,
        z=9.2,
        ext=_ext("icon", key),
    ))
    wrench = _line(
        ir,
        f"pipeline_context_icon_{key}_wrench_handle",
        x0 + w * 0.30,
        y1 - h * 0.22,
        x1 - w * 0.25,
        y0 + h * 0.30,
        color,
        9.5,
        key,
    )
    wrench.update({"thickness": 2.4, "line_width": 2.4})
    jaw_a = _line(ir, f"pipeline_context_icon_{key}_wrench_jaw_a",
                  x1 - w * 0.28, y0 + h * 0.31, x1 - w * 0.18, y0 + h * 0.22,
                  color, 9.6, key)
    jaw_b = _line(ir, f"pipeline_context_icon_{key}_wrench_jaw_b",
                  x1 - w * 0.28, y0 + h * 0.31, x1 - w * 0.18, y0 + h * 0.40,
                  color, 9.6, key)
    for part in (jaw_a, jaw_b):
        part.update({"thickness": 2.0, "line_width": 2.0})
    els.extend([wrench, jaw_a, jaw_b])
    return els


def _scatter_icon(ir: dict, key: str, box: list[float], color: str) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    pts = []
    dots = []
    for i in range(12):
        t = i / 11
        px = w * (0.18 + 0.68 * t)
        py = h * (0.78 - 0.50 * t + 0.10 * math.sin(t * math.pi * 2))
        pts.append([px, py])
        dots.append({"cx": px + (i % 3 - 1) * 2.0, "cy": py + ((i * 5) % 5 - 2) * 2.0, "r": 1.8, "color": color})
    return [
        _line(ir, f"pipeline_context_icon_{key}_axis_x", x0 + w * 0.12, y1 - h * 0.14, x1 - w * 0.08, y1 - h * 0.14, "#aab3bf", 8.8, key),
        _line(ir, f"pipeline_context_icon_{key}_axis_y", x0 + w * 0.12, y1 - h * 0.14, x0 + w * 0.12, y0 + h * 0.10, "#aab3bf", 8.8, key),
        IR.element(
            id=f"pipeline_context_icon_{key}_plot",
            type="dotcloud",
            bbox=box,
            provenance=IR.provenance("PipelineContextAgent", "scatter_icon", r),
            confidence=0.82,
            dots=dots,
            paths=[{"points": pts, "closed": False, "line": color, "line_width": 1.5}],
            z=9.0,
            ext=_ext("icon", key),
        ),
    ]


def _bell_icon(ir: dict, key: str, box: list[float], color: str) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    pts = []
    for i in range(25):
        t = i / 24
        x = w * (0.12 + 0.76 * t)
        g = math.exp(-((t - 0.50) ** 2) / 0.035)
        y = h * (0.78 - 0.56 * g)
        pts.append([x, y])
    els = [
        IR.element(
            id=f"pipeline_context_icon_{key}_bell",
            type="dotcloud",
            bbox=box,
            provenance=IR.provenance("PipelineContextAgent", "bell_icon", r),
            confidence=0.82,
            dots=[],
            paths=[{"points": pts, "closed": False, "line": color, "line_width": 1.9}],
            z=9.0,
            ext=_ext("icon", key),
        ),
        _line(ir, f"pipeline_context_icon_{key}_axis", x0 + w * 0.13, y1 - h * 0.18, x1 - w * 0.10, y1 - h * 0.18, "#aab3bf", 8.8, key),
    ]
    ci = _line(ir, f"pipeline_context_icon_{key}_ci", x0 + w * 0.25, y1 - h * 0.11, x1 - w * 0.23, y1 - h * 0.11, color, 9.0, key)
    ci.update({"dash": True, "thickness": 1.7, "line_width": 1.7})
    arrow = _line(ir, f"pipeline_context_icon_{key}_ci_arrow", x0 + w * 0.40, y1 - h * 0.11, x1 - w * 0.36, y1 - h * 0.11, color, 9.2, key)
    arrow.update({"type": "arrow", "thickness": 1.9, "line_width": 1.9})
    els.extend([ci, arrow])
    return els


def _line(ir: dict, eid: str, x0: float, y0: float, x1: float, y1: float,
          color: str, z: float, key: str) -> dict:
    r = ir.get("round", 0)
    return IR.element(
        id=eid,
        type="line",
        bbox=[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
        provenance=IR.provenance("PipelineContextAgent", "pipeline_icon_line", r),
        confidence=0.80,
        points=[x0, y0, x1, y1],
        color=color,
        thickness=1.2,
        z=z,
        ext=_ext("icon", key),
    )


def _remove_orphans(ir: dict, region: list[float]) -> set[str]:
    removable = {"text", "formula", "rounded_rect", "rect", "icon", "arrow", "line", "freeform", "dotcloud"}
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if eid.startswith("pipeline_context_") or eid.startswith("proc_") or eid.startswith("auditor_") or not bbox:
            keep.append(el)
            continue
        if el.get("type") in removable and _center_inside(bbox, region):
            removed.add(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _remove_generic_flow_outputs(ir: dict) -> set[str]:
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        if eid.startswith("generic_flow_"):
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


def _bbox_overlap_fraction(a: list | tuple | None, b: list | tuple | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / area


def _generic_ext(role: str, key: str, task: dict) -> dict:
    return {
        "component": "generic_flow_pipeline",
        "component_key": key,
        "component_role": "process_card" if role == "text" else role,
        "strategy": {
            "region_id": task.get("region_id") or "generic_flow_pipeline",
            "kind": task.get("kind") or "pipeline_context_row",
            "primary_method": "pipeline_context_layout",
            "fallback_methods": ["text_style", "native_trace"],
            "preferred_agent": "PipelineContextAgent",
        },
    }


def _ext(role: str, key: str = "") -> dict:
    return {
        "component": "pipeline_context",
        "component_key": key,
        "component_role": role,
        "strategy": {
            "region_id": "region_pipeline_context",
            "kind": "pipeline_context_row",
            "primary_method": "pipeline_context_layout",
            "fallback_methods": ["text_style", "native_trace"],
            "preferred_agent": "PipelineContextAgent",
        },
    }
