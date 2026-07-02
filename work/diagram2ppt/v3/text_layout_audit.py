"""Rendered text layout diagnostics.

OCR tells us what the text says, but the native renderer still has to match
where the glyph ink lands, how large it is, and what color it uses.  This
module compares each editable text/formula element in the original image and
the current render, then emits planner-readable corrections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class InkStats:
    bbox: tuple[float, float, float, float]
    area_frac: float
    color: tuple[float, float, float]


def audit(
    elements: list[dict],
    original: Image.Image,
    rendered: Image.Image,
) -> tuple[list[dict], dict[str, Any]]:
    """Return text-layout mismatch defects and aggregate metrics."""
    defects: list[dict] = []
    errors: list[float] = []
    template_errors: list[float] = []
    color_mismatches = 0
    template_mismatches = 0

    for el in elements:
        if el.get("type") not in {"text", "formula"} or not el.get("bbox"):
            continue
        if _skip_element(el):
            continue
        diag = diagnose_element(el, original, rendered)
        if not diag:
            continue
        if _text_color_locked(el):
            diag = _suppress_color_repair(diag)
        if _protected_role(el):
            template_errors.append(float(diag["layout_error"]))
            if diag["needs_repair"]:
                template_mismatches += 1
                role = str((((el.get("ext") or {}).get("typography") or {}).get("role")) or "")
                defects.append({
                    "id": f"defect_text_template_{el['id']}",
                    "type": "text_template_mismatch",
                    "element_id": el["id"],
                    "bbox": el["bbox"],
                    "severity": round(min(1.0, float(diag["layout_error"])), 3),
                    "reason": f"template text slot mismatch role={role} {_reason(diag)}",
                    "suggested_agent": "TemplateSlotAgent",
                    "text_layout": diag,
                    "template_role": role,
                })
            continue
        errors.append(float(diag["layout_error"]))
        if diag.get("color_distance", 0.0) >= 58.0:
            color_mismatches += 1
        if not diag["needs_repair"]:
            continue
        defects.append({
            "id": f"defect_text_layout_{el['id']}",
            "type": "text_layout_mismatch",
            "element_id": el["id"],
            "bbox": el["bbox"],
            "severity": round(min(1.0, float(diag["layout_error"])), 3),
            "reason": _reason(diag),
            "suggested_agent": "TextLayoutAgent",
            "text_layout": diag,
        })

    metrics = {
        "text_layout_mismatch_count": sum(
            1 for d in defects if d.get("type") == "text_layout_mismatch"
        ),
        "text_layout_error": round(float(np.mean(errors)) if errors else 0.0, 4),
        "text_color_mismatch_count": color_mismatches,
        "text_template_mismatch_count": template_mismatches,
        "text_template_error": round(float(np.mean(template_errors)) if template_errors else 0.0, 4),
    }
    return defects, metrics


def diagnose_element(
    el: dict,
    original: Image.Image,
    rendered: Image.Image,
) -> dict[str, Any] | None:
    """Compare original/rendered glyph ink for one text-like element."""
    x0, y0, x1, y1 = _clean_bbox(el["bbox"], original.width, original.height)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    ocrop = original.crop((x0, y0, x1, y1)).convert("RGB")
    rcrop = rendered.crop((x0, y0, x1, y1)).convert("RGB")
    orig = _ink_stats(ocrop)
    rend = _ink_stats(rcrop)
    if not orig or not rend:
        return None

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    ob = orig.bbox
    rb = rend.bbox
    ocx, ocy = (ob[0] + ob[2]) * 0.5, (ob[1] + ob[3]) * 0.5
    rcx, rcy = (rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5
    ow, oh = max(1.0, ob[2] - ob[0]), max(1.0, ob[3] - ob[1])
    rw, rh = max(1.0, rb[2] - rb[0]), max(1.0, rb[3] - rb[1])

    dx = ocx - rcx
    dy = ocy - rcy
    height_scale = _clamp(oh / rh, 0.72, 1.34)
    width_scale = _clamp(ow / rw, 0.72, 1.34)
    font_scale = _clamp((height_scale * 0.72 + width_scale * 0.28), 0.76, 1.28)
    color_distance = _rgb_distance(orig.color, rend.color)

    shift_error = min(1.0, (abs(dx) / bw + abs(dy) / bh) * 2.6)
    size_error = min(1.0, abs(np.log(max(0.2, height_scale))) * 1.8
                     + abs(np.log(max(0.2, width_scale))) * 0.7)
    color_error = min(1.0, color_distance / 150.0)
    area_error = min(1.0, abs(orig.area_frac - rend.area_frac) * 12.0)
    layout_error = 0.42 * shift_error + 0.34 * size_error + 0.16 * color_error + 0.08 * area_error

    needs_repair = (
        abs(dx) > max(4.0, bw * 0.035)
        or abs(dy) > max(3.0, bh * 0.045)
        or abs(height_scale - 1.0) > 0.13
        or abs(width_scale - 1.0) > 0.18
        or color_distance >= 58.0
    )
    if layout_error < 0.085:
        needs_repair = False

    return {
        "original_ink_bbox": [round(x0 + ob[0], 2), round(y0 + ob[1], 2),
                              round(x0 + ob[2], 2), round(y0 + ob[3], 2)],
        "rendered_ink_bbox": [round(x0 + rb[0], 2), round(y0 + rb[1], 2),
                              round(x0 + rb[2], 2), round(y0 + rb[3], 2)],
        "shift_px": [round(dx, 2), round(dy, 2)],
        "font_scale": round(font_scale, 4),
        "height_scale": round(height_scale, 4),
        "width_scale": round(width_scale, 4),
        "original_color": _hex(orig.color),
        "rendered_color": _hex(rend.color),
        "color_distance": round(color_distance, 2),
        "layout_error": round(layout_error, 4),
        "needs_repair": needs_repair,
    }


def _ink_stats(crop: Image.Image) -> InkStats | None:
    arr = np.asarray(crop.convert("RGB")).astype(np.int16)
    h, w = arr.shape[:2]
    if h < 4 or w < 4:
        return None
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    gray = np.asarray(crop.convert("L")).astype(np.int16)
    bg_l = float(np.median(np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])))
    ldist = np.abs(gray - bg_l)
    mask = (dist > 28.0) | (ldist > 24.0)
    # Drop tiny anti-alias specks.
    ys, xs = np.where(mask)
    if len(xs) < max(8, int(w * h * 0.002)):
        return None
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    ink_pixels = arr[mask].astype(np.float32)
    # Prefer the darkest/most saturated pixels for color; border and light grid
    # lines can otherwise dominate tiny math labels.
    lum = ink_pixels.mean(axis=1)
    sat = ink_pixels.max(axis=1) - ink_pixels.min(axis=1)
    order = np.argsort(lum - sat * 0.35)
    sample = ink_pixels[order[: max(1, min(len(order), 60))]]
    color = tuple(float(v) for v in np.median(sample, axis=0))
    return InkStats(
        bbox=(x0, y0, x1, y1),
        area_frac=float(mask.sum()) / float(w * h),
        color=color,
    )


def _skip_element(el: dict) -> bool:
    text = str(el.get("text") or el.get("latex") or "").strip()
    if not text:
        return True
    # Single punctuation-like fragments are usually decorative tick remnants;
    # let chart/component agents own them.
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    return len(text) <= 1 and letters + digits == 0


def _text_color_locked(el: dict) -> bool:
    ext = el.get("ext") or {}
    if ext.get("text_color_locked"):
        return True
    contract = ext.get("text_contract") or {}
    if isinstance(contract, dict) and contract.get("text_color_locked"):
        return True
    role = str((ext.get("typography") or {}).get("role") or "")
    return role in {
        "action_body",
        "action_body_emphasis",
        "action_body_math",
        "action_report_body",
        "action_report_body_emphasis",
    }


def _suppress_color_repair(diag: dict[str, Any]) -> dict[str, Any]:
    out = dict(diag)
    ignored = float(out.get("color_distance") or 0.0)
    out["color_distance_ignored"] = ignored
    out["color_distance"] = 0.0
    color_penalty = min(1.0, ignored / 150.0) * 0.16
    out["layout_error"] = round(max(0.0, float(out.get("layout_error") or 0.0) - color_penalty), 4)
    dx, dy = out.get("shift_px", [0.0, 0.0])
    height_scale = float(out.get("height_scale") or 1.0)
    width_scale = float(out.get("width_scale") or 1.0)
    needs = (
        abs(float(dx)) > 4.0
        or abs(float(dy)) > 3.0
        or abs(height_scale - 1.0) > 0.13
        or abs(width_scale - 1.0) > 0.18
    )
    out["needs_repair"] = bool(needs and float(out.get("layout_error") or 0.0) >= 0.085)
    return out


def _protected_role(el: dict) -> bool:
    role = str((((el.get("ext") or {}).get("typography") or {}).get("role")) or "")
    return role in {
        "slide_title",
        "solution_title",
        "section_title",
        "subtitle",
        "caption",
        "process_title",
        "auditor_title",
        "auditor_group_label",
        "chart_title",
        "chart_title_q",
        "chart_title_sub",
        "chart_title_rest",
        "failure_title",
        "action_title",
        "action_report_title",
        "checklist_body",
        "covariate_label",
        "covariate_text",
        "covariate_math",
        "axis_math",
        "vector_label",
        "surface_vector_math",
        "surface_theta_math",
        "ci_axis_label",
        "risk_label",
        "risk_label_math",
        "risk_q_math",
    }


def _clean_bbox(bbox: list, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox[:4]]
    return (
        max(0, min(width - 1, x0)),
        max(0, min(height - 1, y0)),
        max(1, min(width, x1)),
        max(1, min(height, y1)),
    )


def _rgb_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = [int(max(0, min(255, round(v)))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _reason(diag: dict[str, Any]) -> str:
    dx, dy = diag.get("shift_px", [0, 0])
    return (
        "text ink layout mismatch "
        f"shift=({dx},{dy}) font_scale={diag.get('font_scale')} "
        f"color_delta={diag.get('color_distance')}"
    )
