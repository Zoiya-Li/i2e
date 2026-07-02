"""AuditorCardAgent: semantic native reconstruction for CATE-CI method cards."""
from __future__ import annotations

import math
from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import ir as IR
from work.diagram2ppt.v3.agents.base import Agent


class AuditorCardAgent(Agent):
    """Rebuild the five auditor method cards with native mini diagrams."""

    name = "AuditorCardAgent"

    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        task = kwargs.get("task") or {}
        if not _is_auditor_task(task):
            return []
        specs = _card_specs(ir, task)
        changed = set(_remove_orphans(ir, specs))
        for spec in specs:
            for el in _card_elements(ir, spec):
                existing = IR.get_element(ir, el["id"])
                if existing:
                    existing.clear()
                    existing.update(el)
                else:
                    ir.setdefault("elements", []).append(el)
                changed.add(el["id"])
        for el in _group_elements(ir, specs):
            existing = IR.get_element(ir, el["id"])
            if existing:
                existing.clear()
                existing.update(el)
            else:
                ir.setdefault("elements", []).append(el)
            changed.add(el["id"])
        ir.setdefault("history", []).append({
            "agent": self.name,
            "action": "native_auditor_card_transaction",
            "round": ir.get("round", 0),
            "changed": sorted(changed),
        })
        return sorted(changed)


SPECS = [
    ("propensity", "1", "Propensity\nModel", "T ~ X → β̂", "#2f7dbd"),
    ("surrogate", "2", "Surrogate\nCATE", "Y ~ X + X·T → τ̃", "#45a9d1"),
    ("heterogeneity", "3", "Heterogeneity\nGradient", "τ̃ ~ X → γ̂", "#3c9b86"),
    ("alignment", "4", "Alignment\nScore", "s(x)= |⟨β̂,γ̂⟩|\n/ ||β̂|| ||γ̂||", "#7e73b9"),
    ("segment", "5", "Segment\n& Flag", "", "#d97735"),
]


def _is_auditor_task(task: dict) -> bool:
    text = " ".join(str(task.get(k) or "") for k in (
        "id", "kind", "region_id", "locked_method", "objective",
    )).lower()
    return "auditor" in text and "card" in text


def _card_specs(ir: dict, task: dict | None = None) -> list[dict]:
    canvas = ir.get("canvas") or {}
    w = float(canvas.get("width_px") or 2508)
    h = float(canvas.get("height_px") or 1322)
    task_bbox = _usable_task_bbox(task, w, h)
    if task_bbox:
        bx0, by0, bx1, by1 = task_bbox
        bw, bh = bx1 - bx0, by1 - by0
        # Visual-region bboxes include the section title, bridges, group label,
        # and some lower overlap with the action-card system.  Derive only the
        # five method-card row from that semantic region instead of falling back
        # to global canvas percentages.
        x0, x1 = bx0 + bw * 0.010, bx1 - bw * 0.018
        y0, y1 = by0 + bh * 0.095, by1 - bh * 0.220
    else:
        x0, x1 = w * 0.492, w * 0.982
        y0, y1 = h * 0.338, h * 0.575
    # This agent owns the whole auditor-card component row.  The slots below are
    # normalized to the card box so title, formula, visual, bridge, and shadow
    # move together instead of being independently nudged by residual repairs.
    gap = w * 0.0070
    card_w = (x1 - x0 - gap * 4) / 5
    out = []
    for idx, (key, num, title, formula, color) in enumerate(SPECS):
        cx0 = x0 + idx * (card_w + gap)
        out.append({
            "key": key,
            "num": num,
            "title": title,
            "formula": formula,
            "color": color,
            "bbox": [cx0, y0, cx0 + card_w, y1],
        })
    return out


def _usable_task_bbox(task: dict | None, width: float, height: float) -> list[float] | None:
    bbox = (task or {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return None
    # Reject tiny defect bboxes; component tasks should cover the right-center
    # visual band.
    if (x1 - x0) < width * 0.30 or (y1 - y0) < height * 0.20:
        return None
    return [
        max(0.0, x0),
        max(0.0, y0),
        min(width, x1),
        min(height, y1),
    ]


FORMULA_LAYOUTS = {
    "propensity": {
        "kind": "sequence",
        "tokens": [
            {"text": "T ~ X → "},
            {"text": "β", "accent": "hat"},
        ],
    },
    "surrogate": {
        "kind": "sequence",
        "tokens": [
            {"text": "Y ~ X + X·T → "},
            {"text": "τ", "accent": "tilde"},
        ],
    },
    "heterogeneity": {
        "kind": "sequence",
        "tokens": [
            {"text": "τ", "accent": "tilde"},
            {"text": " ~ X → "},
            {"text": "γ", "accent": "hat"},
        ],
    },
    "alignment_prefix": {
        "kind": "sequence",
        "tokens": [{"text": "s(x)="}],
    },
    "alignment_num": {
        "kind": "sequence",
        "tokens": [
            {"text": "|(x·"},
            {"text": "β", "accent": "hat"},
            {"text": ")(x·"},
            {"text": "γ", "accent": "hat"},
            {"text": ")|"},
        ],
    },
    "alignment_den": {
        "kind": "sequence",
        "tokens": [
            {"text": "||"},
            {"text": "β", "accent": "hat"},
            {"text": "||  ||"},
            {"text": "γ", "accent": "hat"},
            {"text": "||"},
        ],
    },
}


def _card_elements(ir: dict, spec: dict) -> list[dict]:
    x0, y0, x1, y1 = [float(v) for v in spec["bbox"]]
    key = spec["key"]
    color = spec["color"]
    r = ir.get("round", 0)
    w, h = x1 - x0, y1 - y0
    els: list[dict] = [
        IR.element(
            id=f"auditor_shadow_{key}",
            type="rounded_rect",
            bbox=[x0 + w * 0.045, y1 - h * 0.135, x1 - w * 0.020, y1 + h * 0.032],
            provenance=IR.provenance("AuditorCardAgent", "auditor_card_shadow", r),
            confidence=0.76,
            fill=color,
            border_color=color,
            border_width=0,
            corner=0.42,
            z=-0.30,
            ext=_ext(spec, "shadow"),
        ),
        IR.element(
            id=f"auditor_card_{key}",
            type="rounded_rect",
            bbox=[x0, y0, x1, y1],
            provenance=IR.provenance("AuditorCardAgent", "auditor_card", r),
            confidence=0.90,
            fill="#ffffff",
            border_color=color,
            border_width=2.6,
            corner=0.28,
            z=-0.12,
            ext=_ext(spec, "card"),
        ),
        IR.element(
            id=f"auditor_num_{key}",
            type="oval",
            bbox=[(x0 + x1) / 2 - 24, y0 - 25, (x0 + x1) / 2 + 24, y0 + 23],
            provenance=IR.provenance("AuditorCardAgent", "auditor_number", r),
            confidence=0.88,
            fill=color,
            border_color=color,
            border_width=1,
            text=spec["num"],
            font="Arial",
            font_size=22,
            bold=True,
            text_color="#ffffff",
            align="center",
            z=10.0,
            ext=_ext(spec, "number"),
        ),
        IR.element(
            id=f"auditor_title_{key}",
            type="text",
            bbox=[x0 + w * 0.075, y0 + h * 0.075, x1 - w * 0.075, y0 + h * 0.260],
            provenance=IR.provenance("AuditorCardAgent", "auditor_title", r),
            confidence=0.88,
            text=spec["title"],
            font="Arial",
            font_size=19,
            bold=True,
            text_color="#111111",
            align="center",
            z=8.0,
            ext=_ext(spec, "title"),
        ),
    ]
    if key == "alignment":
        els.extend(_alignment_formula_elements(ir, spec, [
            x0 + w * 0.045,
            y0 + h * 0.255,
            x1 - w * 0.045,
            y0 + h * 0.500,
        ]))
    elif spec["formula"]:
        els.append(IR.element(
            id=f"auditor_formula_{key}",
            type="formula",
            bbox=[x0 + w * 0.050, y0 + h * 0.278, x1 - w * 0.050, y0 + h * 0.455],
            provenance=IR.provenance("AuditorCardAgent", "auditor_formula", r),
            confidence=0.82,
            text=spec["formula"],
            latex=spec["formula"],
            font="Cambria Math",
            font_size=_formula_size_for(key),
            text_color="#222222",
            align="center",
            z=8.0,
            ext={
                **_formula_ext(spec, "formula", FORMULA_LAYOUTS.get(key)),
                "typography_locked": True,
            },
        ))
    box = [x0 + w * 0.095, y0 + h * 0.485, x1 - w * 0.085, y1 - h * 0.075]
    if key == "segment":
        box = [x0 + w * 0.070, y0 + h * 0.445, x1 - w * 0.060, y1 - h * 0.105]
    if key == "propensity":
        els.extend(_line_chart(ir, spec, box, color, variant="curve"))
    elif key == "surrogate":
        els.extend(_line_chart(ir, spec, box, color, variant="scatter"))
    elif key == "heterogeneity":
        els.extend(_heterogeneity_surface(ir, spec, box, color))
    elif key == "alignment":
        els.extend(_alignment_visual(ir, spec, box))
    else:
        els.extend(_segment_visual(ir, spec, box))
    return els


def _formula_size_for(key: str) -> float:
    if key == "surrogate":
        return 21.0
    if key == "heterogeneity":
        return 24.5
    return 26.0


def _alignment_formula_elements(ir: dict, spec: dict, box: list[float]) -> list[dict]:
    """Editable native fraction for the alignment-score card."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    prefix_w = w * 0.235
    frac_x0 = x0 + prefix_w
    frac_x1 = x1 - w * 0.010
    line_y = y0 + h * 0.525
    return [
        _formula_text(
            "auditor_formula_alignment_prefix",
            [x0, y0 + h * 0.145, x0 + prefix_w, y0 + h * 0.740],
            "s(x)=",
            r,
            spec,
            size=24,
            align="right",
        ),
        _formula_text(
            "auditor_formula_alignment_num",
            [frac_x0, y0, frac_x1, y0 + h * 0.465],
            "|(x·β̂)(x·γ̂)|",
            r,
            spec,
            size=22,
            align="center",
        ),
        IR.element(
            id="auditor_formula_alignment_rule",
            type="line",
            bbox=[frac_x0 + w * 0.022, line_y - 1.0, frac_x1 - w * 0.010, line_y + 1.0],
            provenance=IR.provenance("AuditorCardAgent", "auditor_formula_fraction_rule", r),
            confidence=0.84,
            points=[frac_x0 + w * 0.022, line_y, frac_x1 - w * 0.010, line_y],
            color="#222222",
            thickness=1.25,
            line_width=1.25,
            z=8.2,
            ext=_ext(spec, "formula_fraction_rule"),
        ),
        _formula_text(
            "auditor_formula_alignment_den",
            [frac_x0, y0 + h * 0.545, frac_x1, y1],
            "||β̂||  ||γ̂||",
            r,
            spec,
            size=21,
            align="center",
        ),
    ]


def _formula_text(eid: str, bbox: list[float], text: str, round_num: int,
                  spec: dict, size: float, align: str = "center") -> dict:
    return IR.element(
        id=eid,
        type="text",
        bbox=bbox,
        provenance=IR.provenance("AuditorCardAgent", "auditor_formula", round_num),
        confidence=0.84,
        text=text,
        font="Cambria Math",
        font_size=size,
        text_color="#222222",
        align=align,
        z=8.0,
        ext={
            **_formula_ext(
                spec,
                "formula",
                FORMULA_LAYOUTS.get(eid.removeprefix("auditor_formula_")),
            ),
            "typography_locked": True,
        },
    )


def _formula_ext(spec: dict, role: str, layout: dict | None) -> dict:
    ext = _ext(spec, role)
    if layout:
        ext["math_layout"] = layout
    return ext


def _line_chart(ir: dict, spec: dict, box: list[float], color: str,
                variant: str) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    pts = []
    dots = []
    for i in range(40):
        t = i / 39
        if variant == "curve":
            y = 0.74 - 0.46 * t + 0.12 * math.sin(t * math.pi * 1.35)
        else:
            y = 0.72 - 0.44 * t + 0.08 * math.sin(t * math.pi * 2.15)
        px, py = w * t, h * max(0.10, min(0.88, y))
        pts.append([px, py])
        if i % 2 == 0 or variant == "scatter":
            dots.append({
                "cx": px + ((i * 17) % 9 - 4) * 1.6,
                "cy": py + ((i * 13) % 11 - 5) * 2.2,
                "r": 1.9 if variant == "curve" else 2.2,
                "color": color if variant == "curve" else ("#75b9d7" if i % 3 else color),
            })
    return [
        _axis_line(ir, spec, "x", [x0, y1 - h * 0.12, x1, y1 - h * 0.12], "#b5c4d1"),
        _axis_line(ir, spec, "y", [x0 + w * 0.08, y0, x0 + w * 0.08, y1], "#b5c4d1"),
        IR.element(
            id=f"auditor_visual_{spec['key']}_dots",
            type="dotcloud",
            bbox=box,
            provenance=IR.provenance("AuditorCardAgent", "auditor_chart_dots", r),
            confidence=0.80,
            dots=dots,
            paths=[{
                "points": pts,
                "closed": False,
                "line": color,
                "line_width": 2.1,
            }],
            z=6.0,
            ext=_ext(spec, "visual_dots"),
        ),
    ]


def _heterogeneity_surface(ir: dict, spec: dict, box: list[float], color: str) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    curves = []
    for band in range(5):
        curve = []
        for i in range(24):
            t = i / 23
            y = h * (0.31 + band * 0.105 + 0.040 * math.sin(t * math.pi * 2.0 + band))
            curve.append([w * (0.03 + 0.94 * t), y])
        curves.append(curve)
    dots = []
    arrows = []
    for row in range(4):
        for col in range(5):
            sx = w * (0.16 + col * 0.17)
            sy = h * (0.22 + row * 0.18)
            dots.append({"cx": sx, "cy": sy, "r": 2.1, "color": color})
            if (row + col) % 2 == 0:
                ax0, ay0 = x0 + sx - 5, y0 + sy + 9
                ax1, ay1 = x0 + sx + 8, y0 + sy - 11
                arrows.append(IR.element(
                    id=f"auditor_visual_{spec['key']}_vec_{row}_{col}",
                    type="arrow",
                    bbox=_ordered_bbox(ax0, ay0, ax1, ay1),
                    provenance=IR.provenance("AuditorCardAgent", "auditor_vector_field", r),
                    confidence=0.74,
                    points=[ax0, ay0, ax1, ay1],
                    color=color,
                    thickness=1.8,
                    z=7.0,
                    ext=_ext(spec, "visual_vector"),
                ))
    return [
        IR.element(
            id=f"auditor_visual_{spec['key']}_surface",
            type="surface",
            bbox=box,
            provenance=IR.provenance("AuditorCardAgent", "auditor_surface_vector_field", r),
            confidence=0.76,
            wave_bands={
                "curves": curves,
                "fills": ["#f2f8f4", "#e9f3ee", "#deebe5", "#d5e5dd"],
            },
            streamlines=curves[1:-1],
            dots=dots,
            z=6.0,
            ext=_ext(spec, "visual_surface"),
        )
    ] + arrows


def _alignment_visual(ir: dict, spec: dict, box: list[float]) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    origin = [x0 + w * 0.24, y0 + h * 0.72]
    beta = [x0 + w * 0.78, y0 + h * 0.22]
    gamma = [x0 + w * 0.92, y0 + h * 0.38]
    arc = []
    for i in range(11):
        a = -0.73 + i * 0.040
        arc.append([w * (0.31 + 0.30 * math.cos(a)), h * (0.72 + 0.30 * math.sin(a))])
    return [
        IR.element(
            id="auditor_visual_alignment_beta",
            type="arrow",
            bbox=_ordered_bbox(origin[0], origin[1], beta[0], beta[1]),
            provenance=IR.provenance("AuditorCardAgent", "auditor_alignment_vector", r),
            confidence=0.82,
            points=[origin[0], origin[1], beta[0], beta[1]],
            color="#1f66d1",
            thickness=4,
            z=7.0,
            ext=_ext(spec, "visual_beta"),
        ),
        IR.element(
            id="auditor_visual_alignment_gamma",
            type="arrow",
            bbox=_ordered_bbox(origin[0], origin[1], gamma[0], gamma[1]),
            provenance=IR.provenance("AuditorCardAgent", "auditor_alignment_vector", r),
            confidence=0.82,
            points=[origin[0], origin[1], gamma[0], gamma[1]],
            color="#16806e",
            thickness=4,
            z=7.0,
            ext=_ext(spec, "visual_gamma"),
        ),
        IR.element(
            id="auditor_visual_alignment_theta_arc",
            type="dotcloud",
            bbox=box,
            provenance=IR.provenance("AuditorCardAgent", "auditor_alignment_theta", r),
            confidence=0.78,
            dots=[],
            paths=[{
                "points": arc,
                "closed": False,
                "line": "#555555",
                "line_width": 1.2,
            }],
            z=7.2,
            ext=_ext(spec, "visual_theta_arc"),
        ),
        IR.element(
            id="auditor_visual_alignment_theta_label",
            type="text",
            bbox=[x0 + w * 0.61, y0 + h * 0.64, x0 + w * 0.80, y0 + h * 0.82],
            provenance=IR.provenance("AuditorCardAgent", "auditor_alignment_theta", r),
            confidence=0.78,
            text="θ",
            font="Arial",
            font_size=17,
            text_color="#444444",
            align="center",
            z=8.0,
            ext=_ext(spec, "visual_theta_label"),
        ),
    ]


def _segment_visual(ir: dict, spec: dict, box: list[float]) -> list[dict]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = ir.get("round", 0)
    palette = ["#9a9a9a", "#e85536", "#e49a24", "#4aa381", "#1b95c9"]
    centers = [(0.27, 0.27), (0.70, 0.28), (0.30, 0.72), (0.66, 0.71), (0.80, 0.52)]
    counts = [34, 34, 24, 30, 18]
    spreads = [(0.24, 0.22), (0.23, 0.21), (0.19, 0.17), (0.21, 0.18), (0.14, 0.16)]
    dots = []
    for group, count in enumerate(counts):
        gx, gy = centers[group]
        sx, sy = spreads[group]
        for j in range(count):
            angle = (j * 2.399963 + group * 0.71) % (math.tau)
            radius = math.sqrt((j + 0.5) / count)
            wobble = 0.78 + 0.22 * math.sin(j * 1.7 + group)
            cx = (gx + math.cos(angle) * sx * radius * wobble) * w
            cy = (gy + math.sin(angle) * sy * radius * (0.86 + 0.14 * math.cos(j))) * h
            if j % 11 == 0:
                dot_r = 6.1
            elif j % 5 == 0:
                dot_r = 4.2
            else:
                dot_r = 2.75
            dots.append({"cx": cx, "cy": cy, "r": dot_r, "color": palette[group]})
    return [
        IR.element(
            id="auditor_visual_segment_dots",
            type="dotcloud",
            bbox=box,
            provenance=IR.provenance("AuditorCardAgent", "auditor_segment_dots", r),
            confidence=0.82,
            dots=dots,
            z=6.0,
            ext=_ext(spec, "visual_dots"),
        ),
    ]


def _group_elements(ir: dict, specs: list[dict]) -> list[dict]:
    r = ir.get("round", 0)
    els: list[dict] = []
    group_x0 = specs[0]["bbox"][0]
    group_x1 = specs[-1]["bbox"][2]
    group_y0 = specs[0]["bbox"][1]
    title_y0 = group_y0 - 95.0
    title_y1 = group_y0 - 38.0
    title_cx = (group_x0 + group_x1) / 2
    line_y = group_y0 - 63.0
    els.extend([
        IR.element(
            id="auditor_section_title",
            type="text",
            bbox=[title_cx - 250.0, title_y0, title_cx + 250.0, title_y1],
            provenance=IR.provenance("AuditorCardAgent", "auditor_section_title", r),
            confidence=0.88,
            text="CATE-CI Auditor",
            font="Times New Roman",
            font_size=34,
            bold=True,
            text_color="#071a4d",
            align="center",
            z=9.0,
            ext=_ext(specs[2], "section_title"),
        ),
        IR.element(
            id="auditor_section_rule_left",
            type="line",
            bbox=[group_x0, line_y - 2.0, title_cx - 280.0, line_y + 2.0],
            provenance=IR.provenance("AuditorCardAgent", "auditor_section_rule", r),
            confidence=0.84,
            points=[group_x0, line_y, title_cx - 280.0, line_y],
            color="#2e6fbc",
            thickness=2.0,
            line_width=2.0,
            z=8.5,
            ext=_ext(specs[2], "section_rule"),
        ),
        IR.element(
            id="auditor_section_rule_right",
            type="line",
            bbox=[title_cx + 280.0, line_y - 2.0, group_x1, line_y + 2.0],
            provenance=IR.provenance("AuditorCardAgent", "auditor_section_rule", r),
            confidence=0.84,
            points=[title_cx + 280.0, line_y, group_x1, line_y],
            color="#2e6fbc",
            thickness=2.0,
            line_width=2.0,
            z=8.5,
            ext=_ext(specs[2], "section_rule"),
        ),
    ])
    for left, right in zip(specs, specs[1:]):
        lx0, ly0, lx1, ly1 = [float(v) for v in left["bbox"]]
        rx0, ry0, rx1, ry1 = [float(v) for v in right["bbox"]]
        y = max(ly1, ry1) - (ly1 - ly0) * 0.08
        els.append(IR.element(
            id=f"auditor_bridge_{left['key']}_{right['key']}",
            type="rounded_rect",
            bbox=[lx1 - 18, y - 8, rx0 + 18, y + 16],
            provenance=IR.provenance("AuditorCardAgent", "auditor_card_bridge", r),
            confidence=0.76,
            fill=right["color"],
            border_color=right["color"],
            border_width=0,
            corner=0.60,
            z=-0.2,
            ext=_ext(right, "bridge"),
        ))
    x0 = specs[0]["bbox"][0]
    x1 = specs[1]["bbox"][2]
    y1 = specs[0]["bbox"][3]
    bracket_y = y1 + 18.0
    bracket_top = y1 + 2.0
    bracket_color = "#1c56b7"
    els.extend([
        IR.element(
            id="auditor_cheap_nuisance_bracket_left",
            type="line",
            bbox=[x0 - 2.0, bracket_top, x0 + 2.0, bracket_y + 2.0],
            provenance=IR.provenance("AuditorCardAgent", "auditor_group_bracket", r),
            confidence=0.82,
            points=[x0, bracket_top, x0, bracket_y],
            color=bracket_color,
            thickness=2.0,
            line_width=2.0,
            z=7.4,
            ext=_ext(specs[0], "group_bracket"),
        ),
        IR.element(
            id="auditor_cheap_nuisance_bracket_mid",
            type="line",
            bbox=[x0, bracket_y - 2.0, x1, bracket_y + 2.0],
            provenance=IR.provenance("AuditorCardAgent", "auditor_group_bracket", r),
            confidence=0.82,
            points=[x0, bracket_y, x1, bracket_y],
            color=bracket_color,
            thickness=2.0,
            line_width=2.0,
            z=7.4,
            ext=_ext(specs[0], "group_bracket"),
        ),
        IR.element(
            id="auditor_cheap_nuisance_bracket_right",
            type="line",
            bbox=[x1 - 2.0, bracket_top, x1 + 2.0, bracket_y + 2.0],
            provenance=IR.provenance("AuditorCardAgent", "auditor_group_bracket", r),
            confidence=0.82,
            points=[x1, bracket_top, x1, bracket_y],
            color=bracket_color,
            thickness=2.0,
            line_width=2.0,
            z=7.4,
            ext=_ext(specs[0], "group_bracket"),
        ),
    ])
    els.append(IR.element(
        id="auditor_cheap_nuisance_label",
        type="text",
        bbox=[x0, y1 + 24, x1, y1 + 60],
        provenance=IR.provenance("AuditorCardAgent", "auditor_group_label", r),
        confidence=0.80,
        text="cheap nuisance models",
        font="Arial",
        font_size=20,
        italic=True,
        text_color="#1c56b7",
        align="center",
        z=8.0,
        ext=_ext(specs[0], "group_label"),
    ))
    return els


def _axis_line(ir: dict, spec: dict, suffix: str, points: list[float], color: str) -> dict:
    r = ir.get("round", 0)
    return IR.element(
        id=f"auditor_visual_{spec['key']}_axis_{suffix}",
        type="line",
        bbox=_ordered_bbox(points[0], points[1], points[2], points[3]),
        provenance=IR.provenance("AuditorCardAgent", "auditor_chart_axis", r),
        confidence=0.70,
        points=points,
        color=color,
        thickness=1.0,
        z=5.8,
        ext=_ext(spec, "visual_axis"),
    )


def _remove_orphans(ir: dict, specs: list[dict]) -> set[str]:
    boxes = [s["bbox"] for s in specs]
    rx0 = min(b[0] for b in boxes) - 40
    ry0 = min(b[1] for b in boxes) - 115
    rx1 = max(b[2] for b in boxes) + 35
    ry1 = max(b[3] for b in boxes) + 40
    removable = {
        "text", "formula", "rounded_rect", "rect", "oval", "icon",
        "chart", "dotcloud", "surface", "freeform", "arrow", "line",
    }
    keep = []
    removed: set[str] = set()
    for el in ir.get("elements", []):
        eid = str(el.get("id") or "")
        bbox = el.get("bbox")
        if (
            eid.startswith("auditor_")
            or eid.startswith("action_card_")
            or eid.startswith("bottom_")
            or eid.startswith("failure_summary_")
            or eid.startswith("pipeline_context_")
            or eid.startswith("proc_")
            or not bbox
        ):
            keep.append(el)
            continue
        if eid.startswith("layout_auditor_") or (
            el.get("type") in removable and _center_inside(bbox, [rx0, ry0, rx1, ry1])
        ):
            removed.add(eid)
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _ordered_bbox(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _center_inside(bbox: list[float], region: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rx0, ry0, rx1, ry1 = [float(v) for v in region]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _ext(spec: dict, role: str) -> dict:
    return {
        "component": "auditor_method_card",
        "component_key": spec["key"],
        "component_role": role,
        "strategy": {
            "region_id": "region_auditor_cards",
            "kind": "auditor_method_cards",
            "primary_method": "auditor_card_layout",
            "fallback_methods": ["chart_parser", "text_style"],
            "preferred_agent": "AuditorCardAgent",
        },
    }
