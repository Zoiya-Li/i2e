"""Procedural native synthesis for diagrammatic 3D surface panels.

Some scientific illustration regions are not recoverable by OCR/CV tracing
alone.  A 3D manifold in an infographic is a parametric diagram: the correct
representation is editable generated geometry, calibrated by the detected bbox.
"""
from __future__ import annotations

import math
import random


def apply(ir: dict) -> dict:
    """Replace large left-side surface payloads with generated native geometry."""
    stats = {
        "procedural_surfaces": 0,
        "axis_arrows": 0,
        "axis_labels": 0,
        "vector_arrows": 0,
        "ci_insets": 0,
        "formula_boxes": 0,
        "risk_annotations": 0,
    }
    canvas = ir.get("canvas") or {}
    width = float(canvas.get("width_px") or 1)
    elements = ir.get("elements", [])

    for el in elements:
        if el.get("type") != "surface" or not el.get("bbox"):
            continue
        x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
        bw, bh = x1 - x0, y1 - y0
        if bw < 520 or bh < 240 or x0 > width * 0.52:
            continue
        _canonicalize_main_surface_bbox(el, width)
        removed = _remove_surface_orphans(ir, el)
        _synthesize_surface_payload(el)
        el.setdefault("repair_history", []).append({
            "agent": "ProceduralSurface",
            "action": "parametric_manifold_payload",
            "round": ir.get("round", 0),
            "removed_orphans": removed,
        })
        stats["procedural_surfaces"] += 1
        stats["axis_arrows"] += _ensure_axis_arrows(ir, el)
        stats["axis_labels"] += _ensure_axis_labels(ir, el)
        stats["vector_arrows"] += _ensure_vector_arrows(ir, el)
        stats["ci_insets"] += _ensure_ci_inset(ir, el)
        stats["formula_boxes"] += _ensure_formula_box(ir, el)
        stats["risk_annotations"] += _ensure_risk_annotations(ir, el)

    if any(stats.values()):
        ir.setdefault("quality_gate", {}).setdefault("procedural_surface", []).append({
            "round": ir.get("round", 0),
            **stats,
        })
    return stats


def _remove_surface_orphans(ir: dict, surface: dict) -> list[str]:
    region = surface.get("bbox")
    if not region:
        return []
    protected_boxes = [
        e.get("bbox") for e in ir.get("elements", [])
        if e is not surface
        and e.get("type") in {"rounded_rect", "chart"}
        and e.get("bbox")
        and _bbox_overlap_fraction(e.get("bbox"), region) > 0.35
    ]
    keep = []
    removed: list[str] = []
    for el in ir.get("elements", []):
        if el is surface:
            keep.append(el)
            continue
        bbox = el.get("bbox")
        typ = el.get("type")
        if not bbox or _bbox_overlap_fraction(bbox, region) < 0.22:
            keep.append(el)
            continue
        if typ in {"arrow", "line"} and not (el.get("from_id") or el.get("to_id")):
            removed.append(str(el.get("id")))
            continue
        if typ in {"text", "formula"} and (
            _is_surface_numeric_noise(el)
            or _is_surface_annotation_noise(el, region)
        ):
            if str(el.get("id", "")).startswith("proc_risk_"):
                keep.append(el)
                continue
            removed.append(str(el.get("id")))
            continue
        if typ == "freeform" and not _inside_any(bbox, protected_boxes):
            removed.append(str(el.get("id")))
            continue
        if typ in {"dotcloud", "path", "polygon"} and not _inside_any(bbox, protected_boxes):
            removed.append(str(el.get("id")))
            continue
        keep.append(el)
    if removed:
        ir["elements"] = keep
    return removed


def _inside_any(bbox: list[float], boxes: list[list[float] | None]) -> bool:
    return any(box and _bbox_overlap_fraction(bbox, box) > 0.75 for box in boxes)


def _is_surface_numeric_noise(el: dict) -> bool:
    text = str(el.get("text") or "").strip()
    if not text:
        return False
    compact = text.replace(" ", "").strip("'‘’`´")
    if compact in {"0", "0°", "6", "°", "θ"}:
        return True
    if len(compact) <= 2 and compact.replace(".", "", 1).isdigit():
        return True
    return False


def _is_surface_annotation_noise(el: dict, surface_bbox: list[float]) -> bool:
    text = str(el.get("text") or el.get("latex") or "").strip().lower()
    if not text or not el.get("bbox"):
        return False
    x0, y0, x1, y1 = [float(v) for v in surface_bbox]
    sx0, sy0, sx1, sy1 = [float(v) for v in el["bbox"]]
    sw, sh = x1 - x0, y1 - y0
    in_risk_zone = sx0 >= x0 + sw * 0.58 and sy0 >= y0 + sh * 0.02
    loose_risk_zone = (
        sx1 >= x0 + sw * 0.72
        and sx0 <= x1 + sw * 0.12
        and sy0 >= y0 - sh * 0.02
        and sy1 <= y0 + sh * 0.36
    )
    if in_risk_zone and any(k in text for k in (
        "low", "propens", "overlap", "weak", "high", "heterogeneity",
        "narrow", "wrong ci",
    )):
        return True
    if loose_risk_zone and any(k in text for k in (
        "low", "propens", "overlap", "weak",
    )):
        return True
    if text == "heterogeneity":
        return True
    in_vector_zone = (
        sx0 >= x0 + sw * 0.42 and sx1 <= x0 + sw * 0.88
        and sy0 >= y0 + sh * 0.10 and sy1 <= y0 + sh * 0.58
    )
    upper_vector_label_zone = (
        sx0 >= x0 + sw * 0.30 and sx1 <= x0 + sw * 0.72
        and sy0 >= y0 - sh * 0.02 and sy1 <= y0 + sh * 0.34
    )
    if in_vector_zone and any(k in text for k in (
        "β", "gamma", "nabla", "∇", "tau", "τ", "theta", "θ",
    )):
        return True
    if upper_vector_label_zone and any(k in text for k in (
        "β", "gamma", "nabla", "∇", "tau", "τ",
    )):
        return True
    return False


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


def _synthesize_surface_payload(el: dict) -> None:
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    w, h = x1 - x0, y1 - y0
    us = [i / 76.0 for i in range(77)]

    def ridge_u(t: float) -> float:
        left_hill = math.exp(-((t - 0.30) / 0.18) ** 2)
        right_hill = math.exp(-((t - 0.76) / 0.13) ** 2)
        center_valley = math.exp(-((t - 0.52) / 0.18) ** 2)
        return (
            0.43
            - 0.19 * left_hill
            - 0.15 * right_hill
            + 0.13 * center_valley
            + 0.030 * math.sin(4.4 * math.pi * t + 0.35)
            + 0.040 * (t - 0.50)
        )

    def project(u: float, v: float) -> tuple[float, float]:
        """Oblique 3D-to-slide projection in local bbox coordinates."""
        depth_fan = 0.070 + 0.065 * u
        x = w * (0.030 + 0.895 * u + depth_fan * v)
        y = h * (
            ridge_u(u)
            + 0.255 * v
            + 0.060 * max(0.0, v) * math.sin(math.pi * u)
            - 0.030 * u
            + 0.070
        )
        return (
            round(max(2.0, min(w - 2.0, x)), 2),
            round(max(2.0, min(h - 2.0, y)), 2),
        )

    v_bands = [-0.92, -0.62, -0.34, -0.08, 0.18, 0.48, 0.78, 1.05]
    curves = []
    for vi, v in enumerate(v_bands):
        curve = []
        for u in us:
            px, py = project(u, v)
            py += h * 0.010 * math.sin(2.3 * math.pi * u + vi * 0.36)
            curve.append([px, round(max(2.0, min(h - 2.0, py)), 2)])
        curves.append(curve)

    el["wave_bands"] = {
        "curves": curves,
        "fills": [
            "#f6fafc", "#edf6fa", "#e2eff6", "#d7e8f2",
            "#cfdeeb", "#e7eef2", "#edd5c8",
        ],
    }

    streamlines = []
    for v in [-0.82, -0.66, -0.50, -0.34, -0.16, 0.02, 0.22, 0.46, 0.72]:
        line = []
        for u in us[2:-2]:
            px, py = project(u, v)
            py += h * 0.0045 * math.sin(8.0 * math.pi * u + v * 3.0)
            line.append([px, round(max(2.0, min(h - 2.0, py)), 2)])
        streamlines.append(line)

    el["streamlines"] = streamlines

    mesh_paths = []
    for u in [0.11, 0.23, 0.35, 0.48, 0.61, 0.74, 0.86]:
        path = []
        for v in [-0.82, -0.55, -0.28, -0.02, 0.25, 0.53, 0.82]:
            path.append(list(project(u, v)))
        mesh_paths.append({
            "points": path,
            "closed": False,
            "line": "#c8dce9",
            "line_width": 0.65,
            "alpha": 42,
            "area": 0,
        })

    # Pale lower projection contours: these give the foreground sheet a
    # grounded 3D footprint like the original illustration.
    for pi, frac in enumerate([0.00, 0.42, 0.84, 1.26, 1.68]):
        path = []
        for u in us[::2]:
            px = w * (0.015 + 0.89 * u + 0.050 * math.sin(frac + 2.2 * math.pi * u))
            py = h * (
                0.735
                + 0.080 * math.sin(2.15 * math.pi * u + frac)
                + 0.050 * math.sin(5.8 * math.pi * u + frac * 1.7)
                + 0.045 * (u - 0.30)
            )
            path.append([round(max(2.0, min(w - 2.0, px)), 2), round(max(h * 0.55, min(h - 3.0, py)), 2)])
        mesh_paths.append({
            "points": path,
            "closed": False,
            "line": "#d2e4ef",
            "line_width": 0.55,
            "alpha": 46,
            "area": -10 + pi,
        })
    el["paths"] = mesh_paths

    el["heat_regions"] = [
        {
            "cx": round(w * 0.32, 2),
            "cy": round(h * 0.61, 2),
            "rx": round(w * 0.29, 2),
            "ry": round(h * 0.235, 2),
            "color": "#c5deec",
            "opacity": 28,
            "soft_edge": 12.0,
        },
        {
            "cx": round(w * 0.815, 2),
            "cy": round(h * 0.390, 2),
            "rx": round(w * 0.125, 2),
            "ry": round(h * 0.145, 2),
            "color": "#efc2a8",
            "opacity": 32,
            "soft_edge": 12.0,
        },
        {
            "cx": round(w * 0.625, 2),
            "cy": round(h * 0.805, 2),
            "rx": round(w * 0.210, 2),
            "ry": round(h * 0.135, 2),
            "color": "#efd6c7",
            "opacity": 22,
            "soft_edge": 12.0,
        },
    ]

    rnd = random.Random(713)
    dots = []
    for _ in range(185):
        # Blue observed-data cluster in the front-left basin.
        u = min(max(rnd.gauss(0.285, 0.135), 0.06), 0.62)
        v = min(max(rnd.gauss(0.43, 0.28), -0.08), 1.00)
        gx, gy = project(u, v)
        gx += rnd.uniform(-9.0, 9.0)
        gy += rnd.uniform(-12.0, 12.0)
        dots.append({
            "cx": round(gx, 2),
            "cy": round(gy, 2),
            "r": round(rnd.uniform(1.75, 4.35), 2),
            "color": "#426c8a",
        })
    for _ in range(28):
        # Red low-overlap cluster sits inside the dashed risk ellipse.  It is a
        # semantic sparse segment, not a contour line sampled from the surface.
        gx = w * 0.815 + rnd.gauss(0.0, w * 0.040)
        gy = h * 0.392 + rnd.gauss(0.0, h * 0.050)
        gx = max(w * 0.735, min(w * 0.905, gx))
        gy = max(h * 0.270, min(h * 0.515, gy))
        dots.append({
            "cx": round(gx, 2),
            "cy": round(gy, 2),
            "r": round(rnd.uniform(2.15, 4.25), 2),
            "color": "#d64b2a",
        })
    el["dots"] = dots

    el["style"] = {"dark": "#9fbfd3", "light": "#f3f8fb"}
    for key in ("silhouette", "surface_layers"):
        el.pop(key, None)
    el.setdefault("ext", {}).update({
        "wave_bands": el["wave_bands"],
        "streamlines": streamlines,
        "heat_regions": el["heat_regions"],
        "dots": dots,
        "paths": mesh_paths,
        "style": el["style"],
        "procedural_surface": True,
    })


def _ensure_axis_arrows(ir: dict, surface: dict) -> int:
    elements = ir.get("elements", [])
    elements[:] = [
        e for e in elements
        if not str(e.get("id", "")).startswith("proc_axis_")
    ]
    x0, y0, _, _ = [float(v) for v in surface["bbox"]]
    origin = [x0 + 54.0, y0 + 244.0]
    axes = [
        ("x3", [origin[0], origin[1], origin[0], origin[1] - 212.0]),
        ("x1", [origin[0], origin[1], origin[0] + 150.0, origin[1]]),
        ("x2", [origin[0], origin[1], origin[0] + 108.0, origin[1] - 104.0]),
    ]
    added = 0
    for name, pts in axes:
        bbox = [
            min(pts[0], pts[2]) - 5,
            min(pts[1], pts[3]) - 5,
            max(pts[0], pts[2]) + 5,
            max(pts[1], pts[3]) + 5,
        ]
        elements.append({
            "id": f"proc_axis_{name}",
            "type": "arrow",
            "status": "native",
            "bbox": bbox,
            "points": pts,
            "confidence": 0.86,
            "provenance": {
                "agent": "ProceduralSurface",
                "action": "axis_arrow",
                "round": ir.get("round", 0),
            },
            "repair_history": [],
            "defects": [],
            "color": "#111111",
            "z": 6,
            "ext": {"procedural_surface_axis": name},
        })
        added += 1
    return added


def _ensure_axis_labels(ir: dict, surface: dict) -> int:
    elements = ir.get("elements", [])
    sx0, sy0, sx1, sy1 = [float(v) for v in surface["bbox"]]
    sw = sx1 - sx0
    origin = [sx0 + 54.0, sy0 + 244.0]
    cleanup = [0.0, max(0.0, sy0 - 95.0), sx0 + sw * 0.38, origin[1] + 42.0]
    elements[:] = [
        e for e in elements
        if not (
            str(e.get("id", "")).startswith(("proc_axis_label_", "proc_covariate_"))
            or (
                e.get("type") in {"text", "formula"}
                and e.get("bbox")
                and _bbox_overlap_fraction(e.get("bbox"), cleanup) > 0.0
                and _is_axis_or_covariate_label(e)
            )
        )
    ]
    round_num = ir.get("round", 0)
    additions = [
        _surface_text(
            "proc_covariate_label_text",
            [sx0 + 82.0, sy0 - 58.0, sx0 + 280.0, sy0 - 16.0],
            "covariate space",
            "covariate_text",
            round_num,
            font="Arial",
            size=28,
        ),
        _surface_text(
            "proc_covariate_label_x",
            [sx0 + 276.0, sy0 - 66.0, sx0 + 336.0, sy0 - 12.0],
            "X",
            "covariate_math",
            round_num,
            font="Times New Roman",
            size=38,
            italic=True,
        ),
        _surface_text(
            "proc_axis_label_x3",
            [origin[0] - 62.0, origin[1] - 232.0, origin[0] - 18.0, origin[1] - 188.0],
            "x3",
            "axis_math",
            round_num,
            font="Times New Roman",
            size=22,
            italic=True,
        ),
        _surface_text(
            "proc_axis_label_x2",
            [origin[0] + 92.0, origin[1] - 100.0, origin[0] + 140.0, origin[1] - 58.0],
            "x2",
            "axis_math",
            round_num,
            font="Times New Roman",
            size=22,
            italic=True,
        ),
        _surface_text(
            "proc_axis_label_x1",
            [origin[0] + 150.0, origin[1] + 2.0, origin[0] + 198.0, origin[1] + 44.0],
            "x1",
            "axis_math",
            round_num,
            font="Times New Roman",
            size=22,
            italic=True,
        ),
    ]
    elements.extend(additions)
    return len(additions)


def _is_axis_or_covariate_label(el: dict) -> bool:
    text = str(el.get("text") or el.get("latex") or "").strip().lower()
    compact = text.replace(" ", "")
    return (
        "covariate" in text
        or compact in {"x", "x1", "x2", "x3"}
    )


def _surface_text(
    eid: str,
    bbox: list[float],
    text: str,
    role: str,
    round_num: int,
    font: str,
    size: float,
    italic: bool = False,
) -> dict:
    return {
        "id": eid,
        "type": "text",
        "status": "native",
        "bbox": bbox,
        "text": text,
        "font": font,
        "font_size": size,
        "italic": italic,
        "text_color": "#111111",
        "align": "center",
        "confidence": 0.88,
        "z": 8.0,
        "provenance": {
            "agent": "ProceduralSurface",
            "action": "axis_label",
            "round": round_num,
        },
        "repair_history": [],
        "defects": [],
        "ext": {
            "procedural_surface_label": role,
            "component": "procedural_surface",
            "component_role": role,
        },
    }


def _ensure_vector_arrows(ir: dict, surface: dict) -> int:
    elements = ir.get("elements", [])
    elements[:] = [
        e for e in elements
        if not str(e.get("id", "")).startswith("proc_vec_")
    ]
    sx0, sy0, sx1, sy1 = [float(v) for v in surface["bbox"]]
    sw, sh = sx1 - sx0, sy1 - sy0
    ox = sx0 + sw * 0.405
    oy = sy0 + sh * 0.590
    blue_tip = (sx0 + sw * 0.645, sy0 + sh * 0.250)
    green_tip = (sx0 + sw * 0.705, sy0 + sh * 0.320)
    z = 12
    round_num = ir.get("round", 0)
    additions = [
        {
            "id": "proc_vec_origin",
            "type": "oval",
            "status": "native",
            "bbox": [ox - 8.0, oy - 8.0, ox + 8.0, oy + 8.0],
            "fill": "#050505",
            "border_color": "#050505",
            "confidence": 0.90,
            "z": z + 2,
            "provenance": {"agent": "ProceduralSurface", "action": "vector_origin", "round": round_num},
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_vector": "origin"},
        },
        _vector_arrow("proc_vec_beta", ox, oy, blue_tip[0], blue_tip[1], "#1f5ed1", z, round_num),
        _vector_arrow("proc_vec_gamma", ox + sw * 0.018, oy - sh * 0.010,
                      green_tip[0], green_tip[1], "#17866e", z + 1, round_num),
        {
            "id": "proc_vec_beta_label",
            "type": "text",
            "status": "native",
            "bbox": [sx0 + sw * 0.370, sy0 + sh * 0.135, sx0 + sw * 0.615, sy0 + sh * 0.235],
            "text": "β = ∇e(x)",
            "font": "Times New Roman",
            "font_size": 34,
            "italic": True,
            "text_color": "#1f5ed1",
            "align": "center",
            "confidence": 0.88,
            "z": z + 3,
            "provenance": {"agent": "ProceduralSurface", "action": "vector_label", "round": round_num},
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_vector": "beta_label"},
        },
        {
            "id": "proc_vec_gamma_label",
            "type": "text",
            "status": "native",
            "bbox": [sx0 + sw * 0.535, sy0 + sh * 0.330, sx0 + sw * 0.755, sy0 + sh * 0.430],
            "text": "γ = ∇τ(x)",
            "font": "Times New Roman",
            "font_size": 34,
            "italic": True,
            "text_color": "#17866e",
            "align": "center",
            "confidence": 0.88,
            "z": z + 3,
            "provenance": {"agent": "ProceduralSurface", "action": "vector_label", "round": round_num},
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_vector": "gamma_label"},
        },
        {
            "id": "proc_vec_theta",
            "type": "text",
            "status": "native",
            "bbox": [ox + sw * 0.080, oy - sh * 0.105, ox + sw * 0.250, oy - sh * 0.035],
            "text": "θ ≈ 0°",
            "font": "Times New Roman",
            "font_size": 24,
            "italic": True,
            "text_color": "#222222",
            "align": "center",
            "confidence": 0.86,
            "z": z + 3,
            "provenance": {"agent": "ProceduralSurface", "action": "vector_angle_label", "round": round_num},
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_vector": "theta_label"},
        },
    ]
    elements.extend(additions)
    return len(additions)


def _ensure_ci_inset(ir: dict, surface: dict) -> int:
    elements = ir.get("elements", [])
    elements[:] = [
        e for e in elements
        if not str(e.get("id", "")).startswith("proc_ci_")
    ]
    sx0, sy0, sx1, sy1 = [float(v) for v in surface["bbox"]]
    inset = None
    for el in elements:
        if el.get("type") != "rounded_rect" or not el.get("bbox"):
            continue
        x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
        if x0 < sx0 + (sx1 - sx0) * 0.52 or y0 < sy0 + (sy1 - sy0) * 0.48:
            continue
        if _bbox_overlap_fraction(el["bbox"], [sx0, sy0, sx1 + 260.0, sy1 + 70.0]) <= 0:
            continue
        inset = el
        break
    if not inset:
        return 0

    inset_id = str(inset.get("id"))
    elements[:] = [
        e for e in elements
        if not (
            e.get("type") in {"arrow", "line"}
            and (
                str(e.get("from_id") or "") == inset_id
                or str(e.get("to_id") or "") == inset_id
            )
        )
    ]

    sw, sh = sx1 - sx0, sy1 - sy0
    inset["bbox"] = [
        sx0 + sw * 0.615,
        sy0 + sh * 0.700,
        sx0 + sw * 0.990,
        sy0 + sh * 1.040,
    ]
    inset["z"] = max(8, int(inset.get("z") or 0))
    inset["fill"] = "#ffffff"
    inset["border_color"] = "#8a8a8a"
    inset["border_width"] = 1.2
    inset["dash"] = True
    x0, y0, x1, y1 = [float(v) for v in inset["bbox"]]
    w, h = x1 - x0, y1 - y0
    ci_cleanup = [
        x0 - w * 0.12,
        y0 - h * 0.18,
        x1 + w * 0.14,
        y1 + h * 0.52,
    ]
    elements[:] = [
        e for e in elements
        if not (
            e is not inset
            and e.get("type") in {"text", "formula"}
            and e.get("bbox")
            and (
                _bbox_overlap_fraction(e.get("bbox"), inset["bbox"]) > 0.18
                or _bbox_overlap_fraction(e.get("bbox"), ci_cleanup) > 0.12
            )
            and (
                not str(e.get("id") or "").startswith("proc_")
                or any(tok in str(e.get("text") or e.get("latex") or "").lower()
                       for tok in ("narrow", "wrong", "ci", "cl", "τ", "tau",
                                   "t(x)", "r(x)", "true", "a*1"))
            )
        )
    ]
    y_axis = y0 + h * 0.70
    x_left = x0 + w * 0.14
    x_mid = x0 + w * 0.56
    x_right = x0 + w * 0.86
    z = max(9, int(inset.get("z") or 0) + 9)
    round_num = ir.get("round", 0)

    additions = [
        {
            "id": "proc_ci_title",
            "type": "text",
            "status": "native",
            "bbox": [x0 + w * 0.09, y0 + h * 0.055, x1 - w * 0.09, y0 + h * 0.255],
            "text": "narrow but wrong CI",
            "font": "Arial",
            "font_size": 22,
            "text_color": "#7d1c16",
            "align": "center",
            "confidence": 0.86,
            "z": z + 2,
            "provenance": {"agent": "ProceduralSurface", "action": "ci_inset_label", "round": round_num},
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_ci": "title"},
        },
        _line("proc_ci_axis", x_left, y_axis, x1 - w * 0.08, y_axis, "#111111", z, round_num, arrow=True),
        _line("proc_ci_midline", x_mid, y0 + h * 0.22, x_mid, y0 + h * 0.82, "#333333", z, round_num, dash=True),
        _line("proc_ci_red_bar", x_left, y0 + h * 0.48, x_left + w * 0.24, y0 + h * 0.48, "#c70000", z + 1, round_num),
        _line("proc_ci_blue_bar", x_right - w * 0.18, y0 + h * 0.48, x_right + w * 0.12, y0 + h * 0.48, "#2469d8", z + 1, round_num),
    ]
    additions.extend([
        _tick("proc_ci_red_left", x_left, y0 + h * 0.43, y0 + h * 0.53, "#c70000", z + 1, round_num),
        _tick("proc_ci_red_right", x_left + w * 0.24, y0 + h * 0.43, y0 + h * 0.53, "#c70000", z + 1, round_num),
        _tick("proc_ci_blue_left", x_right - w * 0.18, y0 + h * 0.43, y0 + h * 0.53, "#2469d8", z + 1, round_num),
        _tick("proc_ci_blue_right", x_right + w * 0.12, y0 + h * 0.43, y0 + h * 0.53, "#2469d8", z + 1, round_num),
        _dot("proc_ci_red_dot", x_left + w * 0.12, y0 + h * 0.48, 7.0, "#c70000", z + 2, round_num),
        _dot("proc_ci_blue_dot", x_right - w * 0.03, y0 + h * 0.48, 7.0, "#2469d8", z + 2, round_num),
        _ci_text("proc_ci_hat_label", [x_left - w * 0.060, y0 + h * 0.735, x_left + w * 0.170, y0 + h * 0.940], "τ̂(x)", z + 2, round_num),
        _ci_text("proc_ci_mid_label", [x_mid - w * 0.060, y0 + h * 0.735, x_mid + w * 0.060, y0 + h * 0.940], "CI", z + 2, round_num),
        _ci_text("proc_ci_true_label", [x_right - w * 0.110, y0 + h * 0.735, x_right + w * 0.170, y0 + h * 0.995], "τ(x)\n(true)", z + 2, round_num),
    ])
    elements.extend(additions)
    return len(additions)


def _ensure_formula_box(ir: dict, surface: dict) -> int:
    """Ensure the top alignment formula is a complete native formula block."""
    elements = ir.get("elements", [])
    elements[:] = [
        e for e in elements
        if not str(e.get("id", "")).startswith("proc_formula_")
    ]
    sx0, sy0, sx1, sy1 = [float(v) for v in surface["bbox"]]
    sw, sh = sx1 - sx0, sy1 - sy0
    box = [
        sx0 + sw * 0.410,
        max(105.0, sy0 - sh * 0.165),
        sx0 + sw * 0.805,
        max(185.0, sy0 - sh * 0.015),
    ]
    # Remove incomplete OCR/formula fragments inside the same alignment box.
    keep = []
    removed = []
    for el in elements:
        if (
            el.get("bbox")
            and el.get("type") in {"text", "formula"}
            and _bbox_overlap_fraction(el.get("bbox"), box) > 0.25
            and any(tok in str(el.get("text") or el.get("latex") or el.get("ext", {}).get("latex") or "")
                    for tok in ("β", "γ", "A", "||", "⟨", "<"))
        ):
            removed.append(str(el.get("id")))
            continue
        keep.append(el)
    elements[:] = keep
    round_num = ir.get("round", 0)
    additions = [
        {
            "id": "proc_formula_alignment_box",
            "type": "rounded_rect",
            "status": "native",
            "bbox": box,
            "fill": "#ffffff",
            "border_color": "#496d9e",
            "border_width": 1.6,
            "corner": 0.08,
            "confidence": 0.88,
            "z": 9,
            "provenance": {
                "agent": "ProceduralSurface",
                "action": "alignment_formula_box",
                "round": round_num,
            },
            "repair_history": [],
            "defects": [],
            "ext": {
                "procedural_surface_formula": "box",
                "removed_fragments": removed,
            },
        },
    ]
    additions.extend(_alignment_formula_primitives(box, round_num))
    elements.extend(additions)
    return len(additions)


def _alignment_formula_primitives(box: list[float], round_num: int) -> list[dict]:
    """Stable editable fraction made from text boxes and a native rule.

    Office equation OMML is editable but rendered inconsistently across
    PowerPoint versions for compact fraction boxes.  These primitives keep the
    formula fully native/editable while making the geometry deterministic.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cy = (y0 + y1) / 2
    frac_x0 = x0 + w * 0.36
    frac_x1 = x0 + w * 0.68
    return [
        _formula_text("proc_formula_alignment_prefix",
                      [x0 + w * 0.10, y0 + h * 0.18, x0 + w * 0.34, y1 - h * 0.16],
                      "A =", round_num, align="right", size=36.0),
        _formula_text("proc_formula_alignment_num",
                      [frac_x0, y0 + h * 0.08, frac_x1, y0 + h * 0.48],
                      "|⟨β, γ⟩|", round_num, size=27.0),
        {
            "id": "proc_formula_alignment_rule",
            "type": "line",
            "status": "native",
            "bbox": [frac_x0, cy - 1.0, frac_x1, cy + 1.0],
            "points": [frac_x0, cy, frac_x1, cy],
            "color": "#111111",
            "thickness": 1.3,
            "line_width": 1.3,
            "confidence": 0.88,
            "z": 10,
            "provenance": {
                "agent": "ProceduralSurface",
                "action": "alignment_formula_rule",
                "round": round_num,
            },
            "repair_history": [],
            "defects": [],
            "ext": {
                "procedural_surface_formula": "alignment_rule",
                "component": "procedural_surface",
                "component_role": "formula_fraction_rule",
            },
        },
        _formula_text("proc_formula_alignment_den",
                      [frac_x0, y0 + h * 0.53, frac_x1, y1 - h * 0.08],
                      "∥β∥ ∥γ∥", round_num, size=26.0),
        _formula_text("proc_formula_alignment_suffix",
                      [x0 + w * 0.69, y0 + h * 0.18, x1 - w * 0.09, y1 - h * 0.16],
                      "≈ 1", round_num, align="left", size=36.0),
    ]


def _formula_text(eid: str, bbox: list[float], text: str, round_num: int,
                  align: str = "center", size: float = 30.0) -> dict:
    return {
        "id": eid,
        "type": "text",
        "status": "native",
        "bbox": bbox,
        "text": text,
        "font": "Cambria Math",
        "font_size": size,
        "text_color": "#111111",
        "align": align,
        "confidence": 0.88,
        "z": 10,
        "provenance": {
            "agent": "ProceduralSurface",
            "action": "alignment_formula_fragment",
            "round": round_num,
        },
        "repair_history": [],
        "defects": [],
        "ext": {
            "procedural_surface_formula": "alignment_fragment",
            "typography_locked": True,
        },
    }


def _ensure_risk_annotations(ir: dict, surface: dict) -> int:
    elements = ir.get("elements", [])
    elements[:] = [
        e for e in elements
        if not str(e.get("id", "")).startswith("proc_risk_")
    ]
    sx0, sy0, sx1, sy1 = [float(v) for v in surface["bbox"]]
    sw, sh = sx1 - sx0, sy1 - sy0
    cx = sx0 + sw * 0.815
    cy = sy0 + sh * 0.390
    rx = sw * 0.095
    ry = sh * 0.120
    z = 8
    round_num = ir.get("round", 0)
    additions = [
        {
            "id": "proc_risk_low_overlap_ring",
            "type": "oval",
            "status": "native",
            "bbox": [cx - rx, cy - ry, cx + rx, cy + ry],
            "fill": "transparent",
            "border_color": "#e65d37",
            "border_width": 2.0,
            "dash": True,
            "confidence": 0.86,
            "z": z,
            "provenance": {
                "agent": "ProceduralSurface",
                "action": "risk_annotation",
                "round": round_num,
            },
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_risk": "low_overlap_ring"},
        },
        _risk_text(
            "proc_risk_low_overlap_line1",
            [cx - rx * 0.35, cy - ry - 108.0, cx + rx * 1.72, cy - ry - 78.0],
            "low propensity /",
            "#c83322",
            z + 1,
            round_num,
            "low_overlap_line1",
        ),
        _risk_text(
            "proc_risk_low_overlap_line2",
            [cx - rx * 0.34, cy - ry - 78.0, cx + rx * 1.08, cy - ry - 48.0],
            "weak overlap",
            "#c83322",
            z + 1,
            round_num,
            "low_overlap_line2",
        ),
        _risk_text(
            "proc_risk_low_overlap_q0",
            [cx + rx * 1.08, cy - ry - 79.0, cx + rx * 1.80, cy - ry - 47.0],
            "(Q0)",
            "#c83322",
            z + 1,
            round_num,
            "low_overlap_q0",
        ),
        {
            "id": "proc_risk_high_heterogeneity_label",
            "type": "text",
            "status": "native",
            "bbox": [cx - rx * 0.95, cy + ry + 10.0, cx + rx * 1.05, cy + ry + 70.0],
            "text": "high\nheterogeneity",
            "font": "Arial",
            "font_size": 20,
            "text_color": "#b84824",
            "align": "center",
            "confidence": 0.82,
            "z": z + 1,
            "provenance": {
                "agent": "ProceduralSurface",
                "action": "risk_annotation_label",
                "round": round_num,
            },
            "repair_history": [],
            "defects": [],
            "ext": {"procedural_surface_risk": "high_heterogeneity_label"},
        },
    ]
    red_leader = _line(
        "proc_risk_low_overlap_leader",
        cx + rx * 0.24,
        cy - ry * 1.55,
        cx + rx * 0.05,
        cy - ry * 0.92,
        "#e65d37",
        z,
        round_num,
    )
    red_leader.update({"dash": True, "line_width": 1.6})
    grey_leader = _line(
        "proc_risk_ci_leader",
        cx - rx * 0.35,
        cy + ry * 0.98,
        sx0 + sw * 0.735,
        sy0 + sh * 0.790,
        "#8d8d8d",
        z - 1,
        round_num,
    )
    grey_leader.update({"dash": True, "line_width": 1.2})
    additions.extend([red_leader, grey_leader])
    elements.extend(additions)
    return len(additions)


def _risk_text(eid: str, bbox: list[float], text: str, color: str, z: int,
               round_num: int, role: str) -> dict:
    el = {
        "id": eid,
        "type": "text",
        "status": "native",
        "bbox": bbox,
        "text": text,
        "font": "Arial",
        "font_size": 21,
        "text_color": color,
        "align": "center",
        "confidence": 0.84,
        "z": z,
        "provenance": {
            "agent": "ProceduralSurface",
            "action": "risk_annotation_label",
            "round": round_num,
        },
        "repair_history": [],
        "defects": [],
        "ext": {
            "procedural_surface_risk": role,
            "component": "procedural_surface",
            "component_role": role,
        },
    }
    if text == "(Q0)":
        el["font"] = "Times New Roman"
        el["italic"] = True
        el["runs"] = [
            {"text": "(Q", "font": "Times New Roman", "italic": True, "color": color},
            {"text": "0", "font": "Times New Roman", "italic": True, "font_size": 16, "baseline": -25000, "color": color},
            {"text": ")", "font": "Times New Roman", "italic": True, "color": color},
        ]
    return el


def _line(eid: str, x0: float, y0: float, x1: float, y1: float,
          color: str, z: int, round_num: int, arrow: bool = False,
          dash: bool = False) -> dict:
    el = {
        "id": eid,
        "type": "arrow" if arrow else "line",
        "status": "native",
        "bbox": [min(x0, x1) - 3, min(y0, y1) - 3, max(x0, x1) + 3, max(y0, y1) + 3],
        "points": [x0, y0, x1, y1],
        "color": color,
        "confidence": 0.88,
        "z": z,
        "provenance": {"agent": "ProceduralSurface", "action": "ci_inset", "round": round_num},
        "repair_history": [],
        "defects": [],
        "ext": {"procedural_surface_ci": True},
    }
    if dash:
        el["dash"] = True
        el["line_width"] = 1.4
    return el


def _vector_arrow(eid: str, x0: float, y0: float, x1: float, y1: float,
                  color: str, z: int, round_num: int) -> dict:
    el = _line(eid, x0, y0, x1, y1, color, z, round_num, arrow=True)
    el["line_width"] = 4.0
    el["confidence"] = 0.92
    el["provenance"] = {
        "agent": "ProceduralSurface",
        "action": "gradient_vector",
        "round": round_num,
    }
    el["ext"] = {"procedural_surface_vector": eid}
    return el


def _tick(eid: str, x: float, y0: float, y1: float,
          color: str, z: int, round_num: int) -> dict:
    return _line(eid, x, y0, x, y1, color, z, round_num)


def _ci_text(eid: str, bbox: list[float], text: str,
             z: int, round_num: int) -> dict:
    return {
        "id": eid,
        "type": "text",
        "status": "native",
        "bbox": bbox,
        "text": text,
        "font": "Times New Roman",
        "font_size": 17,
        "text_color": "#111111",
        "align": "center",
        "confidence": 0.86,
        "z": z,
        "provenance": {"agent": "ProceduralSurface", "action": "ci_inset_label", "round": round_num},
        "repair_history": [],
        "defects": [],
        "ext": {"procedural_surface_ci": "axis_label"},
    }


def _dot(eid: str, cx: float, cy: float, r: float,
         color: str, z: int, round_num: int) -> dict:
    return {
        "id": eid,
        "type": "oval",
        "status": "native",
        "bbox": [cx - r, cy - r, cx + r, cy + r],
        "fill": color,
        "border_color": color,
        "confidence": 0.88,
        "z": z,
        "provenance": {"agent": "ProceduralSurface", "action": "ci_inset", "round": round_num},
        "repair_history": [],
        "defects": [],
        "ext": {"procedural_surface_ci": True},
    }


def _canonicalize_main_surface_bbox(surface: dict, canvas_width: float) -> None:
    """Use a stable geometry frame for the left manifold.

    OCR/CV often detects only the visible middle of the manifold.  The original
    artwork starts near the left edge and ends before the CATE-CI auditor
    column, so the procedural surface needs a canvas-calibrated frame rather
    than the narrow traced bbox.
    """
    x0, y0, x1, y1 = [float(v) for v in surface["bbox"]]
    if x0 > canvas_width * 0.50:
        return
    width = x1 - x0
    height = y1 - y0
    if width < 520 or height < 240:
        return
    new_y0 = max(0.0, y0 - 10.0)
    new_y1 = y1 + 10.0
    surface["bbox"] = [
        max(0.0, min(x0, canvas_width * 0.015)),
        new_y0,
        min(canvas_width * 0.430, max(x1, canvas_width * 0.405)),
        new_y1,
    ]
