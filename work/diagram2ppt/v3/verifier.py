"""v3 Verifier: compares the true-rendered PPTX against the original image.

Produces structured defects and global metrics that the Planner and Attribution
Agent use to decide whether to accept, rollback, or repair.
"""
from __future__ import annotations

from PIL import Image

from work.diagram2ppt.v2 import diff as v2_diff
from work.diagram2ppt.v2.render import render as v2_proxy_render
from . import ir as IR, text_layout_audit


CRITICAL_TEXT_RESIDUAL = 0.5
CRITICAL_SHAPE_RESIDUAL = 0.45
CRITICAL_CONNECTOR_INK = 0.35
COVERAGE_THRESHOLD = 0.97


def verify(ir: dict, original_path: str, rendered_png_path: str) -> dict:
    """Run verification and return metrics + defects.

    Args:
        ir: v3 Global Native IR.
        original_path: path to the original reference image.
        rendered_png_path: path to the true-rendered PNG of the rebuilt PPTX.

    Returns:
        Dict with global metrics and a list of defects.
    """
    original = Image.open(original_path).convert("RGB")
    rendered = Image.open(rendered_png_path).convert("RGB")

    # Ensure same size for pixel-wise comparison.
    if rendered.size != original.size:
        rendered = rendered.resize(original.size, Image.LANCZOS)

    defects: list[dict] = []

    # 1. Per-element residuals.
    elements = ir.get("elements", [])
    shape_map = {e["id"]: e for e in elements if "bbox" in e}
    for el in elements:
        if "bbox" not in el:
            continue
        bbox = el["bbox"]
        children = v2_diff.children_of(el, elements)
        try:
            residual = v2_diff.element_residual(original, rendered, bbox, exclude=children)
        except Exception:
            residual = 1.0

        el["residual"] = residual

        threshold = CRITICAL_SHAPE_RESIDUAL
        if el.get("type") in ("text", "formula"):
            try:
                residual = v2_diff.text_residual(original, rendered, bbox)
            except Exception:
                residual = 1.0
            el["residual"] = residual
            threshold = CRITICAL_TEXT_RESIDUAL

        if residual >= threshold:
            defects.append({
                "id": f"defect_residual_{el['id']}",
                "type": "high_residual",
                "element_id": el["id"],
                "bbox": bbox,
                "severity": round(min(1.0, residual), 3),
                "reason": f"{el.get('type')} residual {residual:.3f} exceeds threshold {threshold:.3f}",
                "suggested_agent": _agent_for_type(el.get("type"), "high_residual", el),
            })

    # 2. Coverage: unexplained ink.
    coverage = v2_diff.coverage(original, _to_v2_ir(ir))
    explained_frac = coverage["explained_frac"]
    for miss in coverage.get("missing", []):
        bbox = miss["bbox"]
        area_frac = miss["area"] / (original.width * original.height)
        severity = min(1.0, area_frac * 100)
        owner = _owner_for_missing(bbox, elements)
        owner_type = owner.get("type") if owner else None
        defects.append({
            "id": f"defect_missing_{len(defects)}",
            "type": "missing_element",
            "element_id": owner.get("id", "") if owner else "",
            "bbox": bbox,
            "severity": round(severity, 3),
            "reason": f"unexplained ink region ({miss['area']} px)",
            "suggested_agent": _agent_for_type(owner_type, "high_residual", owner)
                               if owner else "ShapeAgent",
        })

    # 3. Connector ink support.
    for el in elements:
        if el.get("type") not in ("arrow", "line"):
            continue
        src = shape_map.get(el.get("from_id") or "")
        dst = shape_map.get(el.get("to_id") or "")
        if not src or not dst:
            continue
        from work.diagram2ppt.v2.render import _center, _edge_point
        start = _edge_point(src, _center(dst))
        end = _edge_point(dst, _center(src))
        ink_frac = v2_diff.connector_ink_fraction(original, start, end)
        if ink_frac < CRITICAL_CONNECTOR_INK:
            defects.append({
                "id": f"defect_connector_{el['id']}",
                "type": "connector_mismatch",
                "element_id": el["id"],
                "bbox": list(start) + list(end),
                "severity": round(1.0 - ink_frac, 3),
                "reason": f"connector ink support {ink_frac:.2f} below threshold {CRITICAL_CONNECTOR_INK:.2f}",
                "suggested_agent": "ConnectorAgent",
            })

    # 4. Overall visual delta (proxy: mean element residual or global SSIM).
    visual_delta = _visual_delta(original, rendered, ir)
    text_layout_defects, text_layout_metrics = text_layout_audit.audit(
        elements, original, rendered)
    existing_defect_ids = {d.get("id") for d in defects}
    defects.extend(
        d for d in text_layout_defects
        if d.get("id") not in existing_defect_ids
    )

    # Attach defects to IR blackboard BEFORE computing metrics.
    # Preserve "skipped" status set by the Planner so skipped defects are not
    # re-selected every round.
    skipped_ids = {d.get("id") for d in ir.get("defects", [])
                   if d.get("status") == "skipped"}
    for d in defects:
        if d.get("id") in skipped_ids:
            d["status"] = "skipped"
    ir["defects"] = defects

    metrics = {
        **IR.metrics(ir),
        "visual_delta": round(visual_delta, 4),
        "coverage_explained": round(explained_frac, 4),
        "coverage_missing_count": len(coverage.get("missing", [])),
        "text_accuracy": _text_accuracy(defects, elements),
        "connector_accuracy": _connector_accuracy(defects, elements),
        **_typography_metrics(elements),
        **text_layout_metrics,
    }

    ir["metrics"] = metrics

    return {
        "metrics": metrics,
        "defects": defects,
        "passed": metrics["critical_defect_count"] == 0
                  and explained_frac >= COVERAGE_THRESHOLD
                  and metrics.get("native_fraction_count", 0) == 1.0,
    }


def _agent_for_type(el_type: str | None, defect_type: str = "",
                     element: dict | None = None) -> str:
    t = el_type or ""
    # Appearance mismatches need a style specialist; missing geometry needs
    # a shape detector.  If a text/formula element already has content, its
    # residual is almost certainly a font/size/color/style mismatch, not a
    # content error, so send it to StyleAgent.
    if defect_type == "high_residual":
        if t == "text":
            if element and element.get("text") and not _looks_like_bad_ocr(element):
                return "StyleAgent"
            return "TextAgent"
        if t == "formula":
            if element and element.get("text"):
                return "StyleAgent"
            return "FormulaAgent"
        if t in ("dotcloud", "surface"):
            return "SurfaceAgent"
        if t in ("rect", "rounded_rect", "oval", "diamond", "hexagon",
                 "parallelogram"):
            return "StyleAgent"
        if t in ("arrow", "line"):
            return "ConnectorAgent"
        if t == "chart":
            return "ChartAgent"
        if t == "icon":
            return "IconAgent"
        if t == "freeform":
            return "VectorizeAgent"
    if defect_type == "missing_element":
        return "ShapeAgent"
    if defect_type == "connector_mismatch":
        return "ConnectorAgent"
    # Fallback for legacy callers.
    mapping = {
        "text": "TextAgent",
        "formula": "FormulaAgent",
        "chart": "ChartAgent",
        "icon": "IconAgent",
        "arrow": "ConnectorAgent",
        "line": "ConnectorAgent",
        "freeform": "VectorizeAgent",
        "rect": "ShapeAgent",
        "rounded_rect": "ShapeAgent",
        "oval": "ShapeAgent",
        "diamond": "ShapeAgent",
        "hexagon": "ShapeAgent",
        "parallelogram": "ShapeAgent",
        "dotcloud": "SurfaceAgent",
        "surface": "SurfaceAgent",
    }
    return mapping.get(t, "LayoutAgent")


def _owner_for_missing(bbox: list, elements: list[dict]) -> dict | None:
    """Return the existing element that most likely owns a missing ink blob."""
    best = None
    best_score = 0.0
    for el in elements:
        eb = el.get("bbox")
        if not eb:
            continue
        score = _overlap_frac(bbox, eb)
        if score > best_score:
            best = el
            best_score = score
    if best_score >= 0.45:
        return best
    return None


def _overlap_frac(a: list, b: list) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return (ix * iy) / area


def _looks_like_bad_ocr(element: dict) -> bool:
    text = (element.get("text") or "").strip()
    if not text:
        return True
    x0, y0, x1, y1 = element.get("bbox", [0, 0, 0, 0])
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    if len(text) <= 3 and w > 80 and h > 35:
        return True
    letters = sum(ch.isalpha() for ch in text)
    if len(text) <= 8 and letters <= 2 and w > 70:
        return True
    return False


def _to_v2_ir(ir: dict) -> dict:
    return {
        "elements": ir.get("elements", []),
    }


def _visual_delta(original: Image.Image, rendered: Image.Image, ir: dict) -> float:
    """Global visual difference estimate."""
    residuals = [e.get("residual", 0.0) for e in ir.get("elements", [])
                 if e.get("residual") is not None]
    if residuals:
        return sum(residuals) / len(residuals)
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        a = np.asarray(original.convert("L"))
        b = np.asarray(rendered.convert("L"))
        return 1.0 - ssim(a, b, data_range=255.0)
    except Exception:
        return 1.0


def _text_accuracy(defects: list[dict], elements: list[dict]) -> float:
    texts = [e for e in elements if e.get("type") in ("text", "formula")]
    if not texts:
        return 1.0
    bad = sum(1 for d in defects if d["type"] == "high_residual"
              and any(d.get("element_id") == e["id"] for e in texts))
    return round(1.0 - bad / len(texts), 4)


def _typography_metrics(elements: list[dict]) -> dict:
    texts = [
        e for e in elements
        if e.get("type") in ("text", "formula") and e.get("bbox")
    ]
    if not texts:
        return {
            "typography_role_fraction": 1.0,
            "typography_overflow_count": 0,
        }
    role_count = 0
    overflow = 0
    for el in texts:
        typo = ((el.get("ext") or {}).get("typography") or {})
        if typo.get("role"):
            role_count += 1
        if _text_likely_overflows(el):
            overflow += 1
    return {
        "typography_role_fraction": round(role_count / len(texts), 4),
        "typography_overflow_count": overflow,
    }


def _text_likely_overflows(el: dict) -> bool:
    text = str(el.get("text") or el.get("latex") or "")
    if not text:
        return False
    x0, y0, x1, y1 = [float(v) for v in el["bbox"]]
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    rotation = abs(float(el.get("rotation") or 0.0)) % 180
    if 70 <= rotation <= 110:
        w, h = h, w
    lines = [ln for ln in text.splitlines() if ln] or [text]
    size = float(el.get("font_size") or 12.0)
    longest = max(len(ln) for ln in lines)
    typo = ((el.get("ext") or {}).get("typography") or {})
    role = str(typo.get("role") or "")
    width_factor = float(
        typo.get("fit_width_factor")
        or (0.45 if el.get("type") == "formula" or role == "formula" else 0.52)
    )
    height_factor = float(typo.get("fit_height_factor") or 0.84)
    estimated_w = longest * size * width_factor
    estimated_h = len(lines) * size / max(0.5, height_factor)
    width_slack = 1.14 if el.get("type") == "formula" or role == "formula" else 1.04
    return estimated_w > w * width_slack or estimated_h > h * 1.08


def _connector_accuracy(defects: list[dict], elements: list[dict]) -> float:
    conns = [e for e in elements if e.get("type") in ("arrow", "line")]
    if not conns:
        return 1.0
    bad = sum(1 for d in defects if d["type"] == "connector_mismatch")
    return round(1.0 - bad / len(conns), 4)
