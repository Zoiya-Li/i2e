"""Stage 2 of the decompose→process→integrate pipeline: per-type processors.

Each handler takes an entity (typed bbox from decompose) + the original image
and fills the svg_export-compatible fields (text, fill, chart spec, icon kind,
dots…). The split is by TOOL, which is the whole point of the redesign:

  text / formula / chart / icon  → VLM on a HIGH-RESOLUTION crop of just that
                                   entity (the global one-shot pass lost these
                                   at low res; a tight crop fixes it at the root)
  shape / container / arrow      → deterministic CV (colors, geometry, skeleton)
  surface / dotcloud             → deterministic CV via vectorize.py (dots,
                                   wave bands, flow lines) — never pixels

Handlers are independent (arrows/surfaces reference only OTHER entities'
bboxes, which decompose already fixed), so process_all() runs them in parallel.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from . import vectorize as V
from .vlm import VLMClient

# -- VLM prompts (content extraction — what decompose deliberately did NOT do) -

TEXT_PROMPT = (
    "Read the text in this image crop EXACTLY as written — same wording, case, "
    "line breaks, punctuation. Transcribe LITERALLY: do NOT autocorrect, "
    "'fix', or substitute unusual, rare, or seemingly-misspelled words (e.g. a "
    "proper noun like 'Cliff' must stay 'Cliff', not 'Unit'); reproduce the "
    "visible characters even if they look odd. Output ONLY the text, nothing "
    "else (no labels, no quotes, no commentary)."
)

# Short, model-agnostic prompt for dedicated OCR models (Qwen-VL, etc.).
OCR_PROMPT = (
    "Transcribe all visible text in this image. Output only the exact text, "
    "preserving line breaks. No explanation, no labels."
)

FORMULA_PROMPT = (
    "Transcribe the math expression in this image using Unicode math characters "
    "so it reads correctly as plain text. Use Greek letters, subscripts, "
    "superscripts, and standard math symbols as they appear. Do NOT list examples, "
    "do NOT echo the instruction, and do NOT add commentary. Output ONLY the "
    "expression itself."
)

OCR_FORMULA_PROMPT = (
    "Transcribe the math expression in this image using Unicode math characters. "
    "Output only the expression, no explanation."
)

CHART_PROMPT = (
    "This crop is a chart. Identify its TYPE and output STRICT JSON only.\n\n"
    "Types:\n"
    "- bar: vertical or horizontal bars with categories + values\n"
    "- line: one or more polylines / curves with axes\n"
    "- scatter: cloud of points, optionally with a regression/trend line\n"
    "- pie: circular wedges\n\n"
    "Output schema (omit fields that do not apply):\n"
    '{\n'
    '  "type": "bar|line|scatter|pie",\n'
    '  "categories": ["...", "..."],\n'
    '  "series": [{"name": "...", "color": "#hex", "values": [numbers]}],\n'
    '  "points": [{"x": number, "y": number}, ...],\n'
    '  "trend": {"slope": number, "intercept": number, "color": "#hex"}\n'
    '}\n\n'
    "For bar charts: list EVERY visible bar, grouped by color as separate "
    "series. Do not omit small bars. Read category labels VERBATIM.\n"
    "For line charts: read axis/category labels VERBATIM and output at least "
    "12 points per series in [0,1] image coordinates.\n"
    "For scatter: output at least 8 representative (x,y) points in [0,1] "
    "image coordinates, and the trend line if visible.\n"
    "Never invent placeholder labels like 'Category 1' or 'Point 1'.\n"
    "If the crop is NOT a chart, output exactly "
    '{"type": "none"}. Output ONLY the JSON, no commentary.'
)


# -- orchestration --------------------------------------------------------

def process_all(entities: list[dict], original, vlm, max_workers: int = 8,
                log=print) -> list[dict]:
    """Run every entity's handler in parallel, filling content in place."""
    ok, fail = Counter(), Counter()
    ocr_model = os.environ.get("I2E_OCR_MODEL")
    use_ocr_prompt = bool(ocr_model)
    ocr = VLMClient(model=ocr_model) if ocr_model else vlm

    # Cost-effective batch OCR for text: group nearby text boxes so one API call
    # transcribes several labels with surrounding context.  Formulas stay
    # individual because their layout is more brittle.
    if os.environ.get("I2E_BATCH_OCR", "0") == "1":
        _batch_ocr_texts(entities, original, ocr or vlm, use_ocr_prompt, log)

    def work(e):
        try:
            _dispatch(e, original, vlm, entities, ocr, use_ocr_prompt)
            ok[e["type"]] += 1
        except Exception as exc:
            fail[e["type"]] += 1
            log(f"  [handler] {e['id']} ({e['type']}): {exc}")
            e["content"] = None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, entities))
    log(f"[handlers] ok={dict(ok)} fail={dict(fail)}")
    return entities


def _dispatch(e: dict, original, vlm, elements: list[dict],
              ocr: VLMClient | None = None,
              use_ocr_prompt: bool = False) -> None:
    t = e["type"]
    # Text/formula may have been filled by decompose batch OCR already.
    if t == "text" and e.get("text") is not None:
        return
    if t == "text":
        _text(e, original, ocr or vlm, use_ocr_prompt)
    elif t == "formula" and e.get("text") is not None:
        return
    elif t == "formula":
        _formula(e, original, ocr or vlm, use_ocr_prompt)
    elif t in ("shape", "container"):
        _shape(e, original)
    elif t == "arrow":
        _arrow(e, original, elements)
    elif t == "chart":
        _chart(e, original, vlm)
    elif t == "icon":
        _icon(e, original, vlm)
    elif t in ("surface", "dotcloud"):
        _surface(e, original, elements)
    e["content"] = "done"


# -- crop helper ----------------------------------------------------------

def _crop(original, bbox, pad=0.05, min_side=0):
    """Padded crop; tiny crops upscaled so text/formula stay legible to the VLM."""
    w, h = original.size
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    box = (max(0, int(x0 - dx)), max(0, int(y0 - dy)),
           min(w, int(x1 + dx)), min(h, int(y1 + dy)))
    c = original.crop(box)
    if min_side and min(c.size) < min_side:
        f = min_side / min(c.size)
        c = c.resize((max(1, int(c.size[0] * f)), max(1, int(c.size[1] * f))),
                     Image.LANCZOS)
    return c, box


# -- VLM handlers ---------------------------------------------------------

def _batch_ocr_texts(entities, original, vlm, use_ocr_prompt, log):
    """Transcribe text entities in spatial batches to save API calls and give
    the model context.  Skips entities that already have text."""
    texts = [e for e in entities if e["type"] == "text" and e.get("text") is None]
    if len(texts) < 2:
        return
    # Sort top-to-bottom, left-to-right for stable batching.
    texts.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    batch_size = int(os.environ.get("I2E_BATCH_OCR_SIZE", "2"))
    prompt = OCR_PROMPT if use_ocr_prompt else TEXT_PROMPT
    prompt += (
        "\n\nThis image shows multiple text crops separated by white space, "
        "arranged left-to-right then top-to-bottom. Transcribe each crop exactly "
        "as written. Output ONLY a numbered list, one line per crop, like:\n"
        "1. first text\n2. second text\n..."
    )
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        n = len(batch)
        # Horizontal strip for clarity; avoids confusing row/column numbering.
        cell_h = 0
        total_w = 0
        crops = []
        for e in batch:
            crop, _ = _crop(original, e["bbox"], pad=0.10, min_side=360)
            crops.append(crop)
            cell_h = max(cell_h, crop.height)
            total_w += crop.width
        pad = 12
        comp_w = total_w + (n + 1) * pad
        comp_h = cell_h + 2 * pad
        comp = Image.new("RGB", (comp_w, comp_h), "white")
        draw = ImageDraw.Draw(comp)
        x = pad
        for idx, crop in enumerate(crops):
            y = pad + (cell_h - crop.height) // 2
            comp.paste(crop, (x, y))
            draw.rectangle([x - 2, y - 2, x + crop.width + 2, y + crop.height + 2],
                           outline="#dddddd", width=1)
            x += crop.width + pad
        try:
            raw = vlm.chat(prompt, comp, max_tokens=800, max_edge=1280,
                           frequency_penalty=0.1)
            lines = _parse_batch(raw, n)
            for e, txt in zip(batch, lines):
                txt = _strip(txt)
                if _is_refusal(txt):
                    txt = ""
                e["text"] = txt
                # Sample color/font from the original crop.
                crop, _ = _crop(original, e["bbox"], pad=0.10, min_side=360)
                e["text_color"] = _sample_ink_color(crop)
                e["font_size"] = max(9, int((e["bbox"][3] - e["bbox"][1]) * 0.62))
                e["align"] = _detect_text_alignment(crop)
        except Exception as exc:
            log(f"  [batch ocr] failed batch {i}: {exc}")


def _parse_batch(raw: str, n: int) -> list[str]:
    """Parse a numbered list of N transcriptions from the model output."""
    out = [""] * n
    if not raw:
        return out
    # Remove code fences.
    s = re.sub(r"^```.*?$|```", "", raw.strip(), flags=re.M).strip()
    # Extract lines that start with "1.", "2.", etc.
    pat = re.compile(r"^\s*(\d+)\s*[.:\)]\s*(.*)$", re.M)
    for m in pat.finditer(s):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            out[idx] = m.group(2).strip()
    return out


def _prep_ocr_crop(crop: Image.Image) -> Image.Image:
    """Optional grayscale + autocontrast + mild sharpen for small diagram text."""
    gray = crop.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    return gray


def _text(e, original, vlm, use_ocr_prompt=False):
    # Slightly more padding and a larger minimum side so small/dense labels stay
    # legible after the VLM downscales the crop.
    crop, _ = _crop(original, e["bbox"], pad=0.10, min_side=360)
    if os.environ.get("I2E_OCR_PREPROCESS"):
        crop = _prep_ocr_crop(crop)
    prompt = OCR_PROMPT if use_ocr_prompt else TEXT_PROMPT
    raw = vlm.chat(prompt, crop, max_tokens=600, max_edge=1280,
                   frequency_penalty=0.1)
    txt = _strip(raw)
    if _is_refusal(txt) or _is_degenerate(txt):
        txt = ""
    e["text"] = txt
    e["text_color"] = _sample_ink_color(crop)
    e["font_size"] = max(9, int((e["bbox"][3] - e["bbox"][1]) * 0.62))
    e["align"] = _detect_text_alignment(crop)


def _formula(e, original, vlm, use_ocr_prompt=False):
    # Always run the VLM on formula crops: Tesseract cannot read Greek letters
    # or math symbols (it turns β into B, ∇ into V, etc.). The VLM does a much
    # better job on these small math expressions.
    crop, _ = _crop(original, e["bbox"], pad=0.10, min_side=360)
    if os.environ.get("I2E_OCR_PREPROCESS"):
        crop = _prep_ocr_crop(crop)
    prompt = OCR_FORMULA_PROMPT if use_ocr_prompt else FORMULA_PROMPT
    raw = vlm.chat(prompt, crop, max_tokens=900, max_edge=1024,
                   frequency_penalty=0.1)
    txt = _strip(raw)
    if _is_refusal(txt) or _is_degenerate(txt):
        txt = ""
    e["text"] = txt
    e["text_color"] = _sample_ink_color(crop)
    e["font_size"] = max(10, int((e["bbox"][3] - e["bbox"][1]) * 0.58))


_FANTASY = re.compile(r"(?i)^(category|point|bar|item|series|value)[ _]?\d+$")


def _extract_trend_line(crop: Image.Image) -> list | None:
    """Try to find a straight trend line in a scatter crop (crop-local px).

    Uses Hough on the ink mask after removing round dots. Returns
    [[x0,y0],[x1,y1]] or None.
    """
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    mask = (np.abs(arr.astype(int) - bg).sum(2) > 50).astype(np.uint8) * 255
    # remove obvious round dots
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = bw / max(1, bh)
        fill = area / max(1, bw * bh)
        if 0.5 <= aspect <= 2.0 and fill >= 0.45 and area <= 200:
            mask[labels == i] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=25,
                            minLineLength=int(0.25 * max(crop.size)),
                            maxLineGap=8)
    if lines is None:
        return None
    best = max(lines, key=lambda l: np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]))[0]
    return [[float(best[0]), float(best[1])], [float(best[2]), float(best[3])]]


def _extract_chart_curves(crop: Image.Image) -> list[list[list[float]]]:
    """Extract prominent curve polylines from a line/scatter chart crop.

    Removes round dots and text, then traces the remaining elongated strokes.
    Returns a list of polylines in crop-local coordinates.
    """
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    mask = (np.abs(arr.astype(int) - bg).sum(2) > 50).astype(np.uint8) * 255
    # Remove round dots and small blobs
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        fill = area / max(1, bw * bh)
        if aspect <= 3.0 and fill >= 0.35 and area <= 250:
            mask[labels == i] = 0
    # Slight morphological close to connect fragmented curve pieces
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curves = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        # Keep only elongated contours (curves, not stray dots)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) / max(1, min(bw, bh)) < 2.5:
            continue
        # Uniform subsample
        pts = cnt.reshape(-1, 2)
        if len(pts) > 40:
            idx = np.linspace(0, len(pts) - 1, 40).astype(int)
            pts = pts[idx]
        curves.append([[float(p[0]), float(p[1])] for p in pts])
    return curves


def _chart(e, original, vlm):
    x0, y0, x1, y1 = [float(v) for v in e.get("bbox", [0, 0, 0, 0])]
    bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
    aspect = max(bw, bh) / max(1.0, min(bw, bh))
    if bw < 18 or bh < 18 or aspect > 12:
        # Bad structure memory occasionally labels a 1px-wide vertical strip as
        # a chart.  Sending that to SiliconFlow produces hard 400 errors
        # ("absolute aspect ratio ...").  Treat it as a non-chart native no-op;
        # real charts in this figure have usable area and modest aspect ratio.
        e["type"] = "dotcloud"
        e["dots"] = []
        e["paths"] = []
        e["content"] = "invalid_chart_geometry"
        return
    crop, _ = _crop(original, e["bbox"], pad=0.04, min_side=320)
    raw = vlm.chat(CHART_PROMPT, crop, max_tokens=1800, max_edge=1024,
                   frequency_penalty=0.1)
    spec = _parse_json(raw) or {"type": "none"}
    ctype = (spec.get("type") or "").lower()
    cats = spec.get("categories") or []

    # Reject hallucinated placeholder labels for bar charts
    if ctype == "bar" and cats and all(_FANTASY.match(str(c).strip()) for c in cats):
        ctype = "none"

    if ctype not in ("bar", "line", "scatter", "pie"):
        # Fallback: render unparseable charts as vectorized dots + curves so we
        # never ship an empty box.
        e["_raw"] = raw[:300]
        from . import vectorize as V
        dots = V.extract_dots(crop, round_only=True, ink_threshold=100,
                              max_dots=120)
        trend = _extract_trend_line(crop)
        curves = _extract_chart_curves(crop)
        # CV-based scatter recovery: if we see enough round dots + a trend/curve,
        # promote to a native scatter chart instead of a vague dotcloud.
        if _looks_like_scatter_chart(e, crop) and len(dots) >= 12 \
                and (trend is not None or curves):
            spec = _dots_to_scatter_spec(crop, dots, trend)
            ctype = "scatter"
            e["chart"] = spec
            return
        e["type"] = "dotcloud"
        e["dots"] = dots
        e["trend"] = trend
        e["curves"] = curves
        return

    for s in spec.get("series", []):
        s.setdefault("color", "#4472c4")
        s.setdefault("values", [])
    spec.setdefault("points", [])
    spec.setdefault("trend", {})
    e["chart"] = spec


def _dots_to_scatter_spec(crop: Image.Image, dots, trend):
    """Build a scatter chart spec from CV dot detections + trend line."""
    cw, ch = crop.size
    cw = max(1, cw)
    ch = max(1, ch)
    points = []
    for d in dots:
        points.append({
            "x": float(d["cx"]) / cw,
            "y": 1.0 - float(d["cy"]) / ch,
            "color": d.get("color", "#4472c4"),
        })
    spec = {"type": "scatter", "points": points, "trend": {}}
    if trend:
        (x1, y1), (x2, y2) = trend
        dx = (x2 - x1)
        if abs(dx) > 1e-3:
            m = -((y2 - y1) * cw) / (dx * ch)
        else:
            m = 0.0
        b = 1.0 - y1 / ch - m * (x1 / cw)
        spec["trend"] = {"slope": float(m), "intercept": float(b),
                         "color": "#4472c4"}
    return spec


def _icon(e, original, vlm):
    crop, _ = _crop(original, e["bbox"], pad=0.08, min_side=160)
    kind, color, _glyph = V._classify_icon(crop, vlm, lambda *a: None)
    # VLM-reported colors often pick anti-aliased gray. Re-sample from the
    # most saturated non-background pixels in the crop for a faithful icon hue.
    arr = np.asarray(crop.convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    pixels = arr.reshape(-1, 3)
    dist = np.abs(pixels - bg).sum(1)
    fg = pixels[dist > 50]
    if len(fg) > 10:
        sat = fg.max(1) - fg.min(1)
        k = max(8, len(fg) // 4)
        saturated = fg[np.argsort(sat)[-k:]]
        sat_med = np.median(saturated, axis=0)
        # Only trust the sample if it is clearly non-gray.
        if max(sat_med) - min(sat_med) > 20:
            color = _hex(sat_med)
    e["icon"] = {"kind": kind, "color": color}


# -- CV handlers ----------------------------------------------------------

def _shape(e, original):
    """Fill/edge colors, border width, dash, and geometry from the pixels."""
    import cv2
    crop, _ = _crop(original, e["bbox"], pad=0.02)
    arr = np.asarray(crop.convert("RGB"))
    H, W, _ = arr.shape
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.abs(arr.astype(int) - bg).sum(2)
    ink = dist > 60

    # fill = interior median if the interior is noticeably non-background
    m = max(2, int(min(H, W) * 0.18))
    interior = arr[m:H - m, m:W - m].reshape(-1, 3)
    if interior.size and np.abs(interior - bg).sum(1).mean() > 35:
        e["fill"] = _hex(np.median(interior, axis=0))
    else:
        e["fill"] = ""

    # Header strip detection for panels: many diagram containers have a colored
    # top band. Detect it and recompute the body fill from the lower region.
    if e.get("type") in ("container", "shape"):
        header_fill, header_h = _detect_header(arr, bg)
        if header_fill:
            e["header_fill"] = header_fill
            e["header_height"] = header_h
            body_region = arr[header_h + max(2, H // 20):int(H * 0.92),
                            m:W - m]
            if body_region.size:
                body_med = np.median(body_region.reshape(-1, 3), axis=0)
                if np.abs(body_med - bg).sum() > 25:
                    e["fill"] = _hex(body_med)
                else:
                    e["fill"] = ""

    # border color from the edge band's non-background pixels.
    # Use the most saturated subset rather than the median, because anti-aliased
    # edges average toward gray and hide the true border color.
    eb = np.concatenate([arr[:max(2, H // 12)].reshape(-1, 3),
                         arr[-max(2, H // 12):].reshape(-1, 3),
                         arr[:, :max(2, W // 12)].reshape(-1, 3),
                         arr[:, -max(2, W // 12):].reshape(-1, 3)])
    nz = eb[np.abs(eb - bg).sum(1) > 60]
    if len(nz) > 24:
        sat = nz.max(1) - nz.min(1)
        k = max(12, len(nz) // 5)
        saturated = nz[np.argsort(sat)[-k:]]
        e["border_color"] = _hex(np.median(saturated, axis=0))
    else:
        e["border_color"] = "#888888"
    e["border_width"] = max(1, int(np.median(_edge_widths(ink))))
    e["dash"] = _is_dashed(ink)

    # Geometry is computed on a *tight* crop so corner-radius estimates are not
    # biased by the 2% padding we add for color sampling.
    tight, tbox = _crop(original, e["bbox"], pad=0.0)
    tarr = np.asarray(tight.convert("RGB"))
    tH, tW, _ = tarr.shape
    tborder = np.concatenate([tarr[0], tarr[-1], tarr[:, 0], tarr[:, -1]])
    tbg = np.median(tborder, axis=0)
    tink = (np.abs(tarr.astype(int) - tbg).sum(2) > 60).astype(bool)

    # Containers and generic shapes in framework diagrams are overwhelmingly
    # rounded rectangles.  Pure CV classification is fragile on thin borders
    # with interior noise, so we default to rounded_rect with a conservative
    # rx based on the box size.  We only fall back to rect when the corner scan
    # proves the corners are sharp.
    original_type = e.get("type")
    if original_type in ("container", "shape"):
        w, h = e["bbox"][2] - e["bbox"][0], e["bbox"][3] - e["bbox"][1]
        rx = _estimate_corner_radius(tink, e["bbox"])
        # If the scan is inconsistent (too large relative to the box), fall back
        # to a diagram-typical radius.
        max_rx = min(w, h) * 0.22
        if rx > max_rx or rx < 4.0:
            rx = max(6.0, min(w, h) * 0.12)
        geom, conf = "rounded_rect", 0.75
    else:
        geom, conf = _geometry(tink, e["bbox"])
    e["type"] = geom
    e["geometry_confidence"] = conf
    if geom == "rounded_rect":
        e["rx"] = rx if original_type in ("container", "shape") \
                   else _estimate_corner_radius(tink, e["bbox"])
    e["bold"] = False


def _detect_header(arr, bg):
    """Detect a colored header strip at the top of a panel/container.

    Returns (header_hex_color, header_height) or (None, None).
    """
    H, W, _ = arr.shape
    if H < 30 or W < 30:
        return None, None
    hh = max(1, int(H * 0.25))
    header_region = arr[:hh, :, :].reshape(-1, 3)
    body_region = arr[int(H * 0.35):int(H * 0.90), :, :].reshape(-1, 3)
    if not header_region.size or not body_region.size:
        return None, None
    header_med = np.median(header_region, axis=0)
    body_med = np.median(body_region, axis=0)
    header_body_diff = float(np.abs(header_med - body_med).sum())
    header_bg_diff = float(np.abs(header_med - bg).sum())
    # Header must differ from body and from background.
    if header_body_diff > 55 and header_bg_diff > 45:
        return _hex(header_med), hh
    return None, None


def _enclosing_element(bbox, elements):
    """Return the smallest container/surface/dotcloud that fully contains bbox."""
    x0, y0, x1, y1 = bbox
    best = None
    for o in elements:
        if o.get("type") not in ("container", "surface", "dotcloud"):
            continue
        if "bbox" not in o:
            continue
        bx0, by0, bx1, by1 = o["bbox"]
        if bx0 <= x0 and by0 <= y0 and bx1 >= x1 and by1 >= y1:
            area = (bx1 - bx0) * (by1 - by0)
            if best is None or area < best["_area"]:
                best = dict(o, _area=area)
    return best


def _arrow(e, original, elements):
    """Big saturated arrows via Hough; else skeleton extremes. Snap endpoints
    to the nearest shape/container so svg_export can draw from→to."""
    bbox = e["bbox"]
    x0, y0, x1, y1 = bbox
    w, h = original.size
    # Drop connectors that are implausibly large (often a false panel seam).
    area = (x1 - x0) * (y1 - y0)
    if area > 0.12 * w * h and (x1 - x0) > 0.5 * w and (y1 - y0) > 0.3 * h:
        e["points"] = None
        return
    fats = V.detect_fat_arrows(original, bbox, max_n=1)
    if fats:
        f = fats[0]
        e["points"] = f["points"]
        e["color"] = f["color"]
        e["thickness"] = f["thickness"]
    else:
        crop, box = _crop(original, bbox, pad=0.04)
        pts, color, th = _skeleton_arrow(crop, box)
        e["points"] = pts
        e["color"] = color
        e["thickness"] = th
    # Do not snap vectors/arrows that live entirely inside a single surface or
    # container — snapping them to the panel boundary destroys their geometry.
    enclosing = _enclosing_element(bbox, elements)
    if enclosing and enclosing.get("type") in ("container", "surface", "dotcloud"):
        pass
    else:
        fid, tid, fpt, tpt = _snap(e["points"], elements)
        if fid:
            e["from_id"], e["to_id"] = fid, tid
            if fpt and tpt:
                e["points"] = [float(fpt[0]), float(fpt[1]),
                               float(tpt[0]), float(tpt[1])]


def _surface(e, original, elements):
    """dots (filtered) + flow lines for svg_surface. Bands are extracted at
    export time by manifold_svg; here we only pre-seed what it reads from el."""
    crop, box = _crop(original, e["bbox"], pad=0.0)
    excl = _local_excludes(e, elements, box)
    if e["type"] == "surface":
        wb = V.extract_wave_bands(crop, exclude=excl)
        if wb and len(wb.get("curves", [])) >= 2:
            e["streamlines"] = V._synth_flow_lines(wb)
    e["dots"] = V.extract_dots(crop, exclude=excl, round_only=True,
                               ink_threshold=130, max_dots=300)

    # Small dotclouds that are really scatter plots (e.g. step-box charts the
    # detector demoted) get promoted to native scatter charts.
    if e["type"] == "dotcloud" and _looks_like_scatter_chart(e, crop):
        dots = V.extract_dots(crop, round_only=True, ink_threshold=100,
                              max_dots=120)
        if len(dots) >= 12:
            trend = _extract_trend_line(crop)
            if trend is not None:
                e["type"] = "chart"
                e["chart"] = _dots_to_scatter_spec(crop, dots, trend)
                e.pop("dots", None)


def _looks_like_scatter_chart(e, crop):
    """Reject large organic regions / 3D surfaces from scatter recovery."""
    x0, y0, x1, y1 = e["bbox"]
    w, h = x1 - x0, y1 - y0
    area = w * h
    return area < 20_000 and w < 200 and h < 140


# -- CV helpers -----------------------------------------------------------

def _hex(arr) -> str:
    """Median color of an Nx3 (or 1x3) RGB array → '#rrggbb'. Accepts either a
    flat length-3 array or an Nx3 stack of pixels."""
    a = np.asarray(arr).reshape(-1, 3)
    m = np.clip(np.median(a, axis=0), 0, 255).astype(int)
    return "#%02x%02x%02x" % (int(m[0]), int(m[1]), int(m[2]))


def _sample_ink_color(crop) -> str:
    g = np.asarray(crop.convert("L"))
    arr = np.asarray(crop.convert("RGB"))
    dark = g < 120
    if dark.sum() < 5:
        return "#000000"
    return _hex(arr[dark].reshape(-1, 3))


def _detect_text_alignment(crop) -> str:
    """Infer 'left' | 'center' | 'right' from the ink distribution in a text crop.

    Uses horizontal projections per text line; robust to multi-line blocks.
    """
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    if arr.size == 0:
        return "center"
    # median background from border
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    ink = (np.abs(arr.astype(int) - bg).sum(2) > 50).astype(np.uint8) * 255
    if ink.sum() == 0:
        return "center"
    # horizontal projection to find text rows
    hproj = ink.sum(axis=1)
    # split into line bands by threshold
    mean_ink = hproj.mean()
    in_line = hproj > max(mean_ink * 0.3, 10)
    lines = []
    start = None
    for i, v in enumerate(in_line):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 3:
                lines.append((start, i))
            start = None
    if start is not None and len(in_line) - start >= 3:
        lines.append((start, len(in_line)))
    if not lines:
        return "center"
    lefts, rights, centers = [], [], []
    W = ink.shape[1]
    for y0, y1 in lines:
        band = ink[y0:y1, :]
        xs = np.where(band.any(axis=0))[0]
        if len(xs) == 0:
            continue
        l, r = xs[0], xs[-1]
        lefts.append(l / W)
        rights.append((W - 1 - r) / W)
        centers.append(((l + r) / 2) / W)
    if not lefts:
        return "center"
    # variation of centers, left margins, right margins
    lc = np.std(centers)
    ll = np.std(lefts)
    lr = np.std(rights)
    # if left margins are tight and right margins vary → left aligned
    if ll < 0.08 and lr > 0.12:
        return "left"
    # if right margins are tight and left margins vary → right aligned
    if lr < 0.08 and ll > 0.12:
        return "right"
    # otherwise center (default for diagrams)
    return "center"


def _edge_widths(ink: np.ndarray) -> list:
    """Ink run-lengths along the mid-row and mid-col — the border's thickness."""
    H, W = ink.shape
    mid_row = ink[H // 2] if H else np.array([])
    mid_col = ink[:, W // 2] if W else np.array([])
    out = []
    for line in (mid_row, mid_col):
        out.extend(r for r in _runs(line.astype(bool)) if r > 0)
    return out or [1]


def _runs(line: np.ndarray) -> list:
    out, cur, prev = [], 0, False
    for v in line:
        if v:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
        prev = v
    if cur:
        out.append(cur)
    return out


def _is_dashed(ink: np.ndarray) -> bool:
    """A dashed border shows ≥3 short ink runs separated by gaps along a line."""
    H, W = ink.shape
    band = ink[max(0, int(H * 0.03)):int(H * 0.13)] if H > 30 else ink[:1]
    line = band.any(0) if band.size else np.zeros(W, bool)
    segs = [r for r in _runs(line) if r > 0]
    return bool(len(segs) >= 3 and (np.mean(segs) < 16 if segs else False))


def _geometry(ink: np.ndarray, bbox: list) -> tuple[str, float]:
    """rect / rounded_rect / oval using convex-hull rectangularity + corner radius.

    Thin-bordered rounded rectangles often have very low absolute corner ink,
    which confuses threshold-only classifiers.  We first separate ellipses from
    rect-like shapes via the convex-hull-area / bbox-area ratio, then use the
    estimated corner radius to decide rect vs rounded_rect.

    Returns (geometry_type, confidence).
    """
    import cv2
    H, W = ink.shape
    if min(H, W) < 8:
        return "rect", 0.5

    # Convert boolean mask to uint8 contour input
    mask = (ink.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "rect", 0.5
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return "rect", 0.55

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    bbox_area = max(1.0, float(W * H))
    rectangularity = hull_area / bbox_area

    # A perfect ellipse has rectangularity ≈ π/4 ≈ 0.785; a rectangle ≈ 1.0.
    # Use a conservative threshold to avoid misclassifying rounded rects as ovals.
    if rectangularity < 0.82:
        return "oval", 0.85

    # Estimate corner radius.  Significant radius -> rounded_rect.
    rx = _estimate_corner_radius(ink, bbox)
    hw, hh = (bbox[2] - bbox[0]) / 2.0, (bbox[3] - bbox[1]) / 2.0
    max_rx = min(hw, hh) * 0.45
    if rx >= 4.0 and rx <= max_rx:
        return "rounded_rect", 0.75

    return "rect", 0.85


def _estimate_corner_radius(ink: np.ndarray, bbox: list) -> float:
    """Estimate the corner radius (rx) of a rounded rectangle in output px.

    Scans diagonally from each corner until ink is hit.  Returns the median
    radius across the four corners, clamped to a sensible range.
    """
    H, W = ink.shape
    if min(H, W) < 16:
        return 10.0
    qx, qy = max(2, W // 4), max(2, H // 4)
    radii = []
    for (corner_x, corner_y, dx, dy) in [
        (0, 0, 1, 1),
        (W - 1, 0, -1, 1),
        (0, H - 1, 1, -1),
        (W - 1, H - 1, -1, -1),
    ]:
        dists = []
        for step in range(1, min(qx, qy)):
            x = corner_x + dx * step
            y = corner_y + dy * step
            if 0 <= x < W and 0 <= y < H and ink[y, x]:
                dists.append(step)
                break
        if dists:
            # distance along diagonal to border start ≈ rx * (sqrt(2)-1)
            radii.append(dists[0] / 0.414)
    if not radii:
        return 10.0
    rx = float(np.median(radii))
    # clamp: not too small, not more than half the smaller dimension
    return max(3.0, min(rx, min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0))


def _skeleton_arrow(crop, box):
    """Two extreme ink points = arrow endpoints; color = median ink."""
    import cv2
    arr = np.asarray(crop.convert("RGB"))
    H, W, _ = arr.shape
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    ink = (np.abs(arr.astype(int) - bg).sum(2) > 60).astype(np.uint8)
    ys, xs = np.where(ink)
    if len(xs) < 5:
        return None, "#808080", 2
    # farthest pair of ink pixels (endpoint candidates)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    i, j = np.unravel_index(np.argmax(d), d.shape)
    a = pts[i] + [box[0], box[1]]
    b = pts[j] + [box[0], box[1]]
    color = _hex(arr[ys, xs].reshape(-1, 3))
    th = V.measure_thickness(_as_image(arr), [a[0], a[1], b[0], b[1]])
    return [float(a[0]), float(a[1]), float(b[0]), float(b[1])], color, max(2, int(th))


def _as_image(arr):
    return Image.fromarray(arr.astype(np.uint8))


_SNAP_TYPES = ("shape", "container", "rect", "rounded_rect", "oval",
               "diamond", "hexagon", "parallelogram")


def _nearest_perimeter_point(px: float, py: float, shape: dict) -> tuple[float, float]:
    """Closest point on the shape's visible perimeter to (px,py).

    Rect/rounded_rect → nearest point on the bbox boundary.
    Oval → nearest point on the ellipse boundary.
    Everything else → bbox boundary (safe fallback).
    """
    x0, y0, x1, y1 = shape["bbox"]
    t = shape.get("type", "rect")
    if t == "oval":
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
        dx, dy = px - cx, py - cy
        if dx == 0 and dy == 0:
            return cx + rx, cy
        # parametric angle for nearest boundary point
        ang = np.arctan2(dy / ry, dx / rx)
        return cx + rx * np.cos(ang), cy + ry * np.sin(ang)

    # rectangle / rounded_rect / diamond / hexagon / fallback
    # nearest point on the rectangle boundary
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # normalize to center-relative half-size
    hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    dx, dy = px - cx, py - cy
    if dx == 0 and dy == 0:
        # arbitrary right edge
        return x1, cy
    # scale so the point lies on the boundary of the unit square
    adx, ady = abs(dx) / max(1.0, hw), abs(dy) / max(1.0, hh)
    if adx > ady:
        sx = hw if dx >= 0 else -hw
        sy = dy * (hw / max(abs(dx), 1e-6))
    else:
        sy = hh if dy >= 0 else -hh
        sx = dx * (hh / max(abs(dy), 1e-6))
    return cx + sx, cy + sy


def _snap(points, elements, radius=80):
    """Match each arrow endpoint to the nearest shape/container *perimeter*.

    Returns (from_id, to_id, from_point, to_point).  Ranking is by distance to
    the target shape's visible perimeter, not to its center, so large shapes
    whose edge happens to be close win over small shapes whose center is close.

    Race-safe: handlers run in parallel, so an arrow may see a shape element
    before OR after _shape overwrites its type with the geometry string. Match
    both the semantic ('shape'/'container') and geometry ('rect'/…) forms."""
    if not points:
        return None, None, None, None
    targets = [o for o in elements
               if o.get("type") in _SNAP_TYPES and "bbox" in o]
    ids, pts = [], []
    for px, py in [(points[0], points[1]), (points[2], points[3])]:
        best_oid, best_pt, best_d = None, None, radius
        for o in targets:
            pt = _nearest_perimeter_point(px, py, o)
            dd = ((px - pt[0]) ** 2 + (py - pt[1]) ** 2) ** 0.5
            if dd < best_d:
                best_d = dd
                best_oid = o["id"]
                best_pt = pt
        ids.append(best_oid)
        pts.append(best_pt)
    return ids[0], ids[1], pts[0], pts[1]


def _local_excludes(e, elements, box):
    """Crop-local boxes of annotations (text/formula/chart) inside this region,
    so they don't pollute the surface's dot/wave extraction."""
    x0, y0 = box[0], box[1]
    out = []
    for o in elements:
        if o is e or o.get("type") not in ("text", "formula", "chart", "icon") \
                or "bbox" not in o:
            continue
        bx0, by0, bx1, by1 = o["bbox"]
        lb = [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]
        if lb[2] > lb[0] and lb[3] > lb[1]:
            out.append(lb)
    return out


# -- output helpers -------------------------------------------------------

def _strip(raw: str) -> str:
    s = re.sub(r"^```.*?$|```", "", (raw or "").strip(), flags=re.M).strip()
    return s.strip().strip('"').strip("'").strip()


_REFUSAL = re.compile(
    r"(?i)(\b(i cannot|i can'?t|i am unable|i'?m unable|i'?m sorry|as an ai|"
    r"cannot fulfill|unable to (read|see|fulfill|extract|provide)|"
    r"the image (is|appears) (completely )?(blank|white|empty)|"
    r"no text (detected|found|visible|readable|to transcribe)|"
    r"no readable text|no visible text|no math expression|did not find|"
    r"find any text|discernible math|discernible text|"
    r"the quick brown fox|lorem ipsum)\b|"
    r"\(no text\)|\(no readable text[^)]*\)|\(no visible text[^)]*\)|"
    r"\(no text is visible in the image[^)]*\)|"
    r"\(the image contains only (graphical|abstract|visual) elements[^)]*\)|"
    r"\(the image appears to be blank or entirely white[^)]*\)|"
    r"\(no discernible (characters|words|text)[^)]*\))")


def _is_refusal(txt: str) -> bool:
    """A VLM non-answer ('I cannot…', 'blank image', the pangram placeholder)."""
    return bool(txt) and len(txt) > 3 and bool(_REFUSAL.search(txt))


def _is_degenerate(txt: str) -> bool:
    """Garbage output like '000000...' or '........' from a failed OCR crop."""
    if not txt or len(txt) < 5:
        return False
    # Long run of the same character.
    if len(set(txt.strip())) == 1 and len(txt) > 8:
        return True
    # Mostly a single repeated character.
    from collections import Counter
    c = Counter(txt)
    if c.most_common(1)[0][1] / len(txt) > 0.85 and len(txt) > 15:
        return True
    # Mostly non-alphanumeric (punctuation/spaces) for long strings.
    alpha_num = sum(1 for ch in txt if ch.isalnum())
    if len(txt) > 30 and alpha_num / len(txt) < 0.25:
        return True
    return False


def _parse_json(raw: str):
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
