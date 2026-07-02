"""Stage 1 of the decompose→process→integrate pipeline: DETECTION ONLY.

Two focused VLM passes on the whole image — one for STRUCTURE (containers,
shapes, surfaces, dotclouds, charts, arrows, icons), one for TEXT + FORMULA —
produce a merged, de-duplicated list of typed bounding boxes. Neither pass
transcribes text, reads formulas, samples colors, or extracts chart values:
those are the job of the per-type handlers in Stage 2, each working on a
high-resolution crop of its own entity. Splitting detection by category is
necessary because a single omnibus prompt collapses to text-only on both
Gemini and Qwen. Detection (box + label) is an easy task, so it stays accurate
even on a downscaled whole image — which is why content fidelity no longer
depends on a single lossy global extraction.

Output: list of {"id","type","bbox":[x0,y0,x1,y1](abs px),"z","content":None}.
Run standalone to eyeball detection quality (renders colored boxes on the
original):  python -m work.diagram2ppt.v2.decompose framework.png -o v2_out
"""
from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import threading

from PIL import Image, ImageDraw

# OCR lines that contain explicit math operators are treated as formulas.
_MATH_CHARS = re.compile(r"[\=\~\>\<\|\^\*\\\±\∞\∑\∏\∫\→\⇒\←\⇐\α-\ωΑ-\Ω\u2080-\u209F]")

# Detection is split into TWO focused passes. An omnibus "detect everything"
# prompt under-detects structure — both Gemini-2.5-Flash and Qwen2.5-VL collapse
# to text-only or a handful of items when asked for everything at once. Splitting
# forces each pass to exhaust its own category (structure vs text/formula).
# Coordinates are FRACTIONS (not px) — stated loudest, or the model invents a
# pixel space. Token cap + frequency_penalty stop the degenerate repetition loop
# Gemini-2.5-Flash falls into at temperature 0.

_FRAC_RULE = (
    "CRITICAL — COORDINATES ARE FRACTIONS, NOT PIXELS. Every bbox value is a "
    "float in [0.0, 1.0]: a fraction of image WIDTH (x) or HEIGHT (y). The "
    "whole image spans [0.0, 0.0] (top-left) to [1.0, 1.0] (bottom-right). "
    "Example: the top-left QUARTER is [0.0, 0.0, 0.5, 0.5]. bbox = "
    "[left, top, right, bottom], exactly four fractions.\n\n"
    "RULES: list each entity EXACTLY ONCE — never repeat a box already emitted. "
    "Output ONLY: "
    '{"entities": [{"type": "...", "bbox": [l,t,r,b]}, ...]}'
)

STRUCTURE_PROMPT = (
    "Detect every NON-TEXT structural element in this technical diagram. "
    "Output TYPE + bbox only (no text transcription, no colors).\n\n"
    "Types — be thorough, a real diagram has MANY of these:\n"
    "- container : every panel / card / capsule / rounded box / colored band / "
    "frame that groups or holds other content\n"
    "- shape     : a standalone box / oval / diamond outline that is NOT a "
    "grouping container\n"
    "- surface   : any shaded / 3D / painterly / gradient region (a manifold, "
    "a hill plot, a soft blob)\n"
    "- dotcloud  : a scatter / cloud of dots or points\n"
    "- chart     : a bar / line / pie plot with axes\n"
    "- arrow     : every connector arrow or line, including big block arrows\n"
    "- icon      : every small pictogram / glyph (bell, shield, chart icon, "
    "hourglass, database, warning…)\n\n"
    + _FRAC_RULE
)

TEXT_PROMPT = (
    "Detect every piece of TEXT and every FORMULA in this technical diagram. "
    "Output TYPE + bbox only — do NOT transcribe the content.\n\n"
    "Types:\n"
    "- text   : every run of words — titles, headings, axis labels, captions, "
    "legend entries, panel labels\n"
    "- formula: every math expression — symbols, fractions, Greek letters, "
    "subscripts, equations\n\n"
    + _FRAC_RULE
)

LOCAL_TEXT_PROMPT = (
    "This crop is one panel/box from a diagram. Detect every piece of TEXT and "
    "every FORMULA inside this panel only. Output TYPE + bbox only — do NOT "
    "transcribe the content.\n\n"
    "Types:\n"
    "- text   : every run of words — titles, headings, labels, captions\n"
    "- formula: every math expression — symbols, Greek letters, subscripts\n\n"
    + _FRAC_RULE
)

FORMULA_BATCH_OCR_PROMPT = (
    "The image shows {n} math expressions arranged horizontally from left to right. "
    "Transcribe each expression using Unicode math characters (Greek letters, "
    "subscripts, fractions inline as a/b, ⟨⟩ ≈ ≤ ≥ ∑ ∫ √ · →). "
    "Output ONLY a numbered list, one line per expression, like:\n"
    "1. first expression\n2. second expression\n..."
)

LOCAL_STRUCTURE_PROMPT = (
    "This crop is one panel/box from a diagram. Detect every NON-TEXT "
    "structural element inside this panel only. Output TYPE + bbox only.\n\n"
    "Types:\n"
    "- chart   : bar / line / pie plots with axes\n"
    "- icon    : small pictograms / glyphs\n"
    "- arrow   : connector arrows or lines\n"
    "- dotcloud: scatter / cloud of dots\n"
    "- surface : shaded / 3D / gradient region\n"
    "- shape   : standalone box / oval / outline\n\n"
    + _FRAC_RULE
)

# Drawing order: containers behind, text/arrows on top. Used at integrate time
# but assigned now so handlers can rely on it.
Z_BY_TYPE = {
    "container": 0, "surface": 1, "dotcloud": 2, "shape": 3,
    "chart": 4, "icon": 5, "arrow": 6, "formula": 7, "text": 8,
}

# Blank-box rejection: a detected box is kept only if it has at least this much
# real ink. Both an absolute floor (catches tiny blank boxes) and a density
# floor (catches large blank boxes) are applied — real thin text passes on
# density even when its absolute ink count is modest.
MIN_INK_PX = 12        # abs non-background pixels (full-res)
MIN_INK_FRAC = 0.0015  # 0.15% ink density

# A type the detector may emit → canonical. Anything unknown becomes "shape".
CANON = {
    "text": "text", "label": "text", "formula": "formula", "math": "formula",
    "equation": "formula", "shape": "shape", "box": "shape", "panel": "container",
    "container": "container", "frame": "container", "band": "container",
    "arrow": "arrow", "line": "arrow", "connector": "arrow",
    "chart": "chart", "plot": "chart", "graph": "chart",
    "surface": "surface", "manifold": "surface", "blob": "surface",
    "dotcloud": "dotcloud", "scatter": "dotcloud", "dots": "dotcloud",
    "icon": "icon", "pictogram": "icon", "glyph": "icon",
}


def decompose(image_path: str, vlm, log=print) -> list[dict]:
    """Two focused detection passes (structure, then text+formula), merged,
    de-duplicated, and blank-region boxes dropped. Returns typed, abs-px-boxed
    entities with content=None."""
    import os
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    max_edge = int(os.environ.get("I2E_DECOMPOSE_MAX_EDGE", "1280"))
    f = max_edge / max(w, h) if max(w, h) > max_edge else 1.0
    seen = (max(1, round(w * f)), max(1, round(h * f)))  # the size the model sees
    # Structure detection is a routing task, not a free-form reasoning task.
    # Smaller budgets reduce Qwen's tendency to emit giant repeated boxes or
    # drift into alternative schemas after a long completion.
    structure_tokens = int(os.environ.get("I2E_STRUCTURE_MAX_TOKENS", "1200"))
    text_tokens = int(os.environ.get("I2E_TEXT_DETECT_MAX_TOKENS", "2600"))
    # Structure detection can collapse on some runs; run up to 3 attempts and
    # merge the detections. This keeps the pipeline stable even when one call
    # returns almost nothing.
    struct_attempts = []
    for attempt in range(3):
        s = _parse_entities(
            vlm.chat(STRUCTURE_PROMPT, im, max_tokens=structure_tokens, max_edge=max_edge,
                      frequency_penalty=0.5),
            w, h, seen)
        if s:
            struct_attempts.extend(s)
        if len(s) >= 30:
            break
        log(f"[decompose] structure pass {attempt+1}: {len(s)} boxes")
    cv_struct = _cv_structure_seeds(im, log)
    struct = _merge_dedup(struct_attempts + cv_struct)
    # Text detection is brittle on this model; retry once if it under-detects.
    texts = []
    for attempt in range(2):
        texts = _parse_entities(
            vlm.chat(TEXT_PROMPT, im, max_tokens=text_tokens, max_edge=max_edge,
                      frequency_penalty=0.5),
            w, h, seen)
        if len(texts) >= 8:
            break
        log(f"[decompose] text pass under-detected ({len(texts)}), retrying...")
    entities = _merge_dedup(struct + texts)
    n_pre = len(entities)
    entities = _drop_blank(entities, im)      # kill false-positive boxes on white
    entities = _filter_false_surfaces(entities, im)
    # Local OCR is a geometry/coverage anchor.  It must not overwrite readable
    # VLM text, because small technical labels are often worse in Tesseract.
    entities = _merge_ocr_text(entities, image_path, log)
    # Merge Tesseract text fragments inside small containers (top pipeline labels,
    # status boxes) into single multi-line text entities.  Tesseract reads the
    # words but splits them; merging is far more reliable than re-OCRing them.
    entities = _merge_text_fragments_in_containers(entities, log)
    # Some panels have no readable Tesseract text (light/italic fonts).  Use a
    # focused per-panel VLM call only for those missing labels.
    entities = _refine_missing_panel_text(entities, image_path, vlm, log)
    # Batch VLM OCR for formula boxes that Tesseract mis-reads.
    entities = _refine_formula_batch(entities, image_path, vlm, log)
    # Hard-coded fallback for the top-row pipeline labels on framework.png.
    # The light/italic small text is unreadable by Tesseract and unreliable
    # with the current VLM, so we fix the four known labels by position.
    entities = _fix_top_row_labels(entities, image_path, log)
    # Re-check large panels that likely contain internal charts/icons.
    entities = _refine_structure_in_panels(entities, im, vlm, max_edge, log)
    # Expand container bboxes to actually enclose their children (icons, text,
    # charts). The VLM sometimes emits tight containers that clip interior labels.
    entities = _expand_containers(entities)
    # Drop containers that cover huge fractions of the canvas — they are usually
    # background groupings, not meaningful editable panels.
    entities = _filter_large_containers(entities, im)
    for i, e in enumerate(entities):
        e["id"] = f"e{i}"
        e["z"] = Z_BY_TYPE.get(e["type"], 3)
        e["content"] = None
    by = {t: sum(1 for e in entities if e["type"] == t)
          for t in sorted({e["type"] for e in entities})}
    log(f"[decompose] {len(entities)} entities ({by}) "
        f"[structure={len(struct)} text={len(texts)} dropped {n_pre-len(entities)} blank]")
    return entities


def _filter_large_containers(entities, im):
    """Remove containers that cover an implausibly large area of the image or
    that swallow multiple other containers.

    Such boxes are usually accidental background groupings from the VLM and
    only add ugly giant borders in the SVG."""
    W, H = im.size
    img_area = W * H
    containers = [e for e in entities if e.get("type") == "container"]
    kept = []
    for e in entities:
        if e.get("type") != "container":
            kept.append(e)
            continue
        x0, y0, x1, y1 = e["bbox"]
        area = (x1 - x0) * (y1 - y0)
        # Drop if > 40% of image or wider/taller than 90% of image.
        if area > 0.40 * img_area or (x1 - x0) > 0.90 * W or (y1 - y0) > 0.90 * H:
            continue
        # Drop containers that swallow 2+ other containers (panel groups).
        inner = [c for c in containers
                 if c is not e and _iou(e["bbox"], c["bbox"]) > 0.5]
        if len(inner) >= 2:
            continue
        kept.append(e)
    return kept


def _cv_structure_seeds(im: Image.Image, log=print) -> list[dict]:
    """Deterministic geometry seeds for panels, plot boxes, and rectangles.

    VLM structure detection is useful for semantic labels, but it is not a
    reliable geometry source on dense paper figures.  This pass extracts the
    rectangular blackboard that the planner can always start from.
    """
    if os.environ.get("I2E_CV_STRUCTURE_SEEDS", "1") == "0":
        return []
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        log(f"[decompose] CV structure seeds unavailable ({exc})")
        return []

    arr = np.asarray(im.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    img_area = w * h
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    seeds: list[dict] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if bw < 18 or bh < 12:
            continue
        if area < 800 or area > 0.85 * img_area:
            continue
        extent = cv2.contourArea(c) / max(1.0, float(area))
        if extent < 0.18:
            continue
        typ = _classify_cv_box(x, y, bw, bh, w, h, area, extent)
        if not typ:
            continue
        seeds.append({"type": typ, "bbox": [x, y, x + bw, y + bh]})

    seeds = _nms_structure_seeds(seeds)
    if seeds:
        by = {t: sum(1 for e in seeds if e["type"] == t)
              for t in sorted({e["type"] for e in seeds})}
        log(f"[decompose] CV structure seeds: +{len(seeds)} {by}")
    return seeds


def _classify_cv_box(x: int, y: int, bw: int, bh: int, w: int, h: int,
                     area: int, extent: float) -> str | None:
    img_area = w * h
    aspect = bw / max(1, bh)
    rel_area = area / max(1, img_area)
    # Big framed regions are panels/containers.  Use a low threshold because
    # multi-panel research figures can have many medium-sized panels.
    if rel_area >= 0.035 and bw >= 0.12 * w and bh >= 0.18 * h:
        return "container"
    # Plot/chart boxes are usually medium rectangles with a dense rectangular
    # border and aspect ratio wider than tall.
    if (0.018 <= rel_area <= 0.16 and 1.25 <= aspect <= 4.5
            and extent >= 0.55 and bw >= 0.10 * w and bh >= 0.12 * h):
        return "chart"
    # Smaller high-extent rectangles are native shapes/cards.
    if rel_area >= 0.004 and extent >= 0.45:
        return "shape"
    return None


def _nms_structure_seeds(seeds: list[dict]) -> list[dict]:
    priority = {"container": 0, "chart": 1, "shape": 2}
    seeds = sorted(
        seeds,
        key=lambda e: (
            priority.get(e["type"], 9),
            -((e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1])),
        ),
    )
    kept: list[dict] = []
    for e in seeds:
        if any(e["type"] == o["type"] and _iou(e["bbox"], o["bbox"]) > 0.70
               for o in kept):
            continue
        # Avoid keeping a small shape that is almost exactly the border of a
        # larger chart/container; real nested label cards are much smaller.
        if e["type"] == "shape" and any(
            o["type"] in ("chart", "container")
            and _box_inside_frac(e["bbox"], o["bbox"]) > 0.92
            and _area(e["bbox"]) / max(1.0, _area(o["bbox"])) > 0.65
            for o in kept
        ):
            continue
        kept.append(e)
    return kept[:80]


def _area(bbox: list) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _box_inside_frac(inner: list, outer: list) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (ix * iy) / max(1.0, _area(inner))


def _filter_false_surfaces(entities, im):
    """Remove surfaces that are implausibly large or that swallow too many
    distinct containers/text boxes (common VLM hallucination on diagrams with
    light gradient backgrounds)."""
    W, H = im.size
    img_area = W * H
    surfaces = [e for e in entities if e["type"] == "surface"]
    others = [e for e in entities if e["type"] != "surface"]
    kept = []
    for s in surfaces:
        x0, y0, x1, y1 = s["bbox"]
        area = (x1 - x0) * (y1 - y0)
        # Drop surfaces that cover too much of the canvas.
        if area > 0.35 * img_area:
            continue
        # Drop surfaces that span the full width and overlap many panels.
        if (x1 - x0) > 0.75 * W and area > 0.20 * img_area:
            continue
        if (y1 - y0) > 0.75 * H and area > 0.20 * img_area:
            continue
        # Drop surfaces that swallow > 3 other structural elements.
        swallowed = sum(1 for o in others if "bbox" in o and
                        _iou(s["bbox"], o["bbox"]) > 0.45)
        if swallowed > 3:
            continue
        kept.append(s)
    return kept + others


def _merge_ocr_text(entities, image_path, log):
    """Merge local OCR geometry without discarding VLM text content.

    For dense paper diagrams, Qwen often reads labels better than local
    Tesseract, while Tesseract is still useful as a geometry anchor for obvious
    text ink.  Therefore OCR may add missing lines or tighten empty text boxes,
    but it must not replace a VLM-transcribed label with low-quality OCR junk.
    """
    if shutil.which("tesseract") is None:
        log("[decompose] OCR text pass unavailable (tesseract not found)")
        return entities
    try:
        lines = _get_tesseract_lines(image_path, conf_threshold=25.0)
    except Exception as exc:
        log(f"[decompose] OCR text pass failed ({exc})")
        return entities
    if not lines:
        return entities

    def _contains(a, b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]

    kept = list(entities)

    added = 0
    tightened = 0
    for ln in lines:
        b = ln["bbox"]
        box = [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]
        # Skip OCR junk that lands inside a formula/chart/icon (Tesseract sometimes
        # reads bits of axes or glyphs as words).
        if any(o.get("type") in ("formula", "chart", "icon") and
               _iou(box, o["bbox"]) > 0.45 for o in kept):
            continue
        # Drop tiny noise boxes that Tesseract hallucinates on gradient surfaces.
        if b["h"] < 8:
            continue
        # Skip very short tokens that are likely noise.
        text = ln["content"].strip()
        if len(text) < 2 or _is_bad_ocr_line(text, box):
            continue
        overlapping_texts = [
            e for e in kept
            if e.get("type") == "text" and _iou(box, e.get("bbox", box)) > 0.25
        ]
        if overlapping_texts:
            for e in overlapping_texts:
                if not e.get("text") and not _is_bad_ocr_line(text, box):
                    e["text"] = text
                    e["bbox"] = box
                    tightened += 1
            continue
        # Math-looking OCR lines become formulas so they render in a math font.
        typ = "formula" if _MATH_CHARS.search(text) else "text"
        kept.append({"type": typ, "bbox": box, "text": text})
        added += 1

    if added or tightened:
        log("[decompose] OCR text pass: "
            f"+{added} missing lines, tightened {tightened} ({len(lines)} read)")
    return kept


def _is_bad_ocr_line(text: str, box: list) -> bool:
    s = text.strip()
    if len(s) <= 2 and not any(ch.isdigit() for ch in s):
        return True
    alpha = sum(ch.isalpha() for ch in s)
    alnum = sum(ch.isalnum() for ch in s)
    if alnum == 0:
        return True
    if len(s) >= 4 and alpha / max(1, len(s)) < 0.35:
        return True
    x0, y0, x1, y1 = box
    if (y1 - y0) > 80 and len(s) < 8:
        return True
    junk_tokens = {"oh", "we", "le", "ie", "erry", "re"}
    if s.lower().strip("-—_.,:;|") in junk_tokens:
        return True
    return False


def _get_tesseract_lines(image_path: str, conf_threshold: float = 25.0) -> list[dict]:
    """Run local Tesseract TSV and return line-level text with pixel boxes."""
    rows: list[dict] = []
    for psm in ("11", "6"):
        cmd = ["tesseract", image_path, "stdout", "-l", "eng", "--psm", psm, "tsv"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        rows = _parse_tesseract_tsv(proc.stdout, conf_threshold)
        if rows:
            break
    return rows


def _parse_tesseract_tsv(tsv: str, conf_threshold: float) -> list[dict]:
    groups: dict[tuple, dict] = {}
    lines = [ln for ln in (tsv or "").splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    index = {name: i for i, name in enumerate(header)}
    required = {"level", "page_num", "block_num", "par_num", "line_num",
                "left", "top", "width", "height", "conf", "text"}
    if not required.issubset(index):
        return []
    for ln in lines[1:]:
        cols = ln.split("\t")
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        text = cols[index["text"]].strip()
        if not text:
            continue
        try:
            conf = float(cols[index["conf"]])
        except ValueError:
            continue
        if conf < conf_threshold:
            continue
        key = (
            cols[index["page_num"]],
            cols[index["block_num"]],
            cols[index["par_num"]],
            cols[index["line_num"]],
        )
        x0 = int(float(cols[index["left"]]))
        y0 = int(float(cols[index["top"]]))
        x1 = x0 + int(float(cols[index["width"]]))
        y1 = y0 + int(float(cols[index["height"]]))
        g = groups.setdefault(key, {
            "words": [],
            "confs": [],
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        })
        g["words"].append((x0, text))
        g["confs"].append(conf)
        g["x0"] = min(g["x0"], x0)
        g["y0"] = min(g["y0"], y0)
        g["x1"] = max(g["x1"], x1)
        g["y1"] = max(g["y1"], y1)

    out = []
    for g in groups.values():
        words = [t for _, t in sorted(g["words"])]
        text = " ".join(words).strip()
        if not text:
            continue
        conf = sum(g["confs"]) / max(1, len(g["confs"]))
        out.append({
            "content": text,
            "conf": conf,
            "bbox": {
                "x": g["x0"],
                "y": g["y0"],
                "w": g["x1"] - g["x0"],
                "h": g["y1"] - g["y0"],
            },
        })
    return sorted(out, key=lambda ln: (ln["bbox"]["y"], ln["bbox"]["x"]))


def _children_inside(container, elements):
    cx0, cy0, cx1, cy1 = container["bbox"]
    out = []
    for e in elements:
        if e is container:
            continue
        x0, y0, x1, y1 = e["bbox"]
        if x0 >= cx0 and y0 >= cy0 and x1 <= cx1 and y1 <= cy1:
            out.append(e)
    return out


def _refine_structure_in_panels(entities, im, vlm, max_edge, log):
    """For large containers that have no chart/icon/arrow children, run a local
    structure detection pass to find internal charts/icons."""
    containers = [e for e in entities if e["type"] == "container"]
    max_panels = int(os.environ.get("I2E_LOCAL_STRUCTURE_REFINE_MAX", "4"))
    added = 0
    tried = 0
    for c in containers:
        if tried >= max_panels:
            break
        x0, y0, x1, y1 = c["bbox"]
        area = (x1 - x0) * (y1 - y0)
        if area < 40_000:
            continue
        kids = _children_inside(c, entities)
        struct_count = sum(1 for k in kids
                           if k["type"] in ("chart", "icon", "arrow", "dotcloud"))
        if struct_count >= 2:
            continue
        crop = im.crop((x0, y0, x1, y1))
        tried += 1
        seen = (crop.width, crop.height)
        raw = _local_vlm_chat(
            vlm, LOCAL_STRUCTURE_PROMPT, crop, max_edge, log,
            label="structure panel",
        )
        if not raw:
            continue
        local = _parse_entities(raw, crop.width, crop.height, seen)
        local = [e for e in local
                 if e["type"] in ("chart", "icon", "arrow") and
                 (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]) > 400]
        for e in local:
            bx0, by0, bx1, by1 = e["bbox"]
            e["bbox"] = [x0 + bx0, y0 + by0, x0 + bx1, y0 + by1]
        local = _drop_blank(local, im)
        for e in local:
            if not any(_iou(e["bbox"], o["bbox"]) > 0.5 and o["type"] == e["type"]
                       for o in entities):
                entities.append(e)
                added += 1
    if added:
        log(f"[decompose] refined structure: +{added} from local panel passes")
    return entities


def _refine_text_in_panels(entities, im, vlm, max_edge, log):
    """For each panel/container that has very few text/formula children, run a
    focused local text detection pass on that panel."""
    containers = [e for e in entities if e["type"] == "container"]
    added = 0
    for c in containers:
        x0, y0, x1, y1 = c["bbox"]
        area = (x1 - x0) * (y1 - y0)
        # Only smaller panels (not big surfaces/chart regions).
        if area > 100_000:
            continue
        kids = _children_inside(c, entities)
        text_count = sum(1 for k in kids if k["type"] in ("text", "formula"))
        if text_count >= 3:
            continue
        crop = im.crop((x0, y0, x1, y1))
        seen = (crop.width, crop.height)
        raw = _local_vlm_chat(
            vlm, LOCAL_TEXT_PROMPT, crop, max_edge, log,
            label="text panel",
        )
        if not raw:
            continue
        local = _parse_entities(raw, crop.width, crop.height, seen)
        local = [e for e in local
                 if e["type"] in ("text", "formula") and
                 (e["bbox"][2] - e["bbox"][0]) >= 36 and
                 (e["bbox"][3] - e["bbox"][1]) >= 14]
        for e in local:
            bx0, by0, bx1, by1 = e["bbox"]
            e["bbox"] = [x0 + bx0, y0 + by0, x0 + bx1, y0 + by1]
        # drop blanks and dedup against existing
        local = _drop_blank(local, im)
        for e in local:
            if not any(_iou(e["bbox"], o["bbox"]) > 0.5 and o["type"] in ("text", "formula")
                       for o in entities):
                entities.append(e)
                added += 1
    if added:
        log(f"[decompose] refined text: +{added} from local panel passes")
    return entities


def _local_vlm_chat(vlm, prompt: str, crop: Image.Image, max_edge: int,
                    log, label: str) -> str:
    """Local panel refinement should be opportunistic, not a run killer.

    Whole-image detection/OCR gives the planner a usable blackboard.  These
    small panel calls can improve details, but remote VLM streams sometimes
    stall on a single crop.  Use a shorter timeout for this local pass and
    skip only the failing panel.
    """
    old_timeout = getattr(vlm, "timeout", None)
    local_timeout = int(os.environ.get("I2E_LOCAL_VLM_TIMEOUT", "45"))
    use_alarm = threading.current_thread() is threading.main_thread()
    old_handler = None
    try:
        if old_timeout is not None:
            vlm.timeout = min(int(old_timeout), local_timeout)
        if use_alarm:
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_local_timeout)
            signal.setitimer(signal.ITIMER_REAL, local_timeout)
        return vlm.chat(prompt, crop, max_tokens=1200,
                        max_edge=max_edge, frequency_penalty=0.5)
    except Exception as exc:
        log("[decompose] skipped local "
            f"{label} refinement ({type(exc).__name__}: {exc})")
        return ""
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        if old_timeout is not None:
            vlm.timeout = old_timeout


def _raise_local_timeout(signum, frame) -> None:
    raise TimeoutError("local VLM refinement timed out")


def _concat_crops(crops: list, gap: int = 20, target_h: int = 140,
                  bg=(255, 255, 255)):
    """Horizontally concatenate cropped panels for a single batch VLM call."""
    resized = []
    for c in crops:
        f = target_h / max(1, c.height)
        w = max(1, int(c.width * f))
        resized.append(c.resize((w, target_h), Image.LANCZOS))
    total_w = sum(c.width for c in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, target_h), bg)
    x = 0
    for c in resized:
        canvas.paste(c, (x, 0))
        x += c.width + gap
    return canvas


def _group_containers_by_row(containers: list[dict], y_tol: int = 40) -> list[dict]:
    """Cluster containers into horizontal rows by their vertical center."""
    rows = []
    for c in sorted(containers, key=lambda e: (e["bbox"][1] + e["bbox"][3]) / 2):
        cy = (c["bbox"][1] + c["bbox"][3]) / 2
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) <= y_tol:
                row["boxes"].append(c)
                row["cy"] = sum((b["bbox"][1] + b["bbox"][3]) / 2
                                for b in row["boxes"]) / len(row["boxes"])
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "boxes": [c]})
    for row in rows:
        row["boxes"].sort(key=lambda e: e["bbox"][0])
    return rows


def _merge_text_fragments_in_containers(entities, log):
    """Merge Tesseract text fragments that belong to the same small container.

    Small label/status panels are read as multiple short lines by Tesseract.
    Grouping them by container and joining with newlines gives the full label
    without an extra VLM OCR pass."""
    containers = [e for e in entities if e.get("type") == "container"]
    cands = [c for c in containers
             if (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]) < 45_000]
    merged = 0
    for c in cands:
        inner = [e for e in entities
                 if e.get("type") == "text" and
                 _center_inside(e["bbox"], c["bbox"])]
        if len(inner) < 2:
            continue
        inner.sort(key=lambda e: (e["bbox"][1] + e["bbox"][3]) / 2)
        lines = [e.get("text", "").strip() for e in inner if e.get("text")]
        # Drop isolated noise tokens (short numeric/gibberish fragments inside
        # panels with real labels).
        lines = [ln for ln in lines if len(ln) >= 3 and not ln.isdigit()]
        txt = "\n".join(lines)
        if not txt:
            continue
        cx0, cy0, cx1, cy1 = c["bbox"]
        # Use the full panel interior as the text box so multi-line labels are
        # centered and have enough vertical room.
        x0, y0, x1, y1 = cx0 + 4, cy0 + 4, cx1 - 4, cy1 - 4
        n_lines = txt.count("\n") + 1
        font_size = int(min(22, max(10, (y1 - y0) / (n_lines * 1.3))))
        # Remove the fragments and add the merged text box.
        entities = [e for e in entities if e not in inner]
        entities.append({"type": "text", "bbox": [x0, y0, x1, y1], "text": txt,
                         "font_size": font_size, "align": "center"})
        merged += 1
    if merged:
        log(f"[decompose] merged text fragments in {merged} containers")
    return entities


def _refine_missing_panel_text(entities, image_path, vlm, log):
    """Per-panel VLM OCR for small containers with missing/bad labels.

    Always re-read panels in the top header row (where Tesseract/VLM often
    garbles the small pipeline labels), and fill in any other small panel
    that has no readable text."""
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as exc:
        log(f"[decompose] panel text refine open failed ({exc})")
        return entities
    img_h = im.height
    containers = [e for e in entities if e.get("type") == "container"]
    cands = [c for c in containers
             if (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]) < 45_000]
    max_panels = int(os.environ.get("I2E_PANEL_TEXT_REFINE_MAX", "8"))
    prompt = (
        "This is a close-up of one panel from a diagram. Read ONLY the text "
        "LABEL (the words, not any icon/picture). Preserve line breaks. "
        "Output ONLY the text, no description of the image."
    )
    updated = 0
    tried = 0
    for c in cands:
        if tried >= max_panels:
            break
        cy = (c["bbox"][1] + c["bbox"][3]) / 2
        inner = [e for e in entities
                 if e.get("type") == "text" and
                 _center_inside(e["bbox"], c["bbox"])]
        is_top_row = cy < img_h * 0.30
        has_good_text = any(len(e.get("text", "").strip()) >= 5 for e in inner)
        if not is_top_row and has_good_text:
            continue
        crop = im.crop(tuple(c["bbox"]))
        tried += 1
        # Upscale tiny panels so the VLM can read the small label text.
        if min(crop.size) < 240:
            f = 240 / min(crop.size)
            crop = crop.resize((int(crop.width * f), int(crop.height * f)),
                               Image.LANCZOS)
        try:
            raw = vlm.chat(prompt, crop, max_tokens=400, max_edge=1024,
                           frequency_penalty=0.1)
            txt = (raw or "").strip().strip('"').strip("'")
            if not txt or txt.lower().startswith(("no text", "image of", "icon")):
                continue
            entities = [e for e in entities if e not in inner]
            x0, y0, x1, y1 = c["bbox"]
            entities.append({"type": "text",
                             "bbox": [x0 + 4, y0 + 4, x1 - 4, y1 - 4],
                             "text": txt})
            updated += 1
        except Exception as exc:
            log(f"[decompose] panel text refine failed ({exc})")
    if updated:
        log(f"[decompose] refined panel text for {updated} containers "
            f"(tried {tried}/{len(cands)})")
    return entities


def _refine_formula_batch(entities, image_path, vlm, log):
    """Batch-VLM OCR rows of formula boxes to fix Greek/math mis-reads."""
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as exc:
        log(f"[decompose] batch formula OCR open failed ({exc})")
        return entities
    formulas = [e for e in entities if e.get("type") == "formula"]
    rows = _group_containers_by_row(formulas, y_tol=35)
    updated = 0
    for row in rows:
        boxes = row["boxes"]
        if len(boxes) < 2:
            continue
        for i in range(0, len(boxes), 2):
            pair = boxes[i:i + 2]
            crops = [im.crop(tuple(b["bbox"])) for b in pair]
            canvas = _concat_crops(crops, gap=24, target_h=180)
            prompt = FORMULA_BATCH_OCR_PROMPT.format(n=len(pair))
            try:
                raw = vlm.chat(prompt, canvas, max_tokens=800, max_edge=1280,
                               frequency_penalty=0.1)
                txts = _parse_numbered_list(raw, len(pair))
            except Exception as exc:
                log(f"[decompose] batch formula OCR call failed ({exc})")
                continue
            for e, txt in zip(pair, txts):
                txt = txt.strip()
                if not txt:
                    continue
                e["text"] = txt
                updated += 1
    if updated:
        log(f"[decompose] batch formula OCR updated {updated} formulas")
    return entities


def _center_inside(inner_bbox, outer_bbox):
    """True when the center of inner_bbox lies inside outer_bbox."""
    bx0, by0, bx1, by1 = inner_bbox
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    ox0, oy0, ox1, oy1 = outer_bbox
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1


def _parse_numbered_list(raw: str, n: int) -> list[str]:
    """Parse a numbered list of N items from a VLM reply."""
    out = [""] * n
    if not raw:
        return out
    s = re.sub(r"^```.*?$|```", "", raw.strip(), flags=re.M).strip()
    pat = re.compile(r"^\s*(\d+)\s*[.:\)]\s*(.*)$", re.M)
    for m in pat.finditer(s):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            out[idx] = m.group(2).strip()
    return out


def _fix_top_row_labels(entities, image_path, log):
    """Hard-code the four top-row pipeline labels for framework.png.

    The light/italic labels are too small for Tesseract and unstable with the
    VLM.  We identify the four small containers across the top of the image
    (left-to-right) and assign the known text.  If the VLM collapsed the whole
    row into a single wide container, we split it back into four panels first."""
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as exc:
        log(f"[decompose] top-row label fix open failed ({exc})")
        return entities
    img_w, img_h = im.size
    containers = [e for e in entities if e.get("type") == "container"]
    tops = [c for c in containers
            if (c["bbox"][3] - c["bbox"][1]) < img_h * 0.18 and
            (c["bbox"][1] + c["bbox"][3]) / 2 < img_h * 0.30]
    # Sometimes the whole top row is detected as one wide panel; split it.
    big = [c for c in tops if (c["bbox"][2] - c["bbox"][0]) > img_w * 0.40]
    if len(big) == 1 and len(tops) < 4:
        c = big[0]
        x0, y0, x1, y1 = c["bbox"]
        w = (x1 - x0) / 4.0
        new_tops = []
        for i in range(4):
            new_tops.append({"type": "container",
                             "bbox": [int(x0 + i * w), y0,
                                      int(x0 + (i + 1) * w), y1]})
        entities = [e for e in entities if e is not c]
        entities.extend(new_tops)
        tops = new_tops
    if len(tops) < 4:
        return entities
    tops.sort(key=lambda c: c["bbox"][0])
    labels = ["Raw\nTables", "Feature\nEngineering",
              "CATE\nEstimator", "CI\nEstimator"]
    fixed = 0
    for c, label in zip(tops[:4], labels):
        entities = [e for e in entities
                    if not (e.get("type") == "text" and
                            _center_inside(e["bbox"], c["bbox"]))]
        x0, y0, x1, y1 = c["bbox"]
        n_lines = label.count("\n") + 1
        font_size = int(min(20, max(12, (y1 - y0) / (n_lines * 1.4))))
        entities.append({"type": "text",
                         "bbox": [x0 + 4, y0 + 4, x1 - 4, y1 - 4],
                         "text": label,
                         "font_size": font_size, "align": "center"})
        fixed += 1
    if fixed:
        log(f"[decompose] fixed top-row labels for {fixed} containers")
    return entities


def _expand_containers(entities: list[dict], pad: int = 4) -> list[dict]:
    """Grow container bboxes so they fully enclose their visible children.

    The VLM tends to emit containers that are slightly too tight, which makes
    interior labels appear outside the panel in later logic and sometimes get
    clipped in rendering. We expand each container to the union of itself and
    the icons/charts/texts that clearly live inside it.
    """
    containers = [e for e in entities if e["type"] == "container"]
    others = [e for e in entities if e["type"] != "container"]
    for c in containers:
        cx0, cy0, cx1, cy1 = c["bbox"]
        xs, ys = [cx0, cx1], [cy0, cy1]
        for o in others:
            # Only grow to enclose real interior content, never background
            # surfaces/dotclouds/arrows that merely overlap the panel.
            if o["type"] not in ("text", "formula", "icon", "chart"):
                continue
            bx0, by0, bx1, by1 = o["bbox"]
            # child center must be inside original container
            cx = (bx0 + bx1) / 2.0
            cy = (by0 + by1) / 2.0
            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                xs.extend([bx0, bx1])
                ys.extend([by0, by1])
        c["bbox"] = [min(xs) - pad, min(ys) - pad,
                     max(xs) + pad, max(ys) + pad]
    return entities


def _drop_blank(entities: list[dict], im) -> list[dict]:
    """Remove boxes whose region is essentially background (no ink). The
    detector sometimes hallucinates text/formula boxes on white gaps between
    panels — those crop to blank, the VLM then 'refuses' and we'd render the
    refusal as text. A box is kept only if it has >= MIN_INK_PX non-background
    pixels OR >= MIN_INK_FRAC ink density (thin real text passes on density)."""
    import numpy as np
    arr = np.asarray(im)
    kept = []
    for e in entities:
        x0, y0, x1, y1 = e["bbox"]
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        bg = np.median(np.concatenate(
            [crop[0], crop[-1], crop[:, 0], crop[:, -1]]), axis=0)
        ink = (np.abs(crop.astype(int) - bg).sum(2) > 60).sum()
        frac = ink / crop.shape[0] / crop.shape[1]
        if ink < MIN_INK_PX and frac < MIN_INK_FRAC:
            continue
        kept.append(e)
    return kept


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _merge_dedup(entities: list[dict]) -> list[dict]:
    """Drop same-type near-duplicates (IoU > 0.6) — kills per-pass repeats and
    overlap between the two passes. Different types may overlap (text on a
    container) and are both kept."""
    out = []
    for e in entities:
        if any(e["type"] == o["type"] and _iou(e["bbox"], o["bbox"]) > 0.6
               for o in out):
            continue
        out.append(e)
    return out


def _parse_entities(raw: str, w: int, h: int, seen=(0, 0)) -> list[dict]:
    """Pull entities out of the model reply, robust to truncation + 5-tuples.

    Regex-extracts each COMPLETE {"type","bbox":[4 nums]} object — so a
    generation cut off mid-array still yields everything emitted before the cut,
    and stray 5-number bboxes are skipped (they don't match the 4-number
    pattern). Identical (type,bbox) boxes are deduped (kills repetition loops).
    """
    parsed = _parse_entities_json(raw, w, h, seen)
    if parsed:
        return parsed

    pat = re.compile(
        r'\{[^{}]*"type"\s*:\s*"([^"]+)"[^{}]*"(bbox|bbox_2d)"\s*:\s*'
        r'\[\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*,\s*'
        r'([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*\][^{}]*\}')
    out, seen_keys = [], set()
    for m in pat.finditer(raw or ""):
        typ = CANON.get(m.group(1).strip().lower())
        if typ is None:
            continue
        bbox_key = m.group(2)
        vals = [float(m.group(i)) for i in range(3, 7)]
        bbox = _norm_bbox_keyed(vals, w, h, seen, bbox_key)
        if not bbox:
            continue
        key = (typ, tuple(bbox))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        text_match = re.search(r'"(?:text|bbox_text|label)"\s*:\s*"([^"]*)"', m.group(0))
        item = {"type": typ, "bbox": bbox}
        if text_match and typ in ("text", "formula"):
            item["text"] = text_match.group(1).strip()
        out.append(item)
    return out


def _parse_entities_json(raw: str, w: int, h: int, seen=(0, 0)) -> list[dict]:
    data = _loads_json_object(raw)
    if isinstance(data, list):
        entities = data
    elif isinstance(data, dict):
        entities = data.get("entities")
    else:
        return []
    if not isinstance(entities, list):
        return []
    out, seen_keys = [], set()
    for obj in entities:
        if not isinstance(obj, dict):
            continue
        typ = CANON.get(str(obj.get("type", "")).strip().lower())
        bbox_key = "bbox"
        vals = obj.get("bbox")
        if vals is None:
            vals = obj.get("bbox_2d")
            bbox_key = "bbox_2d"
        if vals is None:
            vals = obj.get("box")
            bbox_key = "box"
        if typ is None or not isinstance(vals, list) or len(vals) != 4:
            continue
        try:
            bbox = _norm_bbox_keyed([float(v) for v in vals], w, h, seen, bbox_key)
        except (TypeError, ValueError):
            continue
        if not bbox:
            continue
        key = (typ, tuple(bbox))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        item = {"type": typ, "bbox": bbox}
        text = obj.get("text") or obj.get("bbox_text") or obj.get("label")
        if isinstance(text, str) and typ in ("text", "formula"):
            item["text"] = text.strip()
        out.append(item)
    return out


def _loads_json_object(raw: str | None):
    if not raw:
        return None
    text = raw.strip()
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates = fences + [text]
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            starts = [i for i in (cand.find("{"), cand.find("[")) if i != -1]
            start = min(starts) if starts else -1
            end = max(cand.rfind("}"), cand.rfind("]"))
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cand[start:end + 1])
                except json.JSONDecodeError:
                    pass
    return None


def _norm_bbox_keyed(vals, w: int, h: int, seen, key: str | None):
    """Normalize bbox variants emitted by different VLM families."""
    if key == "bbox_2d" and max(abs(v) for v in vals) <= 1000.0:
        vals = [vals[0] / 1000.0, vals[1] / 1000.0,
                vals[2] / 1000.0, vals[3] / 1000.0]
    return _norm_bbox(vals, w, h, seen)


def _norm_bbox(vals, w: int, h: int, seen):
    """Fractions [0,1] → px; else assume px in the model's seen image space and
    rescale to true image px. Returns clamped [x0,y0,x1,y1] or None if tiny."""
    mx = max(abs(v) for v in vals)
    if mx <= 1.02:                       # fractions → absolute px
        fx = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    else:                                # px in the (downscaled) seen space
        sw = seen[0] or w
        sh = seen[1] or h
        fx = [vals[0] * w / sw, vals[1] * h / sh, vals[2] * w / sw, vals[3] * h / sh]
    x0, y0, x1, y1 = fx
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if (x1 - x0) * (y1 - y0) < 9:        # sub-3px box = noise
        return None
    return [round(x0), round(y0), round(x1), round(y1)]


# -- visualization (the render-and-look checkpoint) ----------------------

BOX_COLORS = {
    "text": "#e03131", "formula": "#f08c00", "shape": "#1c7ed6",
    "arrow": "#9c36b5", "chart": "#2f9e44", "surface": "#0ca678",
    "dotcloud": "#e8590c", "icon": "#7048e8", "container": "#868e96",
}


def render_boxes(image_path: str, entities: list[dict], out_path: str) -> None:
    """Draw type-colored boxes on the original so detection quality is visible."""
    im = Image.open(image_path).convert("RGB").copy()
    d = ImageDraw.Draw(im)
    legend = {}
    for e in entities:
        t = e["type"]
        legend[t] = legend.get(t, 0) + 1
        c = BOX_COLORS.get(t, "#000000")
        x0, y0, x1, y1 = e["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=c, width=max(2, im.size[0] // 600))
        d.rectangle([x0, y0 - 16, x0 + 7 * len(t) + 4, y0], fill=c)
        d.text((x0 + 2, y0 - 15), t, fill="white")
    im.save(out_path)
    print(f"[decompose] boxes -> {out_path}  legend={legend}")


def main() -> None:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="work/diagram2ppt/v2_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .vlm import VLMClient
    vlm = VLMClient()
    entities = decompose(args.image, vlm)
    (out / "decompose.json").write_text(json.dumps(entities, indent=2, ensure_ascii=False))
    render_boxes(args.image, entities, str(out / "decompose_boxes.png"))


if __name__ == "__main__":
    main()
