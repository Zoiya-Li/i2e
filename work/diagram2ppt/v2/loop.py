"""Phase 3 orchestrator: analyze → render → diff → refine/identify → converge.

Each round:
  1. render the IR, score every native boxed element (1-SSIM in its bbox);
  2. elements over threshold with tries left → REFINE on a padded CROP of the
     original (crops give the VLM far better geometry than full-image asks);
     out of tries → demote to a pixel-faithful raster_crop;
  3. ink the IR doesn't cover → cluster → IDENTIFY each region on its crop →
     new elements (regions the VLM can't structure become crops directly).

Termination: a round with nothing to refine, nothing demoted, and coverage
above target — or max_rounds. After the loop every still-bad native element
is demoted, so shipped fidelity never depends on the loop having converged.
"""
from __future__ import annotations

import json

from PIL import Image

from . import diff as diff_mod
from . import ir as ir_mod
from . import parsing
from .render import render

GLOBAL_PROMPT = (
    "You are extracting the structure of a technical diagram so it can be "
    "rebuilt with native editable shapes.\n"
    "List EVERY visual element as JSON: {\"elements\": [...]}.\n"
    "Element fields: id, type, x, y, width, height (0-1 fractions of the "
    "image), text, fill (#hex or none), border_color, text_color, bold, "
    "font_size (fraction of image height).\n"
    "Types: rect, rounded_rect, oval, diamond, hexagon, parallelogram, text, "
    "arrow, line, raster.\n"
    "For arrow/line give from_id/to_id of connected shapes, or points "
    "[x0,y0,x1,y1] fractions if free-standing.\n"
    "Use type 'raster' for photographs, 3D plots, charts with axes, complex "
    "artwork — anything that is NOT a simple shape or text (give its bbox).\n"
    "Be exhaustive: include panel frames, small labels, legends, formulas "
    "(formulas are 'raster' for now). Output ONLY the JSON, COMPACT "
    "single-line format (no pretty-printing), omit fields that are none/false."
)

REFINE_PROMPT = (
    "This image is a CROP from a larger diagram. It contains one element "
    "that was extracted as:\n{current}\n"
    "Correct it. Return JSON {{\"elements\": [<one corrected element>]}} with "
    "the same schema; x/y/width/height must be 0-1 fractions OF THIS CROP. "
    "Fix bbox tightness, colors (#hex), exact text content, type. If this is "
    "actually a photograph/plot/complex artwork, set type 'raster'. "
    "Output ONLY the JSON."
)

IDENTIFY_PROMPT = (
    "This image is a CROP from a larger technical diagram. Its content was "
    "MISSED by a previous extraction pass.\n"
    "List every element you see as JSON {\"elements\": [...]} — fields: id, "
    "type, x, y, width, height (0-1 fractions OF THIS CROP), text, fill, "
    "border_color, text_color, font_size (fraction of crop height).\n"
    "Types: rect, rounded_rect, oval, diamond, hexagon, parallelogram, text, "
    "arrow, line, raster. Use 'raster' for anything that is not a simple "
    "shape or text (photos, plots, formulas, artwork). Output ONLY the JSON."
)


def run_loop(image_path: str, vlm, out_dir: str,
             max_rounds: int = 3,
             residual_threshold: float = 0.45,
             max_tries: int = 2,
             coverage_target: float = 0.97,
             identify_per_round: int = 6,
             ocr_lines: list | None = None,
             log=print) -> dict:
    """Run the full iterative extraction. Returns the final IR (also saved)."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGB")
    w, h = original.size
    ir = ir_mod.new_ir(image_path, w, h)

    # ---- Phase 0: global pass ------------------------------------------
    log(f"[global] analyzing {image_path} ({w}x{h})")
    raw = vlm.chat(GLOBAL_PROMPT, original)
    ir["elements"] = ir_mod.from_vlm_elements(parsing.parse_elements(raw), w, h)
    log(f"[global] {len(ir['elements'])} elements")

    # ---- Phase 1 specialist: OCR owns text geometry ----------------------
    if ocr_lines:
        from .ocr_snap import snap_text
        snap_text(ir, ocr_lines, log=log)

    # ---- Phase 3: iterate ----------------------------------------------
    for rnd in range(1, max_rounds + 1):
        rendered = render(ir, original)
        changed = 0

        # 2a. score + refine/demote natives
        for el in ir["elements"]:
            if el["status"] != "native" or "bbox" not in el:
                continue
            el["residual"] = _score(el, original, rendered, ir["elements"])
            if el["residual"] <= _threshold(el, residual_threshold):
                continue
            if el["tries"] >= max_tries:
                ir_mod.demote(el)
                changed += 1
                log(f"  [demote] {el['id']} residual={el['residual']}")
                continue
            el["tries"] += 1
            changed += 1
            try:
                _refine(el, original, vlm)
                log(f"  [refine] {el['id']} try={el['tries']} residual was {el['residual']}")
            except Exception as e:
                log(f"  [refine] {el['id']} FAILED ({e}) — will demote next round")
                el["tries"] = max_tries

        # 2b. coverage → identify missed content
        cov = diff_mod.coverage(original, ir)
        added = 0
        if cov["explained_frac"] < coverage_target:
            for region in cov["missing"][:identify_per_round]:
                try:
                    added += _identify(region, original, ir, vlm, rnd)
                except Exception as e:
                    log(f"  [identify] {region['bbox']} FAILED ({e}) — crop fallback")
                    _add_crop(region["bbox"], ir, rnd)
                    added += 1
        if added:
            changed += added
            log(f"  [identify] +{added} elements")

        m = ir_mod.metrics(ir)
        m.update(round=rnd, coverage=cov["explained_frac"], vlm_calls=getattr(vlm, "calls", None))
        ir["history"].append(m)
        ir_mod.save(ir, str(out / f"checkpoint.round{rnd}.ir.json"))
        log(f"[round {rnd}] {json.dumps(m)}")
        if changed == 0 and cov["explained_frac"] >= coverage_target:
            break

    # ---- fidelity gate: nothing bad ships native -------------------------
    rendered = render(ir, original)
    for el in ir["elements"]:
        if el["status"] != "native" or "bbox" not in el:
            continue
        el["residual"] = _score(el, original, rendered, ir["elements"])
        if el["residual"] > _threshold(el, residual_threshold):
            ir_mod.demote(el)
            log(f"  [final-demote] {el['id']} residual={el['residual']}")

    # hallucinated connectors have no ink under them; duplicates render twice
    diff_mod.prune_connectors(original, ir, log=log)
    from .ocr_snap import dedupe_text
    dedupe_text(ir, log=log)

    final_cov = diff_mod.coverage(original, ir)
    m = ir_mod.metrics(ir)
    m.update(round="final", coverage=final_cov["explained_frac"],
             vlm_calls=getattr(vlm, "calls", None))
    ir["history"].append(m)

    ir_mod.save(ir, str(out / "diagram.ir.json"))
    render(ir, original).save(out / "render.png")
    (out / "report.json").write_text(json.dumps(ir["history"], indent=2))
    log(f"[done] {json.dumps(m)}")
    return ir


# -- scoring ------------------------------------------------------------

TEXT_THRESHOLD = 0.62   # edge-F1 scale: font swap ~0.3-0.5, misplaced ~0.8+


def _score(el: dict, original, rendered, elements: list) -> float:
    """Type-aware residual: text by edge overlap (font-tolerant), shapes by
    SSIM on their own shell (children neutralized)."""
    if el["type"] == "text":
        return diff_mod.text_residual(original, rendered, el["bbox"])
    exclude = diff_mod.children_of(el, elements)
    return diff_mod.element_residual(original, rendered, el["bbox"],
                                     exclude=exclude)


def _threshold(el: dict, shape_threshold: float) -> float:
    return TEXT_THRESHOLD if el["type"] == "text" else shape_threshold


# -- steps --------------------------------------------------------------

def _padded_crop(original: Image.Image, bbox: list, pad: float = 0.15):
    w, h = original.size
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    box = (int(max(0, x0 - dx)), int(max(0, y0 - dy)),
           int(min(w, x1 + dx)), int(min(h, y1 + dy)))
    return original.crop(box), box


def _refine(el: dict, original: Image.Image, vlm) -> None:
    crop, box = _padded_crop(original, el["bbox"])
    cw, ch = crop.size
    current = {k: el.get(k) for k in
               ("type", "text", "fill", "border_color", "text_color", "bold")}
    raw = vlm.chat(REFINE_PROMPT.format(current=json.dumps(current, ensure_ascii=False)),
                   crop, max_edge=vlm.CROP_MAX_EDGE if hasattr(vlm, "CROP_MAX_EDGE") else 1024)
    fixed = ir_mod.from_vlm_elements(parsing.parse_elements(raw)[:1], cw, ch)
    if not fixed:
        raise ValueError("refine returned no element")
    fx = fixed[0]
    if "bbox" in fx:  # map crop-relative bbox back to image coords
        bx0, by0, bx1, by1 = fx["bbox"]
        el["bbox"] = ir_mod.clamp_bbox(
            [box[0] + bx0, box[1] + by0, box[0] + bx1, box[1] + by1],
            original.size[0], original.size[1])
    for k in ("text", "fill", "border_color", "text_color", "bold", "font_size"):
        if fx.get(k) not in (None, ""):
            el[k] = fx[k]
    if fx["type"] == "raster_crop":
        ir_mod.demote(el)
    elif fx["type"] in ir_mod.SHAPE_TYPES | {"text"}:
        el["type"] = fx["type"]


def _identify(region: dict, original: Image.Image, ir: dict, vlm, rnd: int) -> int:
    crop, box = _padded_crop(original, region["bbox"], pad=0.08)
    cw, ch = crop.size
    raw = vlm.chat(IDENTIFY_PROMPT, crop,
                   max_edge=vlm.CROP_MAX_EDGE if hasattr(vlm, "CROP_MAX_EDGE") else 1024)
    found = ir_mod.from_vlm_elements(parsing.parse_elements(raw), cw, ch,
                                     id_prefix=f"r{rnd}")
    existing = {e["id"] for e in ir["elements"]}
    n = 0
    for el in found:
        if "bbox" in el:
            x0, y0, x1, y1 = el["bbox"]
            el["bbox"] = ir_mod.clamp_bbox(
                [box[0] + x0, box[1] + y0, box[0] + x1, box[1] + y1],
                original.size[0], original.size[1])
        elif "points" not in el:
            continue  # connector with neither endpoints nor ids: useless
        while el["id"] in existing:
            el["id"] += "x"
        existing.add(el["id"])
        ir["elements"].append(el)
        n += 1
    if n == 0:  # VLM saw nothing structurable — keep the pixels anyway
        _add_crop(region["bbox"], ir, rnd)
        n = 1
    return n


def _add_crop(bbox: list, ir: dict, rnd: int) -> None:
    ids = {e["id"] for e in ir["elements"]}
    i = 1
    while f"crop-{rnd}-{i}" in ids:
        i += 1
    ir["elements"].append({
        "id": f"crop-{rnd}-{i}", "type": "raster_crop", "status": "demoted",
        "tries": 0, "residual": 0.0, "z": 0,
        "bbox": [float(v) for v in bbox], "text": "", "fill": "",
        "border_color": "", "text_color": "", "bold": False,
        "font_size": None, "ext": {"original_type": "unidentified"},
    })
