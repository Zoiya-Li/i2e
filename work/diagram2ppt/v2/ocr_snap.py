"""OCR specialist for text geometry — the first type-routed expert.

The VLM gets text CONTENT right but estimates geometry; that mismatch was the
top demotion cause on framework.png (30/93). Doctrine (same as ocr/detect.py):
content from the VLM, geometry from OCR. Plus one v2 extension: confident OCR
lines that no element claims become NEW text elements — text ink should never
ship as a screenshot.

OCR runs remotely (local pip is SSL-blocked): RapidOCR inside docker
29e8e3afb73f on the A800 box, via work/remote.py base64 transport. See
ocr_runner.py deployed at /home/lzy/AAAI_2026/i2e/ocr/.
"""
from __future__ import annotations

import difflib
import json

from . import ir as ir_mod

REMOTE_OCR_DIR = "/home/lzy/AAAI_2026/i2e/ocr"
OCR_CMD = ("cd {d} && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=pylibs2:pylibs "
           "python3 ocr_runner.py {img}")

MATCH_SIM = 0.5        # min content similarity to claim an OCR line
ADD_CONF = 0.8         # min OCR confidence to add an unclaimed line
RASTER_OVERLAP = 0.6   # lines mostly inside raster crops stay baked there


def fetch_ocr_lines(image_path: str) -> list[dict]:
    """Push the image to the remote box, run RapidOCR, return line dicts."""
    from work import remote

    name = "ocr_input.png"
    remote.push(image_path, f"{REMOTE_OCR_DIR}/{name}")
    raw = remote.run(OCR_CMD.format(d=REMOTE_OCR_DIR, img=name), timeout=600)
    start = raw.find('{"lines"')
    if start == -1:
        raise RuntimeError(f"no OCR JSON in remote output: {raw[:300]!r}")
    return json.loads(raw[start:])["lines"]


def _norm(s: str | None) -> str:
    return "".join((s or "").split()).lower()


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def snap_text(ir: dict, lines: list[dict], log=print) -> dict:
    """Fuse OCR lines into the IR. Returns {"snapped": n, "added": n}.

    Per native text element: match its content (each \\n-separated piece
    independently) to unclaimed OCR lines; on match, bbox = union of the
    matched lines' boxes, font_size = their median height. Afterwards,
    confident unclaimed lines outside raster regions become new elements.
    """
    w, h = ir["image"]["width"], ir["image"]["height"]
    used: set[int] = set()

    texts = [e for e in ir["elements"]
             if e["type"] == "text" and e["status"] == "native"]
    snapped = 0
    for el in texts:
        pieces = [p for p in (el.get("text") or "").split("\n") if p.strip()]
        if not pieces:
            continue
        boxes = []
        for piece in pieces:
            tgt = _norm(piece)
            best_i, best_s = None, 0.0
            for i, ln in enumerate(lines):
                if i in used:
                    continue
                s = _sim(tgt, _norm(ln["text"]))
                if s > best_s:
                    best_i, best_s = i, s
            if best_i is not None and best_s >= MATCH_SIM:
                used.add(best_i)
                boxes.append(lines[best_i]["bbox"])
        if not boxes:
            continue
        el["bbox"] = ir_mod.clamp_bbox(
            [min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes)], w, h)
        heights = sorted(b[3] - b[1] for b in boxes)
        el["font_size"] = round(heights[len(heights) // 2] * 0.82, 1)
        el["ext"]["ocr"] = "snap"
        snapped += 1

    # unclaimed confident lines -> new text elements. Skip raster interiors
    # AND content another element already carries (shape labels etc.) — the
    # duplicate would render twice in the deck.
    rasters = [e["bbox"] for e in ir["elements"]
               if e["type"] in ("raster_crop",) and "bbox" in e]
    existing = {e["id"] for e in ir["elements"]}
    added = 0
    for i, ln in enumerate(lines):
        if i in used or ln.get("conf", 0) < ADD_CONF:
            continue
        if len(_norm(ln["text"])) < 2 or _inside_any(ln["bbox"], rasters):
            continue
        if _claimed_by_element(ln, ir["elements"]):
            continue
        eid = f"ocr-{i}"
        while eid in existing:
            eid += "x"
        existing.add(eid)
        bb = ir_mod.clamp_bbox(list(ln["bbox"]), w, h)
        ir["elements"].append({
            "id": eid, "type": "text", "status": "native", "tries": 0,
            "residual": None, "z": 50, "bbox": bb,
            "text": ln["text"], "fill": "", "border_color": "",
            "text_color": "", "bold": False,
            "font_size": round((bb[3] - bb[1]) * 0.82, 1),
            "ext": {"ocr": "added", "conf": ln.get("conf")},
        })
        added += 1

    log(f"[ocr] snapped {snapped}/{len(texts)} texts, added {added} lines "
        f"({len(lines)} OCR lines total)")
    return {"snapped": snapped, "added": added}


def _overlaps(a: list, b: list) -> bool:
    """Boxes overlap, or either center sits inside the other."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix > 0 and iy > 0:
        return True
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return (b[0] <= acx <= b[2] and b[1] <= acy <= b[3]) or \
           (a[0] <= bcx <= a[2] and a[1] <= bcy <= a[3])


def _claimed_by_element(line: dict, elements: list, sim: float = 0.7) -> bool:
    t = _norm(line["text"])
    for e in elements:
        if "bbox" not in e or not e.get("text"):
            continue
        if not _overlaps(line["bbox"], e["bbox"]):
            continue
        etext = _norm(e["text"])
        if _sim(t, etext) >= sim or (len(t) > 3 and t in etext):
            return True
    return False


def dedupe_text(ir: dict, sim: float = 0.75, log=print) -> int:
    """Drop standalone text elements whose content another overlapping
    element already carries (a shape's own label, or an earlier text).

    Preference order: shape-embedded text wins over standalone text;
    between two standalone texts, OCR-added loses, otherwise the later one.
    """
    els = ir["elements"]
    drop: set[int] = set()
    texts = [(i, e) for i, e in enumerate(els)
             if e["type"] == "text" and e.get("text")]
    for k, (i, t) in enumerate(texts):
        if i in drop:
            continue
        tn = _norm(t["text"])
        for j, e in enumerate(els):
            if j == i or j in drop or "bbox" not in e or not e.get("text"):
                continue
            if not _overlaps(t["bbox"], e["bbox"]):
                continue
            en = _norm(e["text"])
            if _sim(tn, en) < sim and not (len(tn) > 3 and tn in en):
                continue
            if e["type"] != "text":
                drop.add(i)            # the shape keeps its label
                break
            if j < i or t.get("ext", {}).get("ocr") == "added":
                drop.add(i)
                break
    if drop:
        ir["elements"] = [e for k, e in enumerate(els) if k not in drop]
        log(f"[dedupe] dropped {len(drop)} duplicate text elements")
    return len(drop)


def _inside_any(bbox: list, regions: list, frac: float = RASTER_OVERLAP) -> bool:
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area == 0:
        return False
    for rx0, ry0, rx1, ry1 in regions:
        ix = max(0.0, min(x1, rx1) - max(x0, rx0))
        iy = max(0.0, min(y1, ry1) - max(y0, ry0))
        if ix * iy / area >= frac:
            return True
    return False
