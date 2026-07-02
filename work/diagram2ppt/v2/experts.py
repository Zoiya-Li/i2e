"""Type specialists for the HARD content: formulas and charts.

The faithful-crop fallback guarantees fidelity but ships screenshots — and
the hard parts of a technical figure (math, plots) all land there. These
experts convert them to genuinely editable PPT objects:

  formula  text/crop with math content → VLM transcribes LaTeX → remote
           latex2mathml+mathml2omml → OMML injected as a native equation
           (double-click-editable in PowerPoint). Fallback: unicode text.
  chart    bar/line plot crop → VLM extracts series data → python-pptx
           native chart at the same bbox. The DATA is VLM-read and
           approximate — ext.approx marks it; the win is editability.

3D renders/photos stay faithful crops: PPT has no native object for them
(format ceiling, not ours).
"""
from __future__ import annotations

import json
import re

from . import parsing

MATH_CHARS = re.compile(r"[αβγδθλστφω∇≈⟨⟩‖∑√∫∂±·×÷≤≥≠]|\\frac|\^|_\{|->|→|\|\|")

FORMULA_PROMPT = (
    "Transcribe the mathematical expression in this image to LaTeX. "
    "Output ONLY the LaTeX source, no $ delimiters, no explanation, no "
    "markdown. If there are multiple lines, join with \\\\ ."
)

CHART_PROMPT = (
    "This image is a statistical plot from a technical figure. Extract its "
    "data as JSON: {\"kind\": \"bar\"|\"line\", \"x_label\": str, "
    "\"y_label\": str, \"categories\": [str, ...], \"series\": [{\"name\": "
    "str, \"color\": \"#hex\", \"values\": [numbers aligned with "
    "categories]}]}. For line plots sample 6-10 x positions as categories "
    "(numbers as strings) and read each curve's y values. Estimate values "
    "carefully from axes. Output ONLY the JSON."
)


# -- candidate selection --------------------------------------------------

def formula_candidates(ir: dict) -> list[dict]:
    """Text elements (native or demoted-from-text) whose content smells of
    math. The VLM re-transcribes from pixels, so garbled text is fine —
    it only needs to flag WHERE the math is."""
    out = []
    for el in ir["elements"]:
        if "bbox" not in el:
            continue
        if el["type"] == "text" and MATH_CHARS.search(el.get("text") or ""):
            out.append(el)
        elif (el["type"] == "raster_crop"
              and el.get("ext", {}).get("original_type") == "text"
              and MATH_CHARS.search(el.get("text") or "")):
            out.append(el)
    return out


def chart_candidates(ir: dict, kinds=("chart", "plot")) -> list[dict]:
    """Demoted raster crops whose id/labels suggest a 2D data plot.
    3D plots are excluded — PPT charts can't represent them."""
    out = []
    for el in ir["elements"]:
        if el["type"] != "raster_crop" or "bbox" not in el:
            continue
        ident = (el["id"] + " " + str(el.get("ext", {}))).lower()
        if "3d" in ident:
            continue
        if any(k in ident for k in kinds):
            out.append(el)
    return out


# -- passes ----------------------------------------------------------------

def formula_pass(ir: dict, original, vlm, candidates=None, log=print) -> int:
    """Transcribe candidates to LaTeX, convert to OMML remotely, retype
    elements as 'formula'. Elements keep working as text if anything fails."""
    from .loop import _padded_crop

    cands = candidates if candidates is not None else formula_candidates(ir)
    if not cands:
        return 0
    latex: dict[str, str] = {}
    for el in cands:
        crop, _ = _padded_crop(original, el["bbox"], pad=0.2)
        try:
            tex = vlm.chat(FORMULA_PROMPT, crop, max_edge=1024).strip()
            tex = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", tex).strip("$ \n")
            # degenerate transcription ("0" for θ≈0°): keep the text element
            if tex and not (len(tex) < 3 and len(el.get("text") or "") > len(tex)):
                latex[el["id"]] = tex
        except Exception as e:
            log(f"  [formula] {el['id']} transcribe FAILED ({e})")

    # split each transcription into lines (\\ separated); title words stay
    # as plain bold text, math segments convert to OMML — the single-line
    # join crammed multi-line capsule formulas (proven in v38)
    items, plans = {}, {}
    for el in cands:
        tex = latex.get(el["id"])
        if not tex:
            continue
        segs = [s.strip() for s in re.split(r"\\\\", tex) if s.strip()]
        plan = []
        for j, seg in enumerate(segs):
            if re.fullmatch(r"(\\text(?:bf)?\{[^}]*\}\s*,?\s*)+", seg):
                plan.append(("text",
                             re.sub(r"\\text(?:bf)?\{([^}]*)\}", r"\1 ", seg).strip()))
            else:
                key = f"{el['id']}#m{j}"
                items[key] = seg
                plan.append(("math", key))
        plans[el["id"]] = plan

    omml = _convert_remote(items, log=log) if items else {}
    n = 0
    for el in cands:
        if el["id"] not in plans:
            continue
        lines = []
        for kind, val in plans[el["id"]]:
            if kind == "text":
                lines.append({"kind": "text", "value": val})
            else:
                o = omml.get(val)
                lines.append({"kind": "omml", "value": o} if o
                             else {"kind": "text", "value": items[val]})
        el["latex"] = latex[el["id"]]
        el["omml_lines"] = lines
        el["type"] = "formula"
        el["status"] = "native"
        el["ext"]["expert"] = "formula"
        n += 1
        log(f"  [formula] {el['id']}: {latex[el['id']][:60]}")
    return n


def _convert_remote(latex: dict, log=print) -> dict:
    """LaTeX -> OMML via the A800 docker runner (local pip is blocked)."""
    import subprocess
    import tempfile

    from work import remote

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"items": latex}, f, ensure_ascii=False)
            tmp = f.name
        remote.push(tmp, f"{remote.REMOTE_ROOT}/ocr/_omml_in.json")
        raw = remote.run(
            "cd /home/lzy/AAAI_2026/i2e/ocr && "
            "PYTHONPATH=pylibs3 python3 omml_runner.py < _omml_in.json",
            timeout=300)
        start = raw.find("{")
        return json.loads(raw[start:]) if start != -1 else {}
    except (subprocess.SubprocessError, RuntimeError, ValueError) as e:
        log(f"  [formula] OMML conversion unavailable ({e}) — unicode fallback")
        return {}


def chart_pass(ir: dict, original, vlm, candidates=None, log=print) -> int:
    """Extract plot data via VLM, retype as 'chart' (built natively later)."""
    from .loop import _padded_crop

    cands = candidates if candidates is not None else chart_candidates(ir)
    n = 0
    for el in cands:
        crop, _ = _padded_crop(original, el["bbox"], pad=0.05)
        try:
            raw = vlm.chat(CHART_PROMPT, crop, max_edge=1024)
            spec = _parse_chart_spec(raw)
        except Exception as e:
            log(f"  [chart] {el['id']} FAILED ({e}) — stays a crop")
            continue
        el["chart"] = spec
        el["type"] = "chart"
        el["status"] = "native"
        el["ext"]["expert"] = "chart"
        el["ext"]["approx"] = True   # VLM-read values, not measured
        n += 1
        log(f"  [chart] {el['id']}: {spec['kind']}, "
            f"{len(spec['series'])} series x {len(spec['categories'])} cats")
    return n


def _parse_chart_spec(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start = raw.find("{")
    obj = parsing._balanced(raw, start, "{", "}") if start != -1 else None
    spec = json.loads(obj if obj else raw)
    if spec.get("kind") not in ("bar", "line"):
        raise ValueError(f"bad chart kind: {spec.get('kind')!r}")
    if not spec.get("categories") or not spec.get("series"):
        raise ValueError("chart spec missing categories/series")
    for s in spec["series"]:
        vals = [float(v) for v in s.get("values", [])]
        if len(vals) != len(spec["categories"]):
            raise ValueError("series length mismatch")
        s["values"] = vals
    return spec


CARD_ROW_PROMPT = (
    "This image is a horizontal row of card-shaped boxes from a diagram. "
    'For each card left-to-right return JSON {"cards": [{"title": str, '
    '"body": str (the smaller text, \\n between lines), "border_color": '
    '"#hex", "title_color": "#hex", "fill": "#hex or none"}]}. '
    "Output ONLY the JSON."
)


def rebuild_card_row(ir: dict, original, vlm, region: list,
                     log=print) -> int:
    """Nuclear option for a card row the extraction mangled (degenerate
    4x2-px boxes, text stranded outside): re-read the whole region with one
    VLM call, then REBUILD the cards on an even grid. Existing icons and
    thumbnails inside the region are kept and re-seated; broken boxes and
    stranded text are replaced."""
    rx0, ry0, rx1, ry1 = region
    raw = vlm.chat(CARD_ROW_PROMPT,
                   original.crop((int(rx0), int(ry0), int(rx1), int(ry1))),
                   max_edge=1024)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    cards = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["cards"]
    n = len(cards)
    if n < 2:
        return 0

    def center_in(e):
        if "bbox" not in e:
            return False
        cx = (e["bbox"][0] + e["bbox"][2]) / 2
        cy = (e["bbox"][1] + e["bbox"][3]) / 2
        return rx0 <= cx <= rx1 and ry0 <= cy <= ry1

    keep_types = ("icon", "dotcloud", "chart", "formula")
    kept = [e for e in ir["elements"] if center_in(e) and e["type"] in keep_types]
    ir["elements"] = [e for e in ir["elements"]
                      if not center_in(e) or e["type"] in keep_types
                      or e["type"] in ("arrow", "line")]

    gap = (rx1 - rx0) * 0.04
    cw = ((rx1 - rx0) - gap * (n - 1)) / n
    ids = {e["id"] for e in ir["elements"]}
    made = 0
    for i, card in enumerate(cards):
        cx0 = rx0 + i * (cw + gap)
        base = {"status": "native", "tries": 0, "residual": None,
                "fill": "", "border_color": "", "text_color": "",
                "bold": False, "font_size": None}
        bid = _fresh(f"card-{i}", ids)
        ir["elements"].append({**base, "id": bid, "type": "rounded_rect",
            "z": 10, "bbox": [cx0, ry0, cx0 + cw, ry1], "text": "",
            "fill": card.get("fill") or "",
            "border_color": card.get("border_color") or "#888888",
            "ext": {"expert": "card-row"}})
        ir["elements"].append({**base, "id": _fresh(f"card-{i}-title", ids),
            "type": "text", "z": 12, "text": card.get("title", ""),
            "bbox": [cx0 + 4, ry0 + (ry1 - ry0) * 0.30,
                     cx0 + cw - 4, ry0 + (ry1 - ry0) * 0.44],
            "text_color": card.get("title_color") or "#333333", "bold": True,
            "ext": {"expert": "card-row"}})
        if card.get("body"):
            ir["elements"].append({**base, "id": _fresh(f"card-{i}-body", ids),
                "type": "text", "z": 12, "text": card["body"],
                "bbox": [cx0 + 4, ry0 + (ry1 - ry0) * 0.48,
                         cx0 + cw - 4, ry0 + (ry1 - ry0) * 0.92],
                "text_color": "#555555", "ext": {"expert": "card-row"}})
        made += 1
    # re-seat kept icons/thumbs onto the icon band of the nearest card
    for e in kept:
        ex = (e["bbox"][0] + e["bbox"][2]) / 2
        i = min(range(n), key=lambda j: abs(rx0 + j * (cw + gap) + cw / 2 - ex))
        cx0 = rx0 + i * (cw + gap)
        w = e["bbox"][2] - e["bbox"][0]
        h = e["bbox"][3] - e["bbox"][1]
        nx0 = cx0 + (cw - w) / 2
        ny0 = ry0 + (ry1 - ry0) * 0.06
        e["bbox"] = [nx0, ny0, nx0 + w, ny0 + h]
    log(f"[card-row] rebuilt {made} cards in {[int(v) for v in region]}")
    return made


def _fresh(stem: str, ids: set) -> str:
    eid = stem
    while eid in ids:
        eid += "x"
    ids.add(eid)
    return eid


# -- hygiene ----------------------------------------------------------------

def sanitize(ir: dict, min_dim: float = 5.0, log=print) -> int:
    """Drop sliver/degenerate crops (coverage-blob artifacts: 0-width icons,
    3px strips). Their ink is negligible and they pollute the deck."""
    before = len(ir["elements"])
    def keep(e):
        if e["type"] not in ("raster_crop",) or "bbox" not in e:
            return True
        if e.get("ext", {}).get("original_type") not in (None, "unidentified", "raster"):
            return True
        x0, y0, x1, y1 = e["bbox"]
        return (x1 - x0) >= min_dim and (y1 - y0) >= min_dim
    ir["elements"] = [e for e in ir["elements"] if keep(e)]
    n = before - len(ir["elements"])
    if n:
        log(f"[sanitize] dropped {n} sliver crops")
    return n
