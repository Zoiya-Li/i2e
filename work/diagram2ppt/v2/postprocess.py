"""Post-extraction orchestration — the full chain from raw IR to deck-ready.

run_loop() produces a faithful-but-rough IR (native shapes/text + raster
crops). Everything that turns that into a designed deck used to live in
throwaway scripts; this module is that chain made first-class so the pipeline
reproduces it end-to-end:

    experts (formula, chart)      hard content → native equations / charts
    vectorize                     every remaining crop → drawn objects
    theme_dotclouds               tint scatter thumbnails to their card color
    drop_junk_text                kill degenerate OCR fragments
    layout (slot, align, center)  repeated-card templates, rows, centering
    typography                    snap all text to a type scale

Everything is GENERAL — no element ids, no hardcoded coordinates. Where the
pipeline can't recover something the manual deck had (hand-written LaTeX,
nudged positions), it simply ships whatever the general pass produced.
"""
from __future__ import annotations

import re


def postprocess(ir: dict, original, vlm, fidelity: str = "hybrid",
                log=print) -> dict:
    """fidelity='hybrid' (default): aesthetic content stays faithful image
    layers, everything else native — looks like the original AND editable
    where it matters. fidelity='all-native': redraw everything (looks crude
    but zero pixels), the earlier 'ban screenshots' mode."""
    from . import experts
    from . import vectorize
    from .align import align_pass, center_pass, slot_pass, typography_pass

    stats = {}
    stats["formulas"] = experts.formula_pass(ir, original, vlm, log=log)
    stats["charts"] = experts.chart_pass(ir, original, vlm, log=log)
    experts.sanitize(ir, log=log)
    stats["dedup"] = dedup_overlapping(ir, log=log)
    stats["group_shapes"] = drop_hollow_group_shapes(ir, log=log)
    stats["icon_repair"] = repair_icons_by_context(ir, log=log)
    stats["vectorize"] = vectorize.vectorize_pass(ir, original, vlm, log=log)
    if fidelity == "hybrid":
        stats["faithful"] = to_hybrid_fidelity(ir, log=log)
    else:
        stats["themed"] = theme_dotclouds(ir, log=log)
    stats["junk"] = drop_junk_text(ir, log=log)
    stats["slots"] = slot_pass(ir, log=log)
    stats["aligned"] = align_pass(ir, log=log)
    stats["centered"] = center_pass(ir, log=log)
    stats["type_levels"] = typography_pass(ir, log=log)
    log(f"[postprocess fidelity={fidelity}] {stats}")
    return stats


# -- general design passes ---------------------------------------------------

def to_hybrid_fidelity(ir: dict, log=print) -> int:
    """SELECTIVE FIDELITY (the real product): aesthetically-complex content
    (surfaces, dense scatter) reads as a crude imitation when redrawn from
    native shapes — so keep it as a faithful image LAYER, while text / boxes
    / arrows / formulas / charts stay native-editable.

    Lossy re-synthesis compounds: 20 elements each ~90% right → the whole
    figure looks fake. Drawing only what redraws cleanly, and photographing
    what doesn't, gives BOTH 'looks like the original' AND 'editable where it
    matters'. Native text/shapes punched out of the layer stay on top, so the
    label over the manifold is still real editable text.
    """
    regions = []
    converted = 0
    for el in ir["elements"]:
        if el.get("type") not in ("dotcloud",) or "bbox" not in el:
            continue
        for k in ("dots", "wave_bands", "streamlines", "silhouette", "style"):
            el.pop(k, None)
        el["type"] = "raster_crop"
        el["status"] = "demoted"
        el.setdefault("ext", {})["fidelity"] = "faithful"
        regions.append((el, el["bbox"]))
        converted += 1
    if not converted:
        return 0

    # arrows / lines / hollow rings drawn ON a faithful surface are part of
    # its art — they're already in the photographed pixels, so a native copy
    # would double. Bake them in (drop the native copy). Text/formula stay
    # native: faithful_crop punches their box so the label floats on top.
    def in_region(cx, cy):
        return any(rx0 <= cx <= rx1 and ry0 <= cy <= ry1
                   for _, (rx0, ry0, rx1, ry1) in regions)

    baked = 0
    keep = []
    for e in ir["elements"]:
        t = e.get("type")
        if t in ("arrow", "line"):
            pts = e.get("points")
            if pts and in_region((pts[0] + pts[2]) / 2, (pts[1] + pts[3]) / 2):
                baked += 1
                continue
        if t == "oval" and "bbox" in e and not e.get("fill"):
            cx = (e["bbox"][0] + e["bbox"][2]) / 2
            cy = (e["bbox"][1] + e["bbox"][3]) / 2
            if in_region(cx, cy):
                baked += 1
                continue
        keep.append(e)
    ir["elements"] = keep
    log(f"[hybrid] {converted} faithful layers, {baked} art elements baked in")
    return converted


def _hex2rgb(h: str):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def theme_dotclouds(ir: dict, log=print) -> int:
    """Scatter thumbnails sitting inside a colored card read as gray mush;
    tint their dots toward the card's accent color (border, else fill). The
    card→thumbnail containment is the only signal used — fully general."""
    boxed = [e for e in ir["elements"] if "bbox" in e]
    cards = [e for e in boxed if e["type"] in ("rect", "rounded_rect")]
    n = 0
    for el in ir["elements"]:
        if el.get("type") != "dotcloud" or "bbox" not in el or not el.get("dots"):
            continue
        host = _smallest_container(el, cards)
        if host is None:
            continue
        accent = host.get("border_color") or host.get("fill")
        rgb = _accent_rgb(accent)
        if rgb is None:
            continue
        dots = sorted(el["dots"], key=lambda d: -d.get("area", 9))[:22]
        for d in dots:
            r, g, b = _hex2rgb(d["color"])
            t = 0.6
            d["color"] = "#%02x%02x%02x" % (
                int(r * (1 - t) + rgb[0] * t), int(g * (1 - t) + rgb[1] * t),
                int(b * (1 - t) + rgb[2] * t))
            d["r"] = min(max(d["r"], 1.8), 2.6)
        el["dots"] = dots
        n += 1
    if n:
        log(f"[theme] tinted {n} thumbnails to their card color")
    return n


def _accent_rgb(hex_str):
    if not hex_str or not str(hex_str).startswith("#"):
        return None
    rgb = _hex2rgb(hex_str)
    if min(rgb) > 220:          # white/near-white fill is no accent
        return None
    return rgb


def _smallest_container(el: dict, cards: list):
    ex0, ey0, ex1, ey1 = el["bbox"]
    ecx, ecy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
    best, best_area = None, float("inf")
    for c in cards:
        if c is el:
            continue
        cx0, cy0, cx1, cy1 = c["bbox"]
        if not (cx0 <= ecx <= cx1 and cy0 <= ecy <= cy1):
            continue
        area = (cx1 - cx0) * (cy1 - cy0)
        if (ex1 - ex0) * (ey1 - ey0) < area < best_area:
            best, best_area = c, area
    return best


def drop_junk_text(ir: dict, log=print) -> int:
    """Remove degenerate text the extractor litters: 1-char non-math
    fragments, and text whose box is mostly inside a native chart (its axis
    ticks/labels, which the chart redraws). Both are content-free noise."""
    charts = [e["bbox"] for e in ir["elements"]
              if e.get("type") == "chart" and "bbox" in e]
    keep, dropped = [], 0
    for e in ir["elements"]:
        t = e.get("type")
        if t == "text" and "bbox" in e:
            txt = str(e.get("text") or "").strip()
            if len(txt) <= 1 and not txt.isalpha():
                dropped += 1
                continue
            if _is_junk_ocr_fragment(txt, e["bbox"]):
                dropped += 1
                continue
            if _mostly_inside(e["bbox"], charts, 0.65):
                dropped += 1
                continue
        keep.append(e)
    if dropped:
        ir["elements"] = keep
        log(f"[junk] dropped {dropped} degenerate text fragments")
    return dropped


def _is_junk_ocr_fragment(txt: str, bbox: list) -> bool:
    """Heuristic OCR-noise filter for tiny false text boxes.

    The framework image often produces fragments like "ee", "oe", "ea" from
    manifold strokes and dot clouds.  They are not editable content; keeping
    them makes the native reconstruction worse and misroutes defects to style
    repair.
    """
    s = txt.strip()
    if not s:
        return True
    compact = s.replace(" ", "")
    lower = compact.lower()
    allowed = {
        "x1", "x2", "x3", "ci", "cl", "q0", "qo", "0°", "1.0", "0.5",
        "0.0", "0.12", "0.78", "t(x)", "s(x)", "cate", "raw", "high",
        "alert", "retain", "defer",
    }
    if lower in allowed:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    mathish = bool(re.search(r"[()=+\-*/<>βγτθ∇≈°_]", compact))
    if mathish or digits >= 2:
        return False
    # Repeated OCR crumbs from curves/dots: ee, oe, ce, ea, ose, tty, etc.
    if len(compact) <= 3 and letters >= 1:
        return True
    if len(compact) <= 5 and letters <= 3 and w > 35 and h <= 22:
        return True
    if lower in {"ee", "eee", "oe", "ce", "ea", "eet", "ose", "tty", "a,t", "@e"}:
        return True
    return False


def dedup_overlapping(ir: dict, log=print) -> int:
    """Drop overlapping near-duplicate text/formula. The extractor + OCR
    snap routinely emit the same label twice (a box's own text AND a free
    text; formula_A AND an OCR fragment of it). After experts converted some
    to formulas, the loop's earlier dedupe can't see them — so re-run here.

    Two text-bearing elements that overlap >40% of the smaller AND whose
    content matches / contains-each-other collapse to the richer one
    (formula > text; more lines > fewer; longer > shorter)."""
    import re

    def norm(s):
        return re.sub(r"\s+", "", str(s or "")).lower()

    def content(e):
        if e.get("omml_lines"):
            return norm("".join(l.get("value", "") for l in e["omml_lines"]
                               if l["kind"] == "text")) or norm(e.get("latex"))
        return norm(e.get("latex") or e.get("text"))

    def richness(e):
        # a shape that carries the label always wins over a free text — the
        # label belongs inside the box (pipeline-row 'Raw' / 'Raw Tables')
        is_container = e.get("type") in ("rect", "rounded_rect", "oval")
        return (is_container, e.get("type") == "formula",
                len(e.get("omml_lines", []) or [e]),
                len(str(e.get("latex") or e.get("text") or "")))

    texty = [e for e in ir["elements"]
             if "bbox" in e and (e.get("text") or e.get("latex")
                                 or e.get("omml_lines"))
             and e.get("type") in ("text", "formula", "rect", "rounded_rect",
                                    "oval")]
    empties = [e for e in ir["elements"]
               if "bbox" in e and e.get("type") in ("rect", "rounded_rect")
               and not (e.get("text") or "").strip()]

    def host(e):   # the empty container a free text lives in, if any
        ex0, ey0, ex1, ey1 = e["bbox"]
        cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
        for c in empties:
            if c is e:
                continue
            cx0, cy0, cx1, cy1 = c["bbox"]
            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                return id(c)
        return None

    drop = set()
    for i, a in enumerate(texty):
        if id(a) in drop:
            continue
        ca = content(a)
        ha = host(a)
        for b in texty[i + 1:]:
            # stacked label fragments ('Raw' over 'Raw Tables') barely overlap
            # but share an empty container — that co-residence is the signal
            co_resident = ha is not None and ha == host(b)
            if id(b) in drop or not (
                    _overlap_frac(a["bbox"], b["bbox"], 0.25)
                    or _center_in_either(a["bbox"], b["bbox"]) or co_resident):
                continue
            cb = content(b)
            if not ca or not cb:
                continue
            related = ca == cb or (len(ca) >= 3 and ca in cb) or \
                (len(cb) >= 3 and cb in ca)
            if not related:
                continue
            winner, loser = (b, a) if richness(a) < richness(b) else (a, b)
            # icon-box keeps the label, but adopt the fuller wording first
            # ('Raw' box + 'Raw Tables' free text → box reads 'Raw Tables')
            if winner.get("type") in ("rect", "rounded_rect", "oval") \
                    and loser.get("type") in ("text", "formula") \
                    and len(str(loser.get("text") or "")) > \
                    len(str(winner.get("text") or "")):
                winner["text"] = loser["text"]
            drop.add(id(loser))
            if id(loser) == id(a):
                break
    if drop:
        ir["elements"] = [e for e in ir["elements"] if id(e) not in drop]
        log(f"[dedup] dropped {len(drop)} overlapping duplicate texts")
    return len(drop)


def _center_in_either(a: list, b: list) -> bool:
    """Either box's center lies inside the other — catches a short fragment
    ('Raw') stacked just above the full label ('Raw Tables')."""
    for p, q in ((a, b), (b, a)):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if q[0] <= cx <= q[2] and q[1] <= cy <= q[3]:
            return True
    return False


def _overlap_frac(a: list, b: list, thresh: float) -> bool:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return False
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and inter / smaller >= thresh


def _mostly_inside(bbox: list, regions: list, frac: float) -> bool:
    x0, y0, x1, y1 = bbox
    area = max(1.0, (x1 - x0) * (y1 - y0))
    for rx0, ry0, rx1, ry1 in regions:
        ix = max(0.0, min(x1, rx1) - max(x0, rx0))
        iy = max(0.0, min(y1, ry1) - max(y0, ry0))
        if ix * iy / area >= frac:
            return True
    return False


def drop_hollow_group_shapes(ir: dict, log=print) -> int:
    """Drop large hollow shapes that are accidental groupings of smaller panels.

    VLMs sometimes emit one big rectangle around a row of panels instead of
    individual panel boxes. If a rounded_rect contains 3+ other rounded_rects
    and is mostly empty space, it is a grouping artifact and should be removed.
    """
    boxed = [e for e in ir["elements"] if "bbox" in e]
    rects = [e for e in boxed
             if e.get("type") in ("rect", "rounded_rect")]
    drop_ids = set()
    for e in rects:
        x0, y0, x1, y1 = e["bbox"]
        area = (x1 - x0) * (y1 - y0)
        inner = []
        inner_area = 0
        for o in rects:
            if o is e or id(o) in drop_ids:
                continue
            ox0, oy0, ox1, oy1 = o["bbox"]
            if ox0 >= x0 and oy0 >= y0 and ox1 <= x1 and oy1 <= y1:
                inner.append(o)
                inner_area += (ox1 - ox0) * (oy1 - oy0)
        if len(inner) >= 2 and inner_area > 0 and area > inner_area * 1.20:
            drop_ids.add(id(e))
    if drop_ids:
        ir["elements"] = [e for e in ir["elements"] if id(e) not in drop_ids]
        log(f"[postprocess] dropped {len(drop_ids)} hollow grouping shape(s)")
    return len(drop_ids)


def repair_icons_by_context(ir: dict, log=print) -> int:
    """Fix obvious icon misclassifications using the text of their parent panel.

    Small icons are easy for VLMs to misread; the panel label is a strong prior.
    """
    ICON_BY_TEXT = [
        ("raw tables", "database"),
        ("feature engineering", "gear"),
        ("cate estimator", "scatter"),
        ("ci estimator", "line"),
        ("retain", "shield"),
        ("defer", "hourglass"),
        ("alert", "warning"),
        ("reliability report", "document"),
    ]
    boxed = [e for e in ir["elements"] if "bbox" in e]
    containers = [e for e in boxed if e.get("type") in ("container", "shape",
                                                         "rect", "rounded_rect")]
    icons = [e for e in boxed if e.get("type") == "icon"]
    fixed = 0
    for ic in icons:
        ix0, iy0, ix1, iy1 = ic["bbox"]
        # find smallest container that contains this icon
        parent = None
        best_area = float("inf")
        for c in containers:
            cx0, cy0, cx1, cy1 = c["bbox"]
            if cx0 <= ix0 and cy0 <= iy0 and cx1 >= ix1 and cy1 >= iy1:
                area = (cx1 - cx0) * (cy1 - cy0)
                if area < best_area:
                    best_area = area
                    parent = c
        if not parent:
            continue
        text = (parent.get("text") or "").lower()
        kind = ic.get("icon", {}).get("kind", "")
        for trigger, correct in ICON_BY_TEXT:
            if trigger in text and kind != correct:
                ic["icon"]["kind"] = correct
                fixed += 1
                break
    if fixed:
        log(f"[postprocess] repaired {fixed} icon(s) by container text")
    return fixed


def separate_overlapping_panels(ir: dict, log=print) -> int:
    """Shrink adjacent rounded_rect panels so their borders do not merge visually.

    VLMs sometimes emit slightly overlapping panel bboxes; a thin stroke on each
    turns into a solid colored block. Nudge overlapping pairs apart horizontally
    (and vertically if needed) while keeping their centers fixed.
    """
    rects = [e for e in ir["elements"]
             if e.get("type") in ("rect", "rounded_rect") and "bbox" in e]
    changed = 0
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            ax0, ay0, ax1, ay1 = a["bbox"]
            bx0, by0, bx1, by1 = b["bbox"]
            # horizontal overlap
            if ax0 < bx1 and bx0 < ax1:
                overlap = min(ax1, bx1) - max(ax0, bx0)
                if 0 < overlap < 25:
                    shrink = overlap / 2 + 1
                    a["bbox"][0] += shrink / 2
                    a["bbox"][2] -= shrink / 2
                    b["bbox"][0] += shrink / 2
                    b["bbox"][2] -= shrink / 2
                    changed += 1
            # vertical overlap
            if ay0 < by1 and by0 < ay1:
                overlap = min(ay1, by1) - max(ay0, by0)
                if 0 < overlap < 25:
                    shrink = overlap / 2 + 1
                    a["bbox"][1] += shrink / 2
                    a["bbox"][3] -= shrink / 2
                    b["bbox"][1] += shrink / 2
                    b["bbox"][3] -= shrink / 2
                    changed += 1
    if changed:
        log(f"[postprocess] separated {changed} overlapping panel pair(s)")
    return changed
