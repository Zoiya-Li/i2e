"""Deterministic layout regularization — rows that LOOK like rows.

VLM bboxes drift a few percent each; humans read the drift instantly
(the v3.3 true-render comparison: five capsules at five heights). Diagrams
are gridded — same-type elements whose centers share a band ARE a row, so:
snap tops/heights to the row median, unify near-equal widths, spread
near-even gaps evenly. Children translate with their container.
"""
from __future__ import annotations

import numpy as np

ALIGNABLE = {"rect", "rounded_rect", "oval", "diamond", "hexagon",
             "parallelogram", "raster_crop", "chart", "dotcloud", "icon"}
MAX_SHIFT_FRAC = 0.04   # never move anything more than 4% of the canvas


def align_pass(ir: dict, log=print) -> int:
    """Regularize same-type rows. Returns number of elements adjusted."""
    w, h = ir["image"]["width"], ir["image"]["height"]
    max_dx, max_dy = w * MAX_SHIFT_FRAC, h * MAX_SHIFT_FRAC
    els = [e for e in ir["elements"]
           if "bbox" in e and e["type"] in ALIGNABLE]

    moved = 0
    for row in _rows(els):
        if len(row) < 3:
            continue
        y0s = [e["bbox"][1] for e in row]
        y1s = [e["bbox"][3] for e in row]
        ty0, ty1 = float(np.median(y0s)), float(np.median(y1s))

        widths = [e["bbox"][2] - e["bbox"][0] for e in row]
        unify_w = float(np.median(widths)) if _cv(widths) < 0.25 else None

        row.sort(key=lambda e: (e["bbox"][0] + e["bbox"][2]) / 2)
        cxs = [(e["bbox"][0] + e["bbox"][2]) / 2 for e in row]
        gaps = np.diff(cxs)
        even = (len(row) >= 3 and gaps.min() > 0 and _cv(list(gaps)) < 0.25)
        targets_cx = (list(np.linspace(cxs[0], cxs[-1], len(row)))
                      if even else cxs)

        for e, tcx in zip(row, targets_cx):
            x0, y0, x1, y1 = e["bbox"]
            bw = unify_w if unify_w else (x1 - x0)
            nx0, nx1 = tcx - bw / 2, tcx + bw / 2
            dx, dy = nx0 - x0, ty0 - y0
            if abs(dx) > max_dx or abs(dy) > max_dy or abs(ty1 - y1) > max_dy:
                continue  # too aggressive — leave this one alone
            new = [nx0, ty0, nx1, ty1]
            if new != list(e["bbox"]):
                _translate_children(ir, e, dx, dy)
                e["bbox"] = [round(v, 1) for v in new]
                moved += 1
    if moved:
        log(f"[align] regularized {moved} elements")
    return moved


def _rows(els: list) -> list[list]:
    """Cluster same-type elements whose vertical centers share a band."""
    out = []
    by_type: dict[str, list] = {}
    for e in els:
        by_type.setdefault(e["type"], []).append(e)
    for group in by_type.values():
        group = sorted(group, key=lambda e: (e["bbox"][1] + e["bbox"][3]) / 2)
        row = [group[0]]
        for e in group[1:]:
            cy = (e["bbox"][1] + e["bbox"][3]) / 2
            ref = row[-1]
            ref_cy = (ref["bbox"][1] + ref["bbox"][3]) / 2
            band = 0.5 * min(e["bbox"][3] - e["bbox"][1],
                             ref["bbox"][3] - ref["bbox"][1])
            if abs(cy - ref_cy) <= band and _similar_height(e, ref):
                row.append(e)
            else:
                out.append(row)
                row = [e]
        out.append(row)
    return out


def _similar_height(a: dict, b: dict, ratio: float = 1.6) -> bool:
    ha = a["bbox"][3] - a["bbox"][1]
    hb = b["bbox"][3] - b["bbox"][1]
    return max(ha, hb) / max(1.0, min(ha, hb)) < ratio


def _cv(vals: list) -> float:
    m = float(np.mean(vals))
    return float(np.std(vals)) / m if m else 999.0


# -- typography ---------------------------------------------------------------

def typography_pass(ir: dict, merge_gap: float = 3.0, log=print) -> dict:
    """Snap every text to a small TYPE SCALE instead of per-box arithmetic.

    Per-element fit-to-box gives every label its own arbitrary size — a
    patchwork. The original figure already has a designed scale; the
    font_size values we carried from OCR/refine ARE that scale plus noise.
    1-D clustering recovers the levels; snapping enforces 'same role, same
    size' across the whole deck.
    """
    sized = []
    for e in ir["elements"]:
        if "bbox" not in e:
            continue
        is_texty = e.get("type") in ("text", "formula") or \
            (str(e.get("text") or "").strip() and
             e.get("type") in ("rect", "rounded_rect", "oval"))
        if not is_texty:
            continue
        fs = e.get("font_size")
        if not fs:
            if e.get("omml_lines"):   # formulas: per RENDERED line, not \n
                lines = max(1, len(e["omml_lines"]))
                fs = (e["bbox"][3] - e["bbox"][1]) / lines * 0.5
            else:
                lines = max(1, str(e.get("text") or "").count("\n") + 1)
                fs = (e["bbox"][3] - e["bbox"][1]) / lines * 0.7
            fs = min(fs, 34.0)   # un-sized big shapes must not invent a level
        sized.append((e, float(fs)))
    if not sized:
        return {}

    # gap-based clustering collapses a continuous spread into one mush level
    # (53 texts → 21px). A canonical ladder is what typography actually is.
    LADDER = [11.0, 13.0, 15.0, 18.0, 21.0, 26.0, 33.0]
    counts: dict = {}
    for e, fs in sized:
        fs = min(max(fs, 9.0), LADDER[-1])   # clamp: no 0.7px or 68px ships
        c = min(LADDER, key=lambda c: abs(c - fs))
        if abs(c - fs) > 0.35 * fs:   # don't drag an outlier across the scale
            c = fs
        e["font_size"] = round(c, 1)
        counts[round(c, 1)] = counts.get(round(c, 1), 0) + 1
    log("[type-scale] " + ", ".join(
        f"{k:.0f}px×{v}" for k, v in sorted(counts.items())))
    return counts


def center_pass(ir: dict, log=print) -> int:
    """Horizontally center content inside its container — the 'simple
    centering you somehow never did'. A text/formula whose center sits inside
    a card and is narrower than it gets cx := container cx (small nudges
    only; >25% offsets are layout intent, not drift)."""
    boxed = [e for e in ir["elements"] if "bbox" in e]
    containers = [e for e in boxed if e["type"] in ("rect", "rounded_rect")
                  and (e["bbox"][2] - e["bbox"][0]) > 60]
    moved = 0
    for e in boxed:
        if e.get("type") not in ("text", "formula"):
            continue
        ex0, ey0, ex1, ey1 = e["bbox"]
        ecx = (ex0 + ex1) / 2
        ecy = (ey0 + ey1) / 2
        host = None
        for c in containers:
            cx0, cy0, cx1, cy1 = c["bbox"]
            if cx0 <= ecx <= cx1 and cy0 <= ecy <= cy1 \
                    and (ex1 - ex0) < (cx1 - cx0) * 0.96:
                if host is None or (cx1 - cx0) < (host["bbox"][2] - host["bbox"][0]):
                    host = c   # smallest containing card wins
        if host is None:
            continue
        hcx = (host["bbox"][0] + host["bbox"][2]) / 2
        off = hcx - ecx
        if 0.5 < abs(off) <= 0.25 * (host["bbox"][2] - host["bbox"][0]):
            e["bbox"] = [ex0 + off, ey0, ex1 + off, ey1]
            moved += 1
    if moved:
        log(f"[center] {moved} texts centered in their cards")
    return moved


# -- slot normalization ------------------------------------------------------

_CLASS = {"text": "content", "formula": "content", "oval": "badge",
          "dotcloud": "thumb", "chart": "thumb", "icon": "icon"}


def slot_pass(ir: dict, log=print) -> int:
    """Repeated-card layouts: learn each inner slot's RELATIVE bbox from the
    healthy siblings, snap every member to it, and pull orphans (content
    whose bbox drifted clear out of its card — the floating 'Alignment
    Score') into the vacant slots.

    Generic mechanism: any row of >=3 same-type same-size containers with
    children defines the template; no element names are consulted.
    """
    els = ir["elements"]
    boxed = [e for e in els if "bbox" in e]
    containers = [e for e in boxed
                  if e["type"] in ("rect", "rounded_rect")
                  and _children_in(e, boxed)]
    fixed = 0
    for row in _container_rows(containers):
        if len(row) < 3:
            continue
        # cluster children by (class, relative-y band) across the row
        slots: dict = {}
        for c in row:
            for k in _children_in(c, boxed):
                cls = _CLASS.get(k["type"])
                if not cls:
                    continue
                rel = _rel_box(k, c)
                key = (cls, round(rel[1] * 4))   # band = quarter of card height
                slots.setdefault(key, []).append((c, k, rel))

        # phase 1: snap the best-fitting member per container to the template
        keepers: set = set()
        plans = []   # (cls, med, vacant containers)
        for (cls, _band), members in slots.items():
            if len(members) < max(2, len(row) // 2):
                continue
            med = np.median(np.array([m[2] for m in members]), axis=0)
            per_c: dict = {}
            for c, k, rel in members:
                per_c.setdefault(id(c), (c, []))[1].append((k, rel))
            seen = set()
            for c, lst in per_c.values():
                lst.sort(key=lambda kr: float(np.abs(np.array(kr[1]) - med).sum()))
                lst[0][0]["bbox"] = _abs_box(med, c)
                keepers.add(id(lst[0][0]))
                seen.add(id(c))
                fixed += 1
            plans.append((cls, med, [c for c in row if id(c) not in seen]))

        # phase 2: adopt only the unmistakable casualties — a content bbox
        # that ballooned to span multiple cards while sitting INSIDE the
        # row's own y-range (the 'Alignment Score' strip). Anything subtler
        # (nearby labels, headers) caused false adoptions in earlier
        # attempts; those stay where they are.
        row_y0 = min(c["bbox"][1] for c in row)
        row_y1 = max(c["bbox"][3] for c in row)
        card_w = float(np.median([c["bbox"][2] - c["bbox"][0] for c in row]))
        for cls, med, vacant in plans:
            if not vacant:
                continue
            strays = [e for e in els
                      if "bbox" in e and _CLASS.get(e["type"]) == cls
                      and id(e) not in keepers
                      and (e["bbox"][2] - e["bbox"][0]) >= 1.5 * card_w
                      and row_y0 <= (e["bbox"][1] + e["bbox"][3]) / 2 <= row_y1]
            # left-to-right pairing preserves sibling order
            for c, o in zip(sorted(vacant, key=lambda c: c["bbox"][0]),
                            sorted(strays, key=lambda e: e["bbox"][0])):
                o["bbox"] = _abs_box(med, c)
                keepers.add(id(o))
                fixed += 1
                log(f"  [slot] adopted {o['id']} into {c['id']}")
    if fixed:
        log(f"[slot] normalized {fixed} children")
    return fixed


def _children_in(c: dict, boxed: list) -> list:
    out = []
    area_c = ((c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]))
    for o in boxed:
        if o is c or o["type"] in ("rect", "rounded_rect"):
            continue
        oarea = (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1])
        if oarea < area_c and _center_in(o, c):
            out.append(o)
    return out


def _center_in(o: dict, c: dict) -> bool:
    cx = (o["bbox"][0] + o["bbox"][2]) / 2
    cy = (o["bbox"][1] + o["bbox"][3]) / 2
    x0, y0, x1, y1 = c["bbox"]
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _rel_box(k: dict, c: dict) -> list:
    x0, y0, x1, y1 = c["bbox"]
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return [(k["bbox"][0] - x0) / w, (k["bbox"][1] - y0) / h,
            (k["bbox"][2] - x0) / w, (k["bbox"][3] - y0) / h]


def _abs_box(rel, c: dict) -> list:
    x0, y0, x1, y1 = c["bbox"]
    w, h = x1 - x0, y1 - y0
    return [round(x0 + rel[0] * w, 1), round(y0 + rel[1] * h, 1),
            round(x0 + rel[2] * w, 1), round(y0 + rel[3] * h, 1)]


def _container_rows(containers: list) -> list[list]:
    """Same-type containers with aligned tops and similar size = one row."""
    rows: list[list] = []
    for c in sorted(containers, key=lambda e: e["bbox"][1]):
        for row in rows:
            r = row[0]
            if (c["type"] == r["type"]
                    and abs(c["bbox"][1] - r["bbox"][1]) < 8
                    and _similar_height(c, r, ratio=1.2)):
                row.append(c)
                break
        else:
            rows.append([c])
    return rows


def _translate_children(ir: dict, container: dict, dx: float, dy: float) -> None:
    """Anything whose center sits inside the container rides along."""
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return
    x0, y0, x1, y1 = container["bbox"]
    for o in ir["elements"]:
        if o is container or "bbox" not in o:
            continue
        ox0, oy0, ox1, oy1 = o["bbox"]
        cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        oarea = (ox1 - ox0) * (oy1 - oy0)
        if (x0 <= cx <= x1 and y0 <= cy <= y1
                and oarea < (x1 - x0) * (y1 - y0)):
            o["bbox"] = [ox0 + dx, oy0 + dy, ox1 + dx, oy1 + dy]
            if o.get("points"):
                p = o["points"]
                o["points"] = [p[0] + dx, p[1] + dy, p[2] + dx, p[3] + dy]
