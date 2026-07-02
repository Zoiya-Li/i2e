"""VLM object labeling for per-object completion prompts.

Each object cutout gets a short noun-phrase label (what it is + material/color)
from the configured vision model, used to build an object-specific Stable
Diffusion prompt so amodal completion reconstructs the RIGHT thing (a cup rim,
a bottle cap, a mint leaf) instead of a generic blob.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from PIL import Image


def _load_dotenv():
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_PROMPT = (
    "This is a single cut-out object on a gray background, taken from a product poster. "
    "Reply with ONLY a short noun phrase (3 to 8 words) naming what it is, including its "
    "material and main color, written as a Stable Diffusion prompt to regenerate the whole "
    "object. No quotes, no punctuation, no extra words. Example: 'dark green glass medicine bottle with cap'."
)


def _b64(pil: Image.Image, bg=(128, 128, 128), max_edge=512) -> str:
    im = pil.convert("RGBA")
    flat = Image.new("RGB", im.size, bg)
    flat.paste(im, mask=im.getchannel("A"))
    s = max_edge / max(flat.size)
    if s < 1:
        flat = flat.resize((max(1, int(flat.width * s)), max(1, int(flat.height * s))))
    buf = io.BytesIO()
    flat.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# small/fast dedicated VL model for labeling (the configured I2E_VLM_MODEL is a
# 397B model — ~60s/call; this MoE VL model is ~1s/call and accurate enough)
LABEL_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def label_object(cutout_path: str, timeout: int = 90) -> str | None:
    """Return a short descriptive label for one object cutout, or None on failure."""
    import requests
    _load_dotenv()
    base = os.environ.get("I2E_VLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("I2E_VLM_API_KEY", "")
    model = os.environ.get("I2E_LABEL_MODEL", LABEL_MODEL)
    if not base or not model:
        return None
    payload = {
        "model": model, "max_tokens": 40, "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(Image.open(cutout_path))}},
        ]}],
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        s = requests.Session()
        s.trust_env = False  # ignore env proxies (incl. SOCKS ALL_PROXY); API host is domestic
        r = s.post(base + "/chat/completions", json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        txt = txt.strip().strip('"').strip("'").strip().rstrip(".")
        return txt[:120] if txt else None
    except Exception:
        return None


_FIND_MISSING_PROMPT = (
    "This image is what remains of a marketing poster AFTER its main objects "
    "were extracted as separate layers (extracted regions look filled-in/blurred). "
    "Identify any visually prominent UN-EXTRACTED elements still visible "
    "(decorative trim, badges, small graphics, isolated items). Reply ONLY with "
    "a JSON array, max 8 items: "
    "[{\"name\":\"short name\",\"bbox\":[x,y,w,h]}, ...]. "
    "Bbox is in pixels of THIS image. No prose, no markdown fences."
)


def find_missing(image_path: str, max_edge: int = 1024, timeout: int = 120) -> list:
    """Send the (downscaled) bg plate to a fast VL model and parse a JSON list
    of suggested un-extracted elements with bboxes (rescaled to original)."""
    import json as _json
    import re
    import requests
    _load_dotenv()
    base = os.environ.get("I2E_VLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("I2E_VLM_API_KEY", "")
    model = os.environ.get("I2E_LABEL_MODEL", LABEL_MODEL)
    if not base or not model:
        return []
    src = Image.open(image_path).convert("RGB")
    W, H = src.size
    s = min(1.0, max_edge / max(W, H))
    sent_w, sent_h = max(1, int(W * s)), max(1, int(H * s))
    flat = src.resize((sent_w, sent_h))
    buf = io.BytesIO(); flat.save(buf, "JPEG", quality=88)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {"model": model, "max_tokens": 1024, "temperature": 0.0,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": _FIND_MISSING_PROMPT},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}}]}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        sess = requests.Session(); sess.trust_env = False
        r = sess.post(base + "/chat/completions", json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return []
    # extract the first JSON array; tolerate ```json fences and prose
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = _json.loads(m.group(0))
    except Exception:
        return []
    scale = 1.0 / s
    out = []
    for it in items[:8]:
        bb = it.get("bbox")
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in bb]
        except Exception:
            continue
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        if w < 10 or h < 10 or w > W or h > H:
            continue
        out.append({"name": str(it.get("name", ""))[:60],
                    "bbox": {"x": max(0.0, x), "y": max(0.0, y),
                             "w": min(W - x, w), "h": min(H - y, h)}})
    return out


_LIST_TYPES_PROMPT = (
    "Look at this marketing poster. List the visually distinct types of elements "
    "a designer would want as separate layers — JUST short noun phrases, no bboxes, "
    "no descriptions. Each phrase 2-6 words, lowercase, include material/color where "
    "obvious. Be exhaustive (include products, logos, badges, icons, decorative trim, "
    "natural objects). Examples: 'ice cream cup', 'mint leaf', 'circular badge', "
    "'feature icon'. Reply ONLY with a JSON array of strings, no prose: "
    "[\"...\",\"...\",...]. Max 20 items, no duplicates."
)


def list_element_types(image_path: str, max_edge: int = 1280, timeout: int = 120) -> list:
    """VLM lists what KINDS of elements the poster has — names only, no bboxes
    (VLMs are reliable at naming but poor at bbox prediction)."""
    import json as _json
    import re
    import requests
    _load_dotenv()
    base = os.environ.get("I2E_VLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("I2E_VLM_API_KEY", "")
    model = os.environ.get("I2E_IDENTIFY_MODEL", LABEL_MODEL)
    if not base or not model:
        return []
    src = Image.open(image_path).convert("RGB")
    W, H = src.size
    s = min(1.0, max_edge / max(W, H))
    sw, sh = max(1, int(W * s)), max(1, int(H * s))
    buf = io.BytesIO(); src.resize((sw, sh)).save(buf, "JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {"model": model, "max_tokens": 600, "temperature": 0.0,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": _LIST_TYPES_PROMPT},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}}]}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        sess = requests.Session(); sess.trust_env = False
        r = sess.post(base + "/chat/completions", json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return []
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = _json.loads(m.group(0))
    except Exception:
        return []
    out = []
    seen = set()
    for it in items[:20]:
        s = str(it).strip().strip('"').lower()
        if 2 <= len(s) <= 60 and s not in seen:
            out.append(s); seen.add(s)
    return out


_IDENTIFY_PROMPT = (
    "You are extracting layers from a marketing poster for a design editor. "
    "List every visually DISTINCT element a designer would want as its own movable layer. "
    "Be exhaustive but each entry must be a coherent object — never split one object into pieces.\n\n"
    "For each element provide:\n"
    "- name: a short descriptive name (3-7 words, include material/color)\n"
    "- bbox: [x,y,w,h] in pixels of THIS image — the tight rectangle around the WHOLE object\n"
    "  (include occluded portions: if the cup is partly behind a scoop, the bbox covers the whole cup)\n"
    "- kind: one of \"product\" (main subject), \"logo\" (brand mark), \"graphic\" (icon/badge/trim/ornament), "
    "\"natural\" (leaf/ice/water/mist)\n\n"
    "Rules: max 25 items; bboxes must not duplicate (IoU>0.7 is a dup); no text — text is handled separately. "
    "Reply ONLY with a JSON array, no prose, no markdown fences:\n"
    "[{\"name\":\"...\",\"bbox\":[x,y,w,h],\"kind\":\"...\"},...]"
)


def identify_elements(image_path: str, max_edge: int = 1280, timeout: int = 180) -> list:
    """VLM-first object identification: returns [{name, bbox{x,y,w,h}, kind}]
    for every distinct element it sees, with bboxes rescaled to the original."""
    import json as _json
    import re
    import requests
    _load_dotenv()
    base = os.environ.get("I2E_VLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("I2E_VLM_API_KEY", "")
    model = os.environ.get("I2E_IDENTIFY_MODEL", LABEL_MODEL)
    if not base or not model:
        return []
    src = Image.open(image_path).convert("RGB")
    W, H = src.size
    s = min(1.0, max_edge / max(W, H))
    sw, sh = max(1, int(W * s)), max(1, int(H * s))
    flat = src.resize((sw, sh))
    buf = io.BytesIO(); flat.save(buf, "JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {"model": model, "max_tokens": 2400, "temperature": 0.0,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": _IDENTIFY_PROMPT},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}}]}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        sess = requests.Session(); sess.trust_env = False
        r = sess.post(base + "/chat/completions", json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"identify_elements: VLM call failed: {e}")
        return []
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = _json.loads(m.group(0))
    except Exception:
        return []
    scale = 1.0 / s
    out = []
    for it in items[:25]:
        bb = it.get("bbox")
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in bb]
        except Exception:
            continue
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        if w < 20 or h < 20:
            continue
        x = max(0.0, min(W - 1, x)); y = max(0.0, min(H - 1, y))
        w = min(W - x, w); h = min(H - y, h)
        out.append({
            "name": str(it.get("name", ""))[:80],
            "kind": (str(it.get("kind", "")).lower() if it.get("kind") else "product"),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
        })
    # dedup by bbox IoU > 0.7
    kept = []
    def _iou(a, b):
        ix = max(0, min(a["x"]+a["w"], b["x"]+b["w"]) - max(a["x"], b["x"]))
        iy = max(0, min(a["y"]+a["h"], b["y"]+b["h"]) - max(a["y"], b["y"]))
        inter = ix * iy
        u = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / u if u > 0 else 0.0
    for it in out:
        if any(_iou(it["bbox"], k["bbox"]) > 0.7 for k in kept):
            continue
        kept.append(it)
    return kept


def label_objects(ir: dict, cut_of) -> int:
    """Label every raster object in-place: sets e['name'] and ext['label'].
    `cut_of(e)` returns the element's cutout path. Returns count labeled."""
    n = 0
    for e in ir["elements"]:
        if e.get("type") != "raster":
            continue
        p = cut_of(e)
        if not p or not os.path.exists(p):
            continue
        lab = label_object(p)
        if lab:
            e["name"] = lab
            e.setdefault("ext", {})["label"] = lab
            n += 1
    return n
