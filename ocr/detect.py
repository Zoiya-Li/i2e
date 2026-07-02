"""Text detection (precise boxes) + fusion into the IR.

`get_text_detector("auto")` returns whichever PP-OCR engine is installed —
RapidOCR (onnxruntime, light) preferred, PaddleOCR as fallback. Both are
OPTIONAL deps; install only the one you want.

`refine_text_with_ocr` fuses OCR boxes into VLM-extracted text elements:
match each VLM text element to an OCR line by CONTENT similarity (the VLM
content is accurate), then SNAP the box to OCR's tight box. Content stays the
VLM's — OCR contributes geometry only.

`get_tesseract_lines` is a zero-install fallback that uses the system
`tesseract` binary (already present on macOS/Homebrew) to return text lines
with tight bounding boxes. It is used by the v2 diagram pipeline to give the
VLM detection pass precise geometry and actual text content.
"""

from __future__ import annotations

import csv
import difflib
import io
import os
import shutil
import subprocess


def _bbox_from_points(points) -> dict:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


class RapidOCRDetector:
    name = "rapid"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # optional dep
        self._engine = RapidOCR()

    def detect(self, image_path: str) -> list[dict]:
        result, _ = self._engine(image_path)
        out = []
        for box, text, score in (result or []):
            out.append({"content": text, "bbox": _bbox_from_points(box), "confidence": float(score)})
        return out


class PaddleOCRDetector:
    name = "paddle"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # optional dep
        self._engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    def detect(self, image_path: str) -> list[dict]:
        res = self._engine.ocr(image_path, cls=True)
        out = []
        for page in (res or []):
            for box, (text, score) in (page or []):
                out.append({"content": text, "bbox": _bbox_from_points(box), "confidence": float(score)})
        return out


def get_text_detector(name: str = "auto"):
    if name in ("auto", "rapid"):
        try:
            return RapidOCRDetector()
        except Exception:
            if name == "rapid":
                raise
    if name in ("auto", "paddle"):
        try:
            return PaddleOCRDetector()
        except Exception:
            if name == "paddle":
                raise
    raise RuntimeError("no OCR engine available — `pip install rapidocr-onnxruntime` (or paddleocr)")


def _norm(s: str | None) -> str:
    return "".join((s or "").split())


def refine_text_with_ocr(elements: list[dict], ocr_lines: list[dict], sim_threshold: float = 0.5) -> list[dict]:
    """Snap each VLM text element's bbox to the OCR line whose recognized text
    best matches the element's (accurate VLM) content. Geometry from OCR,
    content kept from the VLM. Each OCR line is consumed at most once."""
    used: set[int] = set()
    for el in elements:
        if el.get("type") != "text":
            continue
        target = _norm((el.get("text") or {}).get("content"))
        best_i, best_score = None, 0.0
        for i, line in enumerate(ocr_lines):
            if i in used:
                continue
            ot = _norm(line.get("content"))
            score = 1.0 if (target and ot and target == ot) else difflib.SequenceMatcher(None, target, ot).ratio()
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= sim_threshold:
            used.add(best_i)
            el["bbox"] = dict(ocr_lines[best_i]["bbox"])  # OCR geometry; VLM content unchanged
            ext = el.get("extraction")
            if isinstance(ext, dict):
                ext["method"] = (ext.get("method", "") + "+ocr").lstrip("+")
    return elements


class TesseractDetector:
    name = "tesseract"

    def __init__(self, psm: int = 11, oem: int = 3,
                 lang: str = "eng", conf_threshold: float = 30.0) -> None:
        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract binary not found in PATH")
        self.psm = psm
        self.oem = oem
        self.lang = lang
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str) -> list[dict]:
        return get_tesseract_lines(image_path, self.psm, self.oem,
                                   self.lang, self.conf_threshold)


def get_tesseract_lines(image_path: str, psm: int = 11, oem: int = 3,
                        lang: str = "eng",
                        conf_threshold: float = 30.0,
                        upscale: float | None = None) -> list[dict]:
    """Run system tesseract on ``image_path`` and return word-grouped text
    lines. Each line is {"content": str, "bbox": {"x":, "y":, "w":, "h":},
    "confidence": float(mean_conf)}.

    Words with confidence below ``conf_threshold`` are dropped. The remaining
    words in the same Tesseract line are concatenated with a single space and
    the line bbox is the tight union of the word boxes.

    ``upscale`` (default from ``I2E_OCR_UPSCALE`` or 1.5) resizes the image
    before recognition, which helps Tesseract read the small labels in diagrams.
    """
    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract binary not found in PATH")

    if upscale is None:
        try:
            upscale = float(os.environ.get("I2E_OCR_UPSCALE", "1.5"))
        except Exception:
            upscale = 1.5
    upscale = max(1.0, upscale)

    if upscale > 1.0:
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        tmp_path = image_path + ".ocr_upscale.png"
        im.resize((max(1, int(w * upscale)), max(1, int(h * upscale))),
                  Image.LANCZOS).save(tmp_path)
        image_path = tmp_path
        inv_scale = 1.0 / upscale
    else:
        inv_scale = 1.0

    cmd = [
        "tesseract", image_path, "stdout",
        "-l", lang,
        "--psm", str(psm),
        "--oem", str(oem),
        "tsv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr[:500]}")

    rows = list(csv.DictReader(io.StringIO(proc.stdout), delimiter="\t"))
    def sc(v):
        return int(float(v) * inv_scale + 0.5)

    words = []
    for r in rows:
        if int(r["level"]) != 5:
            continue
        conf = float(r["conf"])
        if conf < conf_threshold:
            continue
        text = r["text"].strip()
        if not text:
            continue
        words.append({
            "text": text,
            "conf": conf,
            "x": sc(r["left"]),
            "y": sc(r["top"]),
            "w": sc(r["width"]),
            "h": sc(r["height"]),
            "block": int(r["block_num"]),
            "par": int(r["par_num"]),
            "line": int(r["line_num"]),
        })

    lines = []
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        grouped.setdefault(key, []).append(w)

    for words_in_line in grouped.values():
        words_in_line.sort(key=lambda w: w["x"])
        content = " ".join(w["text"] for w in words_in_line)
        x0 = min(w["x"] for w in words_in_line)
        y0 = min(w["y"] for w in words_in_line)
        x1 = max(w["x"] + w["w"] for w in words_in_line)
        y1 = max(w["y"] + w["h"] for w in words_in_line)
        conf_mean = sum(w["conf"] for w in words_in_line) / len(words_in_line)
        lines.append({
            "content": content,
            "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "confidence": conf_mean,
        })

    lines.sort(key=lambda ln: (ln["bbox"]["y"], ln["bbox"]["x"]))
    return lines
