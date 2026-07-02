"""Background reconstruction: mask out every foreground element and fill the
holes, yielding a clean background plate written to the IR's background asset_ref.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw


class Inpainter(Protocol):
    name: str
    def fill(self, image: Image.Image, mask: Image.Image) -> Image.Image: ...


class OpenCVInpainter:
    """Classical Telea inpainting. Excellent for flat/gradient marketing
    backgrounds (our v1 scope); photographic fills want LaMa (later)."""

    name = "opencv"

    def __init__(self) -> None:
        import cv2  # from opencv-python (already a transitive dep)
        self._cv2 = cv2

    def fill(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import numpy as np
        cv2 = self._cv2
        arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        m = np.array(mask.convert("L"))
        res = cv2.inpaint(arr, m, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


class FlatFillInpainter:
    """Fill masked pixels with the median of the unmasked pixels. Exact on solid
    backgrounds, clean on near-flat ones (our v1 marketing scope) — avoids the
    interior tonal drift Telea shows on large flat holes. No deps beyond numpy."""

    name = "flat"

    def fill(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import numpy as np
        arr = np.array(image.convert("RGB"))
        m = np.array(mask.convert("L")) > 127
        unmasked = arr[~m]
        if unmasked.size == 0:
            return image
        med = np.median(unmasked.reshape(-1, 3), axis=0).astype(np.uint8)
        out = arr.copy()
        out[m] = med
        return Image.fromarray(out)


class LamaInpainter:
    """LaMa large-mask inpainting — for PHOTOGRAPHIC backgrounds (mint/ice/smoke)
    that flat/opencv can't rebuild. Heavy: pulls torch + a ~200MB model on first
    use. The right tool for composited posters."""

    name = "lama"

    def __init__(self) -> None:
        from simple_lama_inpainting import SimpleLama  # optional heavy dep (torch)
        self._lama = SimpleLama()  # downloads the big-lama model on first init

    def fill(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        out = self._lama(image.convert("RGB"), mask.convert("L"))
        return out if isinstance(out, Image.Image) else Image.fromarray(out)


def get_inpainter(name: str = "auto"):
    if name == "flat":
        return FlatFillInpainter()
    if name in ("auto", "opencv"):
        try:
            return OpenCVInpainter()
        except Exception:
            if name == "opencv":
                raise
    if name == "lama":
        return LamaInpainter()
    raise RuntimeError("no inpainter available — opencv-python should be installed")


def reconstruct_background(ir: dict, image_path: str, inpainter: Inpainter,
                          out_path: str, pad: int = 4, dilate: int = 6) -> str:
    """Inpaint away all non-background elements; write the clean plate and point
    the background element's asset_ref at it. Mask = union of foreground bboxes.
    The mask is DILATED — LaMa regenerates objects from any exposed edge, so the
    mask must fully cover each element + margin (lesson from the real poster)."""
    from PIL import ImageFilter
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    for el in ir["elements"]:
        if el.get("type") == "background":
            continue
        b = el["bbox"]
        md.rectangle([b["x"] - pad, b["y"] - pad, b["x"] + b["w"] + pad, b["y"] + b["h"] + pad], fill=255)
    if dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilate + 1))

    clean = inpainter.fill(img, mask)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clean.save(out_path)

    for el in ir["elements"]:
        if el.get("type") == "background":
            el.setdefault("background", {})["asset_ref"] = str(out_path)
            ext = el.get("extraction")
            if isinstance(ext, dict):
                ext["method"] = (ext.get("method", "") + "+" + inpainter.name).lstrip("+")
    return str(out_path)
