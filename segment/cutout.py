"""Foreground cutout + asset realization.

`get_segmenter("auto")` returns rembg (light, onnxruntime) if installed.
`realize_assets` runs the segmenter on each raster element's bbox, writes the
real RGBA cutout + mask to disk, and replaces the IR's placeholder asset_ref /
mask_ref with the real paths. Marks `extraction.method += "+<segmenter>"`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol


class Segmenter(Protocol):
    name: str
    def cutout(self, image_path: str, bbox: dict) -> dict: ...


def _png(im) -> bytes:
    buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()


def _clamp_crop(image_path: str, bbox: dict):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    x0, y0 = max(0, int(bbox["x"])), max(0, int(bbox["y"]))
    x1, y1 = min(W, int(bbox["x"] + bbox["w"])), min(H, int(bbox["y"] + bbox["h"]))
    return img.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))


class RembgSegmenter:
    name = "rembg"

    def __init__(self) -> None:
        from rembg import remove  # optional dep
        self._remove = remove

    def cutout(self, image_path: str, bbox: dict) -> dict:
        crop = _clamp_crop(image_path, bbox)
        rgba = self._remove(crop).convert("RGBA")   # PIL in -> RGBA PIL out
        return {"rgba": _png(rgba), "mask": _png(rgba.getchannel("A"))}


class Sam2Segmenter:
    """Box-PROMPTABLE segmentation (MobileSAM / SAM-2 family via ultralytics).
    Cuts the SPECIFIC object inside the prompt box — unlike rembg's single
    salient subject, so it separates multiple products on one poster. Default
    mobile_sam.pt (~38MB, CPU-ok); swap "sam2_b.pt" for the true SAM-2 checkpoint.
    Heavy (torch) — isolate in its own env/process behind the seam."""

    name = "sam2"

    def __init__(self, model: str = "mobile_sam.pt") -> None:
        from ultralytics import SAM  # optional heavy dep (torch)
        self._model = SAM(model)

    def cutout(self, image_path: str, bbox: dict) -> dict:
        import numpy as np
        from PIL import Image
        b = bbox
        box = [float(b["x"]), float(b["y"]), float(b["x"] + b["w"]), float(b["y"] + b["h"])]
        r = self._model(image_path, bboxes=[box], verbose=False)
        m = r[0].masks.data[0].cpu().numpy()
        img = Image.open(image_path).convert("RGBA")
        mk = Image.fromarray((m * 255).astype("uint8")).resize(img.size)
        rgba = img.copy(); rgba.putalpha(mk)
        cut = rgba.crop((int(b["x"]), int(b["y"]), int(b["x"] + b["w"]), int(b["y"] + b["h"])))
        return {"rgba": _png(cut), "mask": _png(cut.getchannel("A"))}


def get_segmenter(name: str = "auto"):
    if name in ("auto", "rembg"):
        try:
            return RembgSegmenter()
        except Exception:
            if name == "rembg":
                raise
    if name == "sam2":
        return Sam2Segmenter()
    raise RuntimeError("no segmenter available — `pip install rembg` (or ultralytics for sam2)")


def realize_assets(ir: dict, image_path: str, out_dir: str, segmenter: Segmenter,
                   types: tuple[str, ...] = ("raster",)) -> int:
    """Produce real cutout assets for matching elements; rewrite their asset
    paths from placeholders to the files actually written. Returns count."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    for el in ir["elements"]:
        if el.get("type") not in types:
            continue
        res = segmenter.cutout(image_path, el["bbox"])
        rgba_p = str(Path(out_dir) / f"{el['id']}.png")
        mask_p = str(Path(out_dir) / f"{el['id']}_mask.png")
        Path(rgba_p).write_bytes(res["rgba"])
        Path(mask_p).write_bytes(res["mask"])
        if el["type"] == "raster":
            el.setdefault("raster", {})
            el["raster"]["asset_ref"] = rgba_p
            el["raster"]["mask_ref"] = mask_p
        ext = el.get("extraction")
        if isinstance(ext, dict):
            ext["method"] = (ext.get("method", "") + "+" + segmenter.name).lstrip("+")
        n += 1
    return n
