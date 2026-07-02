"""Offline test for asset realization (no segmentation engine needed). A stub
segmenter stands in for rembg/SAM; verifies realize_assets writes real cutout
files and rewrites the IR's placeholder asset paths to them.

    python tests/segment_smoke.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from extractor.assemble import assemble_ir, validate_ir  # noqa: E402
from extractor.providers import MockProvider  # noqa: E402
from segment.cutout import realize_assets  # noqa: E402


class StubSegmenter:
    name = "stub"

    def cutout(self, image_path, bbox):
        rgba = Image.new("RGBA", (40, 40), (200, 30, 30, 255))   # a fake cutout
        def png(im):
            b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()
        return {"rgba": png(rgba), "mask": png(rgba.getchannel("A"))}


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        img = str(Path(d) / "card.png")
        Image.new("RGB", (1080, 1350), (18, 38, 30)).save(img)

        ir = assemble_ir(MockProvider().extract(img), image_path=img, generator="mock",
                         provider_name="mock", model_version="mock-0.1", method="mock:vlm-extract")

        # before: raster asset_ref is a placeholder path that doesn't exist
        raster = next(e for e in ir["elements"] if e["type"] == "raster")
        assert not Path(raster["raster"]["asset_ref"]).exists()

        n = realize_assets(ir, img, str(Path(d) / "assets"), StubSegmenter())
        print(f"[segment] realized {n} cutout asset(s)")
        assert n == 1, n

        raster = next(e for e in ir["elements"] if e["type"] == "raster")
        ap = Path(raster["raster"]["asset_ref"]); mp = Path(raster["raster"]["mask_ref"])
        assert ap.exists() and ap.stat().st_size > 0, ap
        assert mp.exists() and mp.stat().st_size > 0, mp
        assert raster["extraction"]["method"].endswith("+stub"), raster["extraction"]["method"]
        validate_ir(ir)  # IR still valid with real asset paths
        print(f"[✓] asset_ref now real file ({ap.name}); method={raster['extraction']['method']}; IR valid")

    print("\nSEGMENT SMOKE OK — raster element -> real RGBA cutout layer, IR rewired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
