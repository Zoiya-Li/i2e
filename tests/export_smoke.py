"""Headless test for Node ④ (lightweight export): edited IR -> shipped PNG.
Asserts the render runs, produces a valid same-size PNG, and that a text edit
actually changes the output. Offline.

    python tests/export_smoke.py
"""

from __future__ import annotations

import copy
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from extractor.assemble import assemble_ir  # noqa: E402
from extractor.providers import MockProvider  # noqa: E402
from render.export import export_png  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        img = str(Path(d) / "card.png")
        Image.new("RGB", (1080, 1350), (18, 38, 30)).save(img)

        predicted = assemble_ir(MockProvider().extract(img), image_path=img, generator="mock",
                                provider_name="mock", model_version="mock-0.1", method="mock:vlm-extract")

        # baseline render (no edit) and an edited render (changed headline copy)
        png0 = export_png(predicted, predicted, img)
        edited = copy.deepcopy(predicted)
        next(e for e in edited["elements"] if e["id"] == "text-1")["text"]["content"] = "夏日钜惠\n限时三天"
        png1 = export_png(edited, predicted, img)

        for name, data in [("baseline", png0), ("edited", png1)]:
            im = Image.open(io.BytesIO(data))
            assert im.format == "PNG" and im.size == (1080, 1350), (name, im.format, im.size)
            print(f"[④] {name} render OK -> valid PNG {im.size}")
        assert png0 != png1, "edited copy did not change the rendered output"
        print("[✓] text edit is reflected in the exported PNG")

    print("\nEXPORT SMOKE OK — edited IR -> shipped PNG (value half of the flywheel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
