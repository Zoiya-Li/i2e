"""Headless test for Node ③→⑤: the editor's save path. Exercises apply_save
(the testable core the HTTP layer wraps) without a browser. Offline.

    python tests/editor_smoke.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from editor.server import apply_save  # noqa: E402
from extractor.assemble import assemble_ir, validate_ir  # noqa: E402
from extractor.providers import MockProvider  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        img = str(Path(d) / "card.png")
        from PIL import Image
        Image.new("RGB", (1080, 1350), (18, 38, 30)).save(img)

        # Node ② prediction (the diff baseline the server holds)
        predicted = assemble_ir(MockProvider().extract(img), image_path=img, generator="mock",
                                provider_name="mock", model_version="mock-0.1", method="mock:vlm-extract")

        # what the browser sends back after the user edits: fix copy + nudge a bbox
        edited = copy.deepcopy(predicted)
        bid = {e["id"]: e for e in edited["elements"]}
        bid["text-1"]["text"]["content"] = "夏日新品\n清凉直降"   # text_content
        bid["raster-1"]["bbox"]["x"] += 18                        # geometry (a bbox nudge)

        out = str(Path(d) / "card.edited.json")
        res = apply_save(predicted, edited, out, session_id="editor-test")

        kinds = sorted(c["kind"] for c in res["corrections"])
        print(f"[③→⑤] save captured {res['count']} corrections: {kinds}")
        assert kinds == ["geometry", "text_content"], kinds

        saved = json.loads(Path(out).read_text())
        validate_ir(saved)                                   # persisted IR is valid
        assert len(saved["corrections"]) == 2, saved["corrections"]
        print("[✓] saved edited IR validates and carries its corrections")

    print("\nEDITOR SMOKE OK — browser edit → backend capture_diff → valid IR + corrections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
