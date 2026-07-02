"""End-to-end smoke test for the MVP slice: Node ② (extract->IR) + Node ⑤
(correction capture) + the flywheel metric. Runs fully offline (mock provider).

    python tests/smoke.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make repo root importable

from bench.flywheel import measure  # noqa: E402
from capture.corrections import append_corrections, capture_diff  # noqa: E402
from extractor.assemble import assemble_ir, validate_ir  # noqa: E402
from extractor.providers import MockProvider  # noqa: E402


def _make_test_image(path: str) -> None:
    from PIL import Image
    Image.new("RGB", (1080, 1350), (20, 40, 30)).save(path)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        img = str(Path(d) / "card.png")
        _make_test_image(img)

        # Node ② : image -> IR (mock), validated inside assemble_ir
        raw = MockProvider().extract(img)
        ir = assemble_ir(raw, image_path=img, generator="mock",
                         provider_name="mock", model_version="mock-0.1", method="mock:vlm-extract")
        validate_ir(ir)
        assert ir["elements"], "no elements extracted"
        nr = [e["id"] for e in ir["elements"] if e["needs_review"]]
        print(f"[②] extracted {len(ir['elements'])} elements; needs_review={nr}")

        # Node ③ (simulated): the user edits to ship — fix headline text + font,
        # nudge the product bbox, drop the logo.
        edited = copy.deepcopy(ir)
        by_id = {e["id"]: e for e in edited["elements"]}
        by_id["text-1"]["text"]["content"] = "夏日新品\n限时开抢"      # text_content
        by_id["text-1"]["text"]["font_family"] = "阿里巴巴普惠体 Bold"  # font
        by_id["raster-1"]["bbox"]["x"] += 12                          # geometry
        edited["elements"] = [e for e in edited["elements"] if e["id"] != "logo-1"]  # element_delete

        # Node ⑤ : capture the edits as (predicted -> corrected) pairs
        corrections = capture_diff(ir, edited, session_id="sess_smoke", user_hash="u_test")
        kinds = sorted(c["kind"] for c in corrections)
        print(f"[⑤] captured {len(corrections)} corrections: {kinds}")
        assert kinds == ["element_delete", "font", "geometry", "text_content"], kinds

        # corrections must be schema-valid: append to the shipped IR and revalidate
        append_corrections(edited, corrections)
        validate_ir(edited)
        print("[✓] edited IR + corrections validate against ir-v1.schema.json")

        # flywheel metric over the shipped IR
        out = Path(d) / "card.ir.json"
        out.write_text(json.dumps(edited, ensure_ascii=False))
        m = measure(d)
        assert m["images"] == 1 and m["corrections_per_image"] == 4.0, m
        print(f"[metric] images={m['images']} corrections/image={m['corrections_per_image']} by_kind={m['by_kind']}")

    print("\nSMOKE OK — flywheel closes one turn: image -> IR -> edit -> corrections -> metric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
