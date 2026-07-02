"""Offline tests for the SVG canonical loop (P3).

These do not require rsvg-convert: the SVG export, minimal fallback, and pixel
diff are all pure. Rasterization is only checked for graceful degradation.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from work.diagram2ppt.v3 import svg_loop


def _ir(elements, w=60, h=40):
    return {"canvas": {"width_px": w, "height_px": h}, "elements": elements}


def test_svg_safe_converts_chart_only():
    safe = svg_loop._svg_safe([
        {"id": "c", "type": "chart", "bbox": [0, 0, 10, 10]},
        {"id": "t", "type": "text", "bbox": [0, 0, 5, 5]},
    ])
    assert safe[0]["type"] == "rounded_rect"  # chart -> placeholder
    assert safe[1]["type"] == "text"          # others untouched


def test_minimal_svg_writes_rects_and_escapes_text():
    ir = _ir([{"id": "e1", "type": "rounded_rect", "bbox": [1, 1, 20, 10],
               "fill": "#fff", "text": {"content": "Hi <b>"}}])
    with TemporaryDirectory() as d:
        res = svg_loop._minimal_svg(ir, Path(d) / "m.svg")
        s = Path(res["svg"]).read_text()
        assert s.startswith("<svg") and "</svg>" in s
        assert "<rect" in s
        assert "&lt;b&gt;" in s  # escaped, not raw markup
        assert res["stats"]["minimal"] is True


def test_svg_from_v3_ir_always_produces_svg():
    ir = _ir([
        {"id": "e1", "type": "rounded_rect", "bbox": [1, 1, 20, 10]},
        {"id": "e2", "type": "text", "bbox": [2, 2, 18, 8], "text": {"content": "x"}},
    ])
    with TemporaryDirectory() as d:
        res = svg_loop.svg_from_v3_ir(ir, Image.new("RGB", (60, 40), "white"), Path(d) / "o.svg")
        assert Path(res["svg"]).exists()
        assert "<svg" in Path(res["svg"]).read_text()
        assert res["renderer"] in ("v2_export_svg", "minimal")


def test_diff_png_identical_is_zero_and_different_is_positive():
    with TemporaryDirectory() as d:
        a = Path(d) / "a.png"
        b = Path(d) / "b.png"
        Image.new("RGB", (10, 10), (120, 120, 120)).save(a)
        Image.new("RGB", (10, 10), (120, 120, 120)).save(b)
        assert svg_loop.diff_png(a, b) == 0.0
        Image.new("RGB", (10, 10), (0, 0, 0)).save(b)
        assert svg_loop.diff_png(a, b) > 0


def test_rasterize_degrades_gracefully_without_tool():
    if svg_loop.has_rasterizer():
        return  # tool present; nothing to assert about the missing-tool path
    with TemporaryDirectory() as d:
        assert svg_loop.rasterize_svg(Path(d) / "x.svg", Path(d) / "x.png") is None
