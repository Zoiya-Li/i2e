"""Offline tests for the §8 multi-dimensional metrics module."""
from __future__ import annotations

from work.diagram2ppt.v3 import metrics


def _ir(elements, w=100, h=100, base=None):
    ir = {"canvas": {"width_px": w, "height_px": h}, "elements": elements}
    if base is not None:
        ir["metrics"] = base
    return ir


def test_native_element_ratio():
    ir = _ir([
        {"type": "text"}, {"type": "rounded_rect"},
        {"type": "raster_crop", "bbox": [0, 0, 10, 10]},
    ])
    assert metrics.native_element_ratio(ir) == round(2 / 3, 4)


def test_native_element_ratio_empty():
    assert metrics.native_element_ratio({"elements": []}) == 0.0


def test_editability_score_is_one_minus_fallback_area():
    # one raster covering a quarter of the canvas -> editability 0.75
    ir = _ir([{"type": "raster_crop", "bbox": [0, 0, 50, 50]}])  # 2500 / 10000
    m = metrics.ir_metrics(ir)
    assert m["fallback_area_ratio"] == 0.25
    assert m["editability_score"] == 0.75
    assert m["fallback_count"] == 1
    assert m["native_element_ratio"] == 0.0


def test_verifier_scores_passthrough():
    ir = _ir([{"type": "text"}], base={"visual_delta": 0.31, "coverage_explained": 0.97,
                                        "text_accuracy": 0.8, "ignored_key": 1})
    m = metrics.ir_metrics(ir)
    assert m["visual_delta"] == 0.31
    assert m["coverage_explained"] == 0.97
    assert m["text_accuracy"] == 0.8
    assert "ignored_key" not in m


def test_all_native_ir_scores_perfect_editability():
    ir = _ir([{"type": "text"}, {"type": "chart"}, {"type": "line"}])
    m = metrics.ir_metrics(ir)
    assert m["native_element_ratio"] == 1.0
    assert m["editability_score"] == 1.0
    assert m["fallback_compliant"] is True
