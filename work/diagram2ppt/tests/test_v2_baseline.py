"""Offline tests: lock the frozen v2 delivery as a regression baseline.

The committed baseline (work/diagram2ppt/v3/baselines/v2_framework.json) is the
durable, checkout-independent record. When the live diagram_final.pptx is also
present, we assert its measured structure still matches the baseline, so an
accidental change to the frozen v2 artifact is caught.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from work.diagram2ppt.v3 import pptx_stats, regression_suite

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_baseline_loads_and_is_well_formed():
    baseline = regression_suite.load_v2_baseline()
    assert baseline is not None, "v2 baseline JSON must be present and parseable"
    assert baseline["schema"] == "v2-baseline-v1"
    for key in ("artifact", "structure", "reported_metrics"):
        assert key in baseline
    structure = baseline["structure"]
    for key in ("slides", "total_shapes_recursive", "shape_histogram",
                "pictures", "omml_math_runs", "native_object_ratio"):
        assert key in structure


def test_baseline_records_measured_reality_not_doc_claim():
    # The docs claimed 0 pictures (v3.3 all-native); the delivered artifact is
    # the hybrid with raster fallbacks. The baseline must record the truth.
    structure = regression_suite.load_v2_baseline()["structure"]
    assert structure["pictures"] == 7
    assert structure["total_shapes_recursive"] == 97
    assert "discrepancy" in regression_suite.load_v2_baseline()


def test_live_pptx_matches_baseline_if_present():
    baseline = regression_suite.load_v2_baseline()
    artifact = REPO_ROOT / baseline["artifact"]["path"]
    if not artifact.exists():
        pytest.skip(f"v2 artifact not on disk: {artifact}")
    measured = pptx_stats.pptx_structure(artifact)
    expected = baseline["structure"]
    assert measured["slides"] == expected["slides"]
    assert measured["total_shapes_recursive"] == expected["total_shapes_recursive"]
    assert measured["pictures"] == expected["pictures"]
    assert measured["omml_math_runs"] == expected["omml_math_runs"]
    assert measured["shape_histogram"] == expected["shape_histogram"]
    assert measured["native_object_ratio"] == expected["native_object_ratio"]


def test_native_object_ratio_excludes_pictures():
    # A pure-native deck scores 1.0; each raster picture drags the ratio down.
    hist_native = {"AUTO_SHAPE": 3, "TEXT_BOX": 1}
    total = sum(hist_native.values())
    native = sum(v for k, v in hist_native.items() if k in pptx_stats._NATIVE_TYPES)
    assert native / total == 1.0
    assert "PICTURE" not in pptx_stats._NATIVE_TYPES


def test_compare_to_baseline_flags_editability():
    # worse-than-baseline deck (more pictures, lower native ratio) must not pass
    worse = regression_suite.compare_to_baseline({"native_object_ratio": 0.5, "pictures": 20})
    assert worse is not None
    assert worse["beats_baseline_editability"] is False
    assert worse["baseline_pictures"] == 7
    # an all-native deck beats the hybrid baseline on editability
    better = regression_suite.compare_to_baseline({"native_object_ratio": 1.0, "pictures": 0})
    assert better["beats_baseline_editability"] is True
    assert better["native_ratio_delta"] > 0
