"""Offline tests for the §9 fallback audit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from work.diagram2ppt.v3 import fallback

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ir(elements, w=100, h=100):
    return {"canvas": {"width_px": w, "height_px": h}, "elements": elements}


def test_is_fallback_by_type_and_editable_flag():
    assert fallback.is_fallback({"type": "raster_crop"})
    assert fallback.is_fallback({"type": "text", "editable": False})
    assert not fallback.is_fallback({"type": "text"})
    # forced-native is NOT a fallback
    assert not fallback.is_fallback({"type": "rounded_rect", "ext": {"forced": True}})


def test_undocumented_fallback_is_flagged():
    ir = _ir([{"id": "r1", "type": "raster_crop", "bbox": [0, 0, 10, 10]}])
    audit = fallback.audit_fallbacks(ir)
    assert audit["fallback_count"] == 1
    assert audit["compliant"] is False
    assert audit["violations"][0]["kind"] == "undocumented"
    assert set(audit["violations"][0]["missing"]) == {"reason", "future_replacement"}


def test_documented_fallback_is_compliant():
    ir = _ir([{
        "id": "r1", "type": "raster_crop", "bbox": [0, 0, 10, 10], "editable": False,
        "confidence": 0.4,
        "ext": {"fallback": {"reason": "dense photo", "future_replacement": "vectorize"}},
    }])
    audit = fallback.audit_fallbacks(ir)
    assert audit["compliant"] is True
    assert audit["fallback_area_ratio"] == pytest.approx(0.01, abs=1e-6)


def test_full_page_fallback_violation():
    ir = _ir([{
        "id": "big", "type": "raster_crop", "bbox": [0, 0, 100, 100], "editable": False,
        "ext": {"fallback": {"reason": "x", "future_replacement": "y"}},
    }])
    audit = fallback.audit_fallbacks(ir, full_page_threshold=0.6)
    kinds = {v["kind"] for v in audit["violations"]}
    assert "full_page" in kinds
    assert audit["fallback_area_ratio"] == 1.0


def test_real_v2_hybrid_ir_has_undocumented_raster_crops():
    ir_path = REPO_ROOT / "work/diagram2ppt/v22_out/diagram.hybrid.ir.json"
    if not ir_path.exists():
        pytest.skip("v2 hybrid IR not on disk")
    audit = fallback.audit_fallbacks(json.loads(ir_path.read_text()))
    assert audit["fallback_count"] == 7
    assert audit["compliant"] is False
    assert all(v["kind"] == "undocumented" for v in audit["violations"])
    assert audit["fallback_area_ratio"] == pytest.approx(0.2647, abs=1e-3)
