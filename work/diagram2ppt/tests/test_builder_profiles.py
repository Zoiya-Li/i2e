"""Offline tests for build profiles (P4): all_native vs product_delivery."""
from __future__ import annotations

import pytest

from work.diagram2ppt.v3 import builder


def _ir(elements, w=100, h=100):
    return {"canvas": {"width_px": w, "height_px": h}, "elements": elements}


def _documented_fallback(bbox):
    return {"id": "f1", "type": "raster_crop", "bbox": bbox, "editable": False,
            "ext": {"fallback": {"reason": "dense photo", "future_replacement": "vectorize"}}}


def test_all_native_blocks_raster():
    blockers = builder.validate_buildable(
        _ir([_documented_fallback([0, 0, 10, 10])]), profile=builder.PROFILE_ALL_NATIVE)
    assert len(blockers) == 1
    assert "non-native" in blockers[0]["reason"]


def test_product_allows_documented_local_fallback():
    blockers = builder.validate_buildable(
        _ir([_documented_fallback([0, 0, 10, 10])]), profile=builder.PROFILE_PRODUCT)
    assert blockers == []


def test_product_blocks_undocumented_fallback():
    el = {"id": "f1", "type": "raster_crop", "bbox": [0, 0, 10, 10]}  # no §9 metadata
    blockers = builder.validate_buildable(_ir([el]), profile=builder.PROFILE_PRODUCT)
    assert len(blockers) == 1
    assert "undocumented" in blockers[0]["reason"]


def test_product_blocks_full_page_fallback():
    blockers = builder.validate_buildable(
        _ir([_documented_fallback([0, 0, 100, 100])]), profile=builder.PROFILE_PRODUCT)
    assert len(blockers) == 1
    assert "full-page" in blockers[0]["reason"]


def test_group_blocked_in_both_profiles():
    ir = _ir([{"id": "g", "type": "group", "bbox": [0, 0, 5, 5]}])
    for profile in (builder.PROFILE_ALL_NATIVE, builder.PROFILE_PRODUCT):
        blockers = builder.validate_buildable(ir, profile=profile)
        assert any(b["element_type"] == "group" for b in blockers)


def test_resolve_profile_env_and_invalid(monkeypatch):
    monkeypatch.setenv("I2E_BUILD_PROFILE", "product_delivery")
    assert builder._resolve_profile(None) == "product_delivery"
    monkeypatch.setenv("I2E_BUILD_PROFILE", "nonsense")
    assert builder._resolve_profile(None) == "all_native"
    assert builder._resolve_profile("all_native") == "all_native"


def test_build_pptx_raises_on_blocked_ir():
    ir = _ir([{"id": "x", "type": "raster_crop", "bbox": [0, 0, 10, 10]}])
    with pytest.raises(builder.BuildBlockedError):
        builder.build_pptx(ir, "/tmp/should-not-be-written.pptx", profile="all_native")
