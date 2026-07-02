"""Regression tests: Correction.kind values emitted by capture must be valid
against the IR v1 schema, so the editor's save path (which validates before
persisting) does not silently drop a user's font-size or asset-swap edit.

These cover the two kinds that were missing from the schema enum:
  text.font_size_px -> "font_size"
  raster.asset_ref  -> "asset_replace"
"""
from __future__ import annotations

import json
from pathlib import Path

from capture.corrections import append_corrections, capture_diff
from extractor.assemble import validate_ir

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "ir" / "example-fengyoujing-poster.ir.json"


def _diff(orig_el: dict, edited_el: dict) -> list[dict]:
    orig = {"id": "img1", "elements": [orig_el]}
    edited = {"id": "img1", "elements": [edited_el]}
    return capture_diff(orig, edited)


def _valid_doc_with(corrections: list[dict]) -> dict:
    doc = json.loads(EXAMPLE.read_text())
    append_corrections(doc, corrections)
    return doc


def test_capture_font_size_correction_validates():
    orig = {"id": "t1", "type": "text", "text": {"content": "A", "font_size_px": 12},
            "extraction": {"model_version": "m1", "confidence": 0.9}}
    edited = {"id": "t1", "type": "text", "text": {"content": "A", "font_size_px": 24},
              "extraction": {"model_version": "m1", "confidence": 0.9}}
    corrs = _diff(orig, edited)
    assert [c["kind"] for c in corrs] == ["font_size"]
    validate_ir(_valid_doc_with(corrs))  # must not raise


def test_capture_asset_replace_correction_validates():
    orig = {"id": "r1", "type": "raster", "raster": {"asset_ref": "a.png"},
            "extraction": {"model_version": "m1", "confidence": 0.5}}
    edited = {"id": "r1", "type": "raster", "raster": {"asset_ref": "b.png"},
              "extraction": {"model_version": "m1", "confidence": 0.5}}
    corrs = _diff(orig, edited)
    assert [c["kind"] for c in corrs] == ["asset_replace"]
    validate_ir(_valid_doc_with(corrs))  # must not raise
