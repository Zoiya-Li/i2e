"""Offline tests for Component IR (P2)."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from work.diagram2ppt.v3.components import build_components, write_component_artifacts


def _ir(elements, defects=None, status="failed", w=100, h=100):
    return {"canvas": {"width_px": w, "height_px": h}, "status": status,
            "elements": elements, "defects": defects or []}


def _sp(regions):
    return {"regions": regions}


def test_build_components_maps_elements_and_status():
    ir = _ir(
        elements=[
            {"id": "e1", "type": "text", "bbox": [0, 0, 10, 10]},
            {"id": "e2", "type": "raster_crop", "bbox": [0, 0, 50, 50]},
            {"id": "e3", "type": "rounded_rect", "bbox": [0, 0, 20, 20]},
        ],
        defects=[{"element_id": "e3", "id": "d1"}],
    )
    sp = _sp([
        {"id": "r0", "kind": "card", "bbox": [0, 0, 60, 60], "element_ids": ["e1", "e2"]},
        {"id": "r1", "kind": "box", "bbox": [0, 0, 20, 20], "element_ids": ["e3"]},
        {"id": "r2", "kind": "empty", "bbox": [0, 0, 5, 5], "element_ids": []},
    ])
    comps = build_components(ir, sp)
    assert [c["id"] for c in comps] == ["comp_r0", "comp_r1", "comp_r2"]

    c0 = comps[0]
    assert c0["status"] == "rendered"
    assert c0["metrics"]["fallback_count"] == 1
    assert c0["metrics"]["native_element_ratio"] == 0.5
    assert c0["provenance"]["region_id"] == "r0"

    c1 = comps[1]
    assert c1["status"] == "audited"
    assert c1["defect_count"] == 1

    c2 = comps[2]
    assert c2["status"] == "planned"
    assert c2["metrics"]["element_count"] == 0


def test_all_raster_component_is_fallback():
    ir = _ir([{"id": "e1", "type": "raster_crop", "bbox": [0, 0, 80, 80]}])
    sp = _sp([{"id": "r", "kind": "surface", "bbox": [0, 0, 80, 80], "element_ids": ["e1"]}])
    comp = build_components(ir, sp)[0]
    assert comp["status"] == "fallback"
    assert comp["metrics"]["fallback_area_ratio"] == 1.0
    assert comp["metrics"]["editability_score"] == 0.0


def test_accepted_ir_yields_accepted_component():
    ir = _ir([{"id": "e1", "type": "text", "bbox": [0, 0, 10, 10]}], status="accepted")
    sp = _sp([{"id": "r", "kind": "card", "bbox": [0, 0, 10, 10], "element_ids": ["e1"]}])
    assert build_components(ir, sp)[0]["status"] == "accepted"


def test_write_component_artifacts_writes_index_and_subir():
    ir = _ir([{"id": "e1", "type": "text", "bbox": [0, 0, 10, 10]}])
    sp = _sp([{"id": "r", "kind": "card", "bbox": [0, 0, 10, 10], "element_ids": ["e1"]}])
    comps = build_components(ir, sp)
    with TemporaryDirectory() as d:
        idx = write_component_artifacts(comps, ir, None, d)
        assert idx["schema"] == "components-v1"
        assert idx["count"] == 1
        assert (Path(d) / "components.json").exists()
        sub = json.loads((Path(d) / "components" / "comp_r" / "component_ir.json").read_text())
        assert [e["id"] for e in sub["elements"]] == ["e1"]
        # no source image passed -> crop is absent, not an error
        assert comps[0]["artifacts"]["component_crop"] is None
        assert comps[0]["artifacts"]["component_ir"].endswith("component_ir.json")
