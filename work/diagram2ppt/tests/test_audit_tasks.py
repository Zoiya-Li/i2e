"""Offline tests for unified executable audit tasks (P5)."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from work.diagram2ppt.v3.audit_tasks import TASK_TYPES, unify_tasks, write_audit_tasks


def test_verifier_defect_type_routing():
    ir = {"defects": [
        {"id": "a", "type": "high_residual", "element_id": "e1", "severity": 0.5, "reason": "r", "suggested_agent": "StyleAgent"},
        {"id": "b", "type": "text_layout_mismatch", "element_id": "e2", "severity": 0.3, "suggested_agent": "TextLayoutAgent"},
        {"id": "c", "type": "high_residual", "element_id": "e3", "severity": 0.9, "suggested_agent": "ChartAgent"},
        {"id": "d", "type": "high_residual", "element_id": "e4", "severity": 0.2, "suggested_agent": "StyleAgent", "strategy": "demote"},
        {"id": "e", "type": "x", "element_id": "e5", "severity": 0.1, "status": "skipped"},
    ]}
    tasks = unify_tasks(ir)
    by_el = {t["element_id"]: t for t in tasks}
    assert by_el["e1"]["type"] == "refine_geometry"
    assert by_el["e2"]["type"] == "refine_text"
    assert by_el["e3"]["type"] == "rebuild_component"
    assert by_el["e4"]["type"] == "apply_fallback"
    assert "e5" not in by_el  # skipped excluded
    # sorted by severity desc
    assert [t["severity"] for t in tasks] == sorted((t["severity"] for t in tasks), reverse=True)
    # per-type acceptance gate present
    assert by_el["e1"]["acceptance"] == {"residual_max": 0.45}
    assert by_el["e2"]["acceptance"] == {"text_accuracy_min": 0.9}


def test_component_id_resolution_and_visual_review():
    ir = {
        "defects": [{"id": "a", "type": "high_residual", "element_id": "e1", "severity": 0.5, "suggested_agent": "StyleAgent"}],
        "visual_review": {"defects": [
            {"id": "v1", "region_id": "r0", "region_label": "chart", "severity": "critical",
             "visual_problem": "flat", "expected_native_expression": "grouped chart",
             "suggested_agents": ["ChartAgent"]},
        ]},
    }
    comps = [{"id": "comp_card_00", "element_ids": ["e1"], "provenance": {"region_id": "r0"}}]
    tasks = unify_tasks(ir, comps)
    v = [t for t in tasks if t["origin"] == "verifier"][0]
    assert v["component_id"] == "comp_card_00"
    vr = [t for t in tasks if t["origin"] == "visual_review"][0]
    assert vr["component_id"] == "comp_card_00"
    assert vr["type"] == "rebuild_component"
    assert vr["severity"] == 1.0
    assert vr["expected_native_expression"] == "grouped chart"


def test_write_audit_tasks_roundtrip():
    ir = {"defects": [{"id": "a", "type": "high_residual", "element_id": "e1", "severity": 0.5, "suggested_agent": "StyleAgent"}]}
    tasks = unify_tasks(ir)
    with TemporaryDirectory() as d:
        payload = write_audit_tasks(tasks, d)
        assert payload["schema"] == "audit-tasks-v1"
        assert set(payload["types"]) == set(TASK_TYPES)
        reloaded = json.loads((Path(d) / "audit_tasks.json").read_text())
        assert reloaded["count"] == 1
        assert reloaded["tasks"][0]["type"] == "refine_geometry"
