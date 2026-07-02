"""Offline tests for the v3 run manifest (P0 stabilization infra).

These verify the "failure is diagnosable" contract without importing the heavy
planner/render stack or hitting any provider: a manifest is always well-formed,
outcomes are classified with the right precedence, and it round-trips to disk.
"""
from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from work.diagram2ppt.v3 import run_manifest as rm


def test_classify_outcome_precedence():
    # interrupt beats everything, even if partial output landed on disk
    assert rm.classify_outcome("failed", True, error=None, interrupted=True) == rm.OUTCOME_INTERRUPTED
    assert rm.classify_outcome("accepted", True, error={"x": 1}, interrupted=True) == rm.OUTCOME_INTERRUPTED
    # error beats accepted/partial
    assert rm.classify_outcome("accepted", True, error={"x": 1}) == rm.OUTCOME_ERROR
    # accepted status wins when clean
    assert rm.classify_outcome("accepted", True) == rm.OUTCOME_ACCEPTED
    # produced output but not accepted -> partial
    assert rm.classify_outcome("failed", True) == rm.OUTCOME_PARTIAL
    # nothing usable -> rejected
    assert rm.classify_outcome("failed", False) == rm.OUTCOME_REJECTED
    assert rm.classify_outcome(None, False) == rm.OUTCOME_REJECTED


def test_exit_code_only_accepted_is_zero():
    assert rm.exit_code(rm.OUTCOME_ACCEPTED) == 0
    for outcome in (rm.OUTCOME_PARTIAL, rm.OUTCOME_REJECTED, rm.OUTCOME_ERROR, rm.OUTCOME_INTERRUPTED):
        assert rm.exit_code(outcome) == 1


def _base_kwargs(**over):
    kwargs = dict(
        image="foo.png",
        out_dir="/tmp/does-not-matter",
        config={"loop": "audit_agent_system", "max_rounds": 5},
        started_at=100.0,
        ended_at=142.5,
    )
    kwargs.update(over)
    return kwargs


def test_build_manifest_accepted():
    m = rm.build_manifest(
        ir={"status": "accepted", "round": 3, "metrics": {"native_fraction": 0.7}, "defects": []},
        artifacts={"diagram_v3.pptx": True},
        **_base_kwargs(),
    )
    assert m["schema"] == rm.SCHEMA_VERSION
    assert m["outcome"] == rm.OUTCOME_ACCEPTED
    assert m["ir_status"] == "accepted"
    assert m["rounds"] == 3
    assert m["elapsed_sec"] == 42.5
    assert m["defect_count"] == 0
    assert rm.exit_code(m["outcome"]) == 0


def test_build_manifest_partial_when_output_but_not_accepted():
    m = rm.build_manifest(
        ir={"status": "failed", "round": 5, "metrics": {"coverage": 0.9}, "defects": [{"id": "d1"}]},
        artifacts={"diagram_v3.pptx": True, "ir_final.json": True},
        **_base_kwargs(),
    )
    assert m["outcome"] == rm.OUTCOME_PARTIAL
    assert m["defect_count"] == 1


def test_build_manifest_rejected_when_no_output():
    m = rm.build_manifest(
        ir={"status": "failed", "metrics": {}, "defects": []},
        artifacts={"diagram_v3.pptx": False},
        **_base_kwargs(),
    )
    assert m["outcome"] == rm.OUTCOME_REJECTED


def test_build_manifest_error_records_traceback():
    err = {"type": "RuntimeError", "message": "boom", "traceback": "Traceback..."}
    m = rm.build_manifest(ir={}, error=err, artifacts={}, **_base_kwargs())
    assert m["outcome"] == rm.OUTCOME_ERROR
    assert m["error"]["type"] == "RuntimeError"
    # early crash with no ir must not blow up
    assert m["ir_status"] is None
    assert m["metrics"] == {}
    assert m["defect_count"] == 0


def test_build_manifest_interrupted():
    m = rm.build_manifest(
        ir={"status": "failed", "metrics": {"x": 1}},
        error={"type": "_Terminated", "message": "received signal 15"},
        interrupted=True,
        artifacts={"diagram_v3.pptx": True},
        **_base_kwargs(),
    )
    assert m["outcome"] == rm.OUTCOME_INTERRUPTED


def test_write_manifest_roundtrip():
    with TemporaryDirectory() as d:
        m = rm.build_manifest(
            ir={"status": "accepted", "metrics": {}, "defects": []},
            artifacts={"diagram_v3.pptx": True},
            **_base_kwargs(out_dir=d),
        )
        path = rm.write_manifest(d, m)
        assert path.name == rm.MANIFEST_FILENAME
        reloaded = json.loads(path.read_text())
        assert reloaded["schema"] == rm.SCHEMA_VERSION
        assert reloaded["outcome"] == rm.OUTCOME_ACCEPTED
