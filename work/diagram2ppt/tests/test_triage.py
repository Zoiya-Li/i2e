"""Offline tests for retrospective v3 output triage classification."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from work.diagram2ppt.v3 import triage


def _mk(d: Path, **files):
    d.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        name = name.replace("__", ".")
        if isinstance(content, (dict, list)):
            (d / name).write_text(json.dumps(content))
        else:
            (d / name).write_bytes(b"x")


def test_accepted_from_ir_final():
    with TemporaryDirectory() as t:
        d = Path(t) / "run"
        _mk(d, ir_final__json={"status": "accepted", "round": 3, "metrics": {}, "defects": []},
            diagram_v3__pptx=b"x")
        assert triage.classify_dir(d)["outcome"] == "accepted"


def test_partial_when_failed_with_pptx():
    with TemporaryDirectory() as t:
        d = Path(t) / "run"
        _mk(d, ir_final__json={"status": "failed", "metrics": {"visual_delta": 0.4}, "defects": [1, 2]},
            diagram_v3__pptx=b"x")
        c = triage.classify_dir(d)
        assert c["outcome"] == "partial"
        assert c["defect_count"] == 2
        assert c["visual_delta"] == 0.4


def test_rejected_when_failed_no_pptx():
    with TemporaryDirectory() as t:
        d = Path(t) / "run"
        _mk(d, ir_final__json={"status": "failed", "metrics": {}, "defects": []})
        assert triage.classify_dir(d)["outcome"] == "rejected"


def test_incomplete_when_trace_but_no_ir_final():
    with TemporaryDirectory() as t:
        d = Path(t) / "run"
        _mk(d, audit_trace__json={"status": None, "events": []})
        assert triage.classify_dir(d)["outcome"] == triage.OUTCOME_INCOMPLETE


def test_run_manifest_takes_precedence():
    with TemporaryDirectory() as t:
        d = Path(t) / "run"
        # manifest says interrupted even though a pptx + failed IR exist
        _mk(d,
            run_manifest__json={"outcome": "interrupted", "ir_status": "failed", "metrics": {}},
            ir_final__json={"status": "failed", "metrics": {}, "defects": []},
            diagram_v3__pptx=b"x")
        c = triage.classify_dir(d)
        assert c["outcome"] == "interrupted"
        assert c["source"] == "run_manifest"


def test_scan_and_summarize():
    with TemporaryDirectory() as t:
        root = Path(t)
        _mk(root / "a", ir_final__json={"status": "accepted", "metrics": {}, "defects": []},
            diagram_v3__pptx=b"x")
        _mk(root / "b", ir_final__json={"status": "failed", "metrics": {}, "defects": []},
            diagram_v3__pptx=b"x")
        _mk(root / "__pycache__", junk=b"x")  # must be pruned
        runs = triage.scan(root)
        outcomes = triage.summarize(runs)
        assert outcomes.get("accepted") == 1
        assert outcomes.get("partial") == 1
        assert len(runs) == 2
