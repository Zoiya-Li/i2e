"""Run manifest for diagram2ppt v3 — one diagnosable summary per run.

P0 stabilization infra (see ``work/diagram2ppt/STATUS.md`` §1.5). The v3
pipeline is still converging and frequently does *not* reach ``status:
accepted`` — it times out, errors on a provider call, or stops with residual
defects. Historically those runs produced no single artifact you could open to
learn *what happened*. This module fixes that: ``run.py`` writes a
``run_manifest.json`` on **every** run, including failures and external
``timeout`` kills, so the acceptance bar becomes "failure is diagnosable"
rather than "quality is perfect".

The manifest is deliberately dependency-free (stdlib only) so it can be built
and unit-tested offline without importing the heavy planner/render stack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "run-manifest-v1"

# Run-level outcomes. This is a superset of the internal IR ``status`` field
# (which only ever takes "accepted"/"failed"): it also distinguishes a clean
# rejection from a crash from an external kill, which the raw IR cannot.
OUTCOME_ACCEPTED = "accepted"        # IR status == accepted
OUTCOME_PARTIAL = "partial"          # ran, produced output, but not accepted
OUTCOME_REJECTED = "rejected"        # ran to completion, produced nothing usable
OUTCOME_ERROR = "error"              # raised an exception mid-run
OUTCOME_INTERRUPTED = "interrupted"  # SIGINT / SIGTERM (e.g. an external `timeout`)

ALL_OUTCOMES = (
    OUTCOME_ACCEPTED,
    OUTCOME_PARTIAL,
    OUTCOME_REJECTED,
    OUTCOME_ERROR,
    OUTCOME_INTERRUPTED,
)

MANIFEST_FILENAME = "run_manifest.json"


def classify_outcome(
    ir_status: Optional[str],
    produced_output: bool,
    error: Any = None,
    interrupted: bool = False,
) -> str:
    """Map raw run state onto a single, stable run-level outcome.

    Precedence is intentional: an interrupt or crash is reported even if some
    partial output happened to land on disk first, because the run did not end
    on its own terms.
    """
    if interrupted:
        return OUTCOME_INTERRUPTED
    if error is not None:
        return OUTCOME_ERROR
    if ir_status == "accepted":
        return OUTCOME_ACCEPTED
    if produced_output:
        return OUTCOME_PARTIAL
    return OUTCOME_REJECTED


# Pipeline stages in progress order; the furthest artifact present marks how far
# a run got before stopping. Keeps a failed run's manifest specific ("stalled at
# render") instead of a bare "partial".
_STAGE_ORDER = [
    ("perceived", "perception_blackboard.json"),
    ("content", "content_tasks.json"),
    ("proposed", "task_graph.json"),
    ("rendered", "diagram_v3.pptx"),
    ("audited", "visual_review_latest.json"),
    ("finalized", "ir_final.json"),
]


def derive_stage(artifacts: Optional[dict]) -> Optional[str]:
    """Return the furthest pipeline stage reached, from artifact presence."""
    artifacts = artifacts or {}
    reached = None
    for stage, artifact in _STAGE_ORDER:
        if artifacts.get(artifact):
            reached = stage
    return reached


def acceptance_blockers(ir: Optional[dict]) -> list:
    """Concrete reasons a run is not production-acceptable, derived from the IR.

    An empty list means nothing blocks acceptance (an accepted, true-PowerPoint
    run with no residual defects).
    """
    ir = ir or {}
    metrics = ir.get("metrics") or {}
    blockers: list = []
    if ir.get("status") != "accepted":
        blockers.append(f"ir_status={ir.get('status') or 'none'}")
    renderer_mode = ir.get("renderer_mode")
    if renderer_mode and renderer_mode != "true_powerpoint":
        blockers.append(f"renderer_mode={renderer_mode} (not true_powerpoint)")
    crit = int(metrics.get("critical_defect_count") or 0)
    if crit:
        blockers.append(f"critical_defect_count={crit}")
    actionable = len([d for d in (ir.get("defects") or []) if d.get("status") != "skipped"])
    if actionable:
        blockers.append(f"actionable_defects={actionable}")
    vr = len((ir.get("visual_review") or {}).get("defects") or [])
    if vr:
        blockers.append(f"visual_review_defects={vr}")
    return blockers


def build_manifest(
    *,
    image: Any,
    out_dir: Any,
    config: dict,
    started_at: float,
    ended_at: float,
    ir: Optional[dict] = None,
    error: Any = None,
    interrupted: bool = False,
    artifacts: Optional[dict] = None,
    renderer_mode: Optional[str] = None,
    last_successful_stage: Optional[str] = None,
) -> dict:
    """Build the manifest dict from whatever run state is available.

    Robust to a missing/partial ``ir`` (early crash) — every field degrades to
    a null/empty default rather than raising.

    Production acceptance gate: a run rendered only through the PIL proxy can
    never be reported as ``accepted`` (the proxy lies about OMML/autofit/fonts,
    per the repo's true-PowerPoint rule); such a run is downgraded to
    ``partial`` even if the IR status says accepted.
    """
    ir = ir or {}
    artifacts = artifacts or {}
    metrics = ir.get("metrics", {}) or {}
    defects = ir.get("defects", []) or []
    produced_output = bool(any(artifacts.values())) or bool(metrics)
    outcome = classify_outcome(ir.get("status"), produced_output, error, interrupted)

    renderer_mode = renderer_mode or ir.get("renderer_mode")
    if outcome == OUTCOME_ACCEPTED and renderer_mode and renderer_mode != "true_powerpoint":
        outcome = OUTCOME_PARTIAL

    ir_view = {**ir, "renderer_mode": renderer_mode}
    return {
        "schema": SCHEMA_VERSION,
        "image": str(image),
        "out_dir": str(out_dir),
        "config": config,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(float(ended_at) - float(started_at), 3),
        "outcome": outcome,
        "ir_status": ir.get("status"),
        "renderer_mode": renderer_mode,
        "memory": ir.get("run_memory"),
        "last_successful_stage": last_successful_stage or derive_stage(artifacts),
        "acceptance_blockers": acceptance_blockers(ir_view),
        "rounds": ir.get("round"),
        "metrics": metrics,
        "defect_count": len(defects),
        "artifacts": artifacts,
        "error": error,
    }


def write_manifest(out_dir: Any, manifest: dict) -> Path:
    """Write the manifest to ``<out_dir>/run_manifest.json`` and return its path."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return path


def exit_code(outcome: str) -> int:
    """Process exit code: 0 only when the run was accepted, else 1."""
    return 0 if outcome == OUTCOME_ACCEPTED else 1
