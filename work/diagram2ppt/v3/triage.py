"""Retrospective triage / index of v3 output directories.

The v3 workspace has accumulated dozens of ``v3_out*`` run directories with no
easy way to tell which reached ``accepted``, which produced partial output, and
which died early — STATUS.md calls this out as "找结果困难".

This tool scans an output root, classifies every leaf run directory by outcome
(preferring a ``run_manifest.json`` when present, otherwise reconstructing the
outcome from ``ir_final.json`` / ``audit_trace.json`` / artifacts), and writes a
JSON + Markdown index.

It is strictly **non-destructive**: it never moves, renames, or deletes
anything (the CLAUDE.md / repo rule against destroying run artifacts). It only
reads and writes its own index files.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from . import run_manifest

# Presence of any of these marks a directory as one v3 run's output.
_RUN_MARKERS = (
    "run_manifest.json",
    "ir_final.json",
    "audit_trace.json",
    "diagram_v3.pptx",
)

# Never descend into these (noise / internals / source / already-archived —
# not active run outputs). Point --root at the archive explicitly to index it.
_PRUNE = {
    "__pycache__", "models", "snapshots", "candidates", "proposal_phase",
    ".git", "v2", "v2_out", "v22_out", "v2_out_1920", "baselines", "agents",
    "providers", "tests", "v3", "archive_202506",
}

OUTCOME_INCOMPLETE = "incomplete"  # trace exists but no terminal IR -> likely interrupted/timeout
OUTCOME_EMPTY = "empty"            # marker dir with nothing usable


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _is_run_dir(d: Path) -> bool:
    return any((d / marker).exists() for marker in _RUN_MARKERS)


def _dir_mtime(d: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))
    except OSError:
        return ""


def classify_dir(d: str | Path) -> dict[str, Any]:
    """Classify a single run directory into a run-level outcome."""
    d = Path(d)
    manifest = _read_json(d / "run_manifest.json")
    ir = _read_json(d / "ir_final.json")
    trace = _read_json(d / "audit_trace.json")
    has_pptx = (d / "diagram_v3.pptx").exists()

    if manifest and manifest.get("outcome"):
        outcome = manifest["outcome"]
        source = "run_manifest"
        ir_status = manifest.get("ir_status")
        metrics = manifest.get("metrics", {}) or {}
    else:
        source = "reconstructed"
        ir_status = (ir or {}).get("status") or (trace or {}).get("status")
        metrics = (ir or {}).get("metrics", {}) or {}
        if ir_status == "accepted":
            outcome = run_manifest.OUTCOME_ACCEPTED
        elif ir is not None:
            outcome = (
                run_manifest.OUTCOME_PARTIAL if has_pptx else run_manifest.OUTCOME_REJECTED
            )
        elif trace is not None:
            # a trace but no final IR: the run stopped before writing ir_final,
            # which in practice means it was killed (timeout) or crashed.
            outcome = OUTCOME_INCOMPLETE
        elif has_pptx:
            outcome = run_manifest.OUTCOME_PARTIAL
        else:
            outcome = OUTCOME_EMPTY

    extra: dict[str, Any] = {}
    if ir is not None:
        from . import metrics as _metrics
        m = _metrics.ir_metrics(ir)
        extra = {
            "native_element_ratio": m["native_element_ratio"],
            "fallback_area_ratio": m["fallback_area_ratio"],
            "editability_score": m["editability_score"],
            "fallback_count": m["fallback_count"],
            "fallback_compliant": m["fallback_compliant"],
        }

    return {
        "dir": str(d),
        "outcome": outcome,
        "source": source,
        "ir_status": ir_status,
        "has_pptx": has_pptx,
        "rounds": (ir or {}).get("round"),
        "defect_count": len((ir or {}).get("defects", []) or []),
        "visual_delta": metrics.get("visual_delta"),
        "native_fraction_area": metrics.get("native_fraction_area"),
        "coverage_explained": metrics.get("coverage_explained"),
        "mtime": _dir_mtime(d),
        **extra,
    }


def scan(root: str | Path) -> list[dict[str, Any]]:
    """Find and classify every leaf run directory under ``root``."""
    root = Path(root)
    runs: list[dict[str, Any]] = []
    for cur, dirs, _files in os.walk(root):
        curp = Path(cur)
        if _is_run_dir(curp):
            runs.append(classify_dir(curp))
            dirs[:] = []  # a run dir's internals are not separate runs
            continue
        dirs[:] = [x for x in dirs if x not in _PRUNE]
    return sorted(runs, key=lambda r: (r["outcome"], r["dir"]))


def summarize(runs: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in runs:
        out[r["outcome"]] = out.get(r["outcome"], 0) + 1
    return out


def to_markdown(runs: list[dict[str, Any]], root: str | Path) -> str:
    summary = summarize(runs)
    lines = [
        f"# v3 output triage — {root}",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(runs)} run dirs · non-destructive index_",
        "",
        "## Summary",
        "",
        "| outcome | count |",
        "|---|---|",
    ]
    for outcome in sorted(summary):
        lines.append(f"| {outcome} | {summary[outcome]} |")
    lines += [
        "",
        "## Runs",
        "",
        "| outcome | dir | pptx | rounds | defects | visual_delta | native_area | coverage | mtime |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            "| {outcome} | {dir} | {pptx} | {rounds} | {defects} | {vd} | {na} | {cov} | {mtime} |".format(
                outcome=r["outcome"],
                dir=Path(r["dir"]).name,
                pptx="Y" if r["has_pptx"] else "-",
                rounds=r["rounds"] if r["rounds"] is not None else "-",
                defects=r["defect_count"],
                vd=r["visual_delta"] if r["visual_delta"] is not None else "-",
                na=r["native_fraction_area"] if r["native_fraction_area"] is not None else "-",
                cov=r["coverage_explained"] if r["coverage_explained"] is not None else "-",
                mtime=r["mtime"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Non-destructive triage/index of v3 output directories.")
    ap.add_argument("-r", "--root", default="work/diagram2ppt",
                    help="output root to scan")
    ap.add_argument("-o", "--out", default="work/diagram2ppt/v3_out_index",
                    help="index path stem (writes <stem>.json and <stem>.md)")
    args = ap.parse_args()

    runs = scan(args.root)
    payload = {
        "root": str(args.root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summarize(runs),
        "runs": runs,
    }
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    stem.with_suffix(".md").write_text(to_markdown(runs, args.root))

    print(f"scanned {len(runs)} run dirs under {args.root}")
    for outcome, count in sorted(payload["summary"].items()):
        print(f"  {outcome}: {count}")
    print(f"index: {stem.with_suffix('.json')} / {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
