"""Regression runner for agentic Image -> PPTX reconstruction.

This module deliberately treats images as data, not code paths.  The same
AuditAgentSystem and method registry run for every case; the suite only records
artifacts and metrics so we can catch "fixed one image, broke another" changes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from . import pptx_stats
from .audit_agent_system import AuditAgentSystem
from .default_agents import register_default_agents
from .planner import DEFAULT_MAX_ROUNDS, Planner


BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "v2_framework.json"


def load_v2_baseline() -> dict[str, Any] | None:
    """Load the frozen, measured v2 regression baseline, or None if absent.

    Every regression report embeds this so v3 cases are always scored against
    the same v2 bar (see work/diagram2ppt/v3/baselines/v2_framework.json).
    """
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def compare_to_baseline(structure: dict[str, Any]) -> dict[str, Any] | None:
    """Compare one produced deck's structure against the frozen v2 baseline.

    The v3 all-native goal is: at least as editable as v2 (no more raster
    pictures, native-object ratio no lower). This surfaces that delta per case.
    """
    baseline = load_v2_baseline()
    if not baseline:
        return None
    b = baseline.get("structure", {})
    ratio = structure.get("native_object_ratio", 0.0)
    b_ratio = b.get("native_object_ratio", 0.0)
    pics = structure.get("pictures", 0)
    b_pics = b.get("pictures", 0)
    return {
        "native_object_ratio": ratio,
        "baseline_native_object_ratio": b_ratio,
        "native_ratio_delta": round(ratio - b_ratio, 4),
        "pictures": pics,
        "baseline_pictures": b_pics,
        "beats_baseline_editability": pics <= b_pics and ratio >= b_ratio,
    }


DEFAULT_CASES = [
    "/Users/lizeyan/Desktop/i2e/framework.png.ocr_upscale.png",
    "/Users/lizeyan/Desktop/i2e/test.png",
]


def run_suite(
    images: list[str],
    out_root: str | Path,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    log=print,
) -> dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report: dict[str, Any] = {
        "version": "diagram2ppt-regression-suite-v1",
        "max_rounds": max_rounds,
        "cases": [],
    }
    for index, image in enumerate(images):
        image_path = Path(image).expanduser()
        case_name = _case_name(image_path, index)
        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_report = {
            "name": case_name,
            "image": str(image_path),
            "out_dir": str(case_dir),
            "started_at": time.time(),
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            planner = Planner(str(image_path), str(case_dir), max_rounds=max_rounds)
            register_default_agents(planner)
            final_ir = AuditAgentSystem(planner, log=log).run()
            metrics = final_ir.get("metrics", {})
            case_report.update({
                "status": final_ir.get("status"),
                "round": final_ir.get("round"),
                "metrics": metrics,
                "defects": len(final_ir.get("defects") or []),
                "agent_contract_results": len(final_ir.get("agent_contract_results") or []),
                "artifacts": _artifacts(case_dir),
            })
            pptx = case_dir / "diagram_v3.pptx"
            if pptx.exists():
                structure = pptx_stats.pptx_structure(pptx)
                case_report["structure"] = structure
                case_report["baseline_comparison"] = compare_to_baseline(structure)
        except Exception as exc:
            case_report.update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": _artifacts(case_dir),
            })
        finally:
            case_report["elapsed_sec"] = round(time.time() - case_report["started_at"], 3)
            report["cases"].append(case_report)
            _write_report(out_root, report, started)
    return _write_report(out_root, report, started)


def main() -> int:
    _set_default_provider_env()
    ap = argparse.ArgumentParser(
        description="Run the same agentic Image -> PPTX pipeline on a regression image set.")
    ap.add_argument("images", nargs="*", default=DEFAULT_CASES,
                    help="input image paths")
    ap.add_argument("-o", "--out", default="work/diagram2ppt/v3_regression",
                    help="output root directory")
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                    help="maximum audit-agent repair rounds per case")
    args = ap.parse_args()
    report = run_suite(args.images, args.out, max_rounds=args.max_rounds)
    print("\n=== diagram2ppt regression suite ===")
    for case in report.get("cases", []):
        metrics = case.get("metrics") or {}
        print(
            f"{case.get('name')}: status={case.get('status')} "
            f"visual_delta={metrics.get('visual_delta')} "
            f"critical={metrics.get('critical_defect_count')} "
            f"out={case.get('out_dir')}"
        )
    print(f"report: {Path(args.out) / 'regression_report.json'}")
    return 0 if all(c.get("status") == "accepted" for c in report.get("cases", [])) else 1


def _write_report(out_root: Path, report: dict[str, Any], started: float) -> dict[str, Any]:
    report["elapsed_sec"] = round(time.time() - started, 3)
    report["summary"] = {
        "cases": len(report.get("cases") or []),
        "accepted": sum(1 for c in report.get("cases", []) if c.get("status") == "accepted"),
        "failed": sum(1 for c in report.get("cases", []) if c.get("status") == "failed"),
        "errors": sum(1 for c in report.get("cases", []) if c.get("status") == "error"),
    }
    report.setdefault("baseline", load_v2_baseline())
    (out_root / "regression_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def _artifacts(case_dir: Path) -> dict[str, str | None]:
    names = [
        "diagram_v3.pptx",
        "diagram_v3.compare.png",
        "audit_trace.json",
        "audit_state_initial.json",
        "task_graph.json",
        "proposal_phase/proposal_report.json",
        "ir_final.json",
    ]
    return {
        name: str(case_dir / name) if (case_dir / name).exists() else None
        for name in names
    }


def _case_name(path: Path, index: int) -> str:
    stem = path.name
    for suffix in (".ocr_upscale.png", ".png", ".jpg", ".jpeg"):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return f"{index:02d}_{safe or 'case'}"


def _set_default_provider_env() -> None:
    os.environ.setdefault("I2E_V3_VLM_PROVIDER", "siliconflow")
    os.environ.setdefault("I2E_V3_OCR_PROVIDER", "siliconflow")
    os.environ.setdefault("I2E_VLM_BASE_URL", "https://api.siliconflow.cn/v1")
    os.environ.setdefault("I2E_VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
    os.environ.setdefault("I2E_VISION_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
    os.environ.setdefault("I2E_PLANNER_MODEL", "Qwen/Qwen3.5-397B-A17B")


if __name__ == "__main__":
    raise SystemExit(main())
