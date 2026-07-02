"""CLI entry point for diagram2ppt v3.

Wraps the audit/planner loop so that **every** invocation writes a diagnosable
``run_manifest.json`` — including runs that time out, are killed by an external
``timeout`` (SIGTERM), raise mid-pipeline, or stop with residual defects. This
is the P0 stabilization contract: a run's failure must always be inspectable
after the fact (see ``run_manifest`` and ``work/diagram2ppt/STATUS.md`` §1.5).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from . import run_manifest
from .audit_agent_system import AuditAgentSystem
from .default_agents import register_default_agents
from .planner import DEFAULT_MAX_ROUNDS, Planner


class _Terminated(Exception):
    """Raised when the process receives SIGTERM (e.g. an external ``timeout``)."""


def _install_sigterm_handler() -> None:
    """Turn SIGTERM into a catchable exception so a `timeout` kill still writes
    a manifest. Best-effort: silently skipped if not on the main thread."""
    def _handler(signum, frame):  # noqa: ARG001 - signature fixed by signal API
        raise _Terminated(f"received signal {signum}")

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        pass


# Artifacts the pipeline may drop; presence is recorded in the manifest so a
# partial/failed run still shows how far it got.
ARTIFACT_NAMES = (
    "perception_blackboard.json",
    "content_tasks.json",
    "task_graph.json",
    "visual_review_latest.json",
    "diagram_v3.pptx",
    "diagram_v3.compare.png",
    "ir_final.json",
    "audit_trace.json",
)


def _artifacts(out_dir: str) -> dict:
    base = Path(out_dir)
    return {name: (base / name).exists() for name in ARTIFACT_NAMES}


def main() -> int:
    os.environ.setdefault("I2E_V3_VLM_PROVIDER", "siliconflow")
    os.environ.setdefault("I2E_V3_OCR_PROVIDER", "siliconflow")
    os.environ.setdefault("I2E_VLM_BASE_URL", "https://api.siliconflow.cn/v1")
    os.environ.setdefault("I2E_VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
    os.environ.setdefault("I2E_VISION_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
    os.environ.setdefault("I2E_PLANNER_MODEL", "Qwen/Qwen3.5-397B-A17B")

    ap = argparse.ArgumentParser(
        description="Planner-orchestrated native Image → PPTX reconstruction (v3)")
    ap.add_argument("image", help="input image path")
    ap.add_argument("-o", "--out", default="work/diagram2ppt/v3_out",
                    help="output directory")
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                    help="maximum repair rounds")
    ap.add_argument("--legacy-planner", action="store_true",
                    help="run the old planner loop instead of the agent audit system")
    args = ap.parse_args()

    config = {
        "loop": "legacy_planner" if args.legacy_planner else "audit_agent_system",
        "max_rounds": args.max_rounds,
        "vlm_provider": os.environ.get("I2E_V3_VLM_PROVIDER"),
        "ocr_provider": os.environ.get("I2E_V3_OCR_PROVIDER"),
        "vlm_model": os.environ.get("I2E_VLM_MODEL"),
        "vision_model": os.environ.get("I2E_VISION_MODEL"),
        "planner_model": os.environ.get("I2E_PLANNER_MODEL"),
    }

    _install_sigterm_handler()
    started = time.time()
    planner = None
    final_ir: dict = {}
    error = None
    interrupted = False
    try:
        planner = Planner(args.image, args.out, max_rounds=args.max_rounds)
        register_default_agents(planner)
        final_ir = (
            planner.run() if args.legacy_planner else AuditAgentSystem(planner).run()
        )
    except (KeyboardInterrupt, _Terminated) as exc:
        interrupted = True
        error = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - top-level diagnostic capture
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        ended = time.time()
        ir_obj = final_ir or (getattr(planner, "ir", None) or {})
        manifest = run_manifest.build_manifest(
            image=args.image,
            out_dir=args.out,
            config=config,
            started_at=started,
            ended_at=ended,
            ir=ir_obj,
            error=error,
            interrupted=interrupted,
            artifacts=_artifacts(args.out),
        )
        try:
            manifest_path = run_manifest.write_manifest(args.out, manifest)
        except Exception:  # noqa: BLE001 - never let manifest I/O mask the run
            manifest_path = None

    print("\n=== v3 reconstruction result ===")
    print(f"outcome: {manifest['outcome']}")
    print(f"ir_status: {manifest.get('ir_status')}")
    print(f"rounds: {manifest.get('rounds')}")
    print(f"metrics: {manifest.get('metrics', {})}")
    print(f"defects: {manifest.get('defect_count')}")
    if error:
        print(f"error: {error.get('type')}: {error.get('message')}")
    print(f"manifest: {manifest_path}")
    print(f"output: {args.out}/diagram_v3.pptx")
    print(f"compare: {args.out}/diagram_v3.compare.png")

    ret = run_manifest.exit_code(manifest["outcome"])
    # v3 imports plotting/rendering stacks that can leave non-daemon threads or
    # child processes alive, causing the CLI to hang after the main thread has
    # finished. os._exit is the pragmatic way to guarantee the process exits.
    os._exit(ret)


if __name__ == "__main__":
    sys.exit(main())
