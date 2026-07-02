"""CLI entry point for diagram2ppt v3.

Wraps the audit/planner loop so that **every** invocation writes a diagnosable
``run_manifest.json`` — including runs that time out, are killed by an external
``timeout`` (SIGTERM), raise mid-pipeline, or stop with residual defects. This
is the P0 stabilization contract: a run's failure must always be inspectable
after the fact (see ``run_manifest`` and ``work/diagram2ppt/STATUS.md`` §1.5).

When a run finalizes (``ir_final.json`` exists) the post-run tools are invoked
best-effort so a finished run also produces the Component IR, unified audit
tasks, and an SVG preview/diff — turning the pipeline into
``ir_final → components → audit_tasks → svg_loop`` without a manual step.
"""
from __future__ import annotations

import argparse
import json
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


# Artifacts the pipeline / post-run tools may drop; presence is recorded in the
# manifest so a partial/failed run still shows how far it got.
ARTIFACT_NAMES = (
    "perception_blackboard.json",
    "content_tasks.json",
    "task_graph.json",
    "visual_review_latest.json",
    "diagram_v3.pptx",
    "diagram_v3.compare.png",
    "ir_final.json",
    "audit_trace.json",
    "components.json",
    "audit_tasks.json",
    "svg_loop.json",
)


def _artifacts(out_dir: str) -> dict:
    base = Path(out_dir)
    return {name: (base / name).exists() for name in ARTIFACT_NAMES}


def _postprocess(out_dir: str) -> dict:
    """Best-effort post-run derivation on a finalized run: Component IR,
    unified audit tasks, and the SVG preview/diff. Never raises."""
    produced: dict = {}
    run_dir = Path(out_dir)
    if not (run_dir / "ir_final.json").exists():
        return produced

    try:
        from . import components as _components
        ir, strategy_plan = _components._load_run(run_dir)
        comps = _components.build_components(ir, strategy_plan)
        _components.write_component_artifacts(
            comps, ir, _components._source_path(ir, run_dir), run_dir)
        produced["components"] = len(comps)
    except Exception:  # noqa: BLE001 - post-run tooling must not fail the run
        pass

    try:
        from . import audit_tasks as _audit_tasks
        ir = json.loads((run_dir / "ir_final.json").read_text())
        comp_index = None
        comp_path = run_dir / "components.json"
        if comp_path.exists():
            comp_index = json.loads(comp_path.read_text()).get("components")
        tasks = _audit_tasks.unify_tasks(ir, comp_index)
        _audit_tasks.write_audit_tasks(tasks, run_dir)
        produced["audit_tasks"] = len(tasks)
    except Exception:  # noqa: BLE001
        pass

    try:
        from . import svg_loop as _svg_loop
        _svg_loop.run_svg_loop(run_dir)
        produced["svg_loop"] = True
    except Exception:  # noqa: BLE001
        pass

    return produced


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
    ap.add_argument("--profile", choices=["all_native", "product_delivery"],
                    default="all_native",
                    help="build profile: all_native (research, zero raster) or "
                         "product_delivery (documented local fallback allowed)")
    ap.add_argument("--no-postprocess", action="store_true",
                    help="skip the components/audit_tasks/svg_loop post-run derivation")
    args = ap.parse_args()
    os.environ["I2E_BUILD_PROFILE"] = args.profile

    config = {
        "loop": "legacy_planner" if args.legacy_planner else "audit_agent_system",
        "max_rounds": args.max_rounds,
        "build_profile": args.profile,
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
    postprocess: dict = {}
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
        if not args.no_postprocess and not interrupted:
            postprocess = _postprocess(args.out)
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
        manifest["postprocess"] = postprocess
        try:
            manifest_path = run_manifest.write_manifest(args.out, manifest)
        except Exception:  # noqa: BLE001 - never let manifest I/O mask the run
            manifest_path = None

    print("\n=== v3 reconstruction result ===")
    print(f"outcome: {manifest['outcome']}")
    print(f"ir_status: {manifest.get('ir_status')}")
    print(f"renderer_mode: {manifest.get('renderer_mode')}")
    print(f"last_successful_stage: {manifest.get('last_successful_stage')}")
    print(f"rounds: {manifest.get('rounds')}")
    print(f"metrics: {manifest.get('metrics', {})}")
    print(f"defects: {manifest.get('defect_count')}")
    print(f"acceptance_blockers: {manifest.get('acceptance_blockers')}")
    if postprocess:
        print(f"postprocess: {postprocess}")
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
