"""PlannerKernel — the v3 state machine kernel.

Phase 2: the kernel can dispatch transitions through registered operators.
For backward compatibility it still wraps a Planner; operators call Planner
methods internally.  The kernel records every transition and writes the state
log after each transition.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .state import RuntimeState, Transition
from . import registry
from .graph import ExecutionGraph, GraphScheduler


class PlannerKernel:
    """Owns RuntimeState, dispatches operators, and records transitions."""

    def __init__(
        self,
        planner: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.planner = planner
        self.state = RuntimeState()
        self._operators = registry.register_operators()
        if config:
            self.state.config = dict(config)
        if planner is not None:
            self._sync_from_planner()

    def _sync_from_planner(self) -> None:
        """Copy observable fields from the wrapped Planner into RuntimeState."""
        p = self.planner
        if p is None:
            return

        self.state.input_image = str(getattr(p, "image_path", ""))
        self.state.out_dir = str(getattr(p, "out_dir", ""))
        self.state.round = int((getattr(p, "ir") or {}).get("round", 0))
        self.state.ir = copy.deepcopy(getattr(p, "ir", None))
        self.state.strategy_plan = copy.deepcopy(
            getattr(p, "strategy_plan", None)
            or (getattr(p, "ir") or {}).get("strategy_plan")
        )
        self.state.metrics = copy.deepcopy(
            (getattr(p, "ir") or {}).get("metrics", {})
        )
        self.state.defects = copy.deepcopy(
            (getattr(p, "ir") or {}).get("defects", [])
        )
        self.state.visual_review = copy.deepcopy(
            (getattr(p, "ir") or {}).get("visual_review")
            or self.state.visual_review
        )
        # renderer_mode is owned by the kernel (set by render/verify operators);
        # only overwrite it if the planner IR has an explicit value.
        _renderer_mode = (getattr(p, "ir") or {}).get("renderer_mode")
        if _renderer_mode is not None:
            self.state.renderer_mode = _renderer_mode
        self.state.run_memory = copy.deepcopy(
            (getattr(p, "ir") or {}).get("run_memory", {})
            or self.state.run_memory
        )
        self.state.last_pptx = str(
            Path(self.state.out_dir) / "diagram_v3.pptx"
            if self.state.out_dir
            else ""
        )

    def transition(
        self,
        op_name: str,
        **inputs: Any,
    ) -> RuntimeState:
        """Execute one operator and record the transition.

        The operator receives ``self`` so it can access the planner and current
        state.  It returns an updated RuntimeState; the kernel sets the stage to
        the operator's target stage and appends a Transition record.
        """
        op = self._operators.get(op_name)
        if op is None:
            raise ValueError(f"unknown operator: {op_name}")

        op.check_preconditions(self, **inputs)
        stage_from = self.state.stage
        error: dict[str, Any] | None = None
        outputs: dict[str, Any] = {}

        try:
            new_state = op.run(self, **inputs)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            new_state = copy.deepcopy(self.state)
            new_state.stage = op.target_stage
            # Re-raise after recording the transition.
            raise
        else:
            if new_state.stage == stage_from:
                new_state.stage = op.target_stage
            outputs = {"summary": _summarize_outputs(new_state)}
        finally:
            self.state = new_state
            self._sync_from_planner()
            transition = Transition.create(
                stage_from=stage_from,
                stage_to=self.state.stage,
                operator=op_name,
                inputs=inputs,
                outputs=outputs,
                error=error,
                artifact_paths=_artifact_paths(self.state),
            )
            self.state.transitions.append(transition)
            self.write_state_log()

        return self.state

    def set_final_stage(
        self,
        stage: str,
        outputs: dict[str, Any] | None = None,
    ) -> Transition:
        """Record a final transition (accepted / failed / interrupted)."""
        self._sync_from_planner()
        outputs = outputs or {}
        # Promote well-known output fields into state so replay restores them.
        if outputs.get("renderer_mode"):
            self.state.renderer_mode = outputs["renderer_mode"]
        return self.record_transition(
            operator="finalize",
            stage_to=stage,
            outputs=outputs,
        )

    def record_transition(
        self,
        operator: str,
        stage_to: str,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        artifact_paths: dict[str, str | None] | None = None,
        error: dict[str, Any] | None = None,
    ) -> Transition:
        """Record a transition and update RuntimeState from the planner."""
        self._sync_from_planner()
        stage_from = self.state.stage
        transition = Transition.create(
            stage_from=stage_from,
            stage_to=stage_to,
            operator=operator,
            inputs=inputs or {},
            outputs=outputs or {},
            artifact_paths=artifact_paths or {},
            error=error,
        )
        self.state.transitions.append(transition)
        self.state.stage = stage_to
        return transition

    def write_state_log(self, out_dir: str | Path | None = None) -> Path | None:
        """Persist the current RuntimeState as ``state_log.json``."""
        out = Path(out_dir or self.state.out_dir or ".")
        if not out:
            return None
        self._sync_from_planner()
        self.state.artifacts = {
            name: (out / name).exists()
            for name in (
                "perception_blackboard.json",
                "strategy_plan.json",
                "strategy_plan_processed.json",
                "processed.json",
                "ir_00_plan.json",
                "ir_final.json",
                "diagram_v3.pptx",
                "diagram_v3.compare.png",
                "visual_review_latest.json",
                "task_graph.json",
                "audit_trace.json",
                "components.json",
                "audit_tasks.json",
                "svg_loop.json",
                "state_log.json",
            )
        }
        return self.state.write(out / "state_log.json")

    def execute_graph(
        self,
        graph: ExecutionGraph,
        cache: dict[str, RuntimeState] | None = None,
    ) -> Any:
        """Execute a dependency graph of operators and return the execution trace.

        This is the graph-kernel entry point: it replaces the linear
        ``transition()`` call with a topologically scheduled DAG.  Independent
        nodes are executed serially here; a parallel executor can be swapped in
        later without changing the graph semantics.
        """
        scheduler = GraphScheduler(self, cache=cache)
        trace = scheduler.execute(graph)
        # Write back any populated cache entries to the caller's dict.
        if cache is not None:
            cache.update(scheduler.cache)
        return trace

    def replay(self, state_log_path: str | Path) -> RuntimeState:
        """Load a previously recorded RuntimeState from ``state_log.json``.

        This restores the kernel to the saved checkpoint without re-invoking any
        operators.  It is useful for debugging and for resuming bookkeeping; it
        does not yet re-run the transition sequence deterministically.
        """
        import json

        path = Path(state_log_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = RuntimeState(
            version=raw.get("version", "runtime-v1"),
            input_image=raw.get("input_image", ""),
            out_dir=raw.get("out_dir", ""),
            round=raw.get("round", 0),
            stage=raw.get("stage", "idle"),
            ir=raw.get("ir"),
            strategy_plan=raw.get("strategy_plan"),
            components=raw.get("components"),
            task_graph=raw.get("task_graph"),
            audit_tasks=raw.get("audit_tasks"),
            defects=raw.get("defects", []),
            visual_review=raw.get("visual_review"),
            metrics=raw.get("metrics", {}),
            renderer_mode=raw.get("renderer_mode"),
            last_render_png=raw.get("last_render_png"),
            last_compare_png=raw.get("last_compare_png"),
            last_pptx=raw.get("last_pptx"),
            last_svg=raw.get("last_svg"),
            last_verify_result=raw.get("last_verify_result"),
            last_proposal_result=raw.get("last_proposal_result"),
            config=raw.get("config", {}),
            run_memory=raw.get("run_memory", {}),
            artifacts=raw.get("artifacts", {}),
        )
        state.transitions = [
            Transition(**t) for t in raw.get("transitions", [])
        ]
        self.state = state
        if self.planner is not None and state.ir is not None:
            self.planner.ir = copy.deepcopy(state.ir)
            if state.strategy_plan is not None:
                self.planner.strategy_plan = copy.deepcopy(state.strategy_plan)
        return self.state


def _summarize_outputs(state: RuntimeState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if state.ir:
        out["ir_status"] = state.ir.get("status")
        out["round"] = state.ir.get("round")
    if state.metrics:
        out["metrics"] = {
            k: state.metrics.get(k)
            for k in (
                "visual_delta",
                "coverage_explained",
                "critical_defect_count",
                "defect_count",
                "text_accuracy",
                "native_fraction_count",
            )
            if k in state.metrics
        }
    if state.defects is not None:
        out["defects"] = len(state.defects)
    if state.components is not None:
        out["components"] = len(state.components)
    if state.audit_tasks is not None:
        out["audit_tasks"] = len(state.audit_tasks)
    return out


def _artifact_paths(state: RuntimeState) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if state.out_dir:
        out["state_log"] = str(Path(state.out_dir) / "state_log.json")
        out["ir_final"] = str(Path(state.out_dir) / "ir_final.json")
        out["pptx"] = str(Path(state.out_dir) / "diagram_v3.pptx")
        out["compare"] = str(Path(state.out_dir) / "diagram_v3.compare.png")
    return out
