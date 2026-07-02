"""PlannerKernel — the v3 state machine kernel.

Phase 1 skeleton: wraps the existing Planner, exposes `RuntimeState`, and
records transitions. It does not yet own control flow; that arrives in Phase 3.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .state import RuntimeState, Transition


class PlannerKernel:
    """Owns RuntimeState and records transitions for a reconstruction run."""

    def __init__(
        self,
        planner: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.planner = planner
        self.state = RuntimeState()
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
        self.state.round = int(getattr(p, "ir", {}).get("round", 0))
        self.state.ir = copy.deepcopy(getattr(p, "ir", None))
        self.state.strategy_plan = copy.deepcopy(
            getattr(p, "strategy_plan", None)
            or (getattr(p, "ir", {}) or {}).get("strategy_plan")
        )
        self.state.metrics = copy.deepcopy(
            (getattr(p, "ir", {}) or {}).get("metrics", {})
        )
        self.state.defects = copy.deepcopy(
            (getattr(p, "ir", {}) or {}).get("defects", [])
        )
        self.state.visual_review = copy.deepcopy(
            (getattr(p, "ir", {}) or {}).get("visual_review")
        )
        self.state.renderer_mode = (
            getattr(p, "ir", {}) or {}
        ).get("renderer_mode")
        self.state.run_memory = copy.deepcopy(
            (getattr(p, "ir", {}) or {}).get("run_memory", {})
        )
        self.state.last_pptx = str(
            Path(self.state.out_dir) / "diagram_v3.pptx"
            if self.state.out_dir
            else ""
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

    def set_final_stage(
        self,
        stage: str,
        outputs: dict[str, Any] | None = None,
    ) -> Transition:
        """Record a final transition (accepted / failed / interrupted)."""
        return self.record_transition(
            operator="finalize",
            stage_to=stage,
            outputs=outputs or {},
        )

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
            )
        }
        return self.state.write(out / "state_log.json")
