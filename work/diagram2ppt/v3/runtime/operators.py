"""Runtime operators for the v3 state machine kernel.

Each operator consumes a ``PlannerKernel`` (and thus ``RuntimeState`` + the
wrapped Planner), performs one deterministic state mutation, and returns an
updated ``RuntimeState``.  Agents are not rewritten here; their existing
Planner-level methods are wrapped so the kernel can own control flow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .state import RuntimeState


class Operator:
    """Base class for kernel operators."""

    name: str = "base"
    target_stage: str = "idle"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        pass

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        raise NotImplementedError

    def _state_copy(self, kernel: Any) -> RuntimeState:
        import copy
        return copy.deepcopy(kernel.state)


class PerceiveOperator(Operator):
    """Bootstrap the blackboard: perception, content handling, initial IR."""

    name = "perceive"
    target_stage = "planning"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("perceive requires a planner")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        ir = kernel.planner.plan()
        state = self._state_copy(kernel)
        state.ir = ir
        state.round = int(ir.get("round", 0))
        state.stage = self.target_stage
        return state


class ComposeOperator(Operator):
    """Select the initial IR by real rendered evidence.

    Today this is folded into ``Planner.plan()``; the operator is a placeholder
    for the future split between perception/strategy and composed IR selection.
    For Phase 2 it simply re-runs the compose step by invoking the planner.
    """

    name = "compose"
    target_stage = "composing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("compose requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("compose requires an existing IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        # Phase 2: compose is still inside plan(); no separate call yet.
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        return state


class RenderVerifyAuditOperator(Operator):
    """Build PPTX, render, verify, and run visual review."""

    name = "render_verify_audit"
    target_stage = "auditing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("render_verify_audit requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("render_verify_audit requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        result = kernel.planner.render_and_verify()
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        state.last_verify_result = result
        state.metrics = dict(state.ir.get("metrics", {}) if state.ir else {})
        state.defects = list(state.ir.get("defects", []) if state.ir else [])
        state.visual_review = (
            (state.ir or {}).get("visual_review") if state.ir else None
        )
        state.renderer_mode = (
            (state.ir or {}).get("renderer_mode") if state.ir else None
        )
        state.last_pptx = str(
            Path(kernel.state.out_dir) / "diagram_v3.pptx" if kernel.state.out_dir else ""
        )
        state.last_compare_png = str(
            Path(kernel.state.out_dir) / "diagram_v3.compare.png"
            if kernel.state.out_dir
            else ""
        )
        state.last_render_png = str(
            Path(kernel.state.out_dir) / "diagram_v3.true.png"
            if kernel.state.out_dir
            else ""
        )
        return state


class TaskGraphOperator(Operator):
    """Build the region task graph from the current blackboard."""

    name = "task_graph"
    target_stage = "refining"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("task_graph requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("task_graph requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        from .. import task_graph

        graph = task_graph.build(kernel.state.ir)
        state = self._state_copy(kernel)
        state.task_graph = graph
        state.stage = self.target_stage
        return state


class ProposalPhaseOperator(Operator):
    """Run multi-agent region proposals and commit verified candidates."""

    name = "proposal_phase"
    target_stage = "refining"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("proposal_phase requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("proposal_phase requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        result = kernel.planner.run_proposal_phase()
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        state.last_proposal_result = result
        state.metrics = dict(state.ir.get("metrics", {}) if state.ir else {})
        state.defects = list(state.ir.get("defects", []) if state.ir else [])
        return state


class ComponentCleanupOperator(Operator):
    """Remove redundant native components after accepted proposals."""

    name = "component_cleanup"
    target_stage = "refining"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("component_cleanup requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("component_cleanup requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        from .. import component_cleanup

        cleanup = component_cleanup.apply(kernel.state.ir, log=kernel.planner.log)
        if cleanup.get("removed"):
            kernel.planner._save_ir("ir_agent_component_cleanup.json")
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        return state


class RepairOperator(Operator):
    """Execute one specialist agent repair round."""

    name = "repair"
    target_stage = "refining"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("repair requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("repair requires an IR")
        if not inputs.get("agent_name") and not inputs.get("defect"):
            raise RuntimeError("repair requires agent_name or defect")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        agent_name = inputs.get("agent_name")
        defect = inputs.get("defect")
        if agent_name is None and defect is not None:
            agent_name = defect.get("suggested_agent")
        if not agent_name:
            raise RuntimeError("repair could not determine agent_name")
        expected_fixes = inputs.get("expected_fixes") or (
            [defect.get("id", "")] if defect else []
        )
        patch = kernel.planner.run_round(
            agent_name,
            defect=defect,
            expected_fixes=expected_fixes,
        )
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        if patch and defect:
            patch["defect"] = defect
        return state


class AcceptOrRollbackOperator(Operator):
    """Decide whether to keep the latest repair patch."""

    name = "rollback_or_accept"
    target_stage = "refining"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("rollback_or_accept requires a planner")
        if kernel.state.ir is None:
            raise RuntimeError("rollback_or_accept requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        decision = kernel.planner.accept_or_rollback()
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        state.metrics = dict(state.ir.get("metrics", {}) if state.ir else {})
        state.defects = list(state.ir.get("defects", []) if state.ir else [])
        return state


class DeriveComponentsOperator(Operator):
    """Derive Component IR from the current IR + strategy plan."""

    name = "derive_components"
    target_stage = "auditing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.state.ir is None:
            raise RuntimeError("derive_components requires an IR")
        if kernel.state.strategy_plan is None:
            raise RuntimeError("derive_components requires a strategy_plan")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        from .. import components as _components

        ir = kernel.state.ir
        strategy_plan = kernel.state.strategy_plan
        comps = _components.build_components(ir, strategy_plan)
        source_path = _components._source_path(ir, kernel.state.out_dir)
        index = _components.write_component_artifacts(
            comps, ir, source_path, kernel.state.out_dir
        )
        state = self._state_copy(kernel)
        state.components = comps
        state.stage = self.target_stage
        return state


class AuditTasksOperator(Operator):
    """Unify verifier + visual_review defects into executable audit tasks."""

    name = "audit_tasks"
    target_stage = "auditing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.state.ir is None:
            raise RuntimeError("audit_tasks requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        from .. import audit_tasks as _audit_tasks

        ir = kernel.state.ir
        comp_index = None
        if kernel.state.components is not None:
            comp_index = kernel.state.components
        elif kernel.state.out_dir:
            comp_path = kernel.state.out_dir / "components.json"
            if comp_path.exists():
                import json
                comp_index = json.loads(comp_path.read_text()).get("components")
        tasks = _audit_tasks.unify_tasks(ir, comp_index)
        _audit_tasks.write_audit_tasks(tasks, kernel.state.out_dir)
        state = self._state_copy(kernel)
        state.audit_tasks = tasks
        state.stage = self.target_stage
        return state


class SvgLoopOperator(Operator):
    """Run IR → SVG → PNG → diff for the current IR."""

    name = "svg_loop"
    target_stage = "auditing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.state.ir is None:
            raise RuntimeError("svg_loop requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        from .. import svg_loop as _svg_loop

        result = _svg_loop.run_svg_loop(kernel.state.out_dir)
        state = self._state_copy(kernel)
        state.last_svg = result.get("svg")
        state.stage = self.target_stage
        return state


class AcceptOperator(Operator):
    """Mark the reconstruction as accepted."""

    name = "accept"
    target_stage = "accepted"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None or kernel.planner.ir is None:
            raise RuntimeError("accept requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        kernel.planner.ir["status"] = "accepted"
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        return state


class FailOperator(Operator):
    """Mark the reconstruction as failed."""

    name = "fail"
    target_stage = "failed"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None or kernel.planner.ir is None:
            raise RuntimeError("fail requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        kernel.planner.ir["status"] = "failed"
        state = self._state_copy(kernel)
        state.stage = self.target_stage
        return state


class LegacyPlannerLoopOperator(Operator):
    """Run the original Planner.run() legacy loop.

    This operator exists so ``--legacy-planner`` can also be expressed as a
    kernel transition while preserving the original behavior unchanged.
    """

    name = "legacy_planner_loop"
    target_stage = "finalizing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.planner is None:
            raise RuntimeError("legacy_planner_loop requires a planner")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        ir = kernel.planner.run()
        state = self._state_copy(kernel)
        state.ir = ir
        state.stage = self.target_stage
        return state


class FinalizeOperator(Operator):
    """Best-effort finalization: derive post-run artifacts if not already done."""

    name = "finalize"
    target_stage = "finalizing"

    def check_preconditions(self, kernel: Any, **inputs: Any) -> None:
        if kernel.state.ir is None:
            raise RuntimeError("finalize requires an IR")

    def run(self, kernel: Any, **inputs: Any) -> RuntimeState:
        state = self._state_copy(kernel)
        # Best-effort post-run derivation.  Individual operators are no-ops if
        # their preconditions are not met, so we can safely chain them.
        try:
            if state.strategy_plan is not None and state.components is None:
                state = DeriveComponentsOperator().run(_KernelView(kernel, state))
        except Exception:
            pass
        try:
            if state.audit_tasks is None:
                state = AuditTasksOperator().run(_KernelView(kernel, state))
        except Exception:
            pass
        try:
            if state.last_svg is None:
                state = SvgLoopOperator().run(_KernelView(kernel, state))
        except Exception:
            pass
        state.stage = self.target_stage
        return state


class _KernelView:
    """Minimal kernel-shaped object used by operators when chaining internally."""

    def __init__(self, kernel: Any, state: RuntimeState) -> None:
        self.kernel = kernel
        self.state = state
        self.planner = getattr(kernel, "planner", None)

    def sync_from_planner(self) -> None:
        """No-op: the parent kernel will sync after the outer operator returns."""

    def write_state_log(self, out_dir: str | Path | None = None) -> Path | None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)
