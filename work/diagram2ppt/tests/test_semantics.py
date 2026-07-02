"""Tests for the formal execution semantics layer (Level 5).

These verify OperatorSpec, StateDelta, EffectAlgebra, ExecutionTraceValidator,
and LoopFixpoint independently of the heavy planner/render stack.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from work.diagram2ppt.v3.runtime import PlannerKernel
from work.diagram2ppt.v3.runtime.graph import DependencyEdge, ExecutionGraph, GraphNode
from work.diagram2ppt.v3.runtime.operators import ImmutableOperator, Operator
from work.diagram2ppt.v3.runtime.semantics import (
    EffectAlgebra,
    ExecutionSemantics,
    ExecutionTraceValidator,
    LoopFixpoint,
    OperatorSpec,
    StateDelta,
    StateSpace,
    TerminationMetric,
)
from work.diagram2ppt.v3.runtime.state import RuntimeState


def test_state_space_from_runtime_state():
    space = StateSpace.from_runtime_state()
    assert "ir" in space.fields
    assert "components" in space.fields
    assert "audit_tasks" in space.fields


def test_operator_spec_from_operator():
    class _Op:
        reads = ("ir",)
        writes = ("components",)
        optional_reads = ("strategy_plan",)
        artifacts = ("components.json",)
        idempotent = True

    spec = OperatorSpec.from_operator(_Op())
    assert spec.reads == frozenset({"ir"})
    assert spec.writes == frozenset({"components"})
    assert spec.optional_reads == frozenset({"strategy_plan"})
    assert spec.artifacts == frozenset({"components.json"})
    assert spec.idempotent is True


def test_operator_parallel_safety():
    a = OperatorSpec(reads=frozenset({"ir"}), writes=frozenset({"components"}))
    b = OperatorSpec(reads=frozenset({"ir"}), writes=frozenset({"audit_tasks"}))
    c = OperatorSpec(reads=frozenset({"ir"}), writes=frozenset({"components"}))

    safe, conflicts = a.can_run_in_parallel_with(b)
    assert safe is True
    assert conflicts == []

    safe, conflicts = a.can_run_in_parallel_with(c)
    assert safe is False
    assert conflicts == ["components"]


def test_state_delta_detects_add_replace_noop():
    d_add = StateDelta.compute("task_graph", None, {"summary": {"tasks": 3}})
    assert d_add.kind == "add"
    assert not d_add.is_noop()

    d_replace = StateDelta.compute("ir", {"x": 1}, {"x": 2})
    assert d_replace.kind == "replace"

    d_noop = StateDelta.compute("ir", {"x": 1}, {"x": 1})
    assert d_noop.kind == "no-op"
    assert d_noop.is_noop()


def test_effect_algebra_composition_and_inverse():
    d1 = StateDelta.compute("x", 1, 2)
    d2 = StateDelta.compute("x", 2, 3)
    composed = EffectAlgebra.compose(d1, d2)
    assert composed is not None
    assert composed.before_hash == d1.before_hash
    assert composed.after_hash == d2.after_hash

    inv = EffectAlgebra.inverse(d1)
    assert inv is not None
    assert inv.before_hash == d1.after_hash
    assert inv.after_hash == d1.before_hash


def test_effect_algebra_detects_conflict():
    d1 = StateDelta.compute("x", 1, 2)
    d2 = StateDelta.compute("x", 1, 3)
    assert EffectAlgebra.conflicts(d1, d2) is True

    d3 = StateDelta.compute("x", 1, 2)
    assert EffectAlgebra.conflicts(d1, d3) is False


def test_trace_validator_detects_parallel_write_conflict():
    kernel = _kernel_with_ir()
    g = ExecutionGraph()
    g.add_node(GraphNode(id="a", operator="write_a", produced_fields=["components"]))
    g.add_node(GraphNode(id="b", operator="write_b", produced_fields=["components"]))

    class _WriteA(Operator):
        name = "write_a"
        target_stage = "auditing"
        reads = ()
        writes = ("components",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.components = [{"id": "a"}]
            state.stage = self.target_stage
            return state

    class _WriteB(Operator):
        name = "write_b"
        target_stage = "auditing"
        reads = ()
        writes = ("components",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.components = [{"id": "b"}]
            state.stage = self.target_stage
            return state

    kernel._operators["write_a"] = _WriteA()
    kernel._operators["write_b"] = _WriteB()

    trace = kernel.execute_graph(g)
    validator = ExecutionTraceValidator()
    specs = {
        "write_a": OperatorSpec(reads=frozenset(), writes=frozenset({"components"})),
        "write_b": OperatorSpec(reads=frozenset(), writes=frozenset({"components"})),
    }
    violations = validator.validate(g, trace, deepcopy(trace.node_results.get("a") or RuntimeState()), specs)
    rules = [v.rule for v in violations]
    assert "parallel_write_conflict" in rules


def test_trace_validator_detects_undeclared_write():
    kernel = _kernel_with_ir()

    class _Sneaky(Operator):
        name = "sneaky"
        target_stage = "auditing"
        reads = ()
        writes = ()

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.components = [{"id": "sneaky"}]  # undeclared write
            state.stage = self.target_stage
            return state

    kernel._operators["sneaky"] = _Sneaky()
    g = ExecutionGraph()
    g.add_node(GraphNode(id="s", operator="sneaky"))
    trace = kernel.execute_graph(g)

    validator = ExecutionTraceValidator()
    specs = {"sneaky": OperatorSpec(reads=frozenset(), writes=frozenset())}
    initial = deepcopy(kernel.state)
    initial.components = None
    violations = validator.validate(g, trace, initial, specs)
    rules = [v.rule for v in violations]
    assert "undeclared_write" in rules


def test_trace_validator_accepts_clean_immutable_pair():
    kernel = _kernel_with_ir()

    class _WriteComponents(ImmutableOperator):
        name = "write_components"
        target_stage = "auditing"
        reads = ()
        writes = ("components",)
        idempotent = True

        def run(self, state, **inputs):
            new = deepcopy(state)
            new.components = [{"id": "comp"}]
            new.stage = self.target_stage
            return new, []

    class _WriteAudit(ImmutableOperator):
        name = "write_audit"
        target_stage = "auditing"
        reads = ()
        writes = ("audit_tasks",)
        idempotent = True

        def run(self, state, **inputs):
            new = deepcopy(state)
            new.audit_tasks = [{"id": "task"}]
            new.stage = self.target_stage
            return new, []

    kernel._operators["write_components"] = _WriteComponents()
    kernel._operators["write_audit"] = _WriteAudit()
    g = ExecutionGraph()
    g.add_node(GraphNode(id="wc", operator="write_components", produced_fields=["components"]))
    g.add_node(GraphNode(id="wa", operator="write_audit", produced_fields=["audit_tasks"]))
    trace = kernel.execute_graph(g)

    semantics = ExecutionSemantics(kernel._operators)
    validator = semantics.validator()
    initial = deepcopy(kernel.state)
    initial.components = None
    initial.audit_tasks = None
    violations = validator.validate(g, trace, initial, semantics.specs)
    assert violations == []


def test_termination_metric_monotonic_and_bounded():
    m = TerminationMetric(name="defects", values=[5, 3, 1, 0], lower_bound=0)
    assert m.is_monotonically_non_increasing() is True
    assert m.is_bounded_below() is True
    ok, msg = m.terminates()
    assert ok is True

    m2 = TerminationMetric(name="bad", values=[1, 2, 1], lower_bound=0)
    assert m2.is_monotonically_non_increasing() is False


def test_loop_fixpoint_evaluates_trace():
    class _Guard(Operator):
        name = "guard"
        target_stage = "refining"
        reads = ("loop_iteration",)
        writes = ("loop_continue",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.loop_continue = state.loop_iteration < 2
            state.stage = self.target_stage
            return state

    class _Body(Operator):
        name = "body"
        target_stage = "refining"
        reads = ()
        writes = ("loop_iteration",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.loop_iteration += 1
            state.stage = self.target_stage
            return state

    kernel = _kernel_with_ir()
    kernel._operators["guard"] = _Guard()
    kernel._operators["body"] = _Body()

    body = ExecutionGraph()
    body.add_node(GraphNode(id="body", operator="body"))
    g = ExecutionGraph()
    g.add_node(GraphNode(id="loop", operator="guard", guard_operator="guard", loop_body=body))
    trace = kernel.execute_graph(g)

    # The scheduler recorded the loop budget at each guard evaluation.
    metric = TerminationMetric(name="loop_budget", values=[5, 4, 3], lower_bound=0)
    fixpoint = LoopFixpoint(guard_node_id="guard", body_graph=body, metric=metric)
    is_valid, reasons = fixpoint.evaluate_trace(trace)
    assert is_valid is True, reasons


def test_loop_fixpoint_rejects_non_terminating_metric():
    class _Guard(Operator):
        name = "guard"
        target_stage = "refining"
        reads = ()
        writes = ("loop_continue",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.loop_continue = False
            state.stage = self.target_stage
            return state

    kernel = _kernel_with_ir()
    kernel._operators["guard"] = _Guard()
    body = ExecutionGraph()
    body.add_node(GraphNode(id="body", operator="guard"))
    g = ExecutionGraph()
    g.add_node(GraphNode(id="loop", operator="guard", guard_operator="guard", loop_body=body))
    trace = kernel.execute_graph(g)

    metric = TerminationMetric(name="bad", values=[1, 2], lower_bound=0)
    fixpoint = LoopFixpoint(guard_node_id="guard", body_graph=body, metric=metric)
    is_valid, reasons = fixpoint.evaluate_trace(trace)
    assert is_valid is False
    assert any("not monotonically" in r for r in reasons)


def test_execution_semantics_builds_specs_from_kernel():
    kernel = _kernel_with_ir()
    semantics = ExecutionSemantics(kernel._operators)
    spec = semantics.spec_for("perceive")
    assert spec is not None
    assert "ir" in spec.writes


def _kernel_with_ir():
    ir = {
        "version": "d2p-3",
        "round": 1,
        "status": "auditing",
        "elements": [],
        "defects": [],
        "strategy_plan": {
            "regions": [{"id": "r1", "kind": "chart", "bbox": [0, 0, 10, 10], "element_ids": []}]
        },
    }
    planner = SimpleNamespace(
        image_path="/img/in.png",
        out_dir="/tmp/semantics_test",
        strategy_plan=ir["strategy_plan"],
        ir=ir,
        max_rounds=5,
    )
    kernel = PlannerKernel(planner)
    kernel.state.input_image = str(planner.image_path)
    kernel.state.out_dir = str(planner.out_dir)
    return kernel
