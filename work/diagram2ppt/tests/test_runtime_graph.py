"""Offline tests for the v3 graph execution kernel.

These build simple ExecutionGraphs, run them through PlannerKernel.execute_graph,
and verify topological ordering, dependency enforcement, and caching.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from work.diagram2ppt.v3.runtime import PlannerKernel
from work.diagram2ppt.v3.runtime.graph import (
    DependencyEdge,
    ExecutionGraph,
    GraphNode,
    GraphScheduler,
)
from work.diagram2ppt.v3.runtime.operators import (
    ImmutableAuditTasksOperator,
    ImmutableOperator,
    ImmutableTaskGraphOperator,
)


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
        out_dir="/tmp/graph_test",
        strategy_plan=ir["strategy_plan"],
        ir=ir,
    )
    kernel = PlannerKernel(planner)
    kernel.state.input_image = str(planner.image_path)
    kernel.state.out_dir = str(planner.out_dir)
    return kernel


def test_topological_order_respects_edges():
    g = ExecutionGraph()
    g.add_node(GraphNode(id="a", operator="immutable_task_graph"))
    g.add_node(GraphNode(id="b", operator="immutable_audit_tasks"))
    g.add_node(GraphNode(id="c", operator="immutable_audit_tasks"))
    g.add_edge(DependencyEdge(source="a", target="b", field="task_graph"))
    g.add_edge(DependencyEdge(source="b", target="c", field="audit_tasks"))
    order = g.topological_order()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_cycle_detection():
    g = ExecutionGraph()
    g.add_node(GraphNode(id="a", operator="x"))
    g.add_node(GraphNode(id="b", operator="x"))
    g.add_edge(DependencyEdge(source="a", target="b"))
    g.add_edge(DependencyEdge(source="b", target="a"))
    try:
        g.topological_order()
        assert False, "expected cycle error"
    except RuntimeError as exc:
        assert "cycle" in str(exc).lower()


def test_independent_groups():
    g = ExecutionGraph()
    g.add_node(GraphNode(id="a", operator="x"))
    g.add_node(GraphNode(id="b", operator="x"))
    g.add_node(GraphNode(id="c", operator="x"))
    g.add_edge(DependencyEdge(source="a", target="c"))
    waves = g.independent_groups()
    assert len(waves) == 2
    assert sorted(waves[0]) == ["a", "b"]
    assert waves[1] == ["c"]


def test_graph_scheduler_runs_task_graph_then_audit_tasks():
    kernel = _kernel_with_ir()
    g = ExecutionGraph()
    g.add_node(GraphNode(
        id="task_graph",
        operator="immutable_task_graph",
        produced_fields=["task_graph"],
        produced_artifacts=["task_graph.json"],
    ))
    g.add_node(GraphNode(
        id="audit_tasks",
        operator="immutable_audit_tasks",
        produced_fields=["audit_tasks"],
        produced_artifacts=["audit_tasks.json"],
    ))
    g.add_edge(DependencyEdge(source="task_graph", target="audit_tasks", field="task_graph"))

    trace = kernel.execute_graph(g)
    assert "task_graph" in trace.node_results
    assert "audit_tasks" in trace.node_results
    assert kernel.state.task_graph is not None
    assert kernel.state.audit_tasks is not None
    assert len(trace.transitions) == 2


def test_graph_scheduler_uses_cache():
    kernel = _kernel_with_ir()
    g = ExecutionGraph()
    g.add_node(GraphNode(
        id="task_graph",
        operator="immutable_task_graph",
        produced_fields=["task_graph"],
    ))

    cache: dict[str, Any] = {}
    trace1 = kernel.execute_graph(g, cache=cache)
    assert "task_graph" not in trace1.cache_hits

    # Same graph on the *same post-execution state* should hit cache.
    trace2 = kernel.execute_graph(g, cache=cache)
    assert "task_graph" in trace2.cache_hits


def test_node_cache_key_changes_with_state():
    node = GraphNode(id="tg", operator="immutable_task_graph")
    from work.diagram2ppt.v3.runtime.state import RuntimeState
    s1 = RuntimeState(input_image="a.png", out_dir="/tmp")
    s2 = RuntimeState(input_image="b.png", out_dir="/tmp")
    assert node.compute_cache_key(s1) != node.compute_cache_key(s2)


def test_graph_to_dict_roundtrip():
    g = ExecutionGraph()
    g.add_node(GraphNode(id="a", operator="x", inputs={"k": 1}))
    d = g.to_dict()
    assert d["nodes"]["a"]["operator"] == "x"
    assert d["nodes"]["a"]["inputs"]["k"] == 1


def test_auto_connect_from_operator_contracts():
    kernel = _kernel_with_ir()
    g = ExecutionGraph()
    g.add_node(GraphNode(id="perceive", operator="perceive"))
    g.add_node(GraphNode(id="render", operator="render_verify_audit"))
    g.add_node(GraphNode(id="derive", operator="derive_components"))
    g.add_node(GraphNode(id="audit", operator="audit_tasks"))
    g.auto_connect(kernel)

    edges = {(e.source, e.target, e.field) for e in g.edges}
    assert ("perceive", "render", "ir") in edges
    assert ("perceive", "derive", "ir") in edges or ("perceive", "derive", "strategy_plan") in edges
    assert ("derive", "audit", "components") in edges

    order = g.topological_order()
    assert order.index("perceive") < order.index("render")
    assert order.index("perceive") < order.index("derive")
    assert order.index("derive") < order.index("audit")


def test_immutable_operator_effects_committed():
    kernel = _kernel_with_ir()
    g = ExecutionGraph()
    g.add_node(GraphNode(
        id="audit",
        operator="immutable_audit_tasks",
        produced_fields=["audit_tasks"],
        produced_artifacts=["audit_tasks.json"],
    ))
    trace = kernel.execute_graph(g)
    assert "audit" in trace.node_results
    assert kernel.state.audit_tasks is not None
    assert (Path(kernel.state.out_dir) / "audit_tasks.json").exists()


def test_parallel_wave_executes_independent_immutable_operators():
    from copy import deepcopy

    class _WriteComponents(ImmutableOperator):
        name = "write_components"
        target_stage = "auditing"
        reads = ("ir",)
        writes = ("components",)
        idempotent = True

        def run(self, state, **inputs):
            new = deepcopy(state)
            new.components = [{"id": "comp_a"}]
            new.stage = self.target_stage
            return new, []

    class _WriteAuditTasks(ImmutableOperator):
        name = "write_audit_tasks2"
        target_stage = "auditing"
        reads = ("ir",)
        writes = ("audit_tasks",)
        idempotent = True

        def run(self, state, **inputs):
            new = deepcopy(state)
            new.audit_tasks = [{"id": "task_1"}]
            new.stage = self.target_stage
            return new, []

    kernel = _kernel_with_ir()
    kernel._operators["write_components"] = _WriteComponents()
    kernel._operators["write_audit_tasks2"] = _WriteAuditTasks()

    g = ExecutionGraph()
    g.add_node(GraphNode(id="wc", operator="write_components", produced_fields=["components"]))
    g.add_node(GraphNode(id="wa", operator="write_audit_tasks2", produced_fields=["audit_tasks"]))

    trace = kernel.execute_graph(g)
    assert kernel.state.components == [{"id": "comp_a"}]
    assert kernel.state.audit_tasks == [{"id": "task_1"}]
    assert len(trace.transitions) == 2


def test_loop_node_executes_body_until_guard_stops():
    from work.diagram2ppt.v3.runtime.operators import Operator

    class _CounterGuard(Operator):
        name = "counter_guard"
        target_stage = "refining"
        reads = ("loop_iteration",)
        writes = ("loop_continue",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.loop_continue = state.loop_iteration < 3
            state.stage = self.target_stage
            return state

    class _CounterBody(Operator):
        name = "counter_body"
        target_stage = "refining"
        reads = ()
        writes = ("loop_iteration",)

        def run(self, kernel, **inputs):
            state = self._state_copy(kernel)
            state.loop_iteration += 1
            state.stage = self.target_stage
            return state

    kernel = _kernel_with_ir()
    kernel._operators["counter_guard"] = _CounterGuard()
    kernel._operators["counter_body"] = _CounterBody()

    body = ExecutionGraph()
    body.add_node(GraphNode(id="body", operator="counter_body"))

    g = ExecutionGraph()
    g.add_node(GraphNode(
        id="guard",
        operator="counter_guard",
        guard_operator="counter_guard",
        loop_body=body,
    ))

    trace = kernel.execute_graph(g)
    assert kernel.state.loop_iteration == 3
    body_transitions = [t for t in trace.transitions if t.operator == "counter_body"]
    guard_transitions = [t for t in trace.transitions if t.operator == "counter_guard"]
    assert len(body_transitions) == 3
    assert len(guard_transitions) == 4
