"""Execution graph semantics for the v3 runtime kernel.

This module upgrades the kernel from a linear transition state machine to a
dependency-graph execution scheduler.  A graph is a DAG of operator nodes;
edges represent data dependencies (reads/writes).  The scheduler topologically
sorts the graph, executes independent nodes in parallel where safe, and caches
results by deterministic node keys.

Key concepts:
  - ExecutionGraph: owns nodes and edges
  - GraphNode: one operator invocation with inputs + declared dependencies
  - DependencyEdge: data dependency between nodes (field or artifact)
  - GraphScheduler: executes the DAG and records a GraphExecutionTrace
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import RuntimeState, Transition
from .contract import ImmutableOperator, commit_effects


@dataclass(frozen=True)
class DependencyEdge:
    """A data dependency between two graph nodes."""

    source: str  # producer node id
    target: str  # consumer node id
    field: str | None = None  # state field dependency
    artifact: str | None = None  # artifact path dependency

    def __post_init__(self) -> None:
        pass


@dataclass
class GraphNode:
    """One operator invocation in an execution graph."""

    id: str
    operator: str
    inputs: dict[str, Any] = field(default_factory=dict)
    stage: str = "idle"  # target stage after this node runs
    cache_key: str | None = None  # deterministic key for memoization
    depends_on: list[str] = field(default_factory=list)  # explicit node ids
    produced_fields: list[str] = field(default_factory=list)
    produced_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Loop support: a loop node expands into guard + repeated body execution.
    loop_body: ExecutionGraph | None = None
    guard_operator: str | None = None
    loop_max_iterations: int | None = None

    def compute_cache_key(self, state: RuntimeState) -> str:
        """Stable key from operator name, inputs, and canonical pre-state.

        The cache key intentionally excludes derived state that this node itself
        produces (task_graph, audit_tasks) and the runtime stage so that
        re-executing the same graph on the same upstream state hits the cache.
        """
        payload = {
            "id": self.id,
            "operator": self.operator,
            "inputs": _jsonable(self.inputs),
            "state": _state_canonical_for_cache(state),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:16]


@dataclass
class ExecutionGraph:
    """A DAG of operator nodes describing one execution plan."""

    version: str = "execution-graph-v1"
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[DependencyEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> "ExecutionGraph":
        self.nodes[node.id] = node
        return self

    def add_edge(self, edge: DependencyEdge) -> "ExecutionGraph":
        # Avoid duplicate edges so topological level computation stays clean.
        for existing in self.edges:
            if (
                existing.source == edge.source
                and existing.target == edge.target
                and existing.field == edge.field
                and existing.artifact == edge.artifact
            ):
                return self
        self.edges.append(edge)
        source_node = self.nodes.get(edge.source)
        target_node = self.nodes.get(edge.target)
        if source_node is not None and target_node is not None:
            if edge.source not in target_node.depends_on:
                target_node.depends_on.append(edge.source)
        return self

    def auto_connect(self, kernel: Any) -> "ExecutionGraph":
        """Add dependency edges from declared operator reads/writes.

        For every node, inspect its operator's ``reads``/``writes`` and the
        node's own ``produced_fields``/``produced_artifacts``.  Add an edge from
        the most recent node that writes a field this node reads.
        """
        producers: dict[str, str] = {}
        for nid, node in self.nodes.items():
            op = kernel._operators.get(node.operator)
            if op is None:
                continue
            reads = set(getattr(op, "reads", ()) or ())
            reads |= set(getattr(op, "optional_reads", ()) or ())
            writes = (
                set(getattr(op, "writes", ()) or ())
                | set(node.produced_fields or ())
            )
            for field in reads:
                if field in producers and producers[field] != nid:
                    self.add_edge(
                        DependencyEdge(source=producers[field], target=nid, field=field)
                    )
            for field in writes:
                producers[field] = nid
        return self

    def topological_order(self) -> list[str]:
        """Return node ids in topological order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        seen_deps: set[tuple[str, str]] = set()
        for edge in self.edges:
            if (edge.source, edge.target) in seen_deps:
                continue
            seen_deps.add((edge.source, edge.target))
            if edge.source in in_degree and edge.target in in_degree:
                in_degree[edge.target] += 1
                adj[edge.source].append(edge.target)
        # Also respect explicit depends_on
        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                if dep in self.nodes and (dep, nid) not in seen_deps:
                    seen_deps.add((dep, nid))
                    in_degree[nid] += 1
                    adj[dep].append(nid)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            queue.sort()  # deterministic tie-break
            nid = queue.pop(0)
            order.append(nid)
            for nxt in adj[nid]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.nodes):
            raise RuntimeError("execution graph contains a cycle")
        return order

    def independent_groups(self) -> list[list[str]]:
        """Group node ids into waves of independent execution.

        Nodes in the same wave have no dependencies on each other and can run
        concurrently (if the executor supports it).
        """
        order = self.topological_order()
        levels: dict[str, int] = {}
        for nid in order:
            node = self.nodes[nid]
            level = 0
            for dep in node.depends_on:
                if dep in levels:
                    level = max(level, levels[dep] + 1)
            for edge in self.edges:
                if edge.target == nid and edge.source in levels:
                    level = max(level, levels[edge.source] + 1)
            levels[nid] = level

        max_level = max(levels.values()) if levels else 0
        waves: list[list[str]] = [[] for _ in range(max_level + 1)]
        for nid, lvl in levels.items():
            waves[lvl].append(nid)
        return [sorted(w) for w in waves if w]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": {
                nid: {
                    "id": n.id,
                    "operator": n.operator,
                    "inputs": _jsonable(n.inputs),
                    "stage": n.stage,
                    "depends_on": n.depends_on,
                    "produced_fields": n.produced_fields,
                    "produced_artifacts": n.produced_artifacts,
                    "metadata": _jsonable(n.metadata),
                    "guard_operator": n.guard_operator,
                    "loop_max_iterations": n.loop_max_iterations,
                    "loop_body": n.loop_body.to_dict() if n.loop_body else None,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "field": e.field,
                    "artifact": e.artifact,
                }
                for e in self.edges
            ],
        }


@dataclass
class GraphExecutionTrace:
    """Record of one graph execution: node results, transitions, final state."""

    graph: ExecutionGraph
    node_results: dict[str, RuntimeState] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_hits: set[str] = field(default_factory=set)
    semantics_violations: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "node_order": self.graph.topological_order(),
            "cache_hits": sorted(self.cache_hits),
            "errors": self.errors,
            "semantics_violations": [
                {"node_id": v.node_id, "rule": v.rule, "message": v.message}
                for v in self.semantics_violations
            ],
            "transitions": [t.to_dict() for t in self.transitions],
        }


class GraphScheduler:
    """Execute an ExecutionGraph against a PlannerKernel.

    The scheduler:
      1. Topologically sorts the graph into independent waves.
      2. Executes each wave, running independent immutable operators in parallel.
      3. Commits immutable-operator effects via ``commit_effects``.
      4. Records the result and any error.
      5. Optionally validates the trace against the execution semantics.
    """

    def __init__(
        self,
        kernel: Any,
        cache: dict[str, RuntimeState] | None = None,
        semantics: Any | None = None,
    ) -> None:
        self.kernel = kernel
        self.cache = cache or {}
        self.semantics = semantics

    def execute(self, graph: ExecutionGraph) -> GraphExecutionTrace:
        trace = GraphExecutionTrace(graph=graph)
        initial_state = deepcopy(self.kernel.state)
        waves = graph.independent_groups()
        for wave in waves:
            self._execute_wave(wave, graph, trace)

        if self.semantics is not None:
            validator = self.semantics.validator()
            operator_specs = self.semantics.specs if isinstance(getattr(self.semantics, "specs", None), dict) else None
            violations = validator.validate(graph, trace, initial_state, operator_specs)
            trace.semantics_violations = violations
            if violations:
                trace.errors["__semantics__"] = {
                    "type": "ExecutionSemanticsViolation",
                    "count": len(violations),
                    "first": {
                        "node_id": violations[0].node_id,
                        "rule": violations[0].rule,
                        "message": violations[0].message,
                    },
                }

        return trace

    def _execute_wave(
        self,
        wave: list[str],
        graph: ExecutionGraph,
        trace: GraphExecutionTrace,
    ) -> None:
        remaining: list[str] = []
        for nid in wave:
            node = graph.nodes[nid]
            if node.loop_body is not None:
                # Loop nodes are always executed serially by the scheduler.
                self._execute_loop_node(node, graph, trace)
                continue
            cache_key = node.compute_cache_key(self.kernel.state)
            if cache_key in self.cache:
                cached_state = self.cache[cache_key]
                self.kernel.state = deepcopy(cached_state)
                trace.node_results[nid] = cached_state
                trace.cache_hits.add(nid)
            else:
                remaining.append(nid)

        if not remaining:
            return

        ops = {nid: self.kernel._operators.get(graph.nodes[nid].operator) for nid in remaining}
        if any(op is None for op in ops.values()):
            missing = [nid for nid, op in ops.items() if op is None]
            raise ValueError(f"unknown operators: {missing}")

        can_parallel = all(isinstance(op, ImmutableOperator) for op in ops.values())
        if can_parallel:
            all_writes: set[str] = set()
            for nid in remaining:
                op = ops[nid]
                node = graph.nodes[nid]
                writes = set(getattr(op, "writes", ()) or ()) | set(node.produced_fields or ())
                if writes & all_writes:
                    can_parallel = False
                    break
                all_writes |= writes

        if can_parallel:
            self._execute_parallel(remaining, graph, trace, ops)
        else:
            self._execute_serial(remaining, graph, trace, ops)

    def _execute_serial(
        self,
        nids: list[str],
        graph: ExecutionGraph,
        trace: GraphExecutionTrace,
        ops: dict[str, Any],
    ) -> None:
        for nid in nids:
            node = graph.nodes[nid]
            op = ops[nid]
            cache_key = node.compute_cache_key(self.kernel.state)
            if cache_key in self.cache:
                cached_state = self.cache[cache_key]
                self.kernel.state = deepcopy(cached_state)
                trace.node_results[nid] = cached_state
                trace.cache_hits.add(nid)
                continue

            stage_from = self.kernel.state.stage
            try:
                if isinstance(op, ImmutableOperator):
                    new_state, effects = self._run_immutable_operator(
                        op, node, deepcopy(self.kernel.state)
                    )
                    self.kernel.state = deepcopy(new_state)
                    summary = commit_effects(
                        effects, self.kernel.planner, self.kernel.state.out_dir
                    )
                    outputs = {"summary": _summarize_state(self.kernel.state), "committed": summary}
                else:
                    self.kernel.transition(node.operator, **node.inputs)
                    outputs = {"summary": _summarize_state(self.kernel.state)}
            except Exception as exc:
                trace.errors[nid] = {"type": type(exc).__name__, "message": str(exc)}
                raise

            trace.node_results[nid] = deepcopy(self.kernel.state)
            self.cache[cache_key] = deepcopy(self.kernel.state)
            trace.transitions.append(
                Transition.create(
                    stage_from=stage_from,
                    stage_to=self.kernel.state.stage,
                    operator=node.operator,
                    inputs=node.inputs,
                    outputs=outputs,
                    artifact_paths=self._artifact_paths_for_node(node, op),
                )
            )

    def _execute_parallel(
        self,
        nids: list[str],
        graph: ExecutionGraph,
        trace: GraphExecutionTrace,
        ops: dict[str, Any],
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import os

        pre_state = deepcopy(self.kernel.state)
        results: dict[str, tuple[RuntimeState, list[Any]]] = {}
        max_workers = min(len(nids), (os.cpu_count() or 2))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_immutable_operator,
                    ops[nid],
                    graph.nodes[nid],
                    deepcopy(pre_state),
                ): nid
                for nid in nids
            }
            for future in futures:
                nid = futures[future]
                results[nid] = future.result()

        common_stage: str | None = None
        all_effects: list[Any] = []
        for nid in nids:
            node = graph.nodes[nid]
            op = ops[nid]
            new_state, effects = results[nid]
            trace.node_results[nid] = deepcopy(new_state)
            self.cache[node.compute_cache_key(pre_state)] = deepcopy(new_state)

        # Merge writes deterministically (sorted by node id).
        for nid in sorted(nids):
            node = graph.nodes[nid]
            op = ops[nid]
            new_state, _ = results[nid]
            writes = set(getattr(op, "writes", ()) or ()) | set(node.produced_fields or ())
            for field in writes:
                if field and hasattr(self.kernel.state, field):
                    setattr(self.kernel.state, field, getattr(new_state, field))
            if common_stage is None:
                common_stage = new_state.stage
            all_effects.extend(effects)
            cache_key = node.compute_cache_key(pre_state)
            self.cache[cache_key] = deepcopy(new_state)
            trace.node_results[nid] = deepcopy(new_state)

        if common_stage is not None:
            self.kernel.state.stage = common_stage

        summary = commit_effects(all_effects, self.kernel.planner, self.kernel.state.out_dir)

        for nid in nids:
            node = graph.nodes[nid]
            op = ops[nid]
            new_state, _ = results[nid]
            trace.transitions.append(
                Transition.create(
                    stage_from=pre_state.stage,
                    stage_to=new_state.stage,
                    operator=node.operator,
                    inputs=node.inputs,
                    outputs={"summary": _summarize_state(new_state), "committed": summary},
                    artifact_paths=self._artifact_paths_for_node(node, op),
                )
            )

    def _execute_loop_node(
        self,
        node: GraphNode,
        graph: ExecutionGraph,
        trace: GraphExecutionTrace,
    ) -> None:
        if node.guard_operator is None or node.loop_body is None:
            raise RuntimeError(f"loop node {node.id} missing guard or body")
        guard_op = self.kernel._operators.get(node.guard_operator)
        if guard_op is None:
            raise ValueError(f"unknown guard operator: {node.guard_operator}")
        max_iter = node.loop_max_iterations or int(
            getattr(self.kernel.planner, "max_rounds", 0) or 100
        )

        from .semantics import LoopFixpoint, TerminationMetric
        metric = TerminationMetric(name=f"loop_{node.id}", lower_bound=0)

        for _ in range(max_iter + 1):
            stage_from = self.kernel.state.stage
            # Record metric before guard runs.
            metric.record(self._loop_budget(self.kernel.state))

            if isinstance(guard_op, ImmutableOperator):
                new_state, effects = self._run_immutable_operator(
                    guard_op, node, deepcopy(self.kernel.state)
                )
                self.kernel.state = deepcopy(new_state)
                commit_effects(effects, self.kernel.planner, self.kernel.state.out_dir)
            else:
                self.kernel.transition(node.guard_operator, **node.inputs)

            trace.transitions.append(
                Transition.create(
                    stage_from=stage_from,
                    stage_to=self.kernel.state.stage,
                    operator=node.guard_operator,
                    inputs=node.inputs,
                    outputs={"summary": _summarize_state(self.kernel.state)},
                    artifact_paths=self._artifact_paths_for_node(node, guard_op),
                )
            )

            if not self.kernel.state.loop_continue:
                break

            body_trace = self.execute(node.loop_body)
            trace.node_results.update(body_trace.node_results)
            trace.transitions.extend(body_trace.transitions)
            trace.errors.update(body_trace.errors)
            trace.cache_hits.update(body_trace.cache_hits)
            trace.semantics_violations.extend(body_trace.semantics_violations)

        # Validate loop as a bounded fixed-point.
        loop_fixpoint = LoopFixpoint(
            guard_node_id=node.guard_operator,
            body_graph=node.loop_body,
            metric=metric,
        )
        is_valid, reasons = loop_fixpoint.evaluate_trace(trace)
        if not is_valid:
            trace.errors[f"__loop_{node.id}__"] = {
                "type": "LoopFixpointViolation",
                "reasons": reasons,
            }

    def _loop_budget(self, state: RuntimeState) -> int:
        """Default termination budget for a loop node."""
        ir = state.ir or {}
        defects = ir.get("defects") or []
        visual = ir.get("visual_review") or {}
        visual_defects = visual.get("defects") or []
        defect_budget = len([d for d in defects if d.get("status") != "skipped"]) + len(visual_defects)
        iteration_budget = max(
            0,
            int(getattr(self.kernel.planner, "max_rounds", 0) or 100) - state.loop_iteration,
        )
        return defect_budget + iteration_budget

    def _run_immutable_operator(
        self,
        op: ImmutableOperator,
        node: GraphNode,
        state: RuntimeState,
    ) -> tuple[RuntimeState, list[Any]]:
        op.check_preconditions(state, **node.inputs)
        return op.run(state, **node.inputs)

    def _artifact_paths_for_node(self, node: GraphNode, op: Any) -> dict[str, str | None]:
        if not self.kernel.state.out_dir:
            return {}
        artifacts = set(getattr(op, "artifacts", ()) or ()) | set(node.produced_artifacts or ())
        return {a: str(Path(self.kernel.state.out_dir) / a) for a in artifacts}


def _summarize_state(state: RuntimeState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if state.ir:
        out["ir_status"] = state.ir.get("status")
        out["round"] = state.ir.get("round")
    if state.metrics:
        out["metrics"] = {k: state.metrics.get(k) for k in (
            "visual_delta", "coverage_explained", "critical_defect_count",
        ) if k in state.metrics}
    if state.task_graph is not None:
        out["task_graph"] = state.task_graph.get("summary", {}).get("tasks")
    if state.audit_tasks is not None:
        out["audit_tasks"] = len(state.audit_tasks)
    if state.loop_continue is not None:
        out["loop_continue"] = state.loop_continue
    return out


def _state_canonical_for_cache(state: RuntimeState) -> dict[str, Any]:
    """Return a JSON-normalized canonical subset of state for hashing.

    Excludes derived fields that operators produce (task_graph, audit_tasks) and
    the runtime stage so that cache keys are stable across re-execution of the
    same graph on the same upstream state.
    """
    return {
        "input_image": state.input_image,
        "out_dir": state.out_dir,
        "round": state.round,
        "ir": _jsonable(state.ir),
        "strategy_plan": _jsonable(state.strategy_plan),
        "components": _jsonable(state.components),
        "renderer_mode": state.renderer_mode,
        "loop_iteration": state.loop_iteration,
        "loop_continue": state.loop_continue,
        "next_action": state.next_action,
    }


def _state_canonical(state: RuntimeState) -> dict[str, Any]:
    """Return a JSON-normalized canonical subset of state for hashing."""
    return {
        "input_image": state.input_image,
        "out_dir": state.out_dir,
        "round": state.round,
        "stage": state.stage,
        "ir": _jsonable(state.ir),
        "strategy_plan": _jsonable(state.strategy_plan),
        "components": _jsonable(state.components),
        "task_graph": _jsonable(state.task_graph),
        "audit_tasks": _jsonable(state.audit_tasks),
        "renderer_mode": state.renderer_mode,
    }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return str(value)
