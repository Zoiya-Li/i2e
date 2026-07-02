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
from .contract import ImmutableOperator


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
        self.edges.append(edge)
        source_node = self.nodes.get(edge.source)
        target_node = self.nodes.get(edge.target)
        if source_node is not None and target_node is not None:
            if edge.source not in target_node.depends_on:
                target_node.depends_on.append(edge.source)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "node_order": self.graph.topological_order(),
            "cache_hits": sorted(self.cache_hits),
            "errors": self.errors,
            "transitions": [t.to_dict() for t in self.transitions],
        }


class GraphScheduler:
    """Execute an ExecutionGraph against a PlannerKernel.

    The scheduler:
      1. Topologically sorts the graph.
      2. For each node, checks a deterministic cache key.
      3. Dispatches the operator via kernel.transition().
      4. Records the result and any error.
    """

    def __init__(self, kernel: Any, cache: dict[str, RuntimeState] | None = None) -> None:
        self.kernel = kernel
        self.cache = cache or {}

    def execute(self, graph: ExecutionGraph) -> GraphExecutionTrace:
        trace = GraphExecutionTrace(graph=graph)
        order = graph.topological_order()
        for nid in order:
            node = graph.nodes[nid]
            cache_key = node.compute_cache_key(self.kernel.state)
            if cache_key in self.cache:
                cached_state = self.cache[cache_key]
                self.kernel.state = cached_state
                trace.node_results[nid] = cached_state
                trace.cache_hits.add(nid)
                continue
            op = self.kernel._operators.get(node.operator)
            if op is None:
                raise ValueError(f"unknown operator: {node.operator}")
            try:
                if isinstance(op, ImmutableOperator):
                    # Immutable operator path: pure reducer on RuntimeState.
                    op.check_preconditions(self.kernel.state, **node.inputs)
                    new_state, _effects = op.run(self.kernel.state, **node.inputs)
                    self.kernel.state = deepcopy(new_state)
                else:
                    # Legacy operator path: operator sees the kernel.
                    new_state = self.kernel.transition(node.operator, **node.inputs)
                trace.node_results[nid] = new_state
                self.cache[cache_key] = deepcopy(new_state)
            except Exception as exc:
                trace.errors[nid] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            trace.transitions.append(
                Transition.create(
                    stage_from=self.kernel.state.stage,
                    stage_to=self.kernel.state.stage,
                    operator=node.operator,
                    inputs=node.inputs,
                    outputs={"summary": _summarize_state(self.kernel.state)},
                    artifact_paths={a: None for a in node.produced_artifacts},
                )
            )
        return trace


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
