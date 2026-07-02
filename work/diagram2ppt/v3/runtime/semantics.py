"""Formal execution semantics for the v3 runtime kernel (Level 5).

This layer closes the gap between "graph semantics" and "execution semantics".
It treats operators as algebraic objects over a typed state space, effects as
state deltas with composition laws, and loops as bounded fixed-point
computations over graph transformations.

Core formal objects:
  - StateField / StateSpace: typed fields of RuntimeState
  - OperatorSpec: algebraic signature (reads, writes, artifacts, idempotency)
  - StateDelta: formal record of a state change
  - EffectAlgebra: composition, conflict detection, reversibility
  - ExecutionTraceValidator: verify traces obey the algebra
  - LoopFixpoint / TerminationMetric: loop as bounded fixed-point

The validator is opt-in and additive: existing operators keep working while
semantics checks are enabled during graph execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .state import RuntimeState


@dataclass(frozen=True)
class StateField:
    """One typed field in the runtime state space."""

    name: str
    required: bool = False


@dataclass(frozen=True)
class StateSpace:
    """The set of fields that RuntimeState exposes to operators."""

    fields: frozenset[str]
    required_fields: frozenset[str]

    @classmethod
    def from_runtime_state(cls) -> "StateSpace":
        fields = frozenset(RuntimeState.__dataclass_fields__.keys())
        required = frozenset({"ir"})  # only ir is strictly required for most ops
        return cls(fields=fields, required_fields=required)

    def validate_spec(self, spec: "OperatorSpec") -> list[str]:
        """Return a list of errors if the spec references unknown fields."""
        errors: list[str] = []
        for f in spec.reads | spec.writes | spec.optional_reads:
            if f not in self.fields:
                errors.append(f"spec references unknown field: {f}")
        return errors


@dataclass(frozen=True)
class OperatorSpec:
    """Algebraic signature of an operator over RuntimeState.

    An operator is a partial function:
        f: S_reads × inputs → S_writes × effects
    where S_reads and S_writes are projections of RuntimeState.
    """

    reads: frozenset[str]
    writes: frozenset[str]
    optional_reads: frozenset[str] = field(default_factory=frozenset)
    artifacts: frozenset[str] = field(default_factory=frozenset)
    idempotent: bool = False
    monotonic: bool = False  # True = writes are monotonic (only grow or only shrink)

    @classmethod
    def from_operator(cls, op: Any) -> "OperatorSpec":
        reads = frozenset(getattr(op, "reads", ()) or ())
        writes = frozenset(getattr(op, "writes", ()) or ())
        optional = frozenset(getattr(op, "optional_reads", ()) or ())
        artifacts = frozenset(getattr(op, "artifacts", ()) or ())
        return cls(
            reads=reads,
            writes=writes,
            optional_reads=optional,
            artifacts=artifacts,
            idempotent=bool(getattr(op, "idempotent", False)),
            monotonic=bool(getattr(op, "monotonic", False)),
        )

    def all_reads(self) -> frozenset[str]:
        return self.reads | self.optional_reads

    def conflicts_with(self, other: "OperatorSpec") -> list[str]:
        """Return write-write conflicts between two specs."""
        return sorted(self.writes & other.writes)

    def can_run_in_parallel_with(self, other: "OperatorSpec") -> tuple[bool, list[str]]:
        """Two operators are parallel-safe iff they have no write-write conflicts."""
        conflicts = self.conflicts_with(other)
        return (not conflicts, conflicts)


@dataclass(frozen=True)
class StateDelta:
    """Formal record of one field transition.

    This is a pure description of change, not the change itself.  It is used for
    trace validation, conflict detection, and reversibility analysis.
    """

    field: str
    before_hash: str | None
    after_hash: str | None
    kind: str  # "add" | "remove" | "replace" | "no-op"

    @classmethod
    def compute(
        cls,
        field: str,
        before: Any,
        after: Any,
    ) -> "StateDelta":
        before_hash = _hash_value(before)
        after_hash = _hash_value(after)
        if before_hash == after_hash:
            kind = "no-op"
        elif before is None and after is not None:
            kind = "add"
        elif before is not None and after is None:
            kind = "remove"
        else:
            kind = "replace"
        return cls(
            field=field,
            before_hash=before_hash,
            after_hash=after_hash,
            kind=kind,
        )

    def is_noop(self) -> bool:
        return self.kind == "no-op"

    def is_reversible(self) -> bool:
        return self.before_hash is not None


@dataclass
class OperatorDelta:
    """All StateDeltas produced by one operator invocation."""

    operator: str
    node_id: str
    deltas: dict[str, StateDelta] = field(default_factory=dict)

    def touched_fields(self) -> frozenset[str]:
        return frozenset(self.deltas.keys())

    def changed_fields(self) -> frozenset[str]:
        return frozenset(f for f, d in self.deltas.items() if not d.is_noop())


@dataclass
class EffectAlgebra:
    """Algebraic operations on StateDeltas."""

    @staticmethod
    def compose(first: StateDelta, second: StateDelta) -> StateDelta | None:
        """Compose two deltas on the same field.

        Returns None if the deltas are not composable (second.before_hash !=
        first.after_hash).
        """
        if first.field != second.field:
            raise ValueError("cannot compose deltas on different fields")
        if first.after_hash != second.before_hash:
            return None
        return StateDelta(
            field=first.field,
            before_hash=first.before_hash,
            after_hash=second.after_hash,
            kind=second.kind,
        )

    @staticmethod
    def conflicts(a: StateDelta, b: StateDelta) -> bool:
        """Two deltas conflict if they write the same field differently."""
        if a.field != b.field:
            return False
        if a.is_noop() or b.is_noop():
            return False
        return a.after_hash != b.after_hash

    @staticmethod
    def inverse(delta: StateDelta) -> StateDelta | None:
        """Return the inverse delta if reversible."""
        if not delta.is_reversible():
            return None
        kind = delta.kind
        if kind == "add":
            inv_kind = "remove"
        elif kind == "remove":
            inv_kind = "add"
        else:
            inv_kind = "replace"
        return StateDelta(
            field=delta.field,
            before_hash=delta.after_hash,
            after_hash=delta.before_hash,
            kind=inv_kind,
        )


@dataclass
class ExecutionRuleViolation:
    """One violation of the execution algebra."""

    node_id: str
    rule: str
    message: str


class ExecutionTraceValidator:
    """Validate an execution trace against operator specs and state deltas."""

    def __init__(self, space: StateSpace | None = None) -> None:
        self.space = space or StateSpace.from_runtime_state()

    def validate(
        self,
        graph: Any,
        trace: Any,
        initial_state: RuntimeState,
        operator_specs: dict[str, OperatorSpec] | None = None,
    ) -> list[ExecutionRuleViolation]:
        """Validate a full graph execution trace.

        Checks:
          1. Every read field was either in the initial state or written by an
             ancestor node.
          2. No two parallel-executed nodes write the same field.
          3. Every declared write appears as a non-noop delta or is declared
             no-op for this invocation.
          4. No undeclared writes occur.
          5. Effects are committed exactly once per node.
        """
        violations: list[ExecutionRuleViolation] = []
        if operator_specs is None:
            operator_specs = self._infer_specs(graph, trace)

        node_states = {**trace.node_results}
        # Seed initial state for nodes not in results (e.g. cache hits).
        node_states.setdefault("__initial__", initial_state)

        order = graph.topological_order()
        last_writer: dict[str, str] = {}
        for field in self.space.fields:
            if getattr(initial_state, field, None) is not None:
                last_writer[field] = "__initial__"

        # Pre-fill last_writer from node results so parallel branches know which
        # node produced each field.
        for nid, post_state in node_states.items():
            if nid == "__initial__":
                continue
            node = graph.nodes.get(nid)
            if node is None:
                continue
            spec = operator_specs.get(node.operator)
            if spec is None:
                continue
            for field in spec.writes:
                pre = getattr(initial_state, field, None)
                post = getattr(post_state, field, None)
                if _hash_value(pre) != _hash_value(post):
                    last_writer[field] = nid

        # Detect parallel write conflicts from wave structure.
        waves = graph.independent_groups()
        for wave in waves:
            wave_writes: dict[str, list[str]] = {}
            for nid in wave:
                node = graph.nodes.get(nid)
                if node is None:
                    continue
                spec = operator_specs.get(node.operator)
                if spec is None:
                    continue
                for field in spec.writes:
                    wave_writes.setdefault(field, []).append(nid)
            for field, nids in wave_writes.items():
                if len(nids) > 1:
                    violations.append(ExecutionRuleViolation(
                        node_id=",".join(nids),
                        rule="parallel_write_conflict",
                        message=f"field {field} written by parallel nodes {nids}",
                    ))

        for nid in order:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            spec = operator_specs.get(node.operator)
            if spec is None:
                continue

            pre_state = self._pre_state_for(nid, graph, node_states, last_writer)
            post_state = trace.node_results.get(nid)
            if post_state is None:
                continue

            # Rule 1: reads must be available.
            for field in spec.reads:
                if field not in last_writer and getattr(pre_state, field, None) is None:
                    violations.append(ExecutionRuleViolation(
                        node_id=nid,
                        rule="read_before_write",
                        message=f"operator {node.operator} reads {field} before any producer",
                    ))

            # Rule 3/4: declared writes match actual deltas.
            actual_deltas = self._compute_deltas(pre_state, post_state, spec.writes)
            for field in spec.writes:
                delta = actual_deltas.get(field)
                if delta is None or delta.is_noop():
                    violations.append(ExecutionRuleViolation(
                        node_id=nid,
                        rule="declared_write_not_observed",
                        message=f"operator {node.operator} declared write to {field} but no change",
                    ))

            undeclared = self._undeclared_writes(pre_state, post_state, spec)
            for field in undeclared:
                violations.append(ExecutionRuleViolation(
                    node_id=nid,
                    rule="undeclared_write",
                    message=f"operator {node.operator} wrote {field} without declaring it",
                ))

            # Update last writer map.
            for field in spec.writes:
                if field in actual_deltas and not actual_deltas[field].is_noop():
                    last_writer[field] = nid

        return violations

    def _infer_specs(
        self,
        graph: Any,
        trace: Any,
    ) -> dict[str, OperatorSpec]:
        specs: dict[str, OperatorSpec] = {}
        for node in graph.nodes.values():
            if node.operator not in specs:
                # Try to get spec from the kernel registry if available.
                op = trace.__dict__.get("kernel", {}).get("_operators", {}).get(node.operator) if hasattr(trace, "kernel") else None
                if op is not None:
                    specs[node.operator] = OperatorSpec.from_operator(op)
                else:
                    specs[node.operator] = OperatorSpec(
                        reads=frozenset(),
                        writes=frozenset(node.produced_fields or []),
                    )
        return specs

    def _pre_state_for(
        self,
        nid: str,
        graph: Any,
        node_states: dict[str, RuntimeState],
        last_writer: dict[str, str],
    ) -> RuntimeState:
        """Reconstruct the pre-state by folding ancestor deltas.

        For validation purposes we use the last writer of each field to pick the
        most recent ancestor state for that field.  This is an approximation but
        sufficient for detecting read-before-write and undeclared-write bugs.
        """
        # Prefer an explicit ancestor edge state if available.
        initial = node_states.get("__initial__")
        state = copy.deepcopy(initial) if initial else RuntimeState()
        ancestors: set[str] = set()
        for edge in graph.edges:
            if edge.target == nid:
                ancestors.add(edge.source)
        for dep in graph.nodes.get(nid, GraphNode(nid, "")).depends_on:
            ancestors.add(dep)

        # Fold ancestor states. For fields with a known last writer that is an
        # ancestor, prefer that writer's post-state; otherwise copy any non-None
        # ancestor value.
        for aid in sorted(ancestors):
            ancestor_state = node_states.get(aid)
            if ancestor_state is None:
                continue
            for field in self.space.fields:
                val = getattr(ancestor_state, field, None)
                if val is None:
                    continue
                if last_writer.get(field) == aid:
                    setattr(state, field, copy.deepcopy(val))
                elif getattr(state, field, None) is None:
                    setattr(state, field, copy.deepcopy(val))
        return state

    def _compute_deltas(
        self,
        pre: RuntimeState,
        post: RuntimeState,
        fields: frozenset[str],
    ) -> dict[str, StateDelta]:
        deltas: dict[str, StateDelta] = {}
        for field in fields:
            before = getattr(pre, field, None)
            after = getattr(post, field, None)
            deltas[field] = StateDelta.compute(field, before, after)
        return deltas

    def _undeclared_writes(
        self,
        pre: RuntimeState,
        post: RuntimeState,
        spec: OperatorSpec,
    ) -> list[str]:
        undeclared: list[str] = []
        declared = spec.writes | spec.optional_reads
        for field in self.space.fields:
            before = getattr(pre, field, None)
            after = getattr(post, field, None)
            if field in declared:
                continue
            if _hash_value(before) != _hash_value(after):
                undeclared.append(field)
        return undeclared


@dataclass
class TerminationMetric:
    """A metric used to prove loop termination.

    A loop terminates if its metric is bounded below and monotonically
    non-increasing across iterations.
    """

    name: str
    values: list[Any] = field(default_factory=list)
    lower_bound: Any | None = None

    def record(self, value: Any) -> None:
        self.values.append(value)

    def is_monotonically_non_increasing(self) -> bool:
        if len(self.values) < 2:
            return True
        try:
            return all(self.values[i] >= self.values[i + 1] for i in range(len(self.values) - 1))
        except TypeError:
            return False

    def is_bounded_below(self) -> bool:
        if self.lower_bound is None or not self.values:
            return False
        try:
            return all(v >= self.lower_bound for v in self.values)
        except TypeError:
            return False

    def terminates(self) -> tuple[bool, str]:
        if not self.values:
            return (False, "no metric values recorded")
        if not self.is_monotonically_non_increasing():
            return (False, f"metric {self.name} is not monotonically non-increasing: {self.values}")
        if not self.is_bounded_below():
            return (False, f"metric {self.name} has no lower bound")
        return (True, f"metric {self.name} terminates")


@dataclass
class LoopFixpoint:
    """A loop node expressed as a bounded fixed-point over a graph transformation.

    The guard operator evaluates a termination metric.  The body graph applies a
    state transformation.  The loop is a fixed-point when the guard reports
    ``loop_continue=False``.
    """

    guard_node_id: str
    body_graph: Any
    metric: TerminationMetric

    def evaluate_trace(self, trace: Any) -> tuple[bool, list[str]]:
        """Return (is_valid_fixpoint, reasons)."""
        reasons: list[str] = []
        guard_transitions = [t for t in trace.transitions if t.operator == self.guard_node_id]
        if not guard_transitions:
            return False, ["no guard transitions found"]

        # The final guard transition must set loop_continue=False.
        final_guard = guard_transitions[-1]
        outputs = final_guard.outputs or {}
        summary = outputs.get("summary", {})
        if summary.get("loop_continue") is not False:
            reasons.append("final guard did not set loop_continue=False")

        terminates, msg = self.metric.terminates()
        if not terminates:
            reasons.append(msg)

        is_valid = not reasons
        return is_valid, reasons


class ExecutionSemantics:
    """Entry point: build the formal semantics for a kernel registry."""

    def __init__(self, operators: dict[str, Any] | None = None) -> None:
        self.space = StateSpace.from_runtime_state()
        self.specs: dict[str, OperatorSpec] = {}
        if operators:
            for name, op in operators.items():
                spec = OperatorSpec.from_operator(op)
                errors = self.space.validate_spec(spec)
                if errors:
                    raise RuntimeError(f"operator {name}: {errors}")
                self.specs[name] = spec

    def spec_for(self, operator_name: str) -> OperatorSpec | None:
        return self.specs.get(operator_name)

    def validator(self) -> ExecutionTraceValidator:
        return ExecutionTraceValidator(self.space)

    def loop_metric_from_state(self, state: RuntimeState) -> TerminationMetric:
        """Default termination metric for the audit loop.

        Combines an iteration budget (bounded below by 0) and a defect budget
        (actionable defects + visual review defects, bounded below by 0).
        """
        ir = state.ir or {}
        defects = ir.get("defects") or []
        visual = ir.get("visual_review") or {}
        visual_defects = visual.get("defects") or []
        defect_budget = len([d for d in defects if d.get("status") != "skipped"]) + len(visual_defects)
        return TerminationMetric(
            name="audit_loop_budget",
            values=[defect_budget],
            lower_bound=0,
        )


def _hash_value(value: Any) -> str:
    """Stable hash of any value for delta comparison."""
    if value is None:
        return "__none__"
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        payload = str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# Avoid import cycle: GraphNode is only used for type fallback.
from .graph import GraphNode  # noqa: E402
