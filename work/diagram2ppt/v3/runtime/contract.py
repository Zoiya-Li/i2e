"""Executable transition contract for the v3 runtime kernel.

This module defines the interfaces needed to move from a "mutable interpreter"
to a deterministic state-machine kernel.  It does NOT yet replace the existing
operators; it provides a parallel, opt-in contract layer that can be adopted
operator-by-operator.

Core idea:
  - State is immutable at operator boundaries.
  - Operators return (new_state, effects) instead of mutating in place.
  - Effects are first-class objects describing what the operator wants to do.
  - The kernel is the only entity allowed to commit effects to the Planner/files.
"""
from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .state import RuntimeState


@dataclass(frozen=True)
class StateRead:
    """Declare that an operator reads a canonical state field."""

    field: str


@dataclass(frozen=True)
class StateWrite:
    """Declare that an operator writes a canonical state field."""

    field: str


@dataclass(frozen=True)
class ArtifactWrite:
    """Declare that an operator writes a file artifact."""

    path: str
    content_type: str = "json"


@dataclass(frozen=True)
class SideEffect:
    """Base class for effects an operator wants the kernel to commit.

    Effects are explicit, serializable, and reversible where possible.
    """

    kind: str


@dataclass(frozen=True)
class WriteFileEffect(SideEffect):
    """Write content to a file."""

    kind: str = field(init=False, default="write_file")
    path: str
    payload: dict[str, Any] | list[Any] | str | bytes
    encoding: str = "utf-8"


@dataclass(frozen=True)
class UpdatePlannereffect(SideEffect):
    """Update a Planner attribute from a pure state change.

    This bridges pure operators back to the legacy Planner without giving
    operators direct mutation rights.
    """

    kind: str = field(init=False, default="update_planner")
    attr: str
    value: Any


@dataclass(frozen=True)
class NoEffect(SideEffect):
    """Explicit no-op effect."""

    def __init__(self) -> None:
        super().__init__(kind="none")


NO_EFFECT = NoEffect()


class ImmutableOperator(ABC):
    """Operator contract for deterministic, replayable state transitions.

    An ImmutableOperator:
      1. Receives a deep-copied RuntimeState and read-only inputs.
      2. Returns a *new* RuntimeState without mutating the input.
      3. Declares which fields it reads/writes and which artifacts it produces.
      4. Returns a list of SideEffects for the kernel to commit.

    The kernel is responsible for:
      - Making the pre-state copy.
      - Applying the returned effects (file writes, planner updates).
      - Recording the Transition with input/output hashes.
      - Detecting undeclared writes via validation hooks.
    """

    name: str = "immutable_base"
    target_stage: str = "idle"

    # Declarative contract. Subclasses override.
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    idempotent: bool = False

    @abstractmethod
    def run(
        self,
        state: RuntimeState,
        **inputs: Any,
    ) -> tuple[RuntimeState, list[SideEffect]]:
        """Compute the next state and requested effects.

        Must not mutate ``state``.
        """
        raise NotImplementedError

    def check_preconditions(self, state: RuntimeState, **inputs: Any) -> None:
        """Validate that required read-fields are present."""
        missing = [r for r in self.reads if getattr(state, r, None) is None]
        if missing:
            raise RuntimeError(
                f"{self.name} requires missing fields: {missing}"
            )

    def transition_hash(
        self,
        state: RuntimeState,
        **inputs: Any,
    ) -> str:
        """Stable hash of the operator call.

        Used for deterministic replay and cache keys.
        """
        payload = {
            "operator": self.name,
            "inputs": _jsonable(inputs),
            "state": _state_hashable(state),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:16]


class Transaction:
    """A single operator invocation under the immutable contract.

    The Transaction records the pre-state, post-state, effects, and hashes
    needed for deterministic replay.  Unlike the legacy Transition, it captures
    enough information to recompute the output from the input.
    """

    def __init__(
        self,
        operator: ImmutableOperator,
        inputs: dict[str, Any],
        pre_state: RuntimeState,
    ) -> None:
        self.operator = operator
        self.inputs = inputs
        self.pre_state = pre_state
        self.pre_hash = state_hash(pre_state)
        self.call_hash = operator.transition_hash(pre_state, **inputs)
        self.post_state: RuntimeState | None = None
        self.effects: list[SideEffect] = []
        self.post_hash: str | None = None
        self.error: dict[str, Any] | None = None

    def execute(self) -> RuntimeState:
        """Run the operator and return the post-state.

        Does not commit effects.  The caller (kernel) decides whether to commit.
        """
        self.operator.check_preconditions(self.pre_state, **self.inputs)
        state_copy = copy.deepcopy(self.pre_state)
        try:
            new_state, effects = self.operator.run(state_copy, **self.inputs)
        except Exception as exc:
            self.error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        self.effects = effects
        self.post_state = new_state
        self.post_hash = state_hash(new_state)
        return new_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator.name,
            "inputs": _jsonable(self.inputs),
            "pre_hash": self.pre_hash,
            "post_hash": self.post_hash,
            "call_hash": self.call_hash,
            "effects": [
                {
                    "kind": e.kind,
                    "path": getattr(e, "path", None),
                    "attr": getattr(e, "attr", None),
                }
                for e in self.effects
            ],
            "declared_writes": list(self.operator.writes),
            "declared_artifacts": list(self.operator.artifacts),
            "error": self.error,
        }


def state_hash(state: RuntimeState) -> str:
    """Stable hash of canonical RuntimeState fields."""
    payload = {
        "version": state.version,
        "input_image": state.input_image,
        "out_dir": state.out_dir,
        "round": state.round,
        "stage": state.stage,
        "ir": _jsonable(state.ir),
        "strategy_plan": _jsonable(state.strategy_plan),
        "renderer_mode": state.renderer_mode,
        "config": _jsonable(state.config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _state_hashable(state: RuntimeState) -> dict[str, Any]:
    """Return a JSON-serializable subset of state for hashing."""
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
    """Best-effort JSON-normalized representation for hashing."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return str(value)


def commit_effects(
    effects: list[SideEffect],
    planner: Any | None,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Apply a list of SideEffects.

    Returns a summary of what was committed.  This is the only place in the
    contract layer that is allowed to perform side effects.
    """
    summary: dict[str, Any] = {"files": [], "planner_updates": []}
    out = Path(out_dir)
    for effect in effects:
        if isinstance(effect, WriteFileEffect):
            path = out / effect.path if not Path(effect.path).is_absolute() else Path(effect.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(effect.payload, (str, bytes)):
                if isinstance(effect.payload, str):
                    path.write_text(effect.payload, encoding=effect.encoding)
                else:
                    path.write_bytes(effect.payload)
            else:
                path.write_text(
                    json.dumps(effect.payload, indent=2, ensure_ascii=False, default=str),
                    encoding=effect.encoding,
                )
            summary["files"].append(str(path))
        elif isinstance(effect, UpdatePlannereffect):
            if planner is not None and hasattr(planner, effect.attr):
                setattr(planner, effect.attr, copy.deepcopy(effect.value))
                summary["planner_updates"].append(effect.attr)
    return summary
