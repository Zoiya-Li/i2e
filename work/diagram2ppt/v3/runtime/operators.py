"""Operator placeholder module for the v3 runtime kernel.

In Phase 1 operators are not yet the primary dispatch mechanism. This module
reserves the namespace and provides a base class / protocol so later phases can
wrap existing Planner methods without renaming imports.
"""
from __future__ import annotations

from typing import Any, Protocol

from .state import RuntimeState


class Operator(Protocol):
    """An operator consumes a RuntimeState and returns an updated one."""

    name: str
    target_stage: str

    def check_preconditions(self, state: RuntimeState, **inputs: Any) -> None: ...
    def run(self, state: RuntimeState, **inputs: Any) -> RuntimeState: ...


class BaseOperator:
    """Minimal concrete base for operators added in Phase 2+."""

    name: str = "base"
    target_stage: str = "idle"

    def check_preconditions(self, state: RuntimeState, **inputs: Any) -> None:
        pass

    def run(self, state: RuntimeState, **inputs: Any) -> RuntimeState:
        return state
