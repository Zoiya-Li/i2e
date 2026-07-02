"""Runtime state-machine kernel for diagram2ppt v3.

This package introduces the single coherent runtime that the rest of v3
modules plug into as operators. Phase 1 only lands the skeleton:
`RuntimeState`, `Transition`, and `PlannerKernel` wrap the existing Planner
without changing control flow.
"""
from __future__ import annotations

from .state import RuntimeState, Transition
from .kernel import PlannerKernel

from .semantics import ExecutionSemantics

__all__ = ["RuntimeState", "Transition", "PlannerKernel", "ExecutionSemantics"]
