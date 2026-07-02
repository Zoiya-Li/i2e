"""Operator registry placeholder for the v3 runtime kernel.

Phase 1 does not yet dispatch through operators. This module reserves the
namespace and provides the registry shape used in Phase 2.
"""
from __future__ import annotations

from typing import Any, Callable

Operator = Callable[[Any, Any], Any]

_REGISTRY: dict[str, Operator] = {}


def register(name: str, op: Operator) -> Operator:
    _REGISTRY[name] = op
    return op


def get(name: str) -> Operator | None:
    return _REGISTRY.get(name)


def all_operators() -> dict[str, Operator]:
    return dict(_REGISTRY)
