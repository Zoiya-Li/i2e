"""Operator registry for the v3 runtime kernel.

All operators are registered here so the kernel can dispatch by name.
"""
from __future__ import annotations

from typing import Any, Callable

from .operators import (
    AcceptOperator,
    AcceptOrRollbackOperator,
    AuditTasksOperator,
    ComponentCleanupOperator,
    ComposeOperator,
    DeriveComponentsOperator,
    FailOperator,
    FinalizeOperator,
    ImmutableAuditTasksOperator,
    ImmutableTaskGraphOperator,
    LegacyPlannerLoopOperator,
    Operator,
    PerceiveOperator,
    ProposalPhaseOperator,
    RepairOperator,
    RenderVerifyAuditOperator,
    SvgLoopOperator,
    TaskGraphOperator,
)


def register_operators() -> dict[str, Operator]:
    """Return the default operator registry."""
    ops: list[Operator] = [
        PerceiveOperator(),
        ComposeOperator(),
        RenderVerifyAuditOperator(),
        TaskGraphOperator(),
        ProposalPhaseOperator(),
        ComponentCleanupOperator(),
        RepairOperator(),
        AcceptOrRollbackOperator(),
        DeriveComponentsOperator(),
        AuditTasksOperator(),
        SvgLoopOperator(),
        AcceptOperator(),
        FailOperator(),
        FinalizeOperator(),
        LegacyPlannerLoopOperator(),
        ImmutableTaskGraphOperator(),
        ImmutableAuditTasksOperator(),
    ]
    return {op.name: op for op in ops}


def get(name: str) -> Operator | None:
    return register_operators().get(name)
