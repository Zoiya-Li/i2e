"""Planner policies for choosing native reconstruction methods.

This is the control-plane contract between perception and specialist agents.
A region is not only assigned to an agent; it is assigned to a rendering
method with constraints.  That prevents metric-winning but semantically wrong
repairs, such as tracing a 3D manifold as arbitrary residual paths.
"""
from __future__ import annotations

from typing import Any

from . import method_registry


METHOD_POLICIES = method_registry.METHOD_POLICIES
REGION_METHOD_BY_KIND = method_registry.METHOD_BY_KIND
VISUAL_REGION_METHOD = method_registry.REGION_ID_HINTS


def policy_for_region(
    kind: str | None = None,
    region_id: str | None = None,
    objective: str | None = None,
    expected_native_expression: str | None = None,
    visual_problem: str | None = None,
) -> dict[str, Any]:
    """Return a copy of the method policy for a semantic region/task."""
    return method_registry.policy_for_region(
        kind=kind,
        region_id=region_id,
        objective=objective,
        expected_native_expression=expected_native_expression,
        visual_problem=visual_problem,
    )


def apply_policy_to_task(task: dict[str, Any]) -> dict[str, Any]:
    """Attach method constraints and reorder agents for a task."""
    visual_defect = task.get("visual_defect") or {}
    policy = policy_for_region(
        kind=task.get("kind"),
        region_id=task.get("region_id"),
        objective=task.get("objective"),
        expected_native_expression=task.get("expected_native_expression"),
        visual_problem=visual_defect.get("visual_problem"),
    )
    if not policy:
        return task

    required = list(policy.get("required_agents") or [])
    fallback = list(policy.get("fallback_agents") or [])
    forbid = set(policy.get("forbid_agents") or [])
    existing = [str(a) for a in task.get("agent_roles", []) if a]
    roles = _dedupe(required + [a for a in existing + fallback if a not in forbid])

    task["agent_roles"] = roles
    task["locked_method"] = policy.get("locked_method") or policy.get("method")
    task["required_agents"] = required
    task["forbid_agents"] = sorted(forbid)
    task["acceptance_policy"] = policy.get("acceptance_policy", "metric_guarded")
    contract = method_registry.contract_for_method(task.get("locked_method"))
    if contract:
        task["representation"] = {
            "method": contract["method"],
            "family": contract.get("representation"),
            "owner_agent": contract.get("owner_agent"),
            "required_agents": list(contract.get("required_agents") or []),
            "forbid_agents": list(contract.get("forbid_agents") or []),
            "acceptance_policy": contract.get("acceptance_policy", task["acceptance_policy"]),
            "native_expression": contract.get("native_expression", ""),
            "visual_evidence": list(contract.get("visual_evidence") or []),
        }
    if policy.get("native_expression") and not task.get("expected_native_expression"):
        task["expected_native_expression"] = policy["native_expression"]
    return task


def candidate_allowed(task: dict[str, Any], roles: list[str]) -> bool:
    """Return whether a candidate role chain is allowed by method constraints."""
    forbid = set(task.get("forbid_agents") or [])
    if any(role in forbid for role in roles):
        return False
    required = set(task.get("required_agents") or [])
    if required and not required.intersection(roles):
        return False
    return True


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
