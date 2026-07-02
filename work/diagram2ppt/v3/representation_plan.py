"""Explicit planner decisions for native reconstruction representations.

The planner should decide the representation family before agents propose
geometry.  A task is not just "fix this bbox"; it is "rebuild this bbox using
procedural 3D geometry" or "chart primitives" or "component layout".  This
module records that contract in the strategy plan, task graph, and IR so later
accept/reject decisions can be audited visually.
"""
from __future__ import annotations

from typing import Any

from . import method_registry, typography


CONTRACTS = method_registry.CONTRACTS
contract_for_method = method_registry.contract_for_method


def attach_to_regions(regions: list[dict[str, Any]]) -> None:
    """Attach representation contracts to strategy regions in-place."""
    for region in regions:
        method = region.get("primary_method")
        contract = contract_for_method(method)
        if not contract:
            continue
        region["representation"] = {
            "method": contract["method"],
            "family": contract["representation"],
            "owner_agent": contract["owner_agent"],
            "required_agents": list(contract.get("required_agents") or []),
            "forbid_agents": list(contract.get("forbid_agents") or []),
            "component_template": contract.get("component_template", ""),
            "acceptance_policy": contract.get("acceptance_policy", "metric_guarded"),
            "native_expression": contract.get("native_expression", ""),
            "visual_evidence": list(contract.get("visual_evidence") or []),
            "typography_contract": typography.contract_for_method(method),
            "reason": region.get("reason", ""),
        }


def from_regions(regions: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    """Build the planner's auditable representation plan."""
    items = []
    for region in regions:
        rep = region.get("representation") or {}
        if not rep:
            continue
        items.append({
            "region_id": region.get("id"),
            "kind": region.get("kind"),
            "bbox": region.get("bbox"),
            **rep,
        })
    return {
        "version": "representation-plan-v1",
        "canvas": {"width": width, "height": height},
        "items": items,
        "summary": {
            "regions": len(items),
            "procedural_3d": sum(1 for item in items if item.get("family") == "procedural_3d"),
            "native_chart": sum(1 for item in items if item.get("family") == "native_chart"),
            "component_system": sum(
                1 for item in items
                if str(item.get("family", "")).endswith("_system")
                or item.get("family") in {"flow_pipeline", "summary_panel", "mini_surface_checklist"}
            ),
        },
    }


def apply_to_task(task: dict[str, Any], region: dict[str, Any] | None = None) -> None:
    """Copy representation metadata onto a task if available."""
    rep = None
    if region is not None:
        rep = region.get("representation")
    if rep is None:
        rep = task.get("representation")
    if not rep:
        return
    task["representation"] = dict(rep)
    task.setdefault("locked_method", rep.get("method"))
    task.setdefault("required_agents", list(rep.get("required_agents") or []))
    task.setdefault("forbid_agents", list(rep.get("forbid_agents") or []))
    task.setdefault("acceptance_policy", rep.get("acceptance_policy", "metric_guarded"))
    if rep.get("typography_contract"):
        task.setdefault("typography_contract", dict(rep["typography_contract"]))
    if rep.get("component_template"):
        task.setdefault("component_template", rep["component_template"])
    if rep.get("native_expression"):
        task.setdefault("expected_native_expression", rep["native_expression"])
