"""Base Agent protocol for diagram2ppt v3.

Every specialist agent inherits from Agent and implements run(), which reads
from and writes to the Global Native IR Blackboard. The agent returns the list
of element ids it changed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class Agent(ABC):
    """Abstract specialist agent."""

    name: str = "BaseAgent"

    @abstractmethod
    def run(self, ir: dict, original: Image.Image, **kwargs: Any) -> list[str]:
        """Execute the agent's task and return ids of changed elements."""
        raise NotImplementedError

    def record_contract_result(
        self,
        ir: dict,
        task: dict | None,
        changed: list[str],
        status: str | None = None,
    ) -> None:
        """Record whether this agent satisfied the planner's method contract."""
        contract = task_contract(task)
        if not contract:
            return
        required = set(contract.get("required_agents") or [])
        satisfied = bool(changed) and (not required or self.name in required)
        ir.setdefault("agent_contract_results", []).append({
            "agent": self.name,
            "method": contract.get("method"),
            "acceptance_policy": contract.get("acceptance_policy"),
            "required_agents": sorted(required),
            "forbid_agents": list(contract.get("forbid_agents") or []),
            "task_id": (task or {}).get("id"),
            "region_id": (task or {}).get("region_id"),
            "changed_count": len(changed),
            "changed": list(changed),
            "satisfied": satisfied,
            "status": status or ("changed" if changed else "no_change"),
        })


def task_contract(task: dict | None) -> dict[str, Any]:
    """Extract a normalized method contract from a planner task."""
    if not task:
        return {}
    rep = task.get("representation") or {}
    method = rep.get("method") or task.get("locked_method")
    if not method:
        return {}
    return {
        "method": method,
        "acceptance_policy": rep.get("acceptance_policy") or task.get("acceptance_policy"),
        "required_agents": rep.get("required_agents") or task.get("required_agents") or [],
        "forbid_agents": rep.get("forbid_agents") or task.get("forbid_agents") or [],
    }
