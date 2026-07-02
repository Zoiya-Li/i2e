"""Top-level agentic audit system for Image -> native PPTX.

This is the user-facing loop: an auditing agent observes the image and current
render, chooses tools, records evidence, and iterates until it accepts or
exhausts its budget.  The Planner remains the blackboard/tool substrate.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

from . import component_cleanup, ir as IR
from .planner import _has_visual_review_defects
from .runtime import ExecutionSemantics
from .runtime.graph import DependencyEdge, ExecutionGraph, GraphNode
from .runtime.operators import Operator


class _AuditInitialStateOperator(Operator):
    """Write the initial audit state once the first render is available."""

    name = "audit_initial_state"
    target_stage = "auditing"
    reads = ("last_verify_result", "ir")
    writes = ()

    def __init__(self, system: "AuditAgentSystem") -> None:
        super().__init__()
        self.system = system

    def run(self, kernel: Any, **inputs: Any) -> Any:
        state = self._state_copy(kernel)
        result = state.last_verify_result or {}
        self.system._write_agent_state(
            "initial", self.system._build_audit_state(result, 0)
        )
        return state


class _AuditLoopBodyOperator(Operator):
    """One iteration of the audit agent loop.

    This operator decides between coordinated proposals and a single repair,
    executes the chosen action via kernel transitions, re-renders, and derives
    post-change artifacts.  It is the loop body inside the execution graph.
    """

    name = "audit_loop_body"
    target_stage = "refining"
    reads = (
        "ir", "last_verify_result", "defects", "visual_review",
        "metrics", "loop_iteration",
    )
    writes = (
        "ir", "metrics", "defects", "visual_review", "last_verify_result",
        "last_proposal_result", "components", "audit_tasks", "last_svg",
        "renderer_mode", "loop_continue", "loop_iteration",
    )

    def __init__(self, system: "AuditAgentSystem") -> None:
        super().__init__()
        self.system = system

    def run(self, kernel: Any, **inputs: Any) -> Any:
        state = self._state_copy(kernel)
        state.loop_iteration += 1
        iteration = state.loop_iteration

        result = state.last_verify_result or {}
        audit_state = self.system._build_audit_state(result, iteration)
        self.system._write_agent_state(f"{iteration:02d}", audit_state)
        decision = self.system._decide_next_action(audit_state)
        self.system._write_decision(f"{iteration:02d}", decision)
        self.system._record("decision", decision)

        action = decision["action"]
        should_stop = False
        if action == "proposal_phase":
            kernel.transition("task_graph")
            kernel.transition("proposal_phase")
            if self.system._current_proposal_result().get("accepted", 0):
                kernel.transition("component_cleanup")
            kernel.transition("render_verify_audit")
            self.system._derive_runtime_artifacts()
        elif action == "single_repair":
            self.system._run_single_repair_kernel(decision)
            self.system._derive_runtime_artifacts()
        elif action == "stop":
            should_stop = True
        else:
            self.system._record("warning", {"reason": f"unknown action {action}"})
            should_stop = True

        new_state = self._state_copy(kernel)
        new_state.loop_iteration = iteration
        new_state.loop_continue = False if should_stop else new_state.loop_continue
        new_state.stage = self.target_stage
        return new_state


class AuditAgentSystem:
    """Agent-centric controller around perception/content/proposal/render tools."""

    version = "audit-agent-system-v1"
    tool_registry = {
        "bootstrap_blackboard": {
            "purpose": "read the source image, collect multi-agent evidence, and build initial IR",
            "writes": ["perception_blackboard.json", "content_tasks.json", "ir_00_planned.json"],
        },
        "render_verify_audit": {
            "purpose": "render native PPTX/SVG output and audit it against the source image",
            "writes": ["diagram_v3.pptx", "diagram_v3.compare.png", "visual_review_latest.json"],
        },
        "proposal_phase": {
            "purpose": "dispatch region tasks to multiple specialist agents and accept only verified candidates",
            "writes": ["task_graph.json", "proposal_phase/proposal_report.json"],
        },
        "single_repair": {
            "purpose": "route a narrow residual defect to one specialist agent",
            "writes": ["patch history", "ir_*_accepted.json", "ir_*_rollback.json"],
        },
        "accept_or_rollback": {
            "purpose": "decide whether the latest patch improved the rendered output",
            "writes": ["ir_*_accepted.json", "ir_*_rollback.json"],
        },
        "component_cleanup": {
            "purpose": "remove redundant native components after accepted region proposals",
            "writes": ["ir_agent_component_cleanup.json"],
        },
    }

    def __init__(
        self,
        planner,
        log: Callable[[str], None] = print,
        kernel: Any | None = None,
    ) -> None:
        self.planner = planner
        self.log = log
        self.trace: list[dict[str, Any]] = []
        self._started = time.time()
        self.kernel = kernel

    def run(self) -> dict:
        """Run the autonomous audit loop and return the final IR."""
        self._observe_source()

        if self.kernel is not None:
            return self._run_with_kernel()
        return self._run_legacy()

    def _run_with_kernel(self) -> dict:
        """Kernel-driven control path using an explicit ExecutionGraph."""
        assert self.kernel is not None
        # Register agent-system-specific operators for this run.
        self.kernel._operators[_AuditInitialStateOperator.name] = _AuditInitialStateOperator(self)
        self.kernel._operators[_AuditLoopBodyOperator.name] = _AuditLoopBodyOperator(self)

        graph = self._build_audit_graph()
        self.kernel.execute_graph(graph, semantics=ExecutionSemantics(self.kernel._operators))
        self._finish()
        return self.planner.ir

    def _build_audit_graph(self) -> ExecutionGraph:
        """Build the full audit agent execution graph.

        The graph is:
            perceive → render_verify_audit → audit_initial_state
            render_verify_audit → [derive_components, audit_tasks, svg_loop]
            artifact nodes → audit_loop
            audit_loop (guard + body) → finalize → acceptance
        """
        assert self.kernel is not None
        graph = ExecutionGraph()

        graph.add_node(GraphNode(id="perceive", operator="perceive", stage="planning"))
        graph.add_node(
            GraphNode(id="render_verify_audit", operator="render_verify_audit", stage="auditing")
        )
        graph.add_node(
            GraphNode(id="audit_initial_state", operator="audit_initial_state", stage="auditing")
        )

        graph.add_node(
            GraphNode(
                id="derive_components",
                operator="derive_components",
                stage="auditing",
                produced_fields=["components"],
                produced_artifacts=["components.json"],
            )
        )
        graph.add_node(
            GraphNode(
                id="audit_tasks",
                operator="audit_tasks",
                stage="auditing",
                produced_fields=["audit_tasks"],
                produced_artifacts=["audit_tasks.json"],
            )
        )
        graph.add_node(
            GraphNode(
                id="svg_loop",
                operator="svg_loop",
                stage="auditing",
                produced_fields=["last_svg"],
                produced_artifacts=["svg_loop.json"],
            )
        )

        body = ExecutionGraph()
        body.add_node(
            GraphNode(id="audit_loop_body", operator="audit_loop_body", stage="refining")
        )
        body.auto_connect(self.kernel)

        graph.add_node(
            GraphNode(
                id="audit_loop",
                operator="audit_loop_guard",
                stage="refining",
                guard_operator="audit_loop_guard",
                loop_body=body,
            )
        )
        graph.add_node(GraphNode(id="finalize", operator="finalize", stage="finalizing"))
        graph.add_node(GraphNode(id="acceptance", operator="acceptance", stage="accepted"))

        graph.auto_connect(self.kernel)

        # Control-flow edges that auto-connect cannot infer (loop/terminal ordering).
        graph.add_edge(
            DependencyEdge(source="render_verify_audit", target="audit_initial_state", field="last_verify_result")
        )
        graph.add_edge(DependencyEdge(source="audit_initial_state", target="audit_loop"))
        graph.add_edge(DependencyEdge(source="derive_components", target="audit_tasks"))
        for nid in ("derive_components", "audit_tasks", "svg_loop"):
            graph.add_edge(DependencyEdge(source=nid, target="audit_loop"))
        graph.add_edge(DependencyEdge(source="audit_loop", target="finalize"))
        graph.add_edge(DependencyEdge(source="finalize", target="acceptance"))

        return graph

    def _derive_runtime_artifacts(self) -> None:
        """Best-effort derive components / audit tasks / SVG after state changes."""
        if self.kernel is None:
            return
        try:
            self.kernel.transition("derive_components")
        except Exception:
            pass
        try:
            self.kernel.transition("audit_tasks")
        except Exception:
            pass
        try:
            self.kernel.transition("svg_loop")
        except Exception:
            pass

    def _run_single_repair_kernel(self, decision: dict[str, Any]) -> dict[str, Any]:
        repair = decision.get("repair")
        if not repair:
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}
        agent_name = repair.get("suggested_agent", "")
        self.planner._agent_attempts[agent_name] = (
            self.planner._agent_attempts.get(agent_name, 0) + 1
        )
        if agent_name not in self.planner.agents:
            self._record("tool_skip", {
                "tool": "single_repair",
                "reason": f"no registered agent for {agent_name}",
                "defect": repair,
            })
            for d in self.planner.ir.get("defects", []):
                if d.get("id") == repair.get("id"):
                    d["status"] = "skipped"
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}

        self.kernel.transition("repair", agent_name=agent_name, defect=repair)
        result = self.kernel.transition("render_verify_audit")
        self.kernel.transition("rollback_or_accept")

        patches = self.planner.ir.get("patches", [])
        last_decision = patches[-1]["decision"] if patches else "accept"
        if last_decision == "rollback":
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}
        return result

    def _run_legacy(self) -> dict:
        """Original Planner-level control path (kernel unavailable)."""
        self._call_tool("bootstrap_blackboard", self.planner.plan)
        result = self._call_tool("render_verify_audit", self.planner.render_and_verify)
        self._write_agent_state("initial", self._build_audit_state(result, 0))

        if self._accept_if_done(result, "initial render passed"):
            return self.planner.ir

        for iteration in range(max(0, int(self.planner.max_rounds))):
            state = self._build_audit_state(result, iteration + 1)
            self._write_agent_state(f"{iteration + 1:02d}", state)
            decision = self._decide_next_action(state)
            self._write_decision(f"{iteration + 1:02d}", decision)
            self._record("decision", decision)

            action = decision["action"]
            if action == "proposal_phase":
                proposal = self._call_tool("proposal_phase", self.planner.run_proposal_phase)
                if proposal.get("accepted", 0):
                    cleanup = self._call_tool(
                        "component_cleanup",
                        lambda: component_cleanup.apply(self.planner.ir, log=self.log),
                    )
                    if cleanup.get("removed"):
                        self.planner._save_ir("ir_agent_component_cleanup.json")
                    result = self._call_tool(
                        "render_verify_audit",
                        self.planner.render_and_verify,
                    )
                else:
                    result = self._call_tool(
                        "render_verify_audit",
                        self.planner.render_and_verify,
                    )
            elif action == "single_repair":
                result = self._run_single_repair(decision)
            elif action == "stop":
                break
            else:
                self._record("warning", {"reason": f"unknown action {action}"})
                break

            if self._accept_if_done(result, f"iteration {iteration + 1} passed"):
                return self.planner.ir

        if not (result or {}).get("passed"):
            result = self._call_tool("final_render_verify", self.planner.render_and_verify)

        self.planner.ir["status"] = "accepted" if result.get("passed") else "failed"
        final_stage = "accepted" if self.planner.ir.get("status") == "accepted" else "failed"
        if self.kernel is not None:
            self.kernel.set_final_stage(
                final_stage,
                outputs={
                    "passed": bool(result.get("passed")),
                    "metrics": _summarize_result(result),
                },
            )
        self._finish()
        return self.planner.ir

    def _run_single_repair(self, decision: dict[str, Any]) -> dict:
        repair = decision.get("repair")
        if not repair:
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}
        agent_name = repair.get("suggested_agent", "")
        self.planner._agent_attempts[agent_name] = (
            self.planner._agent_attempts.get(agent_name, 0) + 1
        )
        if agent_name not in self.planner.agents:
            self._record("tool_skip", {
                "tool": "single_repair",
                "reason": f"no registered agent for {agent_name}",
                "defect": repair,
            })
            for d in self.planner.ir.get("defects", []):
                if d.get("id") == repair.get("id"):
                    d["status"] = "skipped"
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}

        patch = self._call_tool(
            f"repair:{agent_name}",
            lambda: self.planner.run_round(
                agent_name,
                defect=repair,
                expected_fixes=[repair.get("id", "")],
            ),
        )
        patch["defect"] = copy.deepcopy(repair)

        if not patch.get("changed"):
            self._record("tool_skip", {
                "tool": f"repair:{agent_name}",
                "reason": "agent made no changes",
                "defect": repair,
            })
            backup = self.planner._patch_preimages.get(patch["patch_id"])
            if backup is not None:
                IR.restore(self.planner.ir, backup)
                self.planner._snapshots = [
                    (r, s) for r, s in self.planner._snapshots
                    if r <= self.planner.ir.get("round", 0)
                ]
            for d in self.planner.ir.get("defects", []):
                if d.get("id") == repair.get("id"):
                    d["status"] = "skipped"
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}

        result = self._call_tool("render_verify_audit", self.planner.render_and_verify)
        if self.planner.ir.get("patches"):
            self.planner.ir["patches"][-1]["metrics_after"] = copy.deepcopy(
                self.planner.ir.get("metrics", {})
            )
        decision = self._call_tool("accept_or_rollback", self.planner.accept_or_rollback)
        if decision == "rollback":
            return {"passed": False, "metrics": self.planner.ir.get("metrics", {})}
        return result

    def _decide_next_action(self, state: dict[str, Any]) -> dict[str, Any]:
        """Choose the next tool from an explicit action slate.

        This keeps planner control auditable: every loop writes what the agent
        saw, which tools were legal, and why one was selected.
        """
        candidates = self._candidate_actions(state)
        ranked = sorted(candidates, key=lambda c: (-c["priority"], c["action"]))
        chosen = copy.deepcopy(ranked[0])
        chosen["candidates"] = candidates
        return chosen

    def _candidate_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        metrics = state.get("metrics", {})
        defects = state.get("defects", {})
        visual = state.get("visual_review", {})
        repair = self.planner._next_repair_task()
        candidates: list[dict[str, Any]] = []

        if defects.get("actionable", 0) == 0 and visual.get("defects", 0) == 0:
            candidates.append({
                "action": "stop",
                "priority": 100,
                "reason": "no actionable residuals remain after rendered visual audit",
                "iteration": state.get("iteration"),
            })

        if (
            state.get("iteration") == 1
            or visual.get("defects", 0) > 0
            or _needs_region_proposals(metrics, state.get("raw_defects", []))
        ):
            candidates.append({
                "action": "proposal_phase",
                "priority": 80,
                "reason": "visible region mismatch or incomplete coverage needs coordinated specialist proposals",
                "iteration": state.get("iteration"),
                "tool": self.tool_registry["proposal_phase"],
            })

        if repair:
            priority = 70
            if defects.get("by_agent", {}).get(repair.get("suggested_agent"), 0) > 1:
                priority += 5
            if visual.get("defects", 0) > 0:
                priority -= 20
            candidates.append({
                "action": "single_repair",
                "priority": priority,
                "reason": "a remaining defect has a concrete owner and specialist route",
                "iteration": state.get("iteration"),
                "repair": copy.deepcopy(repair),
                "tool": self.tool_registry["single_repair"],
            })

        if not candidates:
            candidates.append({
                "action": "stop",
                "priority": 0,
                "reason": "no legal tool remains for the current audit state",
                "iteration": state.get("iteration"),
            })
        return candidates

    def _build_audit_state(self, result: dict[str, Any] | None, iteration: int) -> dict[str, Any]:
        ir = self.planner.ir or {}
        defects = [d for d in ir.get("defects", []) if d.get("status") != "skipped"]
        visual_review = ir.get("visual_review") or {}
        state = {
            "version": self.version,
            "iteration": iteration,
            "image": {
                "path": str(self.planner.image_path),
                "width": self.planner.original.width,
                "height": self.planner.original.height,
            },
            "result": _summarize_result(result),
            "metrics": copy.deepcopy(ir.get("metrics", {})),
            "defects": {
                "actionable": len(defects),
                "by_type": _count_by(defects, "type"),
                "by_agent": _count_by(defects, "suggested_agent"),
            },
            "raw_defects": copy.deepcopy(defects[:12]),
            "visual_review": {
                "defects": len(visual_review.get("defects") or []),
                "summary": visual_review.get("summary", ""),
                "top_findings": copy.deepcopy((visual_review.get("defects") or [])[:5]),
            },
            "registered_agents": sorted(self.planner.agents.keys()),
            "agent_attempts": copy.deepcopy(getattr(self.planner, "_agent_attempts", {})),
            "artifacts": _known_artifacts(self.planner.out_dir),
            "tool_registry": copy.deepcopy(self.tool_registry),
        }
        return state

    def _accept_if_done(self, result: dict[str, Any] | None, reason: str) -> bool:
        passed = False
        if result is not None:
            passed = bool(result.get("passed"))
        elif self.kernel is not None:
            passed = bool(
                (self.kernel.state.last_verify_result or {}).get("passed")
            )
        if passed and not _has_visual_review_defects(self.planner.ir):
            self.planner.ir["status"] = "accepted"
            self._record("decision", {"action": "accept", "reason": reason})
            if self.kernel is not None:
                self.kernel.transition("accept")
            self._finish()
            return True
        return False

    def _current_verify_result(self) -> dict[str, Any]:
        if self.kernel is not None:
            return self.kernel.state.last_verify_result or {}
        return {}

    def _current_proposal_result(self) -> dict[str, Any]:
        if self.kernel is not None:
            return self.kernel.state.last_proposal_result or {}
        return {}

    def _result_passed(self, result: dict[str, Any] | None) -> bool:
        if result is not None:
            return bool(result.get("passed"))
        return bool(self._current_verify_result().get("passed"))

    def _observe_source(self) -> None:
        original = self.planner.original
        self._record("observation", {
            "image": str(self.planner.image_path),
            "width": original.width,
            "height": original.height,
            "tool_registry": copy.deepcopy(self.tool_registry),
        })

    def _call_tool(self, name: str, fn: Callable[[], Any]) -> Any:
        start = time.time()
        self.log(f"[AuditAgent] tool={name}")
        try:
            result = fn()
            self._record("tool_result", {
                "tool": name,
                "status": "ok",
                "elapsed_sec": round(time.time() - start, 3),
                "summary": _summarize_result(result),
                "artifacts": _known_artifacts(self.planner.out_dir),
            })
            if self.kernel is not None:
                self.kernel.record_transition(
                    operator=name,
                    stage_to=_tool_stage(name),
                    outputs={"summary": _summarize_result(result)},
                    artifact_paths=_artifact_paths(name, self.planner.out_dir),
                )
            return result
        except Exception as exc:
            self._record("tool_result", {
                "tool": name,
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
            if self.kernel is not None:
                self.kernel.record_transition(
                    operator=name,
                    stage_to=_tool_stage(name),
                    outputs={"summary": _summarize_result(None)},
                    error={"type": type(exc).__name__, "message": str(exc)},
                    artifact_paths=_artifact_paths(name, self.planner.out_dir),
                )
            self._write_trace()
            raise

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self.trace.append({
            "index": len(self.trace),
            "kind": kind,
            "t_sec": round(time.time() - self._started, 3),
            **payload,
        })
        self._write_trace()

    def _finish(self) -> None:
        self.planner.ir.setdefault("audit_agent", {})["version"] = self.version
        self.planner.ir["audit_agent"]["trace_path"] = str(
            self.planner.out_dir / "audit_trace.json")
        self.planner._save_ir("ir_final.json")
        self._write_trace()

    def _write_trace(self) -> None:
        self.planner.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "image": str(self.planner.image_path),
            "status": (self.planner.ir or {}).get("status"),
            "events": self.trace,
        }
        (self.planner.out_dir / "audit_trace.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    def _write_agent_state(self, tag: str, state: dict[str, Any]) -> None:
        self.planner.out_dir.mkdir(parents=True, exist_ok=True)
        (self.planner.out_dir / f"audit_state_{tag}.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str))

    def _write_decision(self, tag: str, decision: dict[str, Any]) -> None:
        self.planner.out_dir.mkdir(parents=True, exist_ok=True)
        (self.planner.out_dir / f"audit_decision_{tag}.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False, default=str))


def _needs_region_proposals(metrics: dict, defects: list[dict]) -> bool:
    if float(metrics.get("coverage_explained", 1.0)) < 0.92:
        return True
    if int(metrics.get("critical_defect_count", 0)) >= 3:
        return True
    return any(d.get("type") == "missing_element" for d in defects)


def _defect_summary(defects: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for defect in defects:
        key = str(defect.get("suggested_agent") or defect.get("type") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        out[value] = out.get(value, 0) + 1
    return out


def _summarize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        out = {}
        for key in (
            "passed", "accepted", "status", "metrics", "summary",
            "tasks", "ok", "failed",
        ):
            if key in result:
                out[key] = result[key]
        if "defects" in result:
            out["defects"] = len(result.get("defects") or [])
        return out or {"keys": sorted(str(k) for k in result.keys())[:12]}
    return {"type": type(result).__name__, "value": str(result)[:200]}


def _known_artifacts(out_dir: Path) -> dict[str, bool]:
    names = [
        "perception_blackboard.json",
        "content_tasks.json",
        "task_graph.json",
        "diagnostics.json",
        "visual_review_latest.json",
        "diagram_v3.pptx",
        "diagram_v3.compare.png",
        "ir_final.json",
    ]
    return {name: (out_dir / name).exists() for name in names}


def _tool_stage(name: str) -> str:
    """Map audit tool names to runtime lifecycle stages."""
    mapping = {
        "bootstrap_blackboard": "planning",
        "render_verify_audit": "auditing",
        "proposal_phase": "refining",
        "single_repair": "refining",
        "accept_or_rollback": "refining",
        "component_cleanup": "refining",
        "final_render_verify": "auditing",
    }
    return mapping.get(name, "refining")


def _artifact_paths(name: str, out_dir: Path) -> dict[str, str | None]:
    """Return the artifact paths a tool is expected to write."""
    out_dir = Path(out_dir)
    paths: dict[str, str | None] = {}
    if name == "bootstrap_blackboard":
        paths["perception_blackboard"] = str(out_dir / "perception_blackboard.json")
        paths["strategy_plan"] = str(out_dir / "strategy_plan_processed.json")
        paths["ir_00_plan"] = str(out_dir / "ir_00_plan.json")
    elif name == "render_verify_audit":
        paths["pptx"] = str(out_dir / "diagram_v3.pptx")
        paths["compare"] = str(out_dir / "diagram_v3.compare.png")
        paths["visual_review"] = str(out_dir / "visual_review_latest.json")
    elif name == "proposal_phase":
        paths["task_graph"] = str(out_dir / "task_graph.json")
        paths["proposal_report"] = str(out_dir / "proposal_phase" / "proposal_report.json")
    elif name == "component_cleanup":
        paths["ir_cleanup"] = str(out_dir / "ir_agent_component_cleanup.json")
    return paths
