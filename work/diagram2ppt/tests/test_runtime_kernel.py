"""Offline tests for the v3 runtime state-machine kernel (Phase 1+).

These verify the kernel skeleton without importing the heavy planner/render
stack or hitting any provider: state serializes, transitions are recorded, and
the kernel writes ``state_log.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from work.diagram2ppt.v3.audit_agent_system import AuditAgentSystem
from work.diagram2ppt.v3.runtime import PlannerKernel, RuntimeState, Transition


def test_transition_to_dict_roundtrip():
    t = Transition.create(
        stage_from="idle",
        stage_to="planning",
        operator="perceive",
        inputs={"image": "foo.png"},
        outputs={"entities": 12},
        artifact_paths={"perception_blackboard": "pb.json"},
    )
    d = t.to_dict()
    assert d["stage_from"] == "idle"
    assert d["stage_to"] == "planning"
    assert d["operator"] == "perceive"
    assert d["inputs"]["image"] == "foo.png"
    assert d["outputs"]["entities"] == 12
    assert d["artifact_paths"]["perception_blackboard"] == "pb.json"
    assert "id" in d and d["id"].startswith("t_")


def test_runtime_state_to_dict_roundtrip():
    s = RuntimeState(
        input_image="foo.png",
        out_dir="/tmp/out",
        stage="planning",
        ir={"version": "d2p-3", "elements": []},
        metrics={"native_fraction": 0.5},
    )
    s.transitions.append(
        Transition.create("idle", "planning", "perceive", outputs={"entities": 5})
    )
    d = s.to_dict()
    assert d["version"] == "runtime-v1"
    assert d["input_image"] == "foo.png"
    assert d["stage"] == "planning"
    assert d["ir"]["version"] == "d2p-3"
    assert d["transitions"][0]["operator"] == "perceive"


def test_runtime_state_write():
    with TemporaryDirectory() as d:
        s = RuntimeState(out_dir=d, stage="accepted")
        s.transitions.append(
            Transition.create("auditing", "accepted", "finalize")
        )
        path = s.write(f"{d}/state_log.json")
        assert path.exists()
        reloaded = json.loads(path.read_text())
        assert reloaded["stage"] == "accepted"
        assert reloaded["transitions"][0]["stage_to"] == "accepted"
        assert "artifacts" in reloaded


def test_planner_kernel_syncs_from_planner():
    planner = SimpleNamespace(
        image_path="/img/framework.png",
        out_dir="/out/v3_out",
        strategy_plan={"regions": []},
        ir={
            "round": 2,
            "status": "auditing",
            "metrics": {"visual_delta": 0.25},
            "defects": [{"id": "d1"}],
            "visual_review": {"defects": []},
            "renderer_mode": "true_powerpoint",
            "run_memory": {"enabled": True},
        },
    )
    kernel = PlannerKernel(planner, config={"max_rounds": 5})
    assert kernel.state.input_image == "/img/framework.png"
    assert kernel.state.out_dir == "/out/v3_out"
    assert kernel.state.round == 2
    assert kernel.state.metrics["visual_delta"] == 0.25
    assert kernel.state.defects[0]["id"] == "d1"
    assert kernel.state.renderer_mode == "true_powerpoint"
    assert kernel.state.config["max_rounds"] == 5


def test_planner_kernel_records_transition():
    planner = SimpleNamespace(
        image_path="/img/framework.png",
        out_dir="/out/v3_out",
        strategy_plan=None,
        ir={"round": 1, "metrics": {"x": 1}, "defects": []},
    )
    kernel = PlannerKernel(planner)
    t = kernel.record_transition(
        operator="render_verify_audit",
        stage_to="auditing",
        outputs={"passed": False},
    )
    assert kernel.state.stage == "auditing"
    assert t.stage_from == "idle"
    assert t.stage_to == "auditing"
    assert kernel.state.transitions[-1] == t


def test_planner_kernel_writes_state_log():
    with TemporaryDirectory() as d:
        out = f"{d}/v3_out"
        planner = SimpleNamespace(
            image_path="/img/framework.png",
            out_dir=out,
            strategy_plan=None,
            ir={"round": 0, "metrics": {}, "defects": []},
        )
        kernel = PlannerKernel(planner)
        kernel.record_transition("perceive", "planning")
        path = kernel.write_state_log(out)
        assert path is not None
        assert path.name == "state_log.json"
        assert path.exists()
        reloaded = json.loads(path.read_text())
        assert reloaded["stage"] == "planning"
        assert reloaded["transitions"][0]["operator"] == "perceive"
        assert "artifacts" in reloaded
        assert reloaded["artifacts"]["ir_final.json"] is False


def test_planner_kernel_set_final_stage():
    planner = SimpleNamespace(
        image_path="/img/framework.png",
        out_dir="/out",
        strategy_plan=None,
        ir={"round": 3, "metrics": {}, "defects": []},
    )
    kernel = PlannerKernel(planner)
    kernel.record_transition("render_verify_audit", "auditing")
    kernel.set_final_stage("accepted", outputs={"passed": True})
    assert kernel.state.stage == "accepted"
    assert kernel.state.transitions[-1].operator == "finalize"
    assert kernel.state.transitions[-1].outputs["passed"] is True


def test_kernel_dispatches_operators():
    calls = []

    class _Planner:
        def __init__(self, out_dir: str) -> None:
            self.image_path = Path(out_dir) / "in.png"
            self.out_dir = Path(out_dir)
            self.original = Image.new("RGB", (100, 100), "white")
            self.ir = None
            self.strategy_plan = None
            self.agents = {}
            self.max_rounds = 5
            self.log = lambda *a: None

        def plan(self):
            self.ir = {
                "version": "d2p-3",
                "round": 0,
                "status": "planning",
                "canvas": {"width_px": 100, "height_px": 100},
                "elements": [],
                "defects": [],
                "metrics": {},
                "strategy_plan": {"regions": []},
            }
            self.strategy_plan = {"regions": []}
            calls.append("plan")
            return self.ir

        def render_and_verify(self):
            self.ir["status"] = "auditing"
            self.ir["metrics"] = {
                "visual_delta": 0.0,
                "coverage_explained": 1.0,
                "critical_defect_count": 0,
            }
            self.ir["defects"] = []
            self.ir["renderer_mode"] = "proxy"
            calls.append("render_and_verify")
            return {"passed": True, "metrics": self.ir["metrics"], "defects": []}

        def _save_ir(self, name: str) -> None:
            (self.out_dir / name).write_text(
                json.dumps(self.ir, indent=2, ensure_ascii=False, default=str)
            )

    with TemporaryDirectory() as td:
        planner = _Planner(td)
        kernel = PlannerKernel(planner, config={"max_rounds": 5})
        final = AuditAgentSystem(planner, log=lambda *_: None, kernel=kernel).run()
        assert final["status"] == "accepted"
        assert "plan" in calls
        assert "render_and_verify" in calls
        state_log = Path(td) / "state_log.json"
        assert state_log.exists()
        log = json.loads(state_log.read_text())
        ops = [t["operator"] for t in log["transitions"]]
        assert "perceive" in ops
        assert "render_verify_audit" in ops
        assert "derive_components" in ops
        assert "audit_tasks" in ops
        assert "svg_loop" in ops
        assert "accept" in ops
        assert log["stage"] == "accepted"


def test_kernel_path_records_verify_and_proposal_results():
    proposal_accepted = {"accepted": 1, "tasks": []}

    class _Planner:
        def __init__(self, out_dir: str) -> None:
            self.image_path = Path(out_dir) / "in.png"
            self.out_dir = Path(out_dir)
            self.original = Image.new("RGB", (100, 100), "white")
            self.ir = None
            self.strategy_plan = {"regions": [{"id": "r1", "kind": "chart", "bbox": [0, 0, 10, 10], "element_ids": []}]}
            self.agents = {}
            self.max_rounds = 5
            self.log = lambda *a: None

        def plan(self):
            self.ir = {
                "version": "d2p-3",
                "round": 0,
                "status": "planning",
                "canvas": {"width_px": 100, "height_px": 100},
                "elements": [],
                "defects": [{"id": "d1", "type": "missing_element", "severity": 0.8}],
                "metrics": {
                    "visual_delta": 0.5,
                    "coverage_explained": 0.9,
                    "critical_defect_count": 1,
                },
                "strategy_plan": self.strategy_plan,
            }
            return self.ir

        def render_and_verify(self):
            defects = self.ir.get("defects", [])
            metrics = self.ir.get("metrics", {})
            passed = (
                int(metrics.get("critical_defect_count", 9999)) == 0
                and float(metrics.get("coverage_explained", 0.0)) >= 0.97
                and not defects
            )
            self.ir["renderer_mode"] = "proxy"
            return {"passed": passed, "metrics": metrics, "defects": defects}

        def run_proposal_phase(self):
            self.ir["status"] = "extracting"
            self.ir["metrics"] = {
                "visual_delta": 0.0,
                "coverage_explained": 1.0,
                "critical_defect_count": 0,
            }
            self.ir["defects"] = []
            return proposal_accepted

        def _next_repair_task(self):
            return None

        def _save_ir(self, name: str) -> None:
            (self.out_dir / name).write_text(
                json.dumps(self.ir, indent=2, ensure_ascii=False, default=str)
            )

    with TemporaryDirectory() as td:
        planner = _Planner(td)
        kernel = PlannerKernel(planner, config={"max_rounds": 5})
        final = AuditAgentSystem(planner, log=lambda *_: None, kernel=kernel).run()
        assert final["status"] == "accepted"
        assert kernel.state.last_verify_result is not None
        assert kernel.state.last_proposal_result == proposal_accepted
        assert Path(td) / "components.json"
        assert Path(td) / "audit_tasks.json"
        assert Path(td) / "svg_loop.json"


def test_kernel_replay_restores_state():
    with TemporaryDirectory() as td:
        planner = SimpleNamespace(
            image_path=Path(td) / "in.png",
            out_dir=td,
            strategy_plan=None,
            ir={"round": 0, "metrics": {}, "defects": []},
        )
        kernel = PlannerKernel(planner)
        kernel.record_transition("perceive", "planning", outputs={"entities": 12})
        kernel.record_transition("render_verify_audit", "auditing")
        kernel.set_final_stage("partial", outputs={"renderer_mode": "proxy"})
        log_path = kernel.write_state_log()
        assert log_path is not None

        fresh_planner = SimpleNamespace(
            image_path=Path(td) / "in.png",
            out_dir=td,
            strategy_plan=None,
            ir=None,
        )
        fresh_kernel = PlannerKernel(fresh_planner)
        restored = fresh_kernel.replay(log_path)
        assert restored.stage == "partial"
        assert restored.renderer_mode == "proxy"
        assert len(restored.transitions) == 3
        assert restored.transitions[0].operator == "perceive"
        assert restored.transitions[2].operator == "finalize"
        assert fresh_planner.ir["round"] == 0


def test_legacy_planner_loop_operator():
    class _Planner:
        def __init__(self, out_dir: str) -> None:
            self.image_path = Path(out_dir) / "in.png"
            self.out_dir = Path(out_dir)
            self.ir = {"status": "idle", "round": 0}
            self.strategy_plan = None

        def run(self):
            self.ir = {"status": "accepted", "round": 1}
            return self.ir

    with TemporaryDirectory() as td:
        planner = _Planner(td)
        kernel = PlannerKernel(planner)
        kernel.transition("legacy_planner_loop")
        assert kernel.state.stage == "finalizing"
        assert kernel.state.ir["status"] == "accepted"
        assert kernel.state.transitions[-1].operator == "legacy_planner_loop"
