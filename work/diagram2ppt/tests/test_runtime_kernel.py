"""Offline tests for the v3 runtime state-machine kernel (Phase 1).

These verify the kernel skeleton without importing the heavy planner/render
stack or hitting any provider: state serializes, transitions are recorded, and
the kernel writes ``state_log.json``.
"""
from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace

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
