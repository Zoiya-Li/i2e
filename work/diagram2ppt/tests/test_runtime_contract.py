"""Offline tests for the v3 executable transition contract.

These verify that ImmutableOperator, Transaction, and commit_effects behave as a
deterministic state-machine layer without touching the Planner or network.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from work.diagram2ppt.v3.runtime.contract import (
    NO_EFFECT,
    ImmutableOperator,
    RuntimeState,
    SideEffect,
    Transaction,
    UpdatePlannereffect,
    WriteFileEffect,
    commit_effects,
    state_hash,
)
from work.diagram2ppt.v3.runtime.operators import (
    ImmutableAuditTasksOperator,
    ImmutableTaskGraphOperator,
)


class _DoubleRoundOperator(ImmutableOperator):
    """Trivial pure operator for contract testing."""

    name = "double_round"
    target_stage = "auditing"
    reads = ("round",)
    writes = ("round",)
    idempotent = False

    def run(self, state: RuntimeState, **inputs: Any) -> tuple[RuntimeState, list[SideEffect]]:
        new_state = RuntimeState(
            input_image=state.input_image,
            out_dir=state.out_dir,
            round=state.round * 2,
            stage=self.target_stage,
            ir=state.ir,
        )
        return new_state, [NO_EFFECT]


def test_state_hash_is_stable_for_equal_states():
    s1 = RuntimeState(input_image="a.png", out_dir="/tmp", round=3, stage="planning")
    s2 = RuntimeState(input_image="a.png", out_dir="/tmp", round=3, stage="planning")
    assert state_hash(s1) == state_hash(s2)


def test_state_hash_changes_with_canonical_field():
    s1 = RuntimeState(input_image="a.png", out_dir="/tmp", round=3, stage="planning")
    s2 = RuntimeState(input_image="a.png", out_dir="/tmp", round=4, stage="planning")
    assert state_hash(s1) != state_hash(s2)


def test_transaction_records_hashes():
    state = RuntimeState(input_image="a.png", out_dir="/tmp", round=2)
    op = _DoubleRoundOperator()
    tx = Transaction(op, {"x": 1}, state)
    post = tx.execute()
    assert tx.pre_hash == state_hash(state)
    assert tx.post_hash == state_hash(post)
    assert tx.call_hash == op.transition_hash(state, x=1)
    assert post.round == 4
    assert tx.to_dict()["operator"] == "double_round"


def test_immutable_operator_does_not_mutate_input():
    state = RuntimeState(input_image="a.png", out_dir="/tmp", round=2)
    op = _DoubleRoundOperator()
    new_state, effects = op.run(state)
    assert state.round == 2
    assert new_state.round == 4
    assert effects == [NO_EFFECT]


def test_commit_effects_writes_file():
    with TemporaryDirectory() as td:
        effects = [WriteFileEffect(path="task_graph.json", payload={"tasks": []})]
        summary = commit_effects(effects, planner=None, out_dir=td)
        written = Path(td) / "task_graph.json"
        assert written.exists()
        assert str(written) in summary["files"]


def test_commit_effects_updates_planner_attribute():
    class _Planner:
        def __init__(self) -> None:
            self.ir = {"round": 0}

    planner = _Planner()
    effects = [UpdatePlannereffect(attr="ir", value={"round": 5})]
    summary = commit_effects(effects, planner=planner, out_dir="/tmp")
    assert planner.ir == {"round": 5}
    assert summary["planner_updates"] == ["ir"]


def test_immutable_task_graph_operator_is_pure():
    ir = {
        "version": "d2p-3",
        "round": 1,
        "status": "auditing",
        "elements": [],
        "defects": [],
        "strategy_plan": {
            "regions": [
                {"id": "r1", "kind": "chart", "bbox": [0, 0, 10, 10], "element_ids": []}
            ]
        },
    }
    state = RuntimeState(input_image="a.png", out_dir="/tmp", round=1, ir=ir)
    op = ImmutableTaskGraphOperator()
    new_state, effects = op.run(state)
    assert state.task_graph is None
    assert new_state.task_graph is not None
    assert new_state.task_graph["summary"]["tasks"] == 1
    assert len(effects) == 1
    assert isinstance(effects[0], WriteFileEffect)
    assert effects[0].path == "task_graph.json"


def test_immutable_audit_tasks_operator_is_pure():
    ir = {
        "version": "d2p-3",
        "round": 1,
        "status": "auditing",
        "elements": [{"id": "e1", "type": "text"}],
        "defects": [
            {
                "id": "d1",
                "type": "text_layout_mismatch",
                "element_id": "e1",
                "suggested_agent": "TextLayoutAgent",
                "severity": 0.8,
                "reason": "wrong",
            }
        ],
    }
    components = [{"id": "c1", "element_ids": ["e1"], "provenance": {"region_id": "r1"}}]
    state = RuntimeState(input_image="a.png", out_dir="/tmp", round=1, ir=ir, components=components)
    op = ImmutableAuditTasksOperator()
    new_state, effects = op.run(state)
    assert state.audit_tasks is None
    assert new_state.audit_tasks is not None
    assert len(new_state.audit_tasks) == 1
    assert new_state.audit_tasks[0]["component_id"] == "c1"
    assert len(effects) == 1
    assert effects[0].path == "audit_tasks.json"


def test_immutable_task_graph_operator_idempotent_hash():
    ir = {
        "version": "d2p-3",
        "round": 1,
        "status": "auditing",
        "elements": [],
        "defects": [],
        "strategy_plan": {
            "regions": [
                {"id": "r1", "kind": "chart", "bbox": [0, 0, 10, 10], "element_ids": []}
            ]
        },
    }
    state = RuntimeState(input_image="a.png", out_dir="/tmp", round=1, ir=ir)
    op = ImmutableTaskGraphOperator()
    h1 = op.transition_hash(state)
    h2 = op.transition_hash(state)
    assert h1 == h2


def test_transaction_error_captured_without_mutation():
    class _FailingOperator(ImmutableOperator):
        name = "fail"
        reads = ()

        def run(self, state: RuntimeState, **inputs: Any):
            raise RuntimeError("boom")

    state = RuntimeState(input_image="a.png", out_dir="/tmp")
    tx = Transaction(_FailingOperator(), {}, state)
    with pytest.raises(RuntimeError, match="boom"):
        tx.execute()
    assert tx.error is not None
    assert tx.error["type"] == "RuntimeError"
    assert tx.post_state is None
