"""Join and removal state records when each step completed."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_fault.admin.cluster_join_readiness import evaluate_join_readiness
from gpu_fault.admin.cluster_join_state import complete_step


def test_each_completed_step_keeps_its_own_timestamp(tmp_path: Path) -> None:
    """``updated_at`` only says when the record last moved; an operator asking
    where a join spent its time needs one timestamp per step."""

    path = tmp_path / "state.json"
    state: dict[str, object] = {"completed_steps": [], "evidence": {}}

    complete_step(path, state, "DISCOVERED", {"target": {}})
    first = dict(state["step_completed_at"])  # type: ignore[call-overload]
    complete_step(path, state, "PRECHECKED")

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(stored["step_completed_at"]) == ["DISCOVERED", "PRECHECKED"], (
        "every completed step must carry a timestamp"
    )
    assert stored["step_completed_at"]["DISCOVERED"] == first["DISCOVERED"], (
        "completing a later step must not rewrite an earlier step's timestamp"
    )
    assert stored["updated_at"] == stored["step_completed_at"]["PRECHECKED"], (
        "the record's updated_at is the latest step's timestamp"
    )


def test_the_readiness_gate_records_what_it_waited_for_and_deferred(
    tmp_path: Path,
) -> None:
    """An operator reading ``step_completed_at.COLLECTORS_READY`` also needs to
    see which collector kinds that timestamp covers and which it left to the
    verify/status path."""

    collectors = {
        "GPU_INVENTORY": {
            "ready": True,
            "last_success_at": "t",
            "unit_state": "active",
        },
        "GPU_METRICS": {
            "ready": False,
            "last_success_at": None,
            "unit_state": "active",
        },
    }
    verdict = evaluate_join_readiness(
        {"nodes": [{"node_id": "node-1", "collectors": collectors}]},
        expected_nodes=["node-1"],
        fleet={"ready": True, "nodes": []},
    )
    path = tmp_path / "state.json"
    state: dict[str, object] = {"completed_steps": [], "evidence": {}}

    complete_step(path, state, "COLLECTORS_READY", verdict.evidence())

    stored = json.loads(path.read_text(encoding="utf-8"))
    evidence = stored["evidence"]["COLLECTORS_READY"]
    assert stored["step_completed_at"]["COLLECTORS_READY"] == stored["updated_at"]
    assert evidence["ready"] is True
    assert "GPU_INVENTORY" in evidence["waited_kinds"]
    assert evidence["deferred_kinds"]["GPU_METRICS"]["verified_as"] == "scheduled"
    assert evidence["deferred_kinds"]["GPU_METRICS"]["reported_nodes"] == 0
    assert evidence["deferred_kinds"]["GPU_METRICS"]["scheduled_nodes"] == 1
