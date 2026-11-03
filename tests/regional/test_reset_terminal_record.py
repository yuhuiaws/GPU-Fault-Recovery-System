"""A reset case keeps its terminal record and observes a restart only after success.

Live 2026-11-03 (GF-REGIONAL-COLLECT-016 D, gpu-eks-cluster): the reset workflow
ended FAILED at RESET_GPU and the product's reset rung rebooted the node, yet
the case error was the restart custody's refusal of the FAILED workflow, and no
file recorded why the reset failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import reset_terminal_record as terminal
from scripts.e2e.regional.regional_commands import RegionalFixtureError

BUSY = (
    "node agent ip-10-0-38-176: ResetProgressError: nvidia-smi exited with "
    "status 255: GPU 00000000:59:00.0: In use by another client (GPU GPU-a, 1 of 1)"
)


def _state(status: str = "FAILED") -> dict[str, Any]:
    executions: list[dict[str, Any]] = [
        {"operation": "VERIFY_NO_GPU_CLIENTS", "status": "SUCCEEDED", "step_index": 4}
    ]
    if status == "FAILED":
        executions.append(
            {
                "operation": "RESET_GPU",
                "status": "FAILED",
                "step_index": 5,
                "error": BUSY,
                "details": {"gpu_reset_busy_attempts": 3},
            }
        )
    return {
        "release_id": "release-a",
        "event": {"event_id": "event-a", "xid": 109},
        "decision": {"disposition": "EXECUTABLE"},
        "incident": {"incident_id": "inc-a", "job_id": "c016-d-1"},
        "workflow": {
            "request_id": "workflow-d",
            "incident_id": "inc-a",
            "status": status,
            "step_executions": executions,
        },
        "commands": [{"command_id": "remote-1", "status": status}],
        "notifications": [],
        "observed_waiting_step_executions": [],
        "queue": {"drained": True},
    }


def _record(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / terminal.TERMINAL_RECORD).read_text())


def test_failed_reset_is_reported_and_the_restart_is_never_observed(
    tmp_path: Path,
) -> None:
    observed: list[dict[str, Any]] = []

    def observe(current: dict[str, Any]) -> dict[str, Any]:
        observed.append(current)
        return {"job_id": "c016-d-1"}

    state = _state()
    with pytest.raises(RegionalFixtureError) as raised:
        terminal.settle_reset_terminal(tmp_path, state, observe)
    message = str(raised.value)
    assert "reset workflow 'workflow-d' ended 'FAILED'" in message
    assert (
        "RESET_GPU (step 5): node agent ip-10-0-38-176: ResetProgressError" in message
    )
    assert "In use by another client" in message
    assert "restart custody adopts only the restart of a SUCCEEDED" in message
    assert observed == [], "the custody is never asked about a failed reset"
    assert not (tmp_path / terminal.POST_RESTART_WORKLOAD).exists(), (
        "no proof is written for a failed reset"
    )
    record = _record(tmp_path)
    assert record["workflow"]["step_executions"][-1]["error"] == BUSY
    assert record["commands"] == [{"command_id": "remote-1", "status": "FAILED"}]
    assert set(record) == set(terminal.TERMINAL_RECORD_KEYS)


def test_succeeded_reset_is_observed_and_both_documents_are_kept(
    tmp_path: Path,
) -> None:
    state = _state("SUCCEEDED")
    proof = {"job_id": "c016-d-1", "pods": [{"uid": "pod-2"}]}
    seen: list[dict[str, Any]] = []

    def observe(current: dict[str, Any]) -> dict[str, Any]:
        seen.append(current)
        assert (tmp_path / terminal.TERMINAL_RECORD).exists(), "record comes first"
        return proof

    assert terminal.settle_reset_terminal(tmp_path, state, observe) == proof
    assert seen == [state]
    assert state["post_restart_workload"] == proof
    assert json.loads((tmp_path / terminal.POST_RESTART_WORKLOAD).read_text()) == proof
    assert _record(tmp_path)["workflow"]["status"] == "SUCCEEDED"


@pytest.mark.parametrize("proof", [{}, None, ["pod"]])
def test_an_empty_post_restart_proof_is_refused_and_never_written(
    tmp_path: Path, proof: Any
) -> None:
    with pytest.raises(RegionalFixtureError, match="post-restart workload proof"):
        terminal.settle_reset_terminal(tmp_path, _state("SUCCEEDED"), lambda _: proof)
    assert not (tmp_path / terminal.POST_RESTART_WORKLOAD).exists(), (
        "a refused proof is never written"
    )
    assert (tmp_path / terminal.TERMINAL_RECORD).exists(), "the record is kept"


def test_a_reset_without_an_observer_keeps_the_record_and_its_old_flow(
    tmp_path: Path,
) -> None:
    state = _state()
    assert terminal.settle_reset_terminal(tmp_path, state, None) is None
    assert "post_restart_workload" not in state
    assert _record(tmp_path)["workflow"]["status"] == "FAILED"


def test_failure_text_names_the_terminal_reason_and_bounds_each_error() -> None:
    state = _state("FAILED")
    state["workflow"]["step_executions"][-1]["error"] = "x" * 2000
    state["workflow"]["terminal_failure_reason"] = "escalated to a node reboot"
    text = terminal.reset_failure(state) or ""
    assert "x" * terminal.FAILURE_TEXT_LIMIT in text
    assert "x" * (terminal.FAILURE_TEXT_LIMIT + 1) not in text
    assert "terminal failure: escalated to a node reboot" in text
    assert terminal.reset_failure(_state("SUCCEEDED")) is None
    blocked = _state("BLOCKED")
    assert terminal.reset_failure(blocked) == (
        "reset workflow 'workflow-d' ended 'BLOCKED' (workflow-terminal.json)"
    )
    assert terminal.reset_failure({}) == (
        "reset workflow None ended None (workflow-terminal.json)"
    )
