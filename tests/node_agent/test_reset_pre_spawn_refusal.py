"""A reset refused before its first spawn is "not attempted", not "failed".

Live (gpu-eks-cluster, DESTR-001 on the previous release): the final ownership
recheck refused the only reset of the command for an ``nvidia-smi`` holder at
``AGENT_PRE_SPAWN``; the record said ``reset_not_attempted=[]`` for a reset
that was never performed and parked as an unknown outcome. The loop now counts
the resets it spawned before a refusal: none spawned is a known "not
attempted" verdict for that GPU; a refusal before the retry of a reset already
refused busy stays "not applied". Driven through the agent's public ``execute``
with the final ownership boundary armed, as the control plane drives it.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.late_ownership import OWNERSHIP_PROTOCOL, OwnershipRefused
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionStatus,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import (
    SECRET,
    FakeRunner,
    Quiesced,
    no_device_clients,
    node_action_executor,
)
from tests.node_agent.test_late_ownership_checkpoint_history import granted
from tests.regional.test_late_ownership_agent_gate import NOW

HOLDER = {
    "gpu_uuid": "GPU-a",
    "pid": "197055",
    "device": "/dev/nvidia0",
    "process_name": "nvidia-smi",
}


def _agent(tmp_path: Path, runner: FakeRunner, *, fabric: bool = False):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "modules").write_text(
        "nvidia_uvm 1507328 0 -\nnvidia 8933376 44 nvidia_uvm\n", encoding="ascii"
    )
    operation = (
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
        if fabric
        else WorkflowOperation.RESET_GPU
    )
    return node_action_executor(
        tmp_path,
        "pre-spawn.db",
        allowed_operations={operation},
        reset_enabled=True,
        fabric_reset_enabled=fabric,
        service_quiesce_enabled=fabric,
        quiesce_manager=Quiesced() if fabric else None,
        now=lambda: NOW,
        runner=runner,
        require_final_ownership=True,
        agent_generation=1,
        proc_root=str(proc),
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {
            "GPU-a": "/dev/nvidia0",
            "GPU-b": "/dev/nvidia1",
            "GPU-c": "/dev/nvidia2",
        },
        device_client_samples=1,
        sleep=lambda _seconds: None,
    )


def _signed(gpu_uuids: list[str], *, fabric: bool = False) -> SignedNodeAction:
    value = NodeActionCommand(
        command_id="pre-spawn/reset",
        workflow_request_id="workflow-reset",
        # The quiesce window stub of the fabric reset knows this incident.
        incident_id="incident-a" if fabric else "incident-reset",
        fencing_token=1,
        operation=(
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            if fabric
            else WorkflowOperation.RESET_GPU
        ),
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=gpu_uuids,
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=90),
        ownership_guard=OWNERSHIP_PROTOCOL,
    )
    return SignedNodeAction(command=value, signature=sign_node_action(value, SECRET))


def _refuse_after(
    agent, grants: int, monkeypatch: pytest.MonkeyPatch, *, expired: bool = False
) -> None:
    """Grant the first ``grants`` ownership checks, refuse every later one --
    for the holder, or (``expired``) as the gate refuses a permit that ran out:
    the boundary's refusal lands before the next spawn."""

    requests = {"count": 0}

    def require(current: NodeActionCommand):
        requests["count"] += 1
        if requests["count"] > grants:
            if expired:
                raise OwnershipRefused("OWNERSHIP_PERMIT_EXPIRED")
            raise OwnershipRefused(
                "OWNERSHIP_FINAL_CLIENTS_CHANGED", device_clients=[HOLDER]
            )
        return granted(current, requests["count"], SECRET)

    monkeypatch.setattr(agent.ownership_gate, "require", require)


def test_a_fabric_reset_refused_before_its_first_spawn_is_not_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESET_ALL_GPUS_NVSWITCHES refused at the boundary before the one fabric
    reset was spawned: the same known verdict RESET_GPU reports, for the whole
    inventory; nothing to confirm, nothing to park on."""

    runner = FakeRunner()
    agent = _agent(tmp_path, runner, fabric=True)
    _refuse_after(agent, 0, monkeypatch, expired=True)

    result = agent.execute(_signed(["GPU-a"], fabric=True))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert result.error == "OwnershipRefused: OWNERSHIP_PERMIT_EXPIRED"
    details = result.details
    assert not any("--gpu-reset" in item for item in runner.commands), (
        "nothing was spawned for the fabric reset"
    )
    assert details["gpu_reset_attempts_spawned"] == 0
    assert details["reset_verdict"] == "not-attempted"
    assert details["reset_scope"] == "ALL_LOCAL_GPUS_AND_NVSWITCHES"
    assert details["reset_not_attempted"] == ["GPU-a"]
    assert details["reset_failed"] == []
    assert details["reset_completed"] == []
    assert details["reset_outcome_unknown"] == []
    assert details["node_action_outcome_known"] is True
    assert details["node_action_not_started"] is True
    assert details["physical_ownership_checks"] == []
    assert "outcome_unknown" not in details
    assert details["manual_confirmation_required"] is True


def test_a_fabric_reset_refused_before_the_retry_of_a_busy_reset_is_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live (gpu-eks-cluster, COLLECT-013, SXID 10003): the fabric reset was
    refused busy, then its retry was refused OWNERSHIP_PERMIT_EXPIRED at the
    boundary. The previous Agent reported it as an unknown outcome; the record
    parked and kept the node's occupancy against the next incident."""

    runner = FakeRunner(reset_error="GPU 00000000:59:00.0: In use by another client")
    agent = _agent(tmp_path, runner, fabric=True)
    _refuse_after(agent, 1, monkeypatch, expired=True)

    result = agent.execute(_signed(["GPU-a"], fabric=True))

    assert result.status is NodeActionStatus.FAILED
    details = result.details
    assert [item for item in runner.commands if "--gpu-reset" in item] == [
        ["nvidia-smi", "--gpu-reset"]
    ], "one fabric reset was spawned and refused busy before the boundary refused"
    assert details["reason"] == "OWNERSHIP_PERMIT_EXPIRED"
    assert details["gpu_reset_attempts_spawned"] == 1
    assert details["reset_verdict"] == "not-applied"
    assert details["reset_failed"] == ["GPU-a"]
    assert details["reset_not_attempted"] == []
    assert details["reset_completed"] == []
    assert details["node_action_outcome_known"] is True
    assert "node_action_not_started" not in details, "a reset was spawned once"
    assert "outcome_unknown" not in details
    assert len(details["physical_ownership_checks"]) == 1
    assert details["physical_calls"] == [
        {"command": "nvidia-smi --gpu-reset", "attempt": 1, "outcome": "busy"},
        {
            "command": "nvidia-smi --gpu-reset",
            "attempt": 2,
            "outcome": "refused: OWNERSHIP_PERMIT_EXPIRED",
        },
    ]


def test_a_refusal_before_the_first_spawn_is_a_not_attempted_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRunner()
    agent = _agent(tmp_path, runner)
    _refuse_after(agent, 0, monkeypatch)

    result = agent.execute(_signed(["GPU-a"]))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    details = result.details
    assert not any("--gpu-reset" in item for item in runner.commands), (
        "nothing was spawned for the GPU"
    )
    assert details["gpu_reset_attempts_spawned"] == 0
    assert details["reset_verdict"] == "not-attempted"
    assert details["reset_not_attempted"] == ["GPU-a"]
    assert details["reset_failed"] == []
    assert details["reset_completed"] == []
    assert details["reset_outcome_unknown"] == []
    assert details["node_action_outcome_known"] is True
    assert details["node_action_not_started"] is True
    assert details["physical_ownership_checks"] == []
    assert "outcome_unknown" not in details
    assert details["persistent_device_clients"][0]["pid"] == "197055"
    assert "GPU-a" in details["gpu_readback"], "the GPU is still read back as evidence"


def test_a_refusal_before_the_retry_of_a_busy_reset_stays_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRunner(reset_error="GPU 00000000:59:00.0: In use by another client")
    agent = _agent(tmp_path, runner)
    _refuse_after(agent, 1, monkeypatch)

    result = agent.execute(_signed(["GPU-a"]))

    assert result.status is NodeActionStatus.FAILED
    details = result.details
    assert [item for item in runner.commands if "--gpu-reset" in item] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
    ], "one reset was spawned and refused busy before the boundary refused"
    assert details["gpu_reset_attempts_spawned"] == 1
    assert details["reset_verdict"] == "not-applied"
    assert details["reset_failed"] == ["GPU-a"]
    assert details["reset_not_attempted"] == []
    assert details["node_action_outcome_known"] is True
    assert "node_action_not_started" not in details, "a reset was spawned once"
    assert "outcome_unknown" not in details
    # What each checkpoint led to, in checkpoint order: the first spawned a
    # reset the driver refused busy, the second was refused before its retry
    # (live the journal showed two granted checkpoints and said nothing).
    assert details["physical_calls"] == [
        {"command": "nvidia-smi --gpu-reset -i GPU-a", "attempt": 1, "outcome": "busy"},
        {
            "command": "nvidia-smi --gpu-reset -i GPU-a",
            "attempt": 2,
            "outcome": "refused: OWNERSHIP_FINAL_CLIENTS_CHANGED",
        },
    ]


def test_the_untouched_gpus_follow_the_refused_one_into_not_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPU-a is reset; GPU-b's first spawn is refused: GPU-b and GPU-c were
    never attempted, GPU-a stays completed, and the command did start."""

    runner = FakeRunner()
    agent = _agent(tmp_path, runner)
    _refuse_after(agent, 1, monkeypatch)

    result = agent.execute(_signed(["GPU-a", "GPU-b", "GPU-c"]))

    assert result.status is NodeActionStatus.FAILED
    details = result.details
    assert [item for item in runner.commands if "--gpu-reset" in item] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
    ]
    assert details["reset_completed"] == ["GPU-a"]
    assert details["reset_failed"] == []
    assert details["reset_not_attempted"] == ["GPU-b", "GPU-c"]
    assert details["reset_verdict"] == "not-attempted"
    assert "node_action_not_started" not in details, "GPU-a was reset"
    assert details["node_action_outcome_known"] is True
