from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    OwnershipChallenge,
    OwnershipPermit,
    OwnershipRefused,
    command_identity,
    execute_with_final_ownership,
    final_ownership_boundary,
    require_physical_ownership,
    sign_permit,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionStatus,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import SECRET, FakeRunner, no_device_clients
from tests.node_agent._support import node_action_executor as executor_fixture
from tests.regional.test_late_ownership_agent_gate import KEY, NOW
from tests.regional.test_late_ownership_agent_gate import command as command_fixture
from tests.regional.test_late_ownership_agent_gate_edges import FakeNodeAgent
from tests.regional.test_late_ownership_agent_gate_edges import (
    owned_agent_io as owned_agent_io,
)

command: Callable[..., NodeActionCommand] = command_fixture
node_executor: Callable[..., NodeActionExecutor] = executor_fixture


def granted(value: NodeActionCommand, sequence: int, secret: str) -> OwnershipPermit:
    challenge = OwnershipChallenge(
        command_id=value.command_id,
        workflow_id=value.workflow_request_id,
        incident_id=value.incident_id,
        node_id=value.node_id,
        agent_generation=value.agent_generation or 1,
        fencing_token=value.fencing_token,
        command_sha256=command_identity(value),
        boot_id="owned-test-boot",
        nonce=f"{sequence:064x}",
        sequence=sequence,
        boundary="AGENT_PRE_SPAWN",
        expires_at=value.expires_at,
    )
    return sign_permit(challenge, secret, allowed=True, reason="OK", now=NOW)


@pytest.mark.parametrize("prior", [0, 1, 2])
@pytest.mark.parametrize(
    "operation",
    [
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.REMEDIATE_DRIVER,
        WorkflowOperation.REMEDIATE_EFA_DRIVER,
    ],
)
def test_refusal_retains_prior_checkpoints_without_claiming_physical_completion(
    monkeypatch: pytest.MonkeyPatch, prior: int, operation: WorkflowOperation
) -> None:
    agent = FakeNodeAgent()
    value = command(operation=operation)
    requests = 0
    effects: list[int] = []

    def require(current: NodeActionCommand) -> OwnershipPermit:
        nonlocal requests
        requests += 1
        if requests > prior:
            raise OwnershipRefused("OWNED_LATER_REFUSAL")
        return granted(current, requests, KEY)

    @final_ownership_boundary
    def handler(current: NodeActionCommand) -> dict[str, Any]:
        for index in range(prior + 1):
            require_physical_ownership()
            effects.append(index)
        raise AssertionError("the refused checkpoint must prevent handler completion")

    monkeypatch.setattr(agent.ownership_gate, "require", require)
    with pytest.raises(OwnershipRefused, match="OWNED_LATER_REFUSAL") as caught:
        execute_with_final_ownership(
            agent, SignedNodeAction(command=value, signature="local-test"), handler
        )

    details = caught.value.action_details
    assert details["reason"] == "OWNED_LATER_REFUSAL"
    assert details["node_action_command_id"] == value.command_id
    assert [item["sequence"] for item in details["physical_ownership_checks"]] == list(
        range(1, prior + 1)
    )
    assert effects == list(range(prior))
    assert "physical_completed" not in details
    if prior:
        assert "node_action_not_started" not in details
        assert details["outcome_unknown"] is True
        assert details["manual_confirmation_required"] is True
    else:
        assert details["node_action_not_started"] is True
        assert "outcome_unknown" not in details
    require_physical_ownership()
    assert requests == prior + 1


@pytest.mark.parametrize("boundary", ["local-fence", "permit-expiry", "delivery"])
def test_later_local_guard_failures_retain_earlier_permission_receipts(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    agent = FakeNodeAgent()
    value = command()
    requests = 0

    def require(current: NodeActionCommand) -> OwnershipPermit:
        nonlocal requests
        requests += 1
        if requests == 2:
            if boundary == "local-fence":
                agent.ledger.accept_fencing = lambda *args: False
            elif boundary == "permit-expiry":
                agent.now = lambda: NOW + timedelta(seconds=4)
        return granted(current, requests, KEY)

    def delivery(permit: OwnershipPermit) -> None:
        if boundary == "delivery" and permit.challenge.sequence == 2:
            raise OwnershipRefused("OWNERSHIP_CALLER_LOST")

    @final_ownership_boundary
    def handler(current: NodeActionCommand) -> dict[str, Any]:
        require_physical_ownership()
        require_physical_ownership()
        raise AssertionError("the second local checkpoint must refuse")

    monkeypatch.setattr(agent.ownership_gate, "require", require)
    monkeypatch.setattr(OwnershipPermit, "validate_delivery", delivery)
    with pytest.raises(OwnershipRefused) as caught:
        execute_with_final_ownership(
            agent, SignedNodeAction(command=value, signature="local-test"), handler
        )

    details = caught.value.action_details
    assert details["outcome_unknown"] is True
    assert "node_action_not_started" not in details
    assert [item["sequence"] for item in details["physical_ownership_checks"]] == [1]


@pytest.mark.parametrize("kind", ["first-refusal", "busy-retry", "second-gpu"])
def test_real_reset_wrapper_and_ledger_preserve_checkpoint_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hardware = FakeRunner(
        reset_error="In use by another client" if kind == "busy-retry" else None
    )
    agent = node_executor(
        tmp_path,
        "checkpoint-history.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        now=lambda: NOW,
        runner=hardware,
        require_final_ownership=True,
        agent_generation=1,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {
            "GPU-a": "/dev/nvidia0",
            "GPU-b": "/dev/nvidia1",
        },
        device_client_samples=1,
        sleep=lambda seconds: None,
    )
    value = NodeActionCommand(
        command_id="owned-checkpoint/reset",
        workflow_request_id="owned-workflow",
        incident_id="owned-incident",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=["GPU-a", "GPU-b"] if kind == "second-gpu" else ["GPU-a"],
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=90),
        ownership_guard=OWNERSHIP_PROTOCOL,
    )
    requests = 0

    def require(current: NodeActionCommand) -> OwnershipPermit:
        nonlocal requests
        requests += 1
        if kind == "first-refusal" or requests > 1:
            raise OwnershipRefused("OWNED_LATER_REFUSAL")
        return granted(current, requests, SECRET)

    monkeypatch.setattr(agent.ownership_gate, "require", require)
    signed = SignedNodeAction(command=value, signature=sign_node_action(value, SECRET))

    result = agent.execute(signed)
    replayed = agent.execute(signed)

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert replayed == result
    assert agent.ledger.get(value.command_id) == result
    assert result.details["reset_completed"] == (
        ["GPU-a"] if kind == "second-gpu" else []
    )
    resets = [item for item in hardware.commands if "--gpu-reset" in item]
    if kind == "first-refusal":
        assert resets == []
        assert result.details["node_action_not_started"] is True
        assert result.details["physical_ownership_checks"] == []
        assert "outcome_unknown" not in result.details
        assert requests == 1
    else:
        assert resets == [["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]]
        assert "node_action_not_started" not in result.details
        assert result.details["outcome_unknown"] is True
        assert result.details["manual_confirmation_required"] is True
        assert [
            item["sequence"] for item in result.details["physical_ownership_checks"]
        ] == [1]
        assert requests == 2
