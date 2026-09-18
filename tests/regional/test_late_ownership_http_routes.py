from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import app as app_module
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    OwnershipChallenge,
    OwnershipRefused,
    command_identity,
    sign_permit,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import (
    SECRET,
    FakeRunner,
    no_device_clients,
    node_action_executor,
)


class ImmediatePool:
    def __init__(self, **kwargs):
        pass

    def submit(self, method, argument):
        future = Future()
        try:
            future.set_result(method(argument))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


def request(guarded=True):
    now = datetime.now(timezone.utc)
    command = NodeActionCommand(
        command_id="owned-route",
        workflow_request_id="owned-workflow",
        incident_id="owned-incident",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU
        if guarded
        else WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=["GPU-a"],
        issued_at=now,
        expires_at=now + timedelta(seconds=90),
        ownership_guard=OWNERSHIP_PROTOCOL if guarded else None,
    )
    challenge = OwnershipChallenge(
        command_id=command.command_id,
        workflow_id=command.workflow_request_id,
        incident_id=command.incident_id,
        node_id="node-a",
        boot_id="owned-boot",
        agent_generation=1,
        fencing_token=1,
        command_sha256=command_identity(command),
        nonce="a" * 64,
        sequence=1,
        boundary="AGENT_PRE_SPAWN",
        expires_at=command.expires_at,
    )
    permit = sign_permit(challenge, SECRET, allowed=True, reason="OK", now=now)
    return SignedNodeAction(
        command=command,
        signature=sign_node_action(command, SECRET),
        ownership_permit=permit,
    )


def agent_for(tmp_path, guarded=True):
    hardware = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "route-ledger.db",
        allowed_operations={
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        },
        reset_enabled=True,
        now=None,
        require_final_ownership=guarded,
        agent_generation=1,
        runner=hardware,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
    )
    return agent, hardware


@pytest.mark.parametrize(
    "defect", ["legacy", "command-id", "command-digest", "missing-state"]
)
def test_public_permit_route_refuses_unknown_scope_without_queueing_hardware(
    tmp_path, monkeypatch, defect
):
    guarded = defect != "legacy"
    agent, hardware = agent_for(tmp_path, guarded)
    envelope = request(guarded)
    if defect in {"command-id", "command-digest"}:
        field = "command_id" if defect == "command-id" else "command_sha256"
        changed = envelope.ownership_permit.challenge.model_copy(
            update={field: "foreign" if field == "command_id" else "f" * 64}
        )
        envelope = envelope.model_copy(
            update={
                "ownership_permit": sign_permit(
                    changed,
                    SECRET,
                    allowed=True,
                    reason="OK",
                    now=datetime.now(timezone.utc),
                )
            }
        )
    elif defect == "missing-state":
        monkeypatch.setattr(agent.ownership_gate, "authorize", lambda permit: None)
    with TestClient(app_module.create_node_agent_app(agent)) as client:
        response = client.post(
            "/v1/node-actions/submit", json=envelope.model_dump(mode="json")
        )
    assert response.status_code == 409
    details = response.json()["detail"]
    assert details["code"] == "OWNERSHIP_REFUSED"
    assert details["retryable"] is False
    assert hardware.commands == []
    assert agent.ledger.get(envelope.command.command_id) is None


@pytest.mark.parametrize("failure", ["ownership", "generic-guarded", "generic-legacy"])
def test_public_route_queue_failures_preserve_manual_refusal_and_legacy_retry(
    tmp_path, monkeypatch, failure
):
    guarded = failure != "generic-legacy"
    agent, hardware = agent_for(tmp_path, guarded)
    envelope = request(guarded).model_copy(update={"ownership_permit": None})
    error = (
        OwnershipRefused("OWNERSHIP_FINAL_FENCE_CHANGED")
        if failure == "ownership"
        else RuntimeError("owned fake queue failure")
    )

    def fail(value):
        raise error

    monkeypatch.setattr(agent, "execute", fail)
    monkeypatch.setattr(app_module, "ThreadPoolExecutor", ImmediatePool)
    with TestClient(app_module.create_node_agent_app(agent)) as client:
        response = client.post(
            "/v1/node-actions/submit", json=envelope.model_dump(mode="json")
        )
    assert response.status_code == 200
    saved = agent.ledger.get(envelope.command.command_id)
    assert saved is not None and saved.status.value == "FAILED"
    assert saved.retryable is (not guarded)
    if guarded:
        assert (
            saved.details["manual_confirmation_required"]
            and saved.details["safety_rejection"]
        )
    else:
        assert saved.details == {}
    assert hardware.commands == []
