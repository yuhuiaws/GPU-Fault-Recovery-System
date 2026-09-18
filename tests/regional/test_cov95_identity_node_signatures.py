from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_action_keys import derive_node_action_secret
from gpu_fault.node_agent import (
    NodeActionCommand,
    NodeActionExecutor,
    NodeActionLedger,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    create_node_agent_app,
    sign_node_action,
    sign_result_query,
)
from tests.node_agent._support import FakeRunner
from tests.regional._cov95_identity_support import offline_guard as offline_guard


def node_fixture(tmp_path: Path) -> tuple[NodeActionExecutor, FakeRunner, str, str]:
    key_a = derive_node_action_secret("m" * 64, "cluster-a", "node-a")
    key_b = derive_node_action_secret("m" * 64, "cluster-a", "node-b")
    runner = FakeRunner()
    agent = NodeActionExecutor(
        secret=key_a,
        node_ids={"node-a"},
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        reset_enabled=False,
        ledger=NodeActionLedger(str(tmp_path / "unit-node-ledger.db")),
        runner=runner,
        device_client_finder=lambda targets: [],
        gpu_device_path_finder=lambda: {"GPU-a": "/unit/nvidia0"},
        sleep=lambda seconds: None,
    )
    return agent, runner, key_a, key_b


def test_wrong_node_key_rejects_command_before_ledger_or_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, runner, key_a, key_b = node_fixture(tmp_path)
    now = datetime.now(timezone.utc)
    command = NodeActionCommand(
        command_id="unit-verify",
        workflow_request_id="unit-workflow",
        incident_id="unit-incident",
        fencing_token=1,
        operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        node_id="node-a",
        gpu_uuids=["GPU-a"],
        issued_at=now,
        expires_at=now + timedelta(minutes=1),
    )
    ledger_calls = []
    history = agent.ledger.attempt_history
    monkeypatch.setattr(
        agent.ledger,
        "attempt_history",
        lambda identifier: ledger_calls.append(identifier) or history(identifier),
    )
    dispatched = []
    execute = agent.execute
    monkeypatch.setattr(
        agent,
        "execute",
        lambda envelope: dispatched.append(envelope.command.command_id)
        or execute(envelope),
    )
    with TestClient(create_node_agent_app(agent)) as client:
        wrong = SignedNodeAction(
            command=command, signature=sign_node_action(command, key_b)
        )
        denied = client.post(
            "/v1/node-actions/submit", json=wrong.model_dump(mode="json")
        )
        assert denied.status_code == 401
        assert denied.json() == {
            "detail": {
                "code": "INVALID_SIGNATURE",
                "message": "invalid node action signature",
                "retryable": False,
                "requires_new_command": False,
            }
        }
        assert ledger_calls == dispatched == runner.commands == []
        assert history(command.command_id) == []

        correct = SignedNodeAction(
            command=command, signature=sign_node_action(command, key_a)
        )
        # Execute the harmless positive control synchronously so no pool work is
        # left pending when the ASGI test exits.
        result = agent.execute(correct)
        assert result.status is NodeActionStatus.SUCCEEDED
        accepted = client.post(
            "/v1/node-actions/submit", json=correct.model_dump(mode="json")
        )
        assert accepted.status_code == 200
        assert accepted.json()["state"] == "SUCCEEDED"
        assert dispatched == [command.command_id]
        assert len(history(command.command_id)) == 1
        assert runner.commands and all(
            "--gpu-reset" not in call for call in runner.commands
        )
    agent.ledger.close()


def test_wrong_node_key_cannot_read_command_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent, runner, key_a, key_b = node_fixture(tmp_path)
    result = NodeActionResult(
        command_id="unit-known",
        operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        status=NodeActionStatus.SUCCEEDED,
        details={"unit_result": True},
    )
    agent.ledger.save(result)
    reads = []
    getter = agent.ledger.get
    monkeypatch.setattr(
        agent.ledger,
        "get",
        lambda identifier: reads.append(identifier) or getter(identifier),
    )
    issued_at = datetime.now(timezone.utc).isoformat()
    with TestClient(create_node_agent_app(agent)) as client:
        responses = []
        for command_id in ("unit-known", "unit-absent"):
            denied = client.get(
                "/v1/node-actions/result",
                params={
                    "command_id": command_id,
                    "issued_at": issued_at,
                    "signature": sign_result_query(command_id, issued_at, key_b),
                },
            )
            assert denied.status_code == 401
            assert denied.json()["detail"] == {
                "code": "INVALID_SIGNATURE",
                "message": "invalid node action result query signature",
                "retryable": False,
                "requires_new_command": False,
            }
            responses.append(denied.json())
        assert responses[0] == responses[1]
        assert reads == runner.commands == []
        accepted = client.get(
            "/v1/node-actions/result",
            params={
                "command_id": "unit-known",
                "issued_at": issued_at,
                "signature": sign_result_query("unit-known", issued_at, key_a),
            },
        )
        assert accepted.status_code == 200
        assert accepted.json()["result"] == result.model_dump(mode="json")
        assert reads == ["unit-known"]
    assert runner.commands == []
    agent.ledger.close()
