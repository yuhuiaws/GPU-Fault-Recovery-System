from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from gpu_fault.node_agent import app as node_app
from tests.node_agent._cov95_runtime_app import ActionPool
from tests.node_agent._cov95_runtime_app import (
    action_pool_fixture as action_pool_fixture,
)
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import (
    NOW,
    command,
    envelope,
    result_params,
    submit_action,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS


def test_stale_fence_is_refused_before_handler_execution_after_async_submission(
    node_factory: Callable[..., NodeActionExecutor], action_pool: ActionPool
) -> None:
    agent = node_factory(allowed_operations={VERIFY})
    latest = command(VERIFY, command_id="unit-newer-fence", fencing_token=2)
    assert agent.execute(envelope(latest)).status is NodeActionStatus.SUCCEEDED
    calls = list(agent.runner.commands)
    stale = command(VERIFY, command_id="unit-old-fence", fencing_token=1)
    with TestClient(node_app.create_node_agent_app(agent)) as client:
        response = submit_action(client, envelope(stale))
        assert response.status_code == 200
        assert response.json()["state"] == "PENDING"
        action_pool.actions[0].complete()
        refused = client.get(
            "/v1/node-actions/result", params=result_params(stale.command_id)
        )
        action_pool.actions[0].notify()
    assert refused.status_code == 404
    assert len(action_pool.actions) == 1
    assert agent.runner.commands == calls
    assert agent.ledger.attempt_history(stale.command_id) == []
    assert len(agent.ledger.attempt_history(latest.command_id)) == 1
    assert agent.counters_snapshot()["completed"] == 1


@pytest.mark.parametrize(
    ("defect", "status", "code", "retryable", "new_command"),
    [
        ("signature", 401, "INVALID_SIGNATURE", False, False),
        ("node", 422, "TARGET_NODE_MISMATCH", False, False),
        ("generation-unknown", 409, "AGENT_GENERATION_UNKNOWN", True, False),
        ("generation-stale", 409, "STALE_AGENT_GENERATION", True, True),
        ("operation", 403, "OPERATION_NOT_ALLOWED", False, False),
        ("future", 422, "INVALID_ISSUED_AT", False, True),
        ("expired", 410, "COMMAND_EXPIRED", True, True),
        ("ttl", 422, "INVALID_TTL", False, True),
    ],
)
def test_invalid_submissions_are_classified_before_pool_or_ledger_dispatch(
    node_factory: Callable[..., NodeActionExecutor],
    action_pool: ActionPool,
    defect: str,
    status: int,
    code: str,
    retryable: bool,
    new_command: bool,
) -> None:
    agent = node_factory(allowed_operations={VERIFY}, agent_generation=2)
    changes: dict[str, Any] = {}
    if defect == "node":
        changes["node_id"] = "node-b"
    elif defect.startswith("generation-"):
        changes["agent_generation"] = 1
        if defect == "generation-unknown":
            agent.agent_generation = None
    elif defect == "operation":
        changes["operation"] = WorkflowOperation.RESET_GPU
    elif defect == "future":
        changes["issued_at"] = NOW + timedelta(seconds=31)
    elif defect == "expired":
        changes["expires_at"] = NOW
    elif defect == "ttl":
        changes["expires_at"] = NOW + timedelta(minutes=6)
    signed = envelope(command(VERIFY).model_copy(update=changes))
    if defect == "signature":
        signed = signed.model_copy(update={"signature": "0" * 64})
    with TestClient(node_app.create_node_agent_app(agent)) as client:
        response = submit_action(client, signed)
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["retryable"] is retryable
    assert detail["requires_new_command"] is new_command
    assert action_pool.actions == []
    assert agent.ledger.attempt_history(signed.command.command_id) == []
    assert agent.counters_snapshot()["rejected"] == 1


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("release", "bound to the current"),
        ("invalid", "ISO-8601"),
        ("naive", "include UTC"),
        ("expired", "has expired"),
        ("too-long", "cannot exceed 14 days"),
    ],
)
def test_unsigned_result_migration_is_release_bound_and_time_limited(
    monkeypatch: pytest.MonkeyPatch,
    node_factory: Callable[..., NodeActionExecutor],
    action_pool: ActionPool,
    defect: str,
    message: str,
) -> None:
    now = datetime.now(timezone.utc)
    expiries = {
        "invalid": "not-a-timestamp",
        "naive": (now + timedelta(days=1)).replace(tzinfo=None).isoformat(),
        "expired": (now - timedelta(seconds=1)).isoformat(),
        "too-long": (now + timedelta(days=15)).isoformat(),
    }
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED", "false")
    monkeypatch.setenv("GPU_FAULT_RELEASE_ID", "runtime-unit-release")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_RELEASE_ID",
        "other-release" if defect == "release" else "runtime-unit-release",
    )
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_EXPIRES_AT",
        expiries.get(defect, (now + timedelta(days=1)).isoformat()),
    )
    with pytest.raises(RuntimeError, match=message):
        node_app.create_node_agent_app(node_factory())
    assert action_pool.actions == []


def test_valid_migration_does_not_accept_an_invalid_provided_signature(
    monkeypatch: pytest.MonkeyPatch,
    node_factory: Callable[..., NodeActionExecutor],
    action_pool: ActionPool,
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED", "false")
    monkeypatch.setenv("GPU_FAULT_RELEASE_ID", "runtime-unit-release")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_RELEASE_ID",
        "runtime-unit-release",
    )
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_EXPIRES_AT",
        (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )
    with TestClient(node_app.create_node_agent_app(node_factory())) as client:
        unknown = client.get(
            "/v1/node-actions/result", params={"command_id": "unknown"}
        )
        forged = client.get(
            "/v1/node-actions/result",
            params={**result_params("unknown"), "signature": "0" * 64},
        )
    assert unknown.status_code == 404
    assert forged.status_code == 401
    assert action_pool.actions == []


def test_pending_duplicates_execute_once_and_reused_targets_are_refused(
    node_factory: Callable[..., NodeActionExecutor], action_pool: ActionPool
) -> None:
    agent = node_factory(allowed_operations={VERIFY})
    signed = envelope(command(VERIFY))
    with TestClient(node_app.create_node_agent_app(agent)) as client:
        first = submit_action(client, signed)
        duplicate = submit_action(client, signed)
        assert first.json()["state"] == duplicate.json()["state"] == "PENDING"
        assert len(action_pool.actions) == 1
        action_pool.actions[0].complete()
        completed = client.get(
            "/v1/node-actions/result", params=result_params(signed.command.command_id)
        )
        action_pool.actions[0].notify()
        replay = submit_action(client, signed)
        changed = submit_action(
            client, envelope(signed.command.model_copy(update={"gpu_uuids": ["GPU-b"]}))
        )
    assert completed.json()["state"] == replay.json()["state"] == "SUCCEEDED"
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "COMMAND_ID_REUSED"
    assert changed.json()["detail"]["requires_new_command"] is False
    assert agent.counters_snapshot()["completed"] == 1
    assert len(agent.ledger.attempt_history(signed.command.command_id)) == 1
    assert action_pool.shutdowns == [{"wait": False, "cancel_futures": False}]


def test_delayed_old_callback_cannot_drop_the_new_retry(
    monkeypatch: pytest.MonkeyPatch,
    node_factory: Callable[..., NodeActionExecutor],
    action_pool: ActionPool,
) -> None:
    agent = node_factory(allowed_operations={VERIFY})
    mark = agent.ledger.mark_in_progress
    attempts = []

    def fail_once(*args: Any, **kwargs: Any) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("synthetic ledger temporarily busy")
        mark(*args, **kwargs)

    monkeypatch.setattr(agent.ledger, "mark_in_progress", fail_once)
    signed = envelope(command(VERIFY))
    params = result_params(signed.command.command_id)
    with TestClient(node_app.create_node_agent_app(agent)) as client:
        assert submit_action(client, signed).json()["state"] == "PENDING"
        first = action_pool.actions[0]
        first.complete()
        failed = client.get("/v1/node-actions/result", params=params)
        assert failed.json()["result"]["attempt"] == 1
        assert failed.json()["result"]["retryable"] is True
        assert submit_action(client, signed).json()["state"] == "PENDING"
        first.notify()
        assert client.get("/v1/node-actions/result", params=params).json() == {
            "command_id": signed.command.command_id,
            "state": "PENDING",
            "result": None,
        }
        action_pool.actions[1].complete()
        action_pool.actions[1].notify()
        complete = client.get("/v1/node-actions/result", params=params)
    assert complete.json()["state"] == "SUCCEEDED"
    assert complete.json()["result"]["attempt"] == 2
    assert len(action_pool.actions) == 2
    assert agent.counters_snapshot()["completed"] == 1
    assert [
        row["attempt"]
        for row in agent.ledger.attempt_history(signed.command.command_id)
    ] == [1, 2]


def test_pool_rejection_leaves_no_result_and_allows_a_fresh_signed_command(
    node_factory: Callable[..., NodeActionExecutor], action_pool: ActionPool
) -> None:
    clock = [NOW]
    agent = node_factory(allowed_operations={VERIFY}, now=lambda: clock[0])
    signed = envelope(command(VERIFY))
    params = result_params(signed.command.command_id)
    with TestClient(node_app.create_node_agent_app(agent)) as client:
        assert submit_action(client, signed).json()["state"] == "PENDING"
        clock[0] += timedelta(minutes=3)
        action_pool.actions[0].complete()
        assert client.get("/v1/node-actions/result", params=params).status_code == 404
        action_pool.actions[0].notify()
        assert agent.ledger.attempt_history(signed.command.command_id) == []
        refreshed = envelope(
            signed.command.model_copy(
                update={
                    "issued_at": clock[0],
                    "expires_at": clock[0] + timedelta(minutes=1),
                }
            )
        )
        assert submit_action(client, refreshed).json()["state"] == "PENDING"
        action_pool.actions[1].complete()
        action_pool.actions[1].notify()
        result = client.get("/v1/node-actions/result", params=params)
    assert result.json()["state"] == "SUCCEEDED"
    assert result.json()["result"]["attempt"] == 1
    assert agent.counters_snapshot()["completed"] == 1


@pytest.mark.parametrize("break_write", [False, True])
def test_unpersisted_pool_failure_is_interrupted_never_a_retryable_action(
    monkeypatch: pytest.MonkeyPatch,
    node_factory: Callable[..., NodeActionExecutor],
    action_pool: ActionPool,
    break_write: bool,
) -> None:
    agent = node_factory(allowed_operations={VERIFY})
    signed = envelope(command(VERIFY))

    def failed(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic node ledger failure")

    with TestClient(node_app.create_node_agent_app(agent)) as client:
        assert submit_action(client, signed).json()["state"] == "PENDING"
        agent.ledger.mark_in_progress(signed.command, 1)
        if break_write:
            monkeypatch.setattr(agent.ledger, "mark_interrupted", failed)
        action_pool.actions[0].set_exception(OSError("synthetic wrapper failure"))
        response = client.get(
            "/v1/node-actions/result", params=result_params(signed.command.command_id)
        )
        action_pool.actions[0].notify()
    assert response.json()["state"] == "INTERRUPTED"
    assert response.json()["result"]["retryable"] is False
    assert response.json()["result"]["attempt"] == 1
    replay = agent.execute(signed)
    assert replay.status is NodeActionStatus.INTERRUPTED
    assert replay.retryable is False
    assert agent.counters_snapshot()["accepted"] == 0


def test_failed_boot_reconcile_keeps_api_alive_but_cannot_authorize_a_reset(
    node_factory: Callable[..., NodeActionExecutor],
    action_pool: ActionPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class QuiesceFailure:
        def reconcile_after_boot(self) -> dict:
            raise OSError("synthetic unreadable quiesce state")

    agent = node_factory(quiesce_manager=QuiesceFailure())
    with TestClient(node_app.create_node_agent_app(agent)) as client:
        assert client.get("/healthz").status_code == 200
        response = submit_action(client, envelope(command(WorkflowOperation.RESET_GPU)))
    assert response.status_code == 403
    assert action_pool.actions == []
    assert "reconcile after boot failed" in caplog.text
