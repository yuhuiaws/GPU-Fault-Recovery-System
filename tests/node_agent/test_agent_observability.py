"""What an operator on the node can see of the agent without the control plane.

The executor used to run every recovery action without a single log line, and
``/healthz`` said ``ok`` with the ledger read-only underneath it. These tests
pin the journald lines (structured, secret-free) and the health payload.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import types
from datetime import datetime, timezone
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import (
    AgentHeartbeatRejected,
    AgentHeartbeatReporter,
    NodeActionStatus,
    create_node_agent_app,
    run,
)
from tests._builders import copy_model

from ._support import (
    NOW,
    SECRET,
    FakeRunner,
    command,
    envelope,
    executor,
    result_params,
    submit_action,
    wait_for_result,
)

EXECUTOR_LOGGER = "gpu_fault.node_agent.executor"


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_a_completed_action_logs_accept_start_and_completion(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = executor(tmp_path, FakeRunner())
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))

    with caplog.at_level(logging.INFO, logger=EXECUTOR_LOGGER):
        result = agent.execute(signed)

    assert result.status is NodeActionStatus.SUCCEEDED
    lines = messages(caplog)
    assert len(lines) >= 3, lines
    for line in lines:
        assert "command_id=workflow/step/node-a" in line, line
        assert "incident_id=incident-a" in line, line
        assert "workflow_request_id=workflow-a" in line, line
        assert "operation=VERIFY_NO_GPU_CLIENTS" in line, line
        assert "node_id=node-a" in line, line
        assert "fencing_token=1" in line, line
        assert "attempt=1" in line, line
        assert signed.signature not in line, "signature must never be logged"
        assert SECRET not in line, "secret must never be logged"
    assert any(
        "status=SUCCEEDED" in line and "duration_ms=" in line for line in lines
    ), lines
    assert all(record.levelno == logging.INFO for record in caplog.records), (
        "command lifecycle lines are INFO, not WARNING"
    )


def test_a_failed_action_logs_the_error_class_and_retryability(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenRunner(FakeRunner):
        def __call__(self, value, **kwargs):
            if "--query-compute-apps" in " ".join(value):
                raise OSError("device transport error")
            return super().__call__(value, **kwargs)

    agent = executor(tmp_path, BrokenRunner())
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))

    with caplog.at_level(logging.INFO, logger=EXECUTOR_LOGGER):
        result = agent.execute(signed)

    assert result.status is NodeActionStatus.FAILED
    failure = next(line for line in messages(caplog) if "status=FAILED" in line)
    assert "error_class=OSError" in failure
    assert "retryable=True" in failure
    assert "duration_ms=" in failure


def test_the_agent_entrypoint_configures_root_logging(monkeypatch) -> None:
    """uvicorn configures only its own loggers; without ``configure_logging``
    every INFO line the executor writes is dropped before journald sees it."""

    calls: list[str] = []
    monkeypatch.setattr(
        "gpu_fault.node_agent.app.configure_logging",
        lambda: calls.append("configure_logging"),
    )
    monkeypatch.setattr(
        "gpu_fault.node_agent.app.validate_gpu_fault_environment",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpu_fault.node_agent.app.create_node_agent_app", lambda: object()
    )
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *_args, **_kwargs: calls.append("uvicorn.run")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    for name in (
        "GPU_FAULT_NODE_AGENT_TLS_CERT",
        "GPU_FAULT_NODE_AGENT_TLS_KEY",
        "GPU_FAULT_NODE_AGENT_TLS_CLIENT_CA",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_AGENT_ALLOW_PLAINTEXT", "true")

    run()

    assert calls == ["configure_logging", "uvicorn.run"]


def test_healthz_reports_the_ledger_counters_and_heartbeat(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner())
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        before = client.get("/healthz")
        submit_action(client, signed)
        executed = wait_for_result(client, signed.command.command_id)
        after = client.get("/healthz")

    assert before.status_code == 200
    assert before.json()["status"] == "ok"
    assert before.json()["ledger"] == {"writable": True}
    assert before.json()["heartbeat"] == {"configured": False}
    assert before.json()["counters"] == {
        "accepted": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
    }
    assert executed.status_code == 200, executed.text
    assert after.json()["counters"] == {
        "accepted": 1,
        "completed": 1,
        "failed": 0,
        "rejected": 0,
    }


def test_healthz_counts_rejections_and_failures(tmp_path) -> None:
    class BrokenRunner(FakeRunner):
        def __call__(self, value, **kwargs):
            if "--query-compute-apps" in " ".join(value):
                raise OSError("device transport error")
            return super().__call__(value, **kwargs)

    agent = executor(tmp_path, BrokenRunner())
    valid = command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
    forged = copy_model(envelope(valid), signature="0" * 64)

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        rejected = client.post(
            "/v1/node-actions/submit", json=forged.model_dump(mode="json")
        )
        signed = envelope(valid)
        submit_action(client, signed)
        failed = wait_for_result(client, signed.command.command_id)
        health = client.get("/healthz").json()

    assert rejected.status_code == 401
    assert failed.json()["state"] == "FAILED", failed.text
    assert health["counters"] == {
        "accepted": 1,
        "completed": 0,
        "failed": 1,
        "rejected": 1,
    }


def test_healthz_is_503_when_the_ledger_cannot_be_written(
    tmp_path, monkeypatch
) -> None:
    agent = executor(tmp_path, FakeRunner())

    def unwritable() -> None:
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(agent.ledger, "probe_writable", unwritable)

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["ledger"]["writable"] is False
    assert payload["ledger"]["error"] == "OperationalError"
    assert "readonly" in payload["ledger"]["reason"]


def reporter(sender: Any, now: Any) -> AgentHeartbeatReporter:
    return AgentHeartbeatReporter(
        control_plane_url="https://control.example",
        secret=SECRET,
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="policy-a",
        runtime_profile_version="profile-a",
        config_digest="b" * 64,
        allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
        boot_id="boot-a",
        interval_seconds=5,
        sender=sender,
        now=now,
    )


def run_heartbeats(instance: AgentHeartbeatReporter, rounds: int) -> None:
    attempts = {"count": 0}
    stop = Event()
    original_wait = stop.wait

    def counting_wait(_timeout: float | None = None) -> bool:
        attempts["count"] += 1
        if attempts["count"] >= rounds:
            stop.set()
        return stop.is_set()

    stop.wait = counting_wait  # type: ignore[method-assign]
    try:
        instance.run(stop)
    finally:
        stop.wait = original_wait  # type: ignore[method-assign]


def test_heartbeat_health_tracks_failures_and_last_success_age(tmp_path) -> None:
    clock = {"now": NOW}

    def refuse(_url: str, _envelope: Any) -> dict[str, Any]:
        raise AgentHeartbeatRejected(503, "control plane draining")

    failing = reporter(refuse, lambda: clock["now"])
    run_heartbeats(failing, 2)
    snapshot = failing.health_snapshot(now=clock["now"])

    assert snapshot["consecutive_failures"] == 2
    assert snapshot["last_success_age_seconds"] is None
    assert snapshot["last_success_at"] is None

    succeeding = reporter(lambda _url, _envelope: {"generation": 3}, lambda: NOW)
    run_heartbeats(succeeding, 1)
    later = datetime(2026, 7, 20, 12, 0, 45, tzinfo=timezone.utc)
    snapshot = succeeding.health_snapshot(now=later)

    assert snapshot["consecutive_failures"] == 0
    assert snapshot["last_success_age_seconds"] == 45.0
    assert snapshot["last_success_at"] == NOW.isoformat()

    agent = executor(tmp_path, FakeRunner())
    with TestClient(
        create_node_agent_app(agent, heartbeat_reporter=succeeding)
    ) as client:
        health = client.get("/healthz").json()

    assert health["status"] == "ok"
    assert health["heartbeat"]["configured"] is True
    assert health["heartbeat"]["consecutive_failures"] == 0
    assert health["heartbeat"]["last_success_at"] == NOW.isoformat()


def test_a_pinned_generation_before_the_first_heartbeat_is_a_retryable_409(
    tmp_path,
) -> None:
    """Until a heartbeat succeeds the agent has no generation. A command pinned
    to one must be answered "not yet", not "wrong agent, mint a new command"."""

    agent = executor(tmp_path, FakeRunner())
    pinned = copy_model(
        command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, command_id="pinned"),
        agent_generation=7,
    )

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        response = client.post(
            "/v1/node-actions/submit", json=envelope(pinned).model_dump(mode="json")
        )
        polled = client.get("/v1/node-actions/result", params=result_params("pinned"))

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "AGENT_GENERATION_UNKNOWN",
        "message": "node action agent generation is not known yet",
        "retryable": True,
        "requires_new_command": False,
    }
    assert polled.status_code == 404
