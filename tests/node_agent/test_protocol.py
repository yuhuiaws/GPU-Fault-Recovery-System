from __future__ import annotations

import gpu_fault.node_agent.executor as executor_module
from tests._builders import copy_model, node_action_result

from ._support import (
    NOW,
    OPERATION_HANDLERS,
    Event,
    FakeRunner,
    NodeActionExecutionState,
    NodeActionExecutor,
    NodeActionLedger,
    NodeActionStatus,
    SignedNodeAction,
    TestClient,
    ThreadPoolExecutor,
    WorkflowOperation,
    command,
    create_node_agent_app,
    datetime,
    envelope,
    executor,
    no_device_clients,
    node_action_executor,
    os,
    print_config_digest,
    pytest,
    result_params,
    timedelta,
    timezone,
)

EXPECTED_NODE_ACTION_OPERATIONS = {
    WorkflowOperation.COLLECT_HUNG_TRIAGE,
    WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
    WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
    WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
    WorkflowOperation.RUN_NVLINK74_WORKFLOW,
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    WorkflowOperation.RESTORE_GPU_SERVICES,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.RESTART_FABRIC_MANAGER,
}


def test_node_action_registry_matches_protocol_golden() -> None:
    assert set(OPERATION_HANDLERS) == EXPECTED_NODE_ACTION_OPERATIONS


def test_node_agent_rejects_invalid_signature_and_expiry(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner())
    value = command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)

    with pytest.raises(ValueError, match="signature"):
        agent.execute(SignedNodeAction(command=value, signature="0" * 64))
    expired = copy_model(
        value, command_id="expired", expires_at=NOW - timedelta(seconds=1)
    )
    with pytest.raises(ValueError, match="expired"):
        agent.execute(envelope(expired))


def test_node_agent_result_query_requires_a_signature(tmp_path, monkeypatch) -> None:
    """The result read is as sensitive as the command that produced it.

    It names the node, the GPU UUIDs, the processes that were holding
    the devices open and the diagnostics that ran. It used to answer any
    caller that could reach the agent's port with nothing but a command
    ID, which is guessable: the ID is derived from the workflow's
    idempotency key.
    """
    agent = executor(tmp_path, FakeRunner())
    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))
    command_id = signed.command.command_id

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submitted = client.post("/v1/node-actions", json=signed.model_dump(mode="json"))
        assert submitted.status_code == 200

        unsigned = client.get(
            "/v1/node-actions/result", params={"command_id": command_id}
        )
        forged = client.get(
            "/v1/node-actions/result",
            params={
                "command_id": command_id,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "signature": "0" * 64,
            },
        )
        # A signature captured off the wire an hour ago is outside the
        # replay window even though the HMAC itself still verifies.
        stale = client.get(
            "/v1/node-actions/result",
            params=result_params(
                command_id,
                issued_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            ),
        )
        # Signed for one command, replayed against another.
        crossed = client.get(
            "/v1/node-actions/result",
            params={
                **result_params(command_id),
                "command_id": "workflow/step/other-node",
            },
        )
        authorized = client.get(
            "/v1/node-actions/result", params=result_params(command_id)
        )

    assert unsigned.status_code == 401
    assert unsigned.json()["detail"]["code"] == "INVALID_SIGNATURE"
    assert "issued_at and signature" in (unsigned.json()["detail"]["message"])
    assert forged.status_code == 401
    assert stale.status_code == 401
    assert "window" in stale.json()["detail"]["message"]
    # Not 404: the cross-command replay never reaches the ledger.
    assert crossed.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["state"] == (NodeActionExecutionState.SUCCEEDED.value)


def test_node_agent_rejects_stale_fencing_token(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner())
    accepted = command(
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS, command_id="new-token", fencing_token=2
    )
    stale = command(
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS, command_id="old-token", fencing_token=1
    )

    assert agent.execute(envelope(accepted)).status is NodeActionStatus.SUCCEEDED
    with pytest.raises(ValueError, match="stale"):
        agent.execute(envelope(stale))


def test_retryable_node_failure_is_not_permanently_cached(tmp_path) -> None:
    class FlakyRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.reset_attempts = 0

        def __call__(self, command, **kwargs):
            if "--gpu-reset" in command:
                self.reset_attempts += 1
                if self.reset_attempts == 1:
                    raise OSError("temporary device transport error")
            return super().__call__(command, **kwargs)

    runner = FlakyRunner()
    agent = node_action_executor(
        tmp_path,
        "retryable.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=runner,
        device_client_finder=no_device_clients,
        device_client_samples=1,
    )
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert first.status is NodeActionStatus.FAILED
    assert first.retryable is True
    assert first.attempt == 1
    assert second.status is NodeActionStatus.SUCCEEDED
    assert second.attempt == 2
    assert agent.ledger.get(signed.command.command_id).attempt == 2
    assert runner.reset_attempts == 2


def test_in_progress_action_becomes_interrupted_after_restart(tmp_path) -> None:
    path = tmp_path / "interrupted.db"
    first = NodeActionLedger(str(path))
    submitted = command(WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE)
    first.mark_in_progress(submitted, 1)
    assert first.get(submitted.command_id) is None

    restarted = NodeActionLedger(str(path))
    result = restarted.get(submitted.command_id)

    assert result is not None
    assert result.status is NodeActionStatus.INTERRUPTED
    assert result.retryable is False
    assert "manual confirmation" in (result.error or "")


def test_cached_retryable_command_still_expires(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner())
    value = command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
    agent.ledger.save(
        node_action_result(
            value.command_id, value.operation, NodeActionStatus.FAILED, retryable=True
        )
    )
    agent.now = lambda: value.expires_at + timedelta(seconds=1)

    with pytest.raises(ValueError, match="expired"):
        agent.execute(envelope(value))

    assert agent.ledger.get(value.command_id).attempt == 1


def test_allowlisted_but_undispatched_operation_fails_closed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        NodeActionExecutor,
        "OPERATIONS",
        {*NodeActionExecutor.OPERATIONS, WorkflowOperation.RESTART_NODE},
    )
    agent = node_action_executor(
        tmp_path, "undispatched.db", allowed_operations={WorkflowOperation.RESTART_NODE}
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESTART_NODE)))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert "UnsupportedOperationError" in result.error
    assert "has no executor branch" in result.error


def test_node_action_ledger_prunes_old_results_and_fencing(tmp_path) -> None:
    ledger = NodeActionLedger(
        str(tmp_path / "retention.db"), retention_seconds=600, max_results=100
    )
    old = NOW - timedelta(minutes=20)
    ledger.save(
        node_action_result(
            "old", WorkflowOperation.VERIFY_NO_GPU_CLIENTS, completed_at=old
        )
    )
    ledger.save(
        node_action_result(
            "new", WorkflowOperation.VERIFY_NO_GPU_CLIENTS, completed_at=NOW
        )
    )
    ledger.accept_fencing("old-incident", 1)
    ledger._db.execute(
        "UPDATE fencing SET updated_at=? WHERE incident_id=?",
        (old.isoformat(), "old-incident"),
    )

    removed = ledger.cleanup(now=NOW)

    assert removed == {"results": 1, "fencing": 1}
    assert ledger.get("old") is None
    assert ledger.get("new") is not None


def test_ledger_save_failure_releases_all_inflight_waiters(
    tmp_path, monkeypatch
) -> None:
    waiter_started = Event()

    class TrackingEvent:
        def __init__(self) -> None:
            self.event = Event()

        def set(self) -> None:
            self.event.set()

        def wait(self, timeout: float | None = None) -> bool:
            waiter_started.set()
            return self.event.wait(timeout=timeout)

    class BlockingRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def __call__(self, command, **kwargs):
            if "--gpu-reset" in command:
                self.started.set()
                self.release.wait(timeout=5)
            return super().__call__(command, **kwargs)

    runner = BlockingRunner()
    monkeypatch.setattr(executor_module, "Event", TrackingEvent)
    agent = executor(tmp_path, runner)
    signed = envelope(command(WorkflowOperation.RESET_GPU))
    monkeypatch.setattr(
        agent.ledger,
        "save",
        lambda _result, **_kwargs: (_ for _ in ()).throw(OSError("ledger is full")),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(agent.execute, signed)
        assert runner.started.wait(timeout=7)
        second = pool.submit(agent.execute, signed)
        assert waiter_started.wait(timeout=7), (
            "second node action did not enter the in-flight wait path"
        )
        runner.release.set()
        with pytest.raises(OSError, match="ledger is full"):
            first.result(timeout=2)
        with pytest.raises(RuntimeError, match="completed without a ledger result"):
            second.result(timeout=5)

    assert agent._inflight == {}


def test_config_digest_cli_needs_no_secret_or_real_ledger(
    capsys, monkeypatch, tmp_path
) -> None:
    """Deployment computes the pin without holding the node secret.

    It runs on a build host, not on a GPU node, so it must not require
    the shared secret and must not touch a running agent's ledger.
    """
    monkeypatch.delenv("GPU_FAULT_NODE_ACTION_SECRET", raising=False)
    monkeypatch.delenv("GPU_FAULT_QUIESCE_STATE_DIR", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "profile-a")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ALLOWED_OPERATIONS",
        "QUIESCE_GPU_SERVICES,RESET_ALL_GPUS_NVSWITCHES",
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", "true")
    monkeypatch.setenv("GPU_FAULT_QUIESCE_SERVICES", "nvidia-dcgm,kubelet")
    ledger = tmp_path / "must-not-exist.db"
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_DB", str(ledger))

    print_config_digest()

    digest = capsys.readouterr().out.strip()
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert not ledger.exists()
    assert os.environ["GPU_FAULT_NODE_ACTION_DB"] == str(ledger)
    assert "GPU_FAULT_NODE_ACTION_SECRET" not in os.environ
    assert "GPU_FAULT_QUIESCE_STATE_DIR" not in os.environ


def test_signed_parameters_cannot_be_changed(tmp_path) -> None:
    agent = node_action_executor(
        tmp_path, "signed.db", allowed_operations={WorkflowOperation.REMEDIATE_DRIVER}
    )
    original = command(
        WorkflowOperation.REMEDIATE_DRIVER, parameters={"target_driver_branch": 575}
    )
    signed = envelope(original)
    tampered = copy_model(
        signed, command=copy_model(original, parameters={"target_driver_branch": 999})
    )
    with pytest.raises(ValueError, match="invalid node action signature"):
        agent.execute(tampered)
