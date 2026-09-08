from __future__ import annotations

import sqlite3
from threading import Event, current_thread

from tests._builders import copy_model, node_action_result

from ._support import (
    NOW,
    CompletedProcess,
    FakeRunner,
    NodeActionExecutionState,
    NodeActionStatus,
    Path,
    ServiceRunner,
    SignedNodeAction,
    TestClient,
    ThreadPoolExecutor,
    WorkflowOperation,
    command,
    create_node_agent_app,
    datetime,
    envelope,
    executor,
    json,
    no_device_clients,
    node_action_executor,
    os,
    pytest,
    result_params,
    submit_action,
    tarfile,
    time,
    timedelta,
    timezone,
    wait_for_result,
)


def test_node_agent_async_submit_and_poll_preserves_long_action(
    tmp_path, monkeypatch
) -> None:
    agent = executor(tmp_path, FakeRunner())

    def slow_execute(value: SignedNodeAction):
        agent.validate_submission(value)
        time.sleep(0.2)
        result = node_action_result(
            value.command.command_id,
            value.command.operation,
            details={"long_action_completed": True},
        )
        agent.ledger.save(result)
        return result

    monkeypatch.setattr(agent, "execute", slow_execute)
    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        started_at = time.monotonic()
        submitted = client.post(
            "/v1/node-actions/submit", json=signed.model_dump(mode="json")
        )
        submit_elapsed = time.monotonic() - started_at
        assert submitted.status_code == 200
        assert submit_elapsed < 0.15
        assert submitted.json()["state"] == NodeActionExecutionState.PENDING.value
        pending = client.get(
            "/v1/node-actions/result", params=result_params(signed.command.command_id)
        )
        assert pending.status_code == 200
        assert pending.json()["state"] == NodeActionExecutionState.PENDING.value

        for _ in range(100):
            completed = client.get(
                "/v1/node-actions/result",
                params=result_params(signed.command.command_id),
            )
            if (
                completed.status_code == 200
                and completed.json()["state"] != NodeActionExecutionState.PENDING.value
            ):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("long action did not complete")
        assert completed.json()["state"] == NodeActionExecutionState.SUCCEEDED.value, (
            completed.json()
        )
        assert completed.json()["result"]["details"] == {"long_action_completed": True}
        assert client.get("/healthz").json()["status"] == "ok"


def test_node_agent_result_query_migration_escape_hatch(tmp_path, monkeypatch) -> None:
    """A fleet rolled before its control plane can still be polled.

    The signing side ships in the same wheel as the verifying side, so
    the documented order is control plane first. This exists for the
    fleet that is already the other way round.
    """
    agent = executor(tmp_path, FakeRunner())
    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED", "false")
    monkeypatch.setenv("GPU_FAULT_RELEASE_ID", "migration-test")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_RELEASE_ID", "migration-test"
    )
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_EXPIRES_AT",
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        wait_for_result(client, signed.command.command_id)
        unsigned = client.get(
            "/v1/node-actions/result", params={"command_id": signed.command.command_id}
        )
        # A signature that is supplied is still verified, so a rolled
        # control plane is not silently trusted less.
        forged = client.get(
            "/v1/node-actions/result",
            params={
                "command_id": signed.command.command_id,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "signature": "0" * 64,
            },
        )

    assert unsigned.status_code == 200
    assert forged.status_code == 401


@pytest.mark.parametrize("token", ["false", "0", "no", "off"])
def test_unsigned_result_query_escape_hatch_requires_expiry(
    tmp_path, monkeypatch, token: str
) -> None:
    """Every disabled token opens the hatch, and the hatch still needs an expiry."""

    agent = executor(tmp_path, FakeRunner())
    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED", token)
    monkeypatch.setenv("GPU_FAULT_RELEASE_ID", "release-a")
    monkeypatch.delenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_EXPIRES_AT", raising=False
    )
    with pytest.raises(RuntimeError, match="bound to the current"):
        create_node_agent_app(agent, heartbeat_reporter=None)


def test_node_agent_returns_structured_validation_errors(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner())
    app = create_node_agent_app(agent, heartbeat_reporter=None)
    valid = command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
    invalid_signature = SignedNodeAction(command=valid, signature="0" * 64)
    expired = copy_model(
        valid, command_id="expired-http", expires_at=NOW - timedelta(seconds=1)
    )

    with TestClient(app) as client:
        rejected = client.post(
            "/v1/node-actions/submit", json=invalid_signature.model_dump(mode="json")
        )
        expired_response = client.post(
            "/v1/node-actions/submit", json=envelope(expired).model_dump(mode="json")
        )

    assert rejected.status_code == 401
    assert rejected.json()["detail"] == {
        "code": "INVALID_SIGNATURE",
        "message": "invalid node action signature",
        "retryable": False,
        "requires_new_command": False,
    }
    assert expired_response.status_code == 410
    assert expired_response.json()["detail"]["code"] == ("COMMAND_EXPIRED")
    assert expired_response.json()["detail"]["retryable"] is True
    assert expired_response.json()["detail"]["requires_new_command"] is True


def test_compute_only_verify_ignores_read_only_device_clients(tmp_path) -> None:
    device_checks = []
    agent = node_action_executor(
        tmp_path,
        "compute-only.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        runner=FakeRunner(),
        device_client_finder=lambda targets: device_checks.append(targets)
        or [{"gpu_uuid": "GPU-a", "pid": "123", "process_name": "dcgm-exporter"}],
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                parameters={"compute_clients_only": True},
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["device_clients_checked"] is False
    assert device_checks == []


def test_compute_only_verify_still_rejects_compute_clients(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner(clients="GPU-a, 456, python\n"))

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                parameters={"compute_clients_only": True},
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED
    assert "GPU compute clients are still active" in result.error


def test_node_agent_rejects_different_agent_generation(tmp_path) -> None:
    agent = executor(tmp_path, FakeRunner())
    agent.set_agent_generation(7)
    wrong = copy_model(
        command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, command_id="wrong-generation"),
        agent_generation=6,
    )
    current = copy_model(wrong, command_id="current-generation", agent_generation=7)

    with pytest.raises(ValueError, match="agent generation"):
        agent.execute(envelope(wrong))
    assert agent.execute(envelope(current)).status is NodeActionStatus.SUCCEEDED
    agent.set_agent_generation(8)
    with pytest.raises(ValueError, match="agent generation"):
        agent.execute(envelope(current))


def test_health_snapshot_restarts_and_verifies_collectors(tmp_path) -> None:
    runner = ServiceRunner(active=set())
    agent = node_action_executor(
        tmp_path,
        "snapshot-actions.db",
        allowed_operations={WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT},
        runner=runner,
        health_snapshot_request_dir=str(tmp_path / "health-snapshot"),
    )

    result = agent.execute(envelope(command(WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT)))

    services = [
        "gpu-fault-metrics-collector.service",
        "gpu-fault-host-collector.service",
    ]
    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details == {
        "snapshot_triggered": True,
        "triggered_at": NOW.isoformat(),
        "restarted_collectors": services,
    }
    assert runner.commands == [
        ["systemctl", "restart", services[0]],
        ["systemctl", "is-active", "--quiet", services[0]],
        ["systemctl", "restart", services[1]],
        ["systemctl", "is-active", "--quiet", services[1]],
    ]
    assert (tmp_path / "health-snapshot" / "gpu.request").read_text(
        encoding="ascii"
    ) == NOW.isoformat() + "\n"
    assert (tmp_path / "health-snapshot" / "host.request").read_text(
        encoding="ascii"
    ) == NOW.isoformat() + "\n"


def test_strace_samples_killed_by_their_own_deadline_are_not_failures(tmp_path) -> None:
    """timeout(1) exits 124 on every completed sample.

    Measured on p5en: a healthy bundle reported one failed capture per
    strace sample, which made ``failed_capture_count`` meaningless.
    """

    pid = os.getpid()

    class HungRunner:
        def __call__(self, argv, **_):
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                return CompletedProcess(
                    argv, 0, stdout=f"{pid}, GPU-a, python\n", stderr=""
                )
            if "strace" in argv:
                prefix = Path(argv[argv.index("-o") + 1])
                prefix.with_suffix(f".{pid}").write_text("clock_nanosleep")
                return CompletedProcess(argv, 124, stdout="", stderr="")
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    output_dir = tmp_path / "diagnostics"
    agent = node_action_executor(
        tmp_path,
        "diagnostics.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        diagnostic_output_dir=str(output_dir),
        expand_python_cgroup_processes=False,
        runner=HungRunner(),
        sleep=lambda _seconds: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                parameters={
                    "capture_process_state": True,
                    "strace_duration_seconds": 1,
                    "strace_sample_count": 2,
                    "strace_sample_interval_seconds": 0,
                },
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    archive = next(output_dir.glob("*.tar.gz"))
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("diagnostics/manifest.json"))
        names = set(bundle.getnames())
    failed = [
        item
        for item in manifest["captures"]
        if item.get("error") or item.get("returncode") not in {0, None}
    ]
    assert not [item for item in failed if str(item["file"]).startswith("strace-")]
    assert result.details["manifest_summary"]["failed_capture_count"] == len(failed)
    traces = [
        item
        for item in manifest["captures"]
        if str(item["file"]).startswith(f"strace-{pid}-sample-")
    ]
    assert len(traces) == 2
    assert all(item["returncode"] == 0 for item in traces)
    assert all(item["raw_returncode"] == 124 for item in traces)
    assert all(item["terminated_by"] == "sample_duration" for item in traces)
    assert all(item["trace_file_count"] == 1 for item in traces)
    assert all("error" not in item for item in traces)
    assert f"diagnostics/strace-{pid}-sample-01.{pid}" in names
    nvlink = next(
        item
        for item in manifest["captures"]
        if item["file"] == "nvidia-smi-nvlink-errors.txt"
    )
    assert nvlink["command"] == ["nvidia-smi", "nvlink", "--errorcounters"]


def test_strace_sample_without_any_trace_file_stays_a_failure(tmp_path) -> None:
    pid = os.getpid()

    class HungRunner:
        def __call__(self, argv, **_):
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                return CompletedProcess(
                    argv, 0, stdout=f"{pid}, GPU-a, python\n", stderr=""
                )
            if "strace" in argv:
                return CompletedProcess(argv, 124, stdout="", stderr="")
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    output_dir = tmp_path / "diagnostics"
    agent = node_action_executor(
        tmp_path,
        "diagnostics.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        diagnostic_output_dir=str(output_dir),
        expand_python_cgroup_processes=False,
        runner=HungRunner(),
        sleep=lambda _seconds: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                parameters={
                    "capture_process_state": True,
                    "strace_duration_seconds": 1,
                    "strace_sample_count": 1,
                    "strace_sample_interval_seconds": 0,
                },
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    archive = next(output_dir.glob("*.tar.gz"))
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("diagnostics/manifest.json"))
    failed = [
        item
        for item in manifest["captures"]
        if item.get("error") or item.get("returncode") not in {0, None}
    ]
    assert [item for item in failed if str(item["file"]).startswith("strace-")]
    assert result.details["manifest_summary"]["failed_capture_count"] == len(failed)
    trace = next(
        item
        for item in manifest["captures"]
        if str(item["file"]).startswith(f"strace-{pid}-sample-")
    )
    assert trace["returncode"] == 124
    assert trace["trace_file_count"] == 0
    assert "no trace file" in trace["error"]


def test_verify_ignores_respawning_short_lived_device_clients(tmp_path) -> None:
    """A holder that reappears under a new pid must not block forever.

    Measured on a HyperPod node: ``nvidia-persistenced`` (no systemd
    unit on these AMIs, so quiesce cannot stop it) and per-second
    ``nvidia-smi`` calls made 96 of 98 ``VERIFY_NO_GPU_CLIENTS``
    attempts fail, each on a different pid, until the quiesce
    maintenance window expired and the GPU was never reset.
    """
    pids = iter(["101", "102", "103"])
    sleeps: list[float] = []
    agent = node_action_executor(
        tmp_path,
        "respawn.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        runner=FakeRunner(),
        device_client_finder=lambda _: [
            {
                "gpu_uuid": "GPU-a",
                "pid": next(pids),
                "process_name": "nvidia-persistenced",
                "device": "/dev/nvidia0",
            }
        ],
        sleep=sleeps.append,
    )

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["verified_no_gpu_clients"] is True
    assert [
        item["process_name"] for item in result.details["transient_device_clients"]
    ] == ["nvidia-persistenced"] * 3
    assert sleeps == [2.0, 2.0]


def test_verify_still_fails_closed_on_persistent_device_client(tmp_path) -> None:
    """Sampling must not weaken the gate for a real, stable holder."""
    agent = node_action_executor(
        tmp_path,
        "persistent.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        runner=FakeRunner(),
        device_client_finder=lambda _: [
            {
                "gpu_uuid": "GPU-a",
                "pid": "4321",
                "process_name": "python3",
                "device": "/dev/nvidia0",
            }
        ],
        sleep=lambda _: None,
    )

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.FAILED
    assert "GPU-a:4321:python3" in result.error


def test_verify_single_sample_matches_legacy_behaviour(tmp_path) -> None:
    """``device_client_samples=1`` keeps the pre-sampling semantics."""
    sleeps: list[float] = []
    agent = node_action_executor(
        tmp_path,
        "single.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        runner=FakeRunner(),
        device_client_finder=lambda _: [
            {
                "gpu_uuid": "GPU-a",
                "pid": "77",
                "process_name": "nvidia-smi",
                "device": "/dev/nvidia0",
            }
        ],
        device_client_samples=1,
        sleep=sleeps.append,
    )

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.FAILED
    assert "GPU-a:77:nvidia-smi" in result.error
    assert sleeps == []


def reset_agent(tmp_path, ledger_name: str, runner, **overrides):
    """A node agent that is allowed to reset a single GPU."""

    return node_action_executor(
        tmp_path,
        ledger_name,
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=runner,
        device_client_finder=no_device_clients,
        device_client_samples=1,
        **overrides,
    )


def reset_invocations(runner: FakeRunner) -> int:
    return len([item for item in runner.commands if "--gpu-reset" in item])


def test_ledger_save_failure_after_reset_never_reexecutes(
    tmp_path, monkeypatch
) -> None:
    """A result that cannot be written must not run the reset again.

    ``/var/lib/gpu-fault`` also holds diagnostic archives, so ENOSPC is real,
    and an operator's ``sqlite3`` session can hold the write lock. The
    IN_PROGRESS marker then stayed on disk, ``ledger.get`` hid it behind
    ``None``, the resubmitted envelope computed attempt 1 again and reset the
    GPU a second time -- and ``/result`` kept answering 404, so the control
    plane kept resubmitting.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    runner = FakeRunner()
    sleeps: list[float] = []
    agent = reset_agent(tmp_path, "lost-result.db", runner, sleep=sleeps.append)
    attempted: list[str] = []

    def unwritable_save(result, **_kwargs):
        attempted.append(result.status.value)
        raise OSError("[Errno 28] No space left on device")

    monkeypatch.setattr(agent.ledger, "save", unwritable_save)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        first = wait_for_result(client, signed.command.command_id)
        submit_action(client, signed)
        second = wait_for_result(client, signed.command.command_id)
        submit_action(client, signed)
        third = wait_for_result(client, signed.command.command_id)

    assert reset_invocations(runner) == 1, (
        f"the GPU was reset more than once: {runner.commands}"
    )
    for answer in (first, second, third):
        assert answer.status_code == 200, (
            f"the poll must answer, not 404: {answer.status_code} {answer.text}"
        )
        assert answer.json()["state"] == NodeActionExecutionState.INTERRUPTED.value, (
            answer.json()
        )
        assert "result could not be persisted" in (
            answer.json()["result"]["error"] or ""
        ), answer.json()
        assert answer.json()["result"]["retryable"] is False, answer.json()
    assert attempted == ["SUCCEEDED"] * 4, (
        f"the result write is tried once and retried three times: {attempted}"
    )
    assert sleeps == [0.2, 0.2, 0.2], f"retry backoff must be injectable: {sleeps}"
    history = agent.ledger.attempt_history(signed.command.command_id)
    assert [row["attempt"] for row in history] == [1], history
    assert [row["state"] for row in history] == ["INTERRUPTED"], history


def test_a_transient_ledger_write_is_retried_before_the_interrupted_marker(
    tmp_path, monkeypatch
) -> None:
    """``database is locked`` for a moment must not lose a good result."""

    runner = FakeRunner()
    sleeps: list[float] = []
    agent = reset_agent(tmp_path, "flaky-write.db", runner, sleep=sleeps.append)
    real_save = agent.ledger.save
    attempted: list[str] = []

    def flaky_save(result, **kwargs):
        attempted.append(result.status.value)
        if len(attempted) == 1:
            raise sqlite3.OperationalError("database is locked")
        real_save(result, **kwargs)

    monkeypatch.setattr(agent.ledger, "save", flaky_save)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.SUCCEEDED, result
    stored = agent.ledger.get(result.command_id)
    assert stored is not None, "the retried write must reach the ledger"
    assert stored.status is NodeActionStatus.SUCCEEDED, stored
    assert stored.attempt == 1, stored
    assert reset_invocations(runner) == 1, runner.commands
    assert sleeps == [0.2], f"one failure means one backoff sleep: {sleeps}"


def test_submit_reruns_a_retryable_failure_as_attempt_two(
    tmp_path, monkeypatch
) -> None:
    """``/submit`` used to answer the resubmit with the stale retryable row.

    ``executor.execute`` runs attempt+1, but ``/submit`` returned the ledger
    row before ever calling it, so every transient device error on RESET_GPU
    left the step in WAITING until its bound.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)

    class FlakyRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.reset_attempts = 0

        def __call__(self, value, **kwargs):
            if "--gpu-reset" in value:
                self.reset_attempts += 1
                if self.reset_attempts == 1:
                    raise OSError("temporary device transport error")
            return super().__call__(value, **kwargs)

    runner = FlakyRunner()
    agent = reset_agent(tmp_path, "http-retry.db", runner)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        failed = wait_for_result(client, signed.command.command_id)
        resubmitted = submit_action(client, signed)
        succeeded = wait_for_result(client, signed.command.command_id)

    assert failed.json()["result"]["status"] == NodeActionStatus.FAILED.value, (
        failed.json()
    )
    assert failed.json()["result"]["retryable"] is True, failed.json()
    assert failed.json()["result"]["attempt"] == 1, failed.json()
    assert resubmitted.json()["state"] != NodeActionExecutionState.FAILED.value, (
        f"the resubmit must not be answered with the stale row: {resubmitted.json()}"
    )
    assert succeeded.json()["state"] == NodeActionExecutionState.SUCCEEDED.value, (
        succeeded.json()
    )
    assert succeeded.json()["result"]["attempt"] == 2, succeeded.json()
    assert runner.reset_attempts == 2, (
        f"the pool never ran attempt 2: {runner.reset_attempts}"
    )


def test_expiry_while_queued_is_a_rejection_not_a_ledger_result(
    tmp_path, monkeypatch
) -> None:
    """A command that expires between ``/submit`` and the worker is no result.

    The pooled ``execute`` re-validates against the wall clock. Persisting
    that ``ValueError`` under the deterministic command_id answered every
    later, freshly signed envelope with a permanent FAILED row.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    validations = {"count": 0}

    def queued_clock() -> datetime:
        validations["count"] += 1
        return NOW if validations["count"] == 1 else NOW + timedelta(minutes=3)

    runner = FakeRunner()
    agent = reset_agent(tmp_path, "queued-expiry.db", runner, now=queued_clock)
    signed = envelope(command(WorkflowOperation.RESET_GPU))
    fresh = envelope(
        copy_model(
            signed.command,
            issued_at=NOW + timedelta(minutes=3),
            expires_at=NOW + timedelta(minutes=5),
        )
    )

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        rejected = wait_for_result(client, signed.command.command_id)
        submit_action(client, fresh)
        answered = wait_for_result(client, fresh.command.command_id)

    assert rejected.status_code == 404, (
        f"a rejection must not become a ledger row: {rejected.text}"
    )
    assert answered.json()["state"] == NodeActionExecutionState.SUCCEEDED.value, (
        f"the freshly signed envelope was answered with the rejection: "
        f"{answered.json()}"
    )
    assert reset_invocations(runner) == 1, runner.commands
    history = agent.ledger.attempt_history(signed.command.command_id)
    assert [row["state"] for row in history] == ["SUCCEEDED"], history
    assert agent.counters_snapshot()["rejected"] == 1, (
        f"one rejected command counts once: {agent.counters_snapshot()}"
    )


def test_same_command_id_with_different_body_is_rejected_not_replayed(
    tmp_path, monkeypatch
) -> None:
    """One command_id is one command.

    The ledger keeps the operation, the GPU UUIDs and a digest of the
    parameters per attempt but never compared them, so a validly signed
    command for GPU-b and GPU-c was answered with the SUCCEEDED row of a
    reset that only ever touched GPU-a.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    runner = FakeRunner()
    agent = reset_agent(tmp_path, "reuse.db", runner)
    signed = envelope(command(WorkflowOperation.RESET_GPU))
    other_targets = envelope(copy_model(signed.command, gpu_uuids=["GPU-b", "GPU-c"]))
    other_parameters = envelope(
        copy_model(signed.command, parameters={"compute_clients_only": True})
    )

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        first = wait_for_result(client, signed.command.command_id)
        reused_targets = submit_action(client, other_targets)
        reused_parameters = submit_action(client, other_parameters)

    assert first.json()["result"]["details"]["reset_gpu_uuids"] == ["GPU-a"], (
        first.json()
    )
    for response in (reused_targets, reused_parameters):
        assert response.status_code == 409, (
            f"a different body must not read the old row: {response.text}"
        )
        assert response.json()["detail"] == {
            "code": "COMMAND_ID_REUSED",
            "message": "node action command_id reused for a different command",
            "retryable": False,
            "requires_new_command": True,
        }, response.json()
    assert reset_invocations(runner) == 1, runner.commands


def test_sync_node_action_route_is_gone(tmp_path, monkeypatch) -> None:
    """The synchronous route disagreed with ``/submit`` about in-flight work.

    Nothing in the product called it -- the transport polls ``/result`` and
    posts ``/submit`` -- but while it ran a command ``/result`` answered 404
    and a ``/submit`` for the same id parked a pool worker in a 2100 s wait.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    agent = executor(tmp_path, FakeRunner())
    app = create_node_agent_app(agent, heartbeat_reporter=None)
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))

    paths = {getattr(item, "path", None) for item in app.routes}
    with TestClient(app) as client:
        response = client.post("/v1/node-actions", json=signed.model_dump(mode="json"))

    assert "/v1/node-actions" not in paths, sorted(str(item) for item in paths)
    assert "/v1/node-actions/submit" in paths, sorted(str(item) for item in paths)
    assert response.status_code == 404, (
        f"the synchronous route still answers: {response.status_code}"
    )
    assert agent.ledger.attempt_history(signed.command.command_id) == [], (
        "the removed route must not have run anything"
    )


def test_a_ledger_that_recovers_never_reruns_a_dispatched_reset(
    tmp_path, monkeypatch
) -> None:
    """Both ledger writes broken, then working, must still reset once.

    The ledger holds no result and no INTERRUPTED marker, so a resubmit landed
    in the pool wrapper's ``except``. That wrapper wrote FAILED
    ``retryable=True`` over the IN_PROGRESS row -- an UPSERT -- and the next
    resubmit read that row, asked for attempt 2 and reset the GPU again.

    While nothing is on disk the poll answers 404 -- the finished result lives
    only in a future the done-callback has already dropped -- and that is what
    makes the control plane resubmit until the ledger can close the attempt.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    runner = FakeRunner()
    agent = reset_agent(tmp_path, "recovering-ledger.db", runner, sleep=lambda _s: None)
    real_mark_interrupted = agent.ledger.mark_interrupted
    writes_fail = {"value": True}

    def unwritable_save(result, **_kwargs):
        raise OSError("[Errno 28] No space left on device")

    def flaky_mark_interrupted(command_id: str, attempt: int, error: str):
        if writes_fail["value"]:
            raise sqlite3.OperationalError("database is locked")
        return real_mark_interrupted(command_id, attempt, error)

    monkeypatch.setattr(agent.ledger, "save", unwritable_save)
    monkeypatch.setattr(agent.ledger, "mark_interrupted", flaky_mark_interrupted)
    signed = envelope(command(WorkflowOperation.RESET_GPU))
    answers = []

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        answers.append(wait_for_result(client, signed.command.command_id))
        submit_action(client, signed)
        answers.append(wait_for_result(client, signed.command.command_id))
        writes_fail["value"] = False
        submit_action(client, signed)
        answers.append(wait_for_result(client, signed.command.command_id))

    assert reset_invocations(runner) == 1, (
        f"the GPU was reset more than once: {runner.commands}"
    )
    payloads = [
        answer.json() if answer.status_code == 200 else None for answer in answers
    ]
    for payload in payloads:
        if payload is None:
            # Nothing is on disk yet, so 404 is honest and the control plane
            # resubmits. What must never happen is a retryable answer.
            continue
        assert payload["state"] != NodeActionExecutionState.FAILED.value, (
            f"a dispatched attempt must not be answered as a failure: {payload}"
        )
        assert payload["result"]["retryable"] is False, (
            f"a dispatched attempt must never be answered as retryable: {payload}"
        )
    assert payloads[-1] is not None, (
        f"the recovered ledger must answer the last poll: {answers[-1].text}"
    )
    assert payloads[-1]["state"] == NodeActionExecutionState.INTERRUPTED.value, (
        payloads[-1]
    )
    history = agent.ledger.attempt_history(signed.command.command_id)
    assert [(row["attempt"], row["state"]) for row in history] == [
        (1, "INTERRUPTED")
    ], f"the recovered write must close attempt 1, not open attempt 2: {history}"


def test_a_lost_in_progress_write_reply_leaves_no_stuck_waiter(
    tmp_path, monkeypatch
) -> None:
    """The marker landed but the call failed: release the in-flight Event.

    ``mark_in_progress`` ran outside the ``try/finally``, so the per-command
    Event stayed registered and unset. Every later submit for that command_id
    then parked a pool worker in the in-flight wait -- 2100 s in production,
    four such answers and the agent accepts nothing else.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    runner = FakeRunner()
    agent = reset_agent(
        tmp_path, "lost-marker-reply.db", runner, inflight_wait_timeout_seconds=10
    )
    real_mark_in_progress = agent.ledger.mark_in_progress
    marked: list[int] = []

    def lossy_mark_in_progress(value, attempt: int, **kwargs):
        real_mark_in_progress(value, attempt, **kwargs)
        marked.append(attempt)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(agent.ledger, "mark_in_progress", lossy_mark_in_progress)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        first = wait_for_result(client, signed.command.command_id)
        started = time.monotonic()
        submit_action(client, signed)
        second = wait_for_result(client, signed.command.command_id)
        elapsed = time.monotonic() - started

    assert marked == [1], f"only attempt 1 may be dispatched: {marked}"
    assert reset_invocations(runner) == 0, (
        f"the handler never ran, so nothing may have reset: {runner.commands}"
    )
    for answer in (first, second):
        payload = answer.json()
        assert answer.status_code == 200, f"the poll must answer: {answer.text}"
        assert payload["state"] == NodeActionExecutionState.INTERRUPTED.value, payload
        assert payload["result"]["retryable"] is False, (
            f"a marked attempt cannot be replayed automatically: {payload}"
        )
    assert elapsed < 3.0, (
        f"the resubmit waited on a leaked in-flight Event: {elapsed:.1f}s"
    )


MINOR_NUMBER_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:53:00.0">
    <uuid>GPU-a</uuid>
    <minor_number>3</minor_number>
  </gpu>
  <gpu id="00000000:64:00.0">
    <uuid>GPU-b</uuid>
    <minor_number>0</minor_number>
  </gpu>
</nvidia_smi_log>
"""

NO_MINOR_NUMBER_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:53:00.0">
    <uuid>GPU-a</uuid>
  </gpu>
  <gpu id="00000000:64:00.0">
    <uuid>GPU-b</uuid>
  </gpu>
</nvidia_smi_log>
"""


class DevicePathRunner:
    """nvidia-smi on a node whose PCI order is not its device-minor order."""

    def __init__(
        self,
        xml: str,
        index: str = "GPU-a, 0\nGPU-b, 1\n",
        inventory: str = "GPU-a\nGPU-b\n",
    ) -> None:
        self.xml = xml
        self.index = index
        self.inventory = inventory
        self.commands: list[list[str]] = []

    def __call__(self, argv, **_):
        self.commands.append(list(argv))
        if argv[:3] == ["nvidia-smi", "-q", "-x"]:
            return CompletedProcess(argv, 0, stdout=self.xml, stderr="")
        if "--query-compute-apps=gpu_uuid,pid,process_name" in argv:
            return CompletedProcess(argv, 0, stdout="", stderr="")
        if "--query-gpu=uuid,index" in argv:
            return CompletedProcess(argv, 0, stdout=self.index, stderr="")
        if "--query-gpu=uuid" in argv:
            return CompletedProcess(argv, 0, stdout=self.inventory, stderr="")
        return CompletedProcess(argv, 0, stdout="", stderr="")

    def index_queries(self) -> int:
        return len([item for item in self.commands if "--query-gpu=uuid,index" in item])

    def xml_queries(self) -> int:
        return len(
            [item for item in self.commands if item[:3] == ["nvidia-smi", "-q", "-x"]]
        )


def _proc_root_holding(tmp_path: Path, device: str, *, pid: str = "222") -> Path:
    proc = tmp_path / "host-proc"
    fd_dir = proc / pid / "fd"
    fd_dir.mkdir(parents=True)
    (proc / pid / "comm").write_text("python\n")
    (fd_dir / "7").symlink_to(device)
    return proc


def _verify_agent(tmp_path, ledger_name: str, runner, proc, **overrides):
    return node_action_executor(
        tmp_path,
        ledger_name,
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        runner=runner,
        proc_root=str(proc),
        sleep=lambda _seconds: None,
        **overrides,
    )


def test_device_path_uses_minor_number_not_index(tmp_path) -> None:
    """``/dev/nvidia{index}`` is a guess; the driver's minor number is the fact.

    nvidia-smi's index is PCI-bus ordered and the device minor is probe
    ordered, which is why NVIDIA reports "Minor Number" at all. With the two
    swapped, ``VERIFY_NO_GPU_CLIENTS`` inspected a device node that belongs to
    a different GPU: it passed while the target GPU had a holder, and the reset
    that followed failed "in use" three times for nothing.
    """

    runner = DevicePathRunner(MINOR_NUMBER_XML)
    proc = _proc_root_holding(tmp_path, "/dev/nvidia3")
    agent = _verify_agent(tmp_path, "minor.db", runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "GPU-a:222:python" in (result.error or ""), result.error


def test_device_path_never_maps_a_uuid_to_another_gpus_node(tmp_path) -> None:
    """A holder of GPU-b's device node must not block a reset of GPU-a."""

    runner = DevicePathRunner(MINOR_NUMBER_XML)
    proc = _proc_root_holding(tmp_path, "/dev/nvidia0")
    agent = _verify_agent(tmp_path, "other-gpu.db", runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["transient_device_clients"] == [], result.details


def test_device_path_falls_back_to_the_index_without_a_minor_number(tmp_path) -> None:
    """A driver whose XML omits the minor number must stay fail-closed."""

    runner = DevicePathRunner(NO_MINOR_NUMBER_XML)
    proc = _proc_root_holding(tmp_path, "/dev/nvidia0")
    agent = _verify_agent(tmp_path, "fallback.db", runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "GPU-a:222:python" in (result.error or ""), result.error
    assert [item for item in runner.commands if "--query-gpu=uuid,index" in item], (
        f"the index query is the fallback and must have run: {runner.commands}"
    )


def test_verify_queries_the_device_map_once_per_verification(tmp_path) -> None:
    """Three samples used to mean three nvidia-smi calls, per GPU, per attempt.

    On an 8-GPU node with persistenced stopped, one busy-retrying reset ran up
    to 96 nvidia-smi invocations just to learn a mapping that cannot change
    while the GPU services are quiesced.
    """

    runner = DevicePathRunner(MINOR_NUMBER_XML)
    proc = _proc_root_holding(tmp_path, "/dev/nvidia0")
    agent = _verify_agent(
        tmp_path, "cached-map.db", runner, proc, device_client_samples=3
    )

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert runner.xml_queries() == 1, runner.commands


MIXED_MINOR_INDEX = "".join(f"GPU-{index}, {index}\n" for index in range(8))

MIXED_MINOR_INVENTORY = "".join(f"GPU-{index}\n" for index in range(8))

DUPLICATE_MINOR_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:53:00.0">
    <uuid>GPU-a</uuid>
    <minor_number>0</minor_number>
  </gpu>
  <gpu id="00000000:64:00.0">
    <uuid>GPU-b</uuid>
    <minor_number>0</minor_number>
  </gpu>
</nvidia_smi_log>
"""


def _mixed_minor_xml(eighth_minor: str) -> str:
    """Eight GPUs, seven with a minor number and one that fell off the bus.

    The seven run their minor numbers opposite to their PCI order, so the index
    map and the minor map disagree about every one of them. ``eighth_minor`` is
    what the eighth GPU reports: ``N/A`` is what nvidia-smi prints for a card
    the driver can no longer reach, and an empty string leaves the element out.
    """

    entries = []
    for index in range(7):
        entries.append(
            f'  <gpu id="00000000:0{index}:00.0">\n'
            f"    <uuid>GPU-{index}</uuid>\n"
            f"    <minor_number>{7 - index}</minor_number>\n"
            "  </gpu>"
        )
    minor = f"    <minor_number>{eighth_minor}</minor_number>\n" if eighth_minor else ""
    entries.append(
        '  <gpu id="00000000:07:00.0">\n    <uuid>GPU-7</uuid>\n' + minor + "  </gpu>"
    )
    return (
        '<?xml version="1.0" ?>\n<nvidia_smi_log>\n'
        + "\n".join(entries)
        + "\n</nvidia_smi_log>\n"
    )


@pytest.mark.parametrize("eighth_minor", ["N/A", ""])
def test_one_unreported_minor_does_not_reindex_its_healthy_siblings(
    tmp_path, eighth_minor: str
) -> None:
    """One GPU off the bus must not move the other seven onto index paths.

    Falling back to the index map for the whole node whenever a single GPU
    answers "N/A" throws away seven correct minor numbers and remaps every
    healthy GPU by PCI order -- the exact mapping that points a UUID at another
    GPU's device node. On this node GPU-0's node is /dev/nvidia7, so the index
    map would have inspected /dev/nvidia0 and passed the reset with a live
    holder on the target.
    """

    runner = DevicePathRunner(_mixed_minor_xml(eighth_minor), index=MIXED_MINOR_INDEX)
    proc = _proc_root_holding(tmp_path, "/dev/nvidia7")
    agent = _verify_agent(tmp_path, "mixed-minor.db", runner, proc)

    result = agent.execute(
        envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, gpu_uuids=["GPU-0"]))
    )

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "GPU-0:222:python" in (result.error or ""), result.error
    assert runner.index_queries() == 0, (
        f"seven usable minor numbers must not fall back to the index: {runner.commands}"
    )


@pytest.mark.parametrize("eighth_minor", ["N/A", ""])
def test_a_target_without_a_minor_number_fails_the_verification_closed(
    tmp_path, eighth_minor: str
) -> None:
    """An unresolvable target is refused, not quietly dropped from the scan.

    ``_verify_no_clients`` filtered a target the device map could not resolve
    out of the scan and then reported "verified": the one gate the destructive
    reset has passed without ever looking at that GPU's device node. Quiesce
    already refuses a target it cannot map; this says the same thing.
    """

    runner = DevicePathRunner(_mixed_minor_xml(eighth_minor), index=MIXED_MINOR_INDEX)
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    agent = _verify_agent(tmp_path, "unresolvable.db", runner, proc)

    result = agent.execute(
        envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, gpu_uuids=["GPU-7"]))
    )

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "cannot resolve device node for GPU-7" in (result.error or ""), result.error


@pytest.mark.parametrize("eighth_minor", ["N/A", ""])
def test_a_verify_without_targets_means_every_gpu_in_the_inventory(
    tmp_path, eighth_minor: str
) -> None:
    """No explicit UUIDs means "all of them", not "all the ones that resolved".

    ``_require_resolvable_targets`` returned immediately on an empty target
    list, and the scan then walked whatever the device map happened to contain.
    On a node where one GPU no longer reports a minor number that is seven
    device nodes out of eight, reported as ``verified_no_gpu_clients`` -- a
    false green in front of a destructive step, and precisely the hole the
    explicit-target check was added to close.
    """

    runner = DevicePathRunner(
        _mixed_minor_xml(eighth_minor),
        index=MIXED_MINOR_INDEX,
        inventory=MIXED_MINOR_INVENTORY,
    )
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    agent = _verify_agent(tmp_path, "implicit-targets.db", runner, proc)

    result = agent.execute(
        envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, gpu_uuids=[]))
    )

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "cannot resolve device node for GPU-7" in (result.error or ""), result.error


def test_a_verify_without_targets_passes_when_the_whole_node_resolves(tmp_path) -> None:
    """The implicit target set must not turn a healthy node into a refusal."""

    runner = DevicePathRunner(MINOR_NUMBER_XML)
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    agent = _verify_agent(tmp_path, "implicit-ok.db", runner, proc)

    result = agent.execute(
        envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, gpu_uuids=[]))
    )

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert [item for item in runner.commands if "--query-gpu=uuid" in item], (
        f"the implicit target set is the local inventory: {runner.commands}"
    )


def test_two_uuids_on_one_device_node_resolve_to_neither(tmp_path) -> None:
    """A report that gives two GPUs the same minor cannot be trusted for either.

    Keeping the last writer would hand one UUID a device node that provably
    belongs to another GPU, which is the failure this whole map exists to stop.
    """

    runner = DevicePathRunner(DUPLICATE_MINOR_XML)
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    agent = _verify_agent(tmp_path, "duplicate-minor.db", runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "cannot resolve device node for GPU-a" in (result.error or ""), result.error


@pytest.mark.parametrize(
    ("xml", "target", "cause"),
    [
        (_mixed_minor_xml("N/A"), "GPU-7", "reports no minor number for it"),
        (MINOR_NUMBER_XML, "GPU-z", "no GPU on this node reports this UUID"),
        (
            DUPLICATE_MINOR_XML,
            "GPU-a",
            "reports device node /dev/nvidia0 for more than one GPU",
        ),
    ],
    ids=["no-minor-number", "not-on-this-node", "duplicate-minor"],
)
def test_an_unresolvable_target_says_why_it_cannot_be_resolved(
    tmp_path, xml: str, target: str, cause: str
) -> None:
    """ "cannot resolve device node for GPU-..." was the whole diagnosis.

    The three causes want three different actions -- a card that fell off the
    bus (drain and RMA the node), a workflow carrying a UUID this node never
    had (a stale plan, or the wrong node in the step), and a driver report that
    gives two GPUs one device node (nothing here can be trusted) -- and the
    refusal that blocks the destructive step named none of them, leaving the
    operator to re-derive the map by hand from ``nvidia-smi -q -x``.
    """

    runner = DevicePathRunner(xml, index=MIXED_MINOR_INDEX)
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    agent = _verify_agent(tmp_path, "unresolvable-cause.db", runner, proc)

    result = agent.execute(
        envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, gpu_uuids=[target]))
    )

    assert result.status is NodeActionStatus.FAILED, result.details
    assert f"cannot resolve device node for {target}" in (result.error or ""), (
        result.error
    )
    assert cause in (result.error or ""), (
        f"the refusal must name why {target} has no device node: {result.error}"
    )


def test_the_device_map_cache_is_per_command_not_per_executor(tmp_path) -> None:
    """A second command must resolve the map itself, not inherit the first's.

    The window's cache used to live on the executor instance, and the action
    pool runs four commands on that one instance: a command that opens its
    window while another one is still inside answers from the older command's
    table, and the ``depth += 1`` / ``-= 1`` counter that was meant to bound
    that is a read-modify-write with no lock -- one lost update pins the depth
    above zero and the table is never dropped again. A driver reload renumbers
    the device nodes, so an inherited table points a UUID at another GPU.

    The first command is held inside its window while the second one resolves,
    which is the interleaving itself.
    """

    runner = DevicePathRunner(MINOR_NUMBER_XML)
    proc = tmp_path / "empty-proc"
    proc.mkdir()
    first_map_resolved = Event()
    second_map_resolved = Event()

    def hold_the_first_command_inside_its_window(_targets):
        if first_map_resolved.is_set():
            second_map_resolved.set()
        else:
            first_map_resolved.set()
            second_map_resolved.wait(timeout=15)
        return []

    agent = _verify_agent(
        tmp_path,
        "cache-scope.db",
        runner,
        proc,
        device_client_finder=hold_the_first_command_inside_its_window,
    )

    def verify(command_id: str):
        return agent.execute(
            envelope(
                command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, command_id=command_id)
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(verify, "workflow/v1/node-a")
        assert first_map_resolved.wait(timeout=15), (
            "the first command never reached its device-client scan"
        )
        second = pool.submit(verify, "workflow/v2/node-a")
        answers = [future.result(timeout=20) for future in (first, second)]

    for answer in answers:
        assert answer.status is NodeActionStatus.SUCCEEDED, answer.error
    assert runner.xml_queries() == 2, (
        f"each command must resolve its own device map: {runner.commands}"
    )


def test_a_second_close_cannot_reopen_a_manual_confirmation(
    tmp_path, monkeypatch
) -> None:
    """Two callers close one attempt: the second must not make it retryable.

    Both the poll and the pool's done-callback close a finished future, so the
    same attempt goes through the wrapper twice. The first close writes the
    INTERRUPTED marker that means "manual confirmation, never replay"; the
    second one then saw a row that already carried a result, fell to the branch
    for an attempt that was never dispatched, and saved a *retryable* FAILED row
    at attempt+1 over the marker. The control plane resubmits a retryable
    failure, so the destructive handler runs a second time -- which is exactly
    what the marker exists to prevent.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    runner = FakeRunner()
    agent = reset_agent(tmp_path, "double-close.db", runner)
    real_mark_interrupted = agent.ledger.mark_interrupted
    submit_returned = Event()
    attempt_closed = Event()
    poll_answered = Event()

    def dispatch_then_crash(value: SignedNodeAction):
        # An attempt on disk with no result: the handler ran and the wrapper
        # around it failed, which is the only state that closes as INTERRUPTED.
        agent.ledger.mark_in_progress(value.command, 1, signature=value.signature)
        submit_returned.wait(timeout=5)
        raise sqlite3.OperationalError("database is locked")

    def close_then_wait_for_the_poll(command_id: str, attempt: int, error: str):
        closed = real_mark_interrupted(command_id, attempt, error)
        attempt_closed.set()
        # Hold the done-callback before it forgets the future, so the poll has
        # to close the same attempt a second time.
        poll_answered.wait(timeout=5)
        return closed

    monkeypatch.setattr(agent, "execute", dispatch_then_crash)
    monkeypatch.setattr(agent.ledger, "mark_interrupted", close_then_wait_for_the_poll)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        submit_returned.set()
        assert attempt_closed.wait(timeout=5), "the attempt was never closed"
        polled = client.get(
            "/v1/node-actions/result", params=result_params(signed.command.command_id)
        )
        poll_answered.set()

    assert polled.status_code == 200, f"the poll must answer: {polled.text}"
    payload = polled.json()
    assert payload["state"] == NodeActionExecutionState.INTERRUPTED.value, payload
    assert payload["result"]["retryable"] is False, (
        f"a closed attempt must not be reopened as retryable: {payload}"
    )
    assert payload["result"]["attempt"] == 1, (
        f"no second attempt was dispatched: {payload}"
    )
    stored = agent.ledger.get(signed.command.command_id)
    assert stored is not None and stored.retryable is False, (
        f"the manual-confirmation marker must survive on disk: {stored}"
    )
    assert reset_invocations(runner) == 0, f"nothing may have reset: {runner.commands}"


def test_a_poll_answers_a_persisted_result_over_a_fabricated_interrupt(
    tmp_path, monkeypatch
) -> None:
    """A row that finished between the two reads must answer with its result.

    The pool wrapper's close path decides on ``latest_row`` and then closes the
    attempt with ``mark_interrupted``, which answers None for a row that
    already carries a result -- exactly what happens when the attempt finishes
    between those two reads. The branch then fabricated an INTERRUPTED answer
    of its own, so a SUCCEEDED reset sitting on disk was reported to the
    control plane as needing manual confirmation: the node stays quarantined
    and an operator is paged for a GPU that is already back. The executor's own
    close path already falls back to the persisted result; the pool wrapper in
    ``app.py`` did not.

    ``latest_row`` is pinned to the pre-result view and ``get`` left honest,
    which is the interleaving itself: two reads of the same row, the second one
    after the result was committed.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    runner = FakeRunner()
    agent = reset_agent(tmp_path, "stale-inprogress-read.db", runner)
    real_execute = agent.execute
    real_latest_row = agent.ledger.latest_row
    poll_done = Event()

    def execute_then_crash(value: SignedNodeAction):
        # The result is written before ``execute`` returns, so anything that
        # fails afterwards -- the pool wrapper included -- fails with a real
        # result already on disk.
        result = real_execute(value)
        raise sqlite3.OperationalError(f"database is locked ({result.status.value})")

    def latest_row_before_the_result(command_id: str):
        row = real_latest_row(command_id)
        if row is None or row[2] is None:
            return row
        if current_thread().name.startswith("gpu-fault-node-action"):
            # The pool's done-callback closes the attempt too, and it drops the
            # future on its way out. Hold it until the poll has answered so the
            # HTTP caller is the one that has to close this attempt.
            poll_done.wait(timeout=5)
        return "IN_PROGRESS", row[1], None

    monkeypatch.setattr(agent, "execute", execute_then_crash)
    monkeypatch.setattr(agent.ledger, "latest_row", latest_row_before_the_result)
    monkeypatch.setattr(
        agent.ledger,
        "get",
        lambda command_id: (real_latest_row(command_id) or (None, None, None))[2],
    )
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submitted = submit_action(client, signed)
        polled = wait_for_result(client, signed.command.command_id)
        poll_done.set()

    assert reset_invocations(runner) == 1, (
        f"the GPU must be reset exactly once: {runner.commands}"
    )
    assert polled.status_code == 200, f"the poll must answer: {polled.text}"
    for answer in (submitted, polled):
        payload = answer.json()
        assert payload["state"] in {
            NodeActionExecutionState.PENDING.value,
            NodeActionExecutionState.SUCCEEDED.value,
        }, f"the persisted SUCCEEDED result is the only answer: {payload}"
    assert polled.json()["state"] == NodeActionExecutionState.SUCCEEDED.value, (
        polled.json()
    )
    history = agent.ledger.attempt_history(signed.command.command_id)
    assert [(row["attempt"], row["state"]) for row in history] == [(1, "SUCCEEDED")], (
        f"a persisted result must not be rewritten or retried: {history}"
    )
