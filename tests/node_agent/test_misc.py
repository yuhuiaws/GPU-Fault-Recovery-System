from __future__ import annotations

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
    WorkflowOperation,
    command,
    create_node_agent_app,
    datetime,
    envelope,
    executor,
    inspect,
    json,
    node_action_executor,
    os,
    pytest,
    result_params,
    tarfile,
    time,
    timedelta,
    timezone,
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
        assert client.get("/healthz").json() == {"status": "ok"}


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
        client.post("/v1/node-actions", json=signed.model_dump(mode="json"))
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


def test_unsigned_result_query_escape_hatch_requires_expiry(
    tmp_path, monkeypatch
) -> None:
    agent = executor(tmp_path, FakeRunner())
    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED", "false")
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


def test_legacy_node_action_endpoint_runs_in_threadpool(tmp_path, monkeypatch) -> None:
    agent = executor(tmp_path, FakeRunner())
    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    app = create_node_agent_app(agent, heartbeat_reporter=None)
    route = next(
        item for item in app.routes if getattr(item, "path", None) == "/v1/node-actions"
    )

    assert not inspect.iscoroutinefunction(route.endpoint)


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
