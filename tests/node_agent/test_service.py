from __future__ import annotations

from ._support import (
    SECRET,
    AgentHeartbeatRejected,
    AgentHeartbeatReporter,
    Event,
    FakeRunner,
    GpuServiceQuiesceManager,
    HTTPError,
    WorkflowOperation,
    agent_config_digest,
    agent_config_payload,
    heartbeat_reporter_from_environment,
    io,
    logging,
    mock,
    node_action_executor,
    pytest,
    run,
)


def test_node_agent_plaintext_requires_explicit_opt_in(monkeypatch) -> None:
    for name in (
        "GPU_FAULT_NODE_AGENT_TLS_CERT",
        "GPU_FAULT_NODE_AGENT_TLS_KEY",
        "GPU_FAULT_NODE_AGENT_TLS_CLIENT_CA",
        "GPU_FAULT_NODE_AGENT_ALLOW_PLAINTEXT",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="TLS is required"):
        run()


def test_device_clients_use_configured_host_proc_root(tmp_path) -> None:
    proc = tmp_path / "host-proc"
    fd_dir = proc / "222" / "fd"
    fd_dir.mkdir(parents=True)
    (proc / "222" / "comm").write_text("python\n")
    (fd_dir / "7").symlink_to("/dev/nvidia0")
    agent = node_action_executor(
        tmp_path,
        "agent.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        proc_root=str(proc),
        runner=FakeRunner(),
        now=None,
    )

    assert agent._device_clients({"GPU-a"}) == [
        {
            "gpu_uuid": "GPU-a",
            "pid": "222",
            "process_name": "python",
            "device": "/dev/nvidia0",
        }
    ]


def test_agent_config_payload_covers_execution_affecting_settings(tmp_path) -> None:
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        failsafe_seconds=30,
        retry_seconds=10,
        container_stop_timeout_seconds=40,
        container_restore_timeout_seconds=240,
        device_sweep_timeout_seconds=25,
    )
    executor = node_action_executor(
        tmp_path,
        "config.db",
        allowed_operations={WorkflowOperation.QUIESCE_GPU_SERVICES},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=manager,
        driver_remediation_command=("/opt/driver", "--apply"),
        firmware_update_command=("/opt/firmware", "--apply"),
        firmware_verify_command=("/opt/firmware", "--verify"),
        device_client_samples=4,
        device_client_sample_interval_seconds=3,
        inflight_wait_timeout_seconds=2200,
        python_stack_tool="/opt/gpu-fault/py-spy",
        now=None,
    )

    payload = agent_config_payload(executor, "profile-a")

    assert payload["driver_remediation_command"] == ["/opt/driver", "--apply"]
    assert payload["firmware_update_command"] == ["/opt/firmware", "--apply"]
    assert payload["firmware_verify_command"] == ["/opt/firmware", "--verify"]
    assert payload["device_client_samples"] == 4
    assert payload["device_client_sample_interval_seconds"] == 3
    assert payload["inflight_wait_timeout_seconds"] == 2200
    assert payload["python_stack_tool"] == "/opt/gpu-fault/py-spy"
    assert payload["quiesce"]["container_restore_timeout_seconds"] == 240
    assert payload["quiesce"]["device_sweep_timeout_seconds"] == 25


def test_heartbeat_digest_comes_from_the_shared_helper(tmp_path, monkeypatch) -> None:
    """Deploy-time pins and heartbeats must use one computation.

    The control-plane pin is only useful if it equals what an agent
    reports. Keeping a second copy of the payload in deploy.sh let a
    3-service quiesce literal drift from the agent's real 6-service
    value, and the resulting mismatch failed the fleet consistency
    gate on every node-owned step -- no reset, no quiesce, ever.
    """
    monkeypatch.setenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", "http://control-plane")
    monkeypatch.setenv("GPU_FAULT_NODE_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_NODE_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "profile-a")
    monkeypatch.setenv("NODE_NAME", "node-a")
    executor = node_action_executor(
        tmp_path,
        "shared.db",
        allowed_operations={
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        },
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=GpuServiceQuiesceManager(
            state_dir=str(tmp_path / "quiesce"),
            services=(
                "nvidia-fabricmanager",
                "nvidia-dcgm",
                "nvidia-persistenced",
                "gpu-fault-metrics-collector",
                "gpu-fault-host-collector",
                "kubelet",
            ),
        ),
        now=None,
    )

    reporter = heartbeat_reporter_from_environment(executor)

    assert reporter is not None
    assert reporter.config_digest == agent_config_digest(executor, "profile-a")


def test_heartbeat_requires_boot_or_instance_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", "http://control-plane")
    monkeypatch.setenv("GPU_FAULT_NODE_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_NODE_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "profile-a")
    monkeypatch.setenv("NODE_NAME", "node-a")
    for name in ("GPU_FAULT_NODE_INSTANCE_ID", "NODE_UID", "EC2_INSTANCE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("gpu_fault.node_agent.heartbeat._read_boot_id", lambda: None)
    executor = node_action_executor(
        tmp_path,
        "identity.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        now=None,
    )

    with pytest.raises(ValueError, match="requires a boot ID or node instance ID"):
        heartbeat_reporter_from_environment(executor)


def test_heartbeat_policy_version_is_refreshed_per_report() -> None:
    versions = iter(["policy-v1", "policy-v2"])
    reports = []
    reporter = AgentHeartbeatReporter(
        control_plane_url="https://control.example",
        secret=SECRET,
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="initial",
        policy_version_provider=lambda: next(versions),
        runtime_profile_version="profile-a",
        config_digest="b" * 64,
        allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
        boot_id="boot-a",
        sender=lambda _url, envelope: (
            reports.append(envelope.heartbeat.policy_version) or {}
        ),
    )

    reporter.report_once()
    reporter.report_once()

    assert reports == ["policy-v1", "policy-v2"]


def test_heartbeat_sends_full_unit_inventory_only_on_change() -> None:
    inventories = iter(
        [
            ["gpu-fault-node-agent.service"],
            ["gpu-fault-node-agent.service"],
            ["gpu-fault-node-agent.service", "gpu-fault-dcgm-exporter.service"],
        ]
    )
    reports = []
    reporter = AgentHeartbeatReporter(
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
        installed_units_provider=lambda: next(inventories),
        boot_id="boot-a",
        sender=lambda _url, envelope: (
            reports.append(envelope.heartbeat.installed_unit_report) or {}
        ),
    )

    reporter.report_once()
    reporter.report_once()
    reporter.report_once()

    assert reports[0].units == ["gpu-fault-node-agent.service"]
    assert reports[1].units is None
    assert reports[1].digest == reports[0].digest
    assert reports[2].units == [
        "gpu-fault-dcgm-exporter.service",
        "gpu-fault-node-agent.service",
    ]
    assert reports[2].digest != reports[1].digest


def test_failed_heartbeat_retries_full_unit_inventory() -> None:
    reports = []

    def sender(_url, envelope):
        reports.append(envelope.heartbeat.installed_unit_report)
        if len(reports) == 1:
            raise RuntimeError("temporary failure")
        return {}

    reporter = AgentHeartbeatReporter(
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
        installed_units_provider=lambda: ["gpu-fault-node-agent.service"],
        boot_id="boot-a",
        sender=sender,
    )

    with pytest.raises(RuntimeError, match="temporary failure"):
        reporter.report_once()
    reporter.report_once()

    assert reports[0].units is not None
    assert reports[1].units == reports[0].units


def test_control_plane_can_request_full_unit_inventory_again() -> None:
    reports = []

    def sender(_url, envelope):
        report = envelope.heartbeat.installed_unit_report
        reports.append(report)
        if len(reports) == 2:
            raise AgentHeartbeatRejected(
                409, "full installed unit inventory is required when digest changes"
            )
        return {}

    reporter = AgentHeartbeatReporter(
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
        installed_units_provider=lambda: ["gpu-fault-node-agent.service"],
        boot_id="boot-a",
        sender=sender,
    )

    reporter.report_once()
    with pytest.raises(AgentHeartbeatRejected):
        reporter.report_once()
    reporter.report_once()

    assert reports[0].units is not None
    assert reports[1].units is None
    assert reports[2].units == reports[0].units


def test_heartbeat_rejection_names_the_reason_and_the_remedy(caplog) -> None:
    """A retired incarnation never clears by retrying.

    ``hyperpod-i-00000000000000001`` sat fenced from 2026-08-12 because
    the journal only ever showed ``HTTP Error 409: Conflict`` with no
    cause and no remedy.
    """

    attempts = []

    def refuse(_url, _envelope):
        attempts.append(1)
        raise AgentHeartbeatRejected(
            409, '{"detail":"agent incarnation has been retired"}'
        )

    reporter = AgentHeartbeatReporter(
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
        sender=refuse,
    )
    stop = Event()

    def stop_after_two(_timeout):
        if len(attempts) >= 2:
            stop.set()

    with caplog.at_level(logging.ERROR):
        stop_wait = stop.wait
        stop.wait = stop_after_two  # type: ignore[method-assign]
        try:
            reporter.run(stop)
        finally:
            stop.wait = stop_wait  # type: ignore[method-assign]

    assert len(attempts) == 2
    messages = [record.getMessage() for record in caplog.records]
    assert all("Traceback" not in message for message in messages), (
        f"heartbeat rejection logs leaked tracebacks: {messages}"
    )
    assert "agent incarnation has been retired" in messages[0]
    assert "reactivate the agent or reboot the node" in messages[0]
    assert "refused 2 time(s) in a row" in messages[1]


def test_heartbeat_rejection_carries_the_control_plane_detail() -> None:
    class Refused(HTTPError):
        def __init__(self) -> None:
            super().__init__(
                "https://control.example/v1/fleet/agents/heartbeat",
                409,
                "Conflict",
                {},  # type: ignore[arg-type]
                io.BytesIO(b'{"detail":"agent incarnation has been retired"}'),
            )

    def urlopen(_request, timeout=None):
        raise Refused()

    with mock.patch("gpu_fault.node_agent.heartbeat.urllib_request.urlopen", urlopen):
        with pytest.raises(AgentHeartbeatRejected) as excinfo:
            AgentHeartbeatReporter._send(
                "https://control.example",
                mock.Mock(model_dump_json=lambda: "{}"),
                cluster_id="cluster-a",
                bearer_token="token-a",
            )

    assert excinfo.value.status == 409
    assert "retired" in excinfo.value.detail


def test_device_client_sampling_config_is_validated(tmp_path) -> None:
    for samples, interval in ((0, 2.0), (11, 2.0), (3, -1.0), (3, 11.0)):
        with pytest.raises(ValueError):
            node_action_executor(
                tmp_path,
                "cfg.db",
                allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
                runner=FakeRunner(),
                device_client_samples=samples,
                device_client_sample_interval_seconds=interval,
            )
