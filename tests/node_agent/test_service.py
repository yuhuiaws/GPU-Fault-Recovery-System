from __future__ import annotations

from gpu_fault.node_agent.heartbeat import CollectorServiceStates, XidPolicyVersion

from ._support import (
    SECRET,
    AgentHeartbeatRejected,
    AgentHeartbeatReporter,
    CompletedProcess,
    Event,
    FakeRunner,
    GpuServiceQuiesceManager,
    HTTPError,
    TimeoutExpired,
    WorkflowOperation,
    agent_config_digest,
    agent_config_payload,
    executor_from_environment,
    heartbeat_reporter_from_environment,
    io,
    logging,
    mock,
    node_action_executor,
    os,
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


@pytest.mark.parametrize("token", ["1", "yes", "on"])
def test_node_agent_destructive_gates_accept_every_enabled_token(
    tmp_path, monkeypatch, token: str
) -> None:
    """``GPU_FAULT_NODE_ALLOW_GPU_RESET=1`` used to leave the reset gate shut."""

    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", SECRET)
    monkeypatch.setenv("NODE_NAME", "node-a")
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOWED_OPERATIONS", "VERIFY_NO_GPU_CLIENTS")
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_DB", str(tmp_path / "actions.db"))
    monkeypatch.delenv("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_GPU_RESET", token)
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_FABRIC_RESET", token)

    agent = executor_from_environment(validate_runtime_paths=False)

    assert agent.reset_enabled is True
    assert agent.fabric_reset_enabled is True


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


def test_heartbeat_caches_unit_enablement_between_ticks() -> None:
    """``systemctl is-enabled`` is asked once, not twice per unit per tick.

    Every tick spawned two ``systemctl`` calls per collector unit with a 5 s
    timeout each, in sequence. While a quiesce holds kubelet down systemd
    answers slowly, so one report took tens of seconds and pushed the next one
    out. Enablement only changes when the installer installs or removes a unit
    -- and the installer restarts this agent -- so it is cached; ``is-active``
    stays live.
    """

    calls: list[list[str]] = []

    def runner(argv, **_):
        calls.append(list(argv))
        answer = "active" if argv[1] == "is-active" else "enabled"
        return CompletedProcess(argv, 0, stdout=answer + "\n", stderr="")

    states = CollectorServiceStates(
        units=lambda: ["gpu-fault-host-collector.service"],
        runner=runner,
        refresh_every=30,
    )

    first = states()
    second = states()

    assert first == second, (first, second)
    assert first["gpu-fault-host-collector.service"].enabled == "enabled", first
    assert first["gpu-fault-host-collector.service"].active == "active", first
    assert [argv[1] for argv in calls] == ["is-active", "is-enabled", "is-active"], (
        calls
    )


def test_heartbeat_refreshes_unit_enablement_when_the_unit_set_changes() -> None:
    """A newly installed unit is asked for -- and only that unit.

    A changed unit set used to re-query the whole node's enablement, which is
    the very burst the cache exists to avoid; the unit that is missing from the
    cache is the only one whose answer is unknown.
    """

    calls: list[list[str]] = []
    units = [["gpu-fault-host-collector.service"]]

    def runner(argv, **_):
        calls.append(list(argv))
        return CompletedProcess(argv, 0, stdout="enabled\n", stderr="")

    states = CollectorServiceStates(
        units=lambda: units[0], runner=runner, refresh_every=30
    )

    states()
    units[0] = [
        "gpu-fault-host-collector.service",
        "gpu-fault-metrics-collector.service",
    ]
    calls.clear()
    second = states()

    assert sorted(second) == units[0], second
    assert [argv for argv in calls if argv[1] == "is-enabled"] == [
        ["systemctl", "is-enabled", "gpu-fault-metrics-collector.service"]
    ], calls


def test_heartbeat_refreshes_unit_enablement_every_refresh_interval() -> None:
    """A hand-run ``systemctl disable`` is still reported, just not per tick."""

    calls: list[list[str]] = []

    def runner(argv, **_):
        calls.append(list(argv))
        return CompletedProcess(argv, 0, stdout="enabled\n", stderr="")

    states = CollectorServiceStates(
        units=lambda: ["gpu-fault-host-collector.service"],
        runner=runner,
        refresh_every=3,
    )

    for _ in range(4):
        states()

    assert [argv[1] for argv in calls].count("is-enabled") == 2, calls
    assert [argv[1] for argv in calls].count("is-active") == 4, calls


def test_heartbeat_unit_enablement_failure_is_not_cached() -> None:
    """A timed-out ``is-enabled`` must not pin "unknown" for 30 ticks."""

    answers = [TimeoutExpired(["systemctl"], 5), None]

    def runner(argv, **_):
        if argv[1] == "is-enabled":
            answer = answers.pop(0) if answers else None
            if answer is not None:
                raise answer
        return CompletedProcess(argv, 0, stdout="enabled\n", stderr="")

    states = CollectorServiceStates(
        units=lambda: ["gpu-fault-host-collector.service"],
        runner=runner,
        refresh_every=30,
    )

    first = states()
    second = states()

    assert first["gpu-fault-host-collector.service"].enabled == "unknown", first
    assert second["gpu-fault-host-collector.service"].enabled == "enabled", second


def test_heartbeat_enablement_failure_re_asks_only_the_failed_unit() -> None:
    """One slow ``systemctl`` must not put the whole node back on the next tick.

    A failed ``is-enabled`` is not cached, and the unit set was then compared
    with the cache: one timeout on a loaded node re-queried every collector
    unit on the following tick -- eight extra 5 s calls on the tick that was
    already running late, which is what pushed the heartbeat interval out.
    """

    calls: list[list[str]] = []
    units = ["gpu-fault-host-collector.service", "gpu-fault-metrics-collector.service"]
    timeouts = [TimeoutExpired(["systemctl"], 5)]

    def runner(argv, **_):
        calls.append(list(argv))
        if argv[1] == "is-enabled" and argv[2] == units[0] and timeouts:
            raise timeouts.pop()
        return CompletedProcess(argv, 0, stdout="enabled\n", stderr="")

    states = CollectorServiceStates(
        units=lambda: units, runner=runner, refresh_every=30
    )

    first = states()
    calls.clear()
    second = states()

    assert first[units[0]].enabled == "unknown", first
    assert second[units[0]].enabled == "enabled", second
    assert [argv for argv in calls if argv[1] == "is-enabled"] == [
        ["systemctl", "is-enabled", units[0]]
    ], calls


def test_heartbeat_reloads_the_xid_policy_only_when_its_mtime_changes(tmp_path) -> None:
    """The policy file was parsed and validated on every heartbeat tick."""

    policy_path = tmp_path / "xid-policy.yaml"
    policy_path.write_text("catalog: v1\n", encoding="utf-8")
    loads: list[str | None] = []

    def loader(path):
        loads.append(path)
        return mock.Mock(mapping_version=f"policy-v{len(loads)}")

    version = XidPolicyVersion(str(policy_path), loader=loader)

    first = version()
    second = version()
    os.utime(policy_path, (1, 1))
    third = version()

    assert (first, second) == ("policy-v1", "policy-v1"), (first, second)
    assert third == "policy-v2", third
    assert loads == [str(policy_path), str(policy_path)], loads


def test_heartbeat_caches_a_unit_systemd_does_not_know() -> None:
    """An uninstalled collector unit must not refresh the whole set per tick."""

    calls: list[list[str]] = []

    def runner(argv, **_):
        calls.append(list(argv))
        return CompletedProcess(argv, 1, stdout="", stderr="Unit not found.\n")

    states = CollectorServiceStates(
        units=lambda: ["gpu-fault-host-collector.service"],
        runner=runner,
        refresh_every=30,
    )

    states()
    states()

    assert [argv[1] for argv in calls].count("is-enabled") == 1, calls
