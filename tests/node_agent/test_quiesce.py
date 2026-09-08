from __future__ import annotations

from ._support import (
    SECRET,
    CalledProcessError,
    CompletedProcess,
    GpuServiceQuiesceManager,
    NodeActionStatus,
    Path,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    executor_from_environment,
    hashlib,
    json,
    no_device_clients,
    node_action_executor,
    pytest,
    quiesce_executor,
    quiesce_manager,
    validate_host_proc_root,
)


def test_quiesce_arms_timer_stops_and_restores_active_services(tmp_path) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager", "nvidia-dcgm", "kubelet"})
    agent = quiesce_executor(tmp_path, runner)

    quiesced = agent.execute(envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES)))
    restored = agent.execute(
        envelope(
            command(
                WorkflowOperation.RESTORE_GPU_SERVICES,
                command_id="workflow/restore/node-a",
            )
        )
    )

    assert quiesced.status is NodeActionStatus.SUCCEEDED
    assert quiesced.details["quiesced"] is True
    assert runner.active == {"nvidia-fabricmanager", "nvidia-dcgm", "kubelet"}
    assert restored.status is NodeActionStatus.SUCCEEDED
    assert restored.details["restored_services"] == [
        "nvidia-fabricmanager",
        "nvidia-dcgm",
        "kubelet",
    ]
    assert restored.details["restore_settle_seconds"] == 0
    assert any(item[0] == "systemd-run" for item in runner.commands)
    stop_services = [
        item[2]
        for item in runner.commands
        if item[:2] == ["systemctl", "stop"] and not item[2].endswith(".timer")
    ]
    assert stop_services == ["kubelet", "nvidia-dcgm", "nvidia-fabricmanager"]


def test_quiesce_stops_exact_container_and_waits_for_restart(tmp_path) -> None:
    old_id = "a" * 64
    new_id = "b" * 64

    class ContainerRunner(ServiceRunner):
        def __init__(self):
            super().__init__(active={"kubelet"})
            self.container_id = old_id
            self.container_running = True

        def __call__(self, value, **kwargs):
            if value[:6] == ["ctr", "-n", "k8s.io", "containers", "list", "--quiet"]:
                self.commands.append(value)
                return CompletedProcess(
                    value, 0, stdout=f"{self.container_id}\n", stderr=""
                )
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "list"]:
                self.commands.append(value)
                rows = "TASK PID STATUS\n"
                if self.container_running:
                    rows += f"{self.container_id} 123 RUNNING\n"
                return CompletedProcess(value, 0, stdout=rows, stderr="")
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "kill"]:
                self.commands.append(value)
                self.container_running = False
                return CompletedProcess(value, 0, stdout="", stderr="")
            result = super().__call__(value, **kwargs)
            if value[:3] == ["systemctl", "start", "kubelet"]:
                self.container_id = new_id
                self.container_running = True
            return result

    runner = ContainerRunner()
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        containers=("aws-hyperpod/health-monitoring-agent",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
    )

    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    result = manager.restore(incident_id="incident-a")

    assert any(
        item[:8]
        == ["ctr", "-n", "k8s.io", "tasks", "kill", "--signal", "SIGTERM", "--all"]
        and item[-1] == old_id
        for item in runner.commands
    )
    assert runner.container_id == new_id
    assert result["restored_services"] == ["kubelet"]


def test_quiesce_accepts_multiple_running_daemonset_containers(tmp_path) -> None:
    first_id = "a" * 64
    second_id = "b" * 64

    class RollingRunner(ServiceRunner):
        def __call__(self, value, **kwargs):
            if value[:6] == ["ctr", "-n", "k8s.io", "containers", "list", "--quiet"]:
                return CompletedProcess(
                    value, 0, stdout=f"{first_id}\n{second_id}\n", stderr=""
                )
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "list"]:
                return CompletedProcess(
                    value,
                    0,
                    stdout=(
                        "TASK PID STATUS\n"
                        f"{first_id} 101 RUNNING\n"
                        f"{second_id} 102 RUNNING\n"
                    ),
                    stderr="",
                )
            return super().__call__(value, **kwargs)

    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        containers=("kube-system/nvidia-device-plugin-ctr",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=RollingRunner(active=set()),
    )

    assert manager._resolve_container_targets() == [
        {"selector": "kube-system/nvidia-device-plugin-ctr", "container_id": first_id},
        {"selector": "kube-system/nvidia-device-plugin-ctr", "container_id": second_id},
    ]


def test_restore_container_timeout_is_warning_and_cancels_timer(
    tmp_path, monkeypatch
) -> None:
    runner = ServiceRunner(active={"kubelet"})
    manager = quiesce_manager(tmp_path, runner)
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    monkeypatch.setattr(
        GpuServiceQuiesceManager,
        "_wait_containers_restored",
        lambda _self, _targets: ["kube-system/nvidia-device-plugin-ctr"],
    )

    result = manager.restore(incident_id="incident-a")

    assert result["restored"] is True
    assert result["container_restore_warning"] is True
    assert result["pending_container_selectors"] == [
        "kube-system/nvidia-device-plugin-ctr"
    ]
    assert not manager._state_path("incident-a").exists()
    assert [
        "systemctl",
        "stop",
        manager._timer_unit("incident-a") + ".timer",
    ] in runner.commands


def test_quiesce_only_restores_services_that_were_active(tmp_path) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager", "kubelet"})
    manager = quiesce_manager(tmp_path, runner)

    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    result = manager.restore(incident_id="incident-a")

    assert result["restored_services"] == ["nvidia-fabricmanager", "kubelet"]
    assert "nvidia-dcgm" not in runner.active


def test_quiesce_and_restore_commands_are_idempotent(tmp_path) -> None:
    runner = ServiceRunner(active={"kubelet"})
    agent = quiesce_executor(tmp_path, runner)
    quiesce_command = envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES))
    restore_command = envelope(
        command(
            WorkflowOperation.RESTORE_GPU_SERVICES, command_id="workflow/restore/node-a"
        )
    )

    assert agent.execute(quiesce_command) == agent.execute(quiesce_command)
    assert agent.execute(restore_command) == agent.execute(restore_command)
    assert sum(item[0] == "systemd-run" for item in runner.commands) == 1
    assert (
        sum(item[:3] == ["systemctl", "start", "kubelet"] for item in runner.commands)
        == 1
    )


def test_quiesce_kills_gpu_holders_systemd_does_not_own(tmp_path, monkeypatch) -> None:
    # nvidia-persistenced started by hand outside systemd: the unit is
    # inactive with MainPID=0, so "systemctl stop" is a silent no-op
    # while the process keeps every /dev/nvidiaN open. Before the sweep
    # this made VERIFY_NO_GPU_CLIENTS fail closed forever.
    runner = ServiceRunner(active={"kubelet"})
    manager = quiesce_manager(tmp_path, runner)
    holder = {
        "pid": "48784",
        "process_name": "nvidia-persiste",
        "devices": "/dev/nvidia0,/dev/nvidia1",
        "cgroup_paths": ["/system.slice/nvidia-persistenced.service"],
    }
    alive = [holder]
    monkeypatch.setattr(
        type(manager), "_device_holders", lambda _self, _targets=None: list(alive)
    )
    original = manager._run

    def runner_with_kill(cmd, *, check, timeout=30):
        if cmd[:2] == ["kill", "--signal"] and cmd[2] == "TERM":
            alive.clear()
        return original(cmd, check=check, timeout=timeout)

    monkeypatch.setattr(manager, "_run", runner_with_kill)

    details = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )

    assert details["swept_device_holders"] == [{**holder, "signal": "TERM"}]
    assert ["kill", "--signal", "TERM", "48784"] in runner.commands
    assert not any(item[:3] == ["kill", "--signal", "KILL"] for item in runner.commands)


def test_quiesce_escalates_to_kill_when_holder_ignores_term(
    tmp_path, monkeypatch
) -> None:
    runner = ServiceRunner(active={"kubelet"})
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        device_sweep_timeout_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
    )
    holder = {
        "pid": "48784",
        "process_name": "nvidia-persiste",
        "devices": "/dev/nvidia0",
        "cgroup_paths": ["/system.slice/nvidia-persistenced.service"],
    }
    monkeypatch.setattr(
        type(manager), "_device_holders", lambda _self, _targets=None: [holder]
    )

    details = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )

    # A holder that survives both signals still reaches QUIESCED:
    # quiesce is advisory, VERIFY_NO_GPU_CLIENTS is the authority that
    # refuses to reset.
    assert details["quiesced"] is True
    assert details["swept_device_holders"] == [{**holder, "signal": "KILL"}]
    assert ["kill", "--signal", "KILL", "48784"] in runner.commands


def test_device_holders_reads_only_gpu_device_nodes(tmp_path, monkeypatch) -> None:
    proc = tmp_path / "proc"
    for pid, name, target in (
        ("101", "nvidia-persiste", "/dev/nvidia0"),
        ("102", "dcgmi", "/dev/nvidiactl"),
        ("103", "bash", "/dev/nvidia-uvm"),
    ):
        fd_dir = proc / pid / "fd"
        fd_dir.mkdir(parents=True)
        (proc / pid / "comm").write_text(f"{name}\n")
        (fd_dir / "3").symlink_to(target)
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        proc_root=str(proc),
        restore_command="/opt/gpu-fault/restore",
    )

    holders = manager._device_holders()

    # /dev/nvidiactl and /dev/nvidia-uvm are not per-GPU device nodes,
    # so they are not what VERIFY_NO_GPU_CLIENTS gates on either.
    assert holders == [
        {
            "pid": "101",
            "process_name": "nvidia-persiste",
            "devices": "/dev/nvidia0",
            "cgroup_paths": [],
        }
    ]


def test_device_holder_sweep_is_limited_to_target_gpu_and_cgroup(tmp_path) -> None:
    proc = tmp_path / "proc"
    for pid, name, target, cgroup in (
        (
            "101",
            "nvidia-persiste",
            "/dev/nvidia0",
            "/system.slice/nvidia-persistenced.service",
        ),
        ("102", "python", "/dev/nvidia0", "/kubepods/pod-a/container-a"),
        ("103", "python", "/dev/nvidia0", "/kubepods/pod-b/container-b"),
        (
            "104",
            "nvidia-persiste",
            "/dev/nvidia1",
            "/system.slice/nvidia-persistenced.service",
        ),
    ):
        fd_dir = proc / pid / "fd"
        fd_dir.mkdir(parents=True)
        (proc / pid / "comm").write_text(f"{name}\n")
        (proc / pid / "cgroup").write_text(f"0::{cgroup}\n")
        (fd_dir / "3").symlink_to(target)
    runner = ServiceRunner(active=set())
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        device_sweep_timeout_seconds=0,
        proc_root=str(proc),
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
    )

    swept, skipped = manager._sweep_device_holders(
        target_device_paths={"/dev/nvidia0"}, workload_cgroup_paths={"/kubepods/pod-a"}
    )

    assert {item["pid"] for item in swept} == {"101", "102"}
    assert {item["pid"] for item in skipped} == {"103"}
    killed_pids = {
        item[-1] for item in runner.commands if item[:2] == ["kill", "--signal"]
    }
    assert killed_pids == {"101", "102"}
    assert "104" not in killed_pids


def test_host_proc_root_rejects_container_pid_namespace(tmp_path, monkeypatch) -> None:
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "1" / "comm").write_text("python\n")

    class Stat:
        st_dev = 1
        st_ino = 2

    monkeypatch.setattr("gpu_fault.node_agent.config.os.stat", lambda _: Stat())

    with pytest.raises(ValueError, match="container pid namespace"):
        validate_host_proc_root(str(proc))


def test_quiesce_stop_failure_leaves_failsafe_state_armed(tmp_path) -> None:
    runner = ServiceRunner(
        active={"nvidia-fabricmanager", "nvidia-dcgm", "kubelet"},
        fail_stop="nvidia-dcgm",
    )
    manager = quiesce_manager(tmp_path, runner)

    with pytest.raises(RuntimeError, match="stop failed"):
        manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")

    state_path = manager._state_path("incident-a")
    assert state_path.exists()
    assert '"phase": "QUIESCE_FAILED"' in state_path.read_text()
    assert any(item[0] == "systemd-run" for item in runner.commands)


def test_restore_failure_keeps_state_for_timer_retry(tmp_path) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager"})
    manager = quiesce_manager(tmp_path, runner)
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    runner.fail_start = "nvidia-fabricmanager"

    with pytest.raises(RuntimeError, match="start failed"):
        manager.restore(incident_id="incident-a")

    assert manager._state_path("incident-a").exists()
    assert not any(
        item[:2] == ["systemctl", "stop"] and item[2].endswith(".timer")
        for item in runner.commands
    )


def test_second_command_id_cannot_reset_inside_one_quiesce_window(tmp_path) -> None:
    # One quiesce window buys exactly one reset. A second command_id for
    # the same incident (an escalation step, or a resubmit of a reset the
    # node already issued) must be refused without touching the GPU.
    runner = ServiceRunner(active={"kubelet"})
    agent = quiesce_executor(tmp_path, runner)
    agent.execute(envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES)))

    first = agent.execute(
        envelope(command(WorkflowOperation.RESET_GPU, command_id="workflow/r1/node-a"))
    )
    second = agent.execute(
        envelope(command(WorkflowOperation.RESET_GPU, command_id="workflow/r2/node-a"))
    )

    assert first.status is NodeActionStatus.SUCCEEDED, first.error
    assert second.status is NodeActionStatus.FAILED, (
        "a second command must not reset inside one quiesce window"
    )
    assert second.retryable is False, "a refused second reset must not be retried"
    assert "already issued" in second.error, second.error
    assert len([item for item in runner.commands if "--gpu-reset" in item]) == 1, (
        "only one nvidia-smi --gpu-reset may run per quiesce window"
    )


def test_the_reset_window_is_claimed_once_and_not_by_remediation(tmp_path) -> None:
    runner = ServiceRunner(active={"kubelet"})
    manager = quiesce_manager(tmp_path, runner)
    quiesced = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )

    # Driver/firmware remediation runs under the quiesce but is not a reset.
    manager.assert_quiesced(incident_id="incident-a", for_reset=False)
    manager.assert_quiesced(incident_id="incident-a", for_reset=False)
    manager.assert_quiesced(incident_id="incident-a", command_id="workflow/r1/node-a")

    # Even the command that claimed the window may not reset twice: after a
    # timeout the first reset's outcome is unknown, so a retry is refused.
    with pytest.raises(RuntimeError, match="already issued"):
        manager.assert_quiesced(
            incident_id="incident-a", command_id="workflow/r1/node-a"
        )

    state_path = Path(quiesced["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["reset_issued"]["command_id"] == "workflow/r1/node-a", state
    assert state["reset_issued"]["issued_at"], "the claim must record when it was made"

    manager.restore(incident_id="incident-a")
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")

    # A real restore plus a fresh quiesce is what buys the next reset.
    manager.assert_quiesced(incident_id="incident-a", command_id="workflow/r2/node-a")


def test_driver_remediation_does_not_consume_the_reset_window(tmp_path) -> None:
    # Driver and firmware remediation are fenced on the same quiesce state as
    # a reset. If they claimed the window's single reset slot, the reset that
    # follows the remediation -- or a second remediation step -- would be
    # refused for no reason.
    executable = tmp_path / "driver-remediate"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()

    class RemediationRunner(ServiceRunner):
        def __call__(self, value, **kwargs):
            if "--query-gpu=driver_version" in value:
                self.commands.append(value)
                return CompletedProcess(value, 0, stdout="575.86.01\n", stderr="")
            return super().__call__(value, **kwargs)

    runner = RemediationRunner(active={"kubelet"})
    manager = quiesce_manager(tmp_path, runner)
    agent = node_action_executor(
        tmp_path,
        "remediate-window.db",
        allowed_operations={
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.REMEDIATE_DRIVER,
            WorkflowOperation.RESET_GPU,
        },
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=manager,
        driver_remediation_enabled=True,
        driver_remediation_command=(str(executable), "--branch", "{target}"),
        driver_remediation_sha256=digest,
        target_driver_branch=575,
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=lambda _: None,
    )
    agent.execute(envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES)))

    remediated = agent.execute(
        envelope(
            command(
                WorkflowOperation.REMEDIATE_DRIVER,
                command_id="workflow/d1/node-a",
                parameters={"target_driver_branch": 575},
            )
        )
    )
    reset = agent.execute(
        envelope(command(WorkflowOperation.RESET_GPU, command_id="workflow/r1/node-a"))
    )

    assert remediated.status is NodeActionStatus.SUCCEEDED, remediated.error
    assert reset.status is NodeActionStatus.SUCCEEDED, reset.error
    assert [item for item in runner.commands if "--gpu-reset" in item] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
    ], "the reset after a remediation must still be allowed exactly once"


def test_failed_quiesce_can_be_retried_after_inline_restore(tmp_path) -> None:
    # A stop that fails leaves QUIESCE_FAILED behind. Before this fix every
    # retry raised "quiesce is incomplete" until the 420 s fail-safe timer
    # fired, so the workflow died with kubelet/FM/DCGM still down.
    runner = ServiceRunner(
        active={"nvidia-fabricmanager", "nvidia-dcgm", "kubelet"},
        fail_stop="nvidia-dcgm",
    )
    manager = quiesce_manager(tmp_path, runner)

    with pytest.raises(RuntimeError, match="stop failed"):
        manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    runner.fail_stop = None

    details = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )

    assert details["quiesced"] is True, "a failed quiesce must be retryable"
    assert "already_quiesced" not in details, (
        "the retry must be a fresh quiesce, not a replay of the failed one"
    )
    assert ["systemctl", "start", "kubelet"] in runner.commands, (
        "the inline restore must start what the failed quiesce stopped"
    )
    assert ["systemctl", "stop", details["timer_unit"]] in runner.commands, (
        "the inline restore must cancel the failed window's fail-safe timer"
    )
    assert sum(item[0] == "systemd-run" for item in runner.commands) == 2, (
        "the retry must arm a fresh fail-safe timer"
    )
    assert runner.active == set(), "the retry must leave every service stopped"


def test_ctr_kill_of_an_exited_task_does_not_fail_quiesce(tmp_path) -> None:
    container_id = "c" * 64

    class ExitedTaskRunner(ServiceRunner):
        """A task that exits between ``tasks list`` and ``tasks kill``."""

        def __init__(self) -> None:
            super().__init__(active={"kubelet"})
            self.task_list_calls = 0

        def __call__(self, value, **kwargs):
            if value[:6] == ["ctr", "-n", "k8s.io", "containers", "list", "--quiet"]:
                self.commands.append(value)
                return CompletedProcess(value, 0, stdout=f"{container_id}\n", stderr="")
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "list"]:
                self.commands.append(value)
                self.task_list_calls += 1
                status = "RUNNING" if self.task_list_calls == 1 else "STOPPED"
                return CompletedProcess(
                    value,
                    0,
                    stdout=f"TASK PID STATUS\n{container_id} 123 {status}\n",
                    stderr="",
                )
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "kill"]:
                self.commands.append(value)
                if kwargs.get("check"):
                    raise CalledProcessError(
                        1, value, stderr="process already finished"
                    )
                return CompletedProcess(
                    value, 1, stdout="", stderr="process already finished"
                )
            return super().__call__(value, **kwargs)

    runner = ExitedTaskRunner()
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        containers=("aws-hyperpod/health-monitoring-agent",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
    )

    details = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )

    assert details["quiesced"] is True, (
        "a task that exited before the kill must not fail the quiesce"
    )
    assert details["stopped_containers"] == [
        {
            "selector": "aws-hyperpod/health-monitoring-agent",
            "container_id": container_id,
        }
    ], details["stopped_containers"]
    assert not any(
        item[:7] == ["ctr", "-n", "k8s.io", "tasks", "kill", "--signal", "SIGKILL"]
        for item in runner.commands
    ), "a task already stopped must not be escalated to SIGKILL"


def test_reset_requires_quiesce_state_when_production_gate_enabled(tmp_path) -> None:
    runner = ServiceRunner(active=set())
    agent = quiesce_executor(tmp_path, runner)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert "requires an active service quiesce state" in result.error
    assert not any("--gpu-reset" in item for item in runner.commands)


@pytest.mark.parametrize(
    ("services", "processes"),
    [(("kubelet;reboot",), ("nv-hostengine",)), (("kubelet",), ("bad process",))],
)
def test_quiesce_rejects_unsafe_names(tmp_path, services, processes) -> None:
    with pytest.raises(ValueError, match="invalid quiesce"):
        GpuServiceQuiesceManager(
            state_dir=str(tmp_path), services=services, processes=processes
        )


def test_long_mutation_requires_extended_quiesce_failsafe(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", SECRET)
    monkeypatch.setenv("NODE_NAME", "node-a")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ALLOWED_OPERATIONS", "QUIESCE_GPU_SERVICES,REMEDIATE_DRIVER"
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", "true")
    monkeypatch.setenv("GPU_FAULT_QUIESCE_FAILSAFE_SECONDS", "420")
    monkeypatch.setenv("GPU_FAULT_QUIESCE_STATE_DIR", str(tmp_path / "quiesce"))
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_DB", str(tmp_path / "actions.db"))

    with pytest.raises(ValueError, match="fail-safe window of at least 2100 seconds"):
        executor_from_environment(validate_runtime_paths=False)
