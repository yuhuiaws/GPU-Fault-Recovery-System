from __future__ import annotations

from tests._builders import copy_model

from ._support import (
    CompletedProcess,
    FakeRunner,
    NodeActionStatus,
    Path,
    Quiesced,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    executor,
    hashlib,
    no_device_clients,
    node_action_executor,
    pytest,
)


def test_gpu_reset_refuses_active_compute_client(tmp_path) -> None:
    runner = FakeRunner("GPU-a, 1234, python\n")
    agent = executor(tmp_path, runner)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert "GPU-a:1234" in result.error
    assert not any("--gpu-reset" in item for item in runner.commands)


def test_gpu_reset_refuses_active_device_client(tmp_path) -> None:
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "device-client.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=runner,
        device_client_finder=lambda _: [
            {
                "gpu_uuid": "GPU-a",
                "pid": "4321",
                "process_name": "nv-hostengine",
                "device": "/dev/nvidia0",
            }
        ],
        sleep=lambda _: None,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert "GPU-a:4321:nv-hostengine" in result.error
    assert not any("--gpu-reset" in item for item in runner.commands)


def test_gpu_reset_reports_nvidia_smi_stderr(tmp_path) -> None:
    runner = FakeRunner(reset_error="GPU 00000000:59:00.0: In use by another client")
    agent = executor(tmp_path, runner)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert "exited with status 255" in result.error
    assert "In use by another client" in result.error
    assert len([item for item in runner.commands if "--gpu-reset" in item]) == 3


def test_gpu_reset_retries_transient_nvidia_smi_client(tmp_path) -> None:
    class BusyOnceRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.busy = True

        def __call__(self, command, **kwargs):
            if "--gpu-reset" in command and self.busy:
                self.busy = False
                self.reset_error = "GPU 00000000:59:00.0: In use by another client"
                try:
                    return super().__call__(command, **kwargs)
                finally:
                    self.reset_error = None
            return super().__call__(command, **kwargs)

    runner = BusyOnceRunner()
    agent = executor(tmp_path, runner)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["reset_attempts"] == 2
    assert [item for item in runner.commands if "--gpu-reset" in item] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"],
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"],
    ]


def test_gpu_reset_is_idempotent_and_rechecks_clients(tmp_path) -> None:
    runner = FakeRunner()
    agent = executor(tmp_path, runner)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert first.status is NodeActionStatus.SUCCEEDED
    assert second == first
    reset_commands = [item for item in runner.commands if "--gpu-reset" in item]
    assert reset_commands == [["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]]


def test_gpu_reset_remains_disabled_without_node_opt_in(tmp_path) -> None:
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "disabled.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert "disabled" in result.error
    assert not runner.commands


def test_full_fabric_reset_verifies_inventory_and_is_idempotent(tmp_path) -> None:
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "fabric-reset.db",
        allowed_operations={WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES},
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )
    signed = envelope(command(WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES))

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert first.status is NodeActionStatus.SUCCEEDED
    assert second == first
    assert first.details["reset_scope"] == ("ALL_LOCAL_GPUS_AND_NVSWITCHES")
    assert [item for item in runner.commands if "--gpu-reset" in item] == [
        ["nvidia-smi", "--gpu-reset"]
    ]


def test_full_fabric_reset_rejects_inventory_mismatch(tmp_path) -> None:
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "fabric-mismatch.db",
        allowed_operations={WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES},
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )
    value = copy_model(
        command(WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES), gpu_uuids=["GPU-other"]
    )

    result = agent.execute(envelope(value))

    assert result.status is NodeActionStatus.FAILED
    assert "does not match local inventory" in result.error
    assert not any("--gpu-reset" in item for item in runner.commands)


def test_fabric_manager_restart_is_fenced_and_verified(tmp_path) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager"})
    agent = node_action_executor(
        tmp_path,
        "fm-actions.db",
        allowed_operations={WorkflowOperation.RESTART_FABRIC_MANAGER},
        fabric_manager_restart_enabled=True,
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESTART_FABRIC_MANAGER)))

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details == {
        "service": "nvidia-fabricmanager",
        "active": True,
        "previous_main_pid": "101",
        "current_main_pid": "101",
    }
    assert [item[:2] for item in runner.commands] == [
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name"],
        ["systemctl", "show"],
        ["systemctl", "restart"],
        ["systemctl", "is-active"],
        ["systemctl", "show"],
    ]


def test_fabric_manager_restart_refuses_active_compute_client(
    tmp_path, monkeypatch
) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager"})
    agent = node_action_executor(
        tmp_path,
        "fm-clients.db",
        allowed_operations={WorkflowOperation.RESTART_FABRIC_MANAGER},
        fabric_manager_restart_enabled=True,
        runner=runner,
    )
    monkeypatch.setattr(
        agent,
        "_compute_clients",
        lambda: [{"gpu_uuid": "GPU-a", "pid": "123", "process_name": "python"}],
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESTART_FABRIC_MANAGER)))

    assert result.status is NodeActionStatus.FAILED
    assert "compute clients are still active" in result.error
    assert not any(item[:2] == ["systemctl", "restart"] for item in runner.commands)


def test_fabric_manager_restart_requires_explicit_enable(tmp_path) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager"})
    agent = node_action_executor(
        tmp_path,
        "fm-disabled.db",
        allowed_operations={WorkflowOperation.RESTART_FABRIC_MANAGER},
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESTART_FABRIC_MANAGER)))

    assert result.status is NodeActionStatus.FAILED
    assert "restart is disabled" in result.error
    assert runner.commands == []


def test_single_gpu_reset_capability_can_fail_closed(tmp_path) -> None:
    agent = node_action_executor(
        tmp_path,
        "reset-capability.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        single_gpu_reset_supported=False,
        now=None,
    )

    with pytest.raises(RuntimeError, match="single-GPU reset is not supported"):
        agent._reset_gpu(["GPU-a"])


def test_driver_remediation_is_pinned_and_target_verified(tmp_path) -> None:
    executable = tmp_path / "driver-remediate"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()

    class Quiesced:
        def assert_quiesced(self, *, incident_id):
            assert incident_id == "incident-a"

    class DriverRunner:
        def __init__(self):
            self.commands = []

        def __call__(self, argv, **_):
            self.commands.append(argv)
            stdout = (
                "575.86.01\n575.86.01\n" if "--query-gpu=driver_version" in argv else ""
            )
            return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    runner = DriverRunner()
    agent = node_action_executor(
        tmp_path,
        "driver.db",
        allowed_operations={WorkflowOperation.REMEDIATE_DRIVER},
        reset_enabled=True,
        service_quiesce_enabled=True,
        driver_remediation_enabled=True,
        driver_remediation_command=(str(executable), "--branch", "{target}"),
        driver_remediation_sha256=digest,
        target_driver_branch=575,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.REMEDIATE_DRIVER,
                parameters={"target_driver_branch": 575},
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert [str(executable), "--branch", "575"] in runner.commands
    assert result.details["verified_driver_branches"] == [575]


def test_efa_driver_remediation_accepts_already_bound_inventory(tmp_path) -> None:
    pci_root = tmp_path / "pci"
    driver = tmp_path / "drivers" / "efa"
    driver.mkdir(parents=True)
    for index in range(2):
        device = pci_root / f"0000:0{index}:00.0"
        device.mkdir(parents=True)
        (device / "vendor").write_text("0x1d0f\n")
        (device / "device").write_text("0xefa2\n")
        (device / "driver").symlink_to(driver)
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "efa-driver.db",
        allowed_operations={WorkflowOperation.REMEDIATE_EFA_DRIVER},
        efa_driver_remediation_enabled=True,
        efa_pci_devices_root=str(pci_root),
        efa_driver_bind_path=str(driver / "bind"),
        runner=runner,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.REMEDIATE_EFA_DRIVER, parameters={"expected_count": 2}
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["already_bound"] is True
    assert result.details["driver_bound_count"] == 2
    assert runner.commands == []


def test_efa_driver_remediation_refuses_missing_pci_device(tmp_path) -> None:
    pci_root = tmp_path / "pci"
    pci_root.mkdir()
    agent = node_action_executor(
        tmp_path,
        "efa-missing.db",
        allowed_operations={WorkflowOperation.REMEDIATE_EFA_DRIVER},
        efa_driver_remediation_enabled=True,
        efa_pci_devices_root=str(pci_root),
        efa_driver_bind_path=str(tmp_path / "bind"),
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.REMEDIATE_EFA_DRIVER, parameters={"expected_count": 1}
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED
    assert "PCI inventory is incomplete" in result.error


def test_efa_driver_remediation_reports_partial_bind_errors(
    tmp_path, monkeypatch
) -> None:
    pci_root = tmp_path / "pci"
    bdfs = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0"]
    for bdf in bdfs:
        device = pci_root / bdf
        device.mkdir(parents=True)
        (device / "vendor").write_text("0x1d0f\n")
        (device / "device").write_text("0xefa2\n")
    bind_path = tmp_path / "drivers" / "efa" / "bind"
    bound = {bdfs[0]}
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "efa-partial.db",
        allowed_operations={WorkflowOperation.REMEDIATE_EFA_DRIVER},
        efa_driver_remediation_enabled=True,
        efa_pci_devices_root=str(pci_root),
        efa_driver_bind_path=str(bind_path),
        runner=runner,
    )
    monkeypatch.setattr(
        agent, "_bound_driver", lambda device: "efa" if device.name in bound else None
    )
    original_write = Path.write_text

    def write_text(path, data, *args, **kwargs):
        if path == bind_path:
            bdf = data.strip()
            if bdf == bdfs[2]:
                raise OSError("simulated probe failure")
            bound.add(bdf)
            return len(data)
        return original_write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.REMEDIATE_EFA_DRIVER, parameters={"expected_count": 3}
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED
    assert bdfs[1] in bound
    assert bdfs[2] not in bound
    assert bdfs[2] in result.error
    assert "simulated probe failure" in result.error
    assert ["modprobe", "efa"] in runner.commands


def test_firmware_update_requires_exact_verified_version(tmp_path) -> None:
    executable = tmp_path / "firmware-update"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    verifier = tmp_path / "firmware-version"
    verifier.write_text("#!/bin/sh\n", encoding="utf-8")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    verifier_digest = hashlib.sha256(verifier.read_bytes()).hexdigest()

    class Quiesced:
        def assert_quiesced(self, *, incident_id):
            assert incident_id == "incident-a"

    def runner(argv, **_):
        stdout = "92.10.14\n" if argv[0] == str(verifier) else ""
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    agent = node_action_executor(
        tmp_path,
        "firmware.db",
        allowed_operations={WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE},
        reset_enabled=True,
        service_quiesce_enabled=True,
        firmware_update_enabled=True,
        firmware_update_command=(str(executable), "--version", "{target}"),
        firmware_update_sha256=digest,
        target_firmware_version="92.10.14",
        firmware_verify_command=(str(verifier),),
        firmware_verify_sha256=verifier_digest,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
                parameters={"target_firmware_version": "92.10.14"},
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["verified_firmware_version"] == "92.10.14"
