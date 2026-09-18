from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import FakeRunner, command, envelope
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def pinned_settings(tmp_path: Path, kind: str) -> dict[str, Any]:
    tool = tmp_path / "unit-tool"
    tool.write_bytes(b"unit pinned executable\n")
    digest = hashlib.sha256(tool.read_bytes()).hexdigest()
    if kind == "driver":
        return {
            "driver_remediation_enabled": True,
            "driver_remediation_command": (str(tool), "{target}"),
            "driver_remediation_sha256": digest,
            "target_driver_branch": 575,
        }
    return {
        "firmware_update_enabled": True,
        "firmware_update_command": (str(tool), "{target}"),
        "firmware_update_sha256": digest,
        "target_firmware_version": "92.10.14",
        "firmware_verify_command": (str(tool), "--verify"),
        "firmware_verify_sha256": digest,
    }


@pytest.mark.parametrize("kind", ["driver", "firmware"])
@pytest.mark.parametrize(
    "defect",
    [
        "empty-command",
        "relative-command",
        "missing-target",
        "missing-hash",
        "bad-hash",
        "missing-file",
        "changed-file",
    ],
)
def test_remediation_configuration_requires_a_pinned_existing_target(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    kind: str,
    defect: str,
) -> None:
    options = pinned_settings(tmp_path, kind)
    prefix = "driver_remediation" if kind == "driver" else "firmware_update"
    if defect == "empty-command":
        options[prefix + "_command"] = ()
    elif defect == "relative-command":
        options[prefix + "_command"] = ("relative-tool",)
    elif defect == "missing-target":
        options[
            "target_driver_branch" if kind == "driver" else "target_firmware_version"
        ] = None
    elif defect == "missing-hash":
        options[prefix + "_sha256"] = None
    elif defect == "bad-hash":
        options[prefix + "_sha256"] = "invalid"
    elif defect == "missing-file":
        options[prefix + "_command"] = (str(tmp_path / "missing"),)
    else:
        (tmp_path / "unit-tool").write_bytes(b"changed after pinning\n")
    runner = FakeRunner()
    with pytest.raises(
        ValueError, match="absolute|target is required|SHA-256|does not exist"
    ):
        node_factory(runner=runner, **options)
    assert runner.commands == []


def test_firmware_update_requires_a_separately_pinned_verification_command(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path
) -> None:
    options = pinned_settings(tmp_path, "firmware")
    options["firmware_verify_command"] = ()
    with pytest.raises(ValueError, match="requires a verification command"):
        node_factory(**options)


@pytest.mark.parametrize("kind", ["driver", "firmware"])
@pytest.mark.parametrize("defect", ["disabled", "target", "not-quiesced"])
def test_runtime_remediation_guards_refuse_before_any_probe_or_installer(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    kind: str,
    defect: str,
) -> None:
    operation = (
        WorkflowOperation.REMEDIATE_DRIVER
        if kind == "driver"
        else WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE
    )
    runner = FakeRunner()
    settings = {} if defect == "disabled" else pinned_settings(tmp_path, kind)
    target = 575 if kind == "driver" else "92.10.14"
    parameters = {
        "target_driver_branch"
        if kind == "driver"
        else "target_firmware_version": "foreign" if defect == "target" else target
    }
    agent = node_factory(allowed_operations={operation}, runner=runner, **settings)
    result = agent.execute(envelope(command(operation, parameters=parameters)))
    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert {
        "disabled": "disabled",
        "target": "does not match",
        "not-quiesced": "requires service quiesce",
    }[defect] in result.error
    assert runner.commands == []


@pytest.mark.parametrize("kind", ["driver", "firmware"])
@pytest.mark.parametrize("matches", [False, True])
def test_remediation_verification_decides_the_record_without_spending_a_reset_claim(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    kind: str,
    matches: bool,
) -> None:
    operation = (
        WorkflowOperation.REMEDIATE_DRIVER
        if kind == "driver"
        else WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE
    )
    settings = pinned_settings(tmp_path, kind)
    requests = []
    claims = []

    class Quiesced:
        def assert_quiesced(self, **kwargs: Any) -> None:
            claims.append(kwargs)

    def runner(argv: list[str], **kwargs: Any) -> CompletedProcess:
        requests.append(argv)
        stdout = ""
        if "--query-gpu=driver_version" in argv:
            stdout = "575.86\n" if matches else "invalid\n"
        if "--verify" in argv:
            stdout = "92.10.14" if matches else "old-version"
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    agent = node_factory(
        allowed_operations={operation},
        runner=runner,
        service_quiesce_enabled=True,
        quiesce_manager=Quiesced(),
        **settings,
    )
    parameters = (
        {"target_driver_branch": 575}
        if kind == "driver"
        else {"target_firmware_version": "92.10.14"}
    )
    signed = envelope(command(operation, parameters=parameters))
    result = agent.execute(signed)
    assert result.status is (
        NodeActionStatus.SUCCEEDED if matches else NodeActionStatus.FAILED
    )
    assert result.retryable is False
    assert claims == [{"incident_id": "incident-a", "for_reset": False}]
    count = len(requests)
    assert agent.execute(signed) == result
    assert len(requests) == count
    assert all("--gpu-reset" not in argv for argv in requests), (
        "driver/firmware remediation must not consume a reset"
    )


@pytest.mark.parametrize(
    "state", ["disabled", "missing-device", "conflicting-driver", "already-bound"]
)
def test_efa_repair_respects_inventory_and_driver_ownership_before_module_load(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path, state: str
) -> None:
    pci = tmp_path / "pci"
    pci.mkdir()
    (pci / "not-a-device").write_text("unit", encoding="ascii")
    (pci / "unreadable-device").mkdir()
    if state in {"conflicting-driver", "already-bound"}:
        device = pci / "0000:00:00.0"
        device.mkdir()
        (device / "vendor").write_text("0x1d0f", encoding="ascii")
        (device / "device").write_text("0xefa2", encoding="ascii")
        driver = tmp_path / ("efa" if state == "already-bound" else "foreign-driver")
        driver.mkdir()
        (device / "driver").symlink_to(driver)
    runner = FakeRunner()
    agent = node_factory(
        allowed_operations={WorkflowOperation.REMEDIATE_EFA_DRIVER},
        efa_driver_remediation_enabled=state != "disabled",
        efa_pci_devices_root=str(pci),
        efa_driver_bind_path=str(tmp_path / "unit-bind"),
        runner=runner,
    )
    result = agent.execute(envelope(command(WorkflowOperation.REMEDIATE_EFA_DRIVER)))
    assert result.status is (
        NodeActionStatus.SUCCEEDED
        if state == "already-bound"
        else NodeActionStatus.FAILED
    )
    assert runner.commands == []
    assert (tmp_path / "unit-bind").exists() is False
    if state == "already-bound":
        assert result.details["already_bound"] is True
    else:
        assert result.retryable is False
