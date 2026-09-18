from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.node_agent import GpuServiceQuiesceManager
from tests.node_agent._support import ServiceRunner
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def manager(tmp_path: Path, runner: Any, **changes: Any) -> GpuServiceQuiesceManager:
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    options = {
        "state_dir": str(tmp_path / "quiesce"),
        "services": ("kubelet",),
        "containers": (),
        "processes": (),
        "settle_seconds": 0,
        "restore_settle_seconds": 0,
        "proc_root": str(proc),
        "runner": runner,
        "sleeper": lambda seconds: None,
        "boot_id_reader": lambda: "boot-new",
        **changes,
    }
    return GpuServiceQuiesceManager(**options)


@pytest.mark.parametrize("defect", ["json", "unicode", "permission", "not-object"])
def test_boot_reconcile_preserves_bad_state_and_restores_independent_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    transport = ServiceRunner(active={"kubelet"})
    service = manager(tmp_path, transport, boot_id_reader=lambda: "boot-old")
    receipt = service.quiesce(
        incident_id="unit-incident", workflow_request_id="unit-workflow"
    )
    valid_path = Path(receipt["state_path"])
    invalid_path = valid_path.parent / "quiesce-000-invalid.json"
    payload = (
        b"{"
        if defect == "json"
        else b"\xff"
        if defect == "unicode"
        else b"[]"
        if defect == "not-object"
        else b"{}"
    )
    invalid_path.write_bytes(payload)
    transport.commands.clear()
    if defect == "permission":
        original = Path.read_text

        def denied(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == invalid_path:
                raise PermissionError("synthetic quiesce state unreadable")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
    report = manager(tmp_path, transport).reconcile_after_boot()
    assert len(report["failed"]) == 1
    assert report["failed"][0]["state_path"] == str(invalid_path)
    assert [item["incident_id"] for item in report["restored"]] == ["unit-incident"]
    assert report["kept"] == []
    assert invalid_path.read_bytes() == payload
    assert not valid_path.exists(), "a separate valid stale window must still restore"
    assert ["systemctl", "start", "kubelet"] in transport.commands
    assert ["systemctl", "stop", receipt["timer_unit"]] in transport.commands


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"failsafe_seconds": 29}, "fail-safe"),
        ({"failsafe_seconds": 3601}, "fail-safe"),
        ({"retry_seconds": 9}, "retry seconds"),
        ({"retry_seconds": 601}, "retry seconds"),
        ({"settle_seconds": -1}, "settle seconds"),
        ({"restore_settle_seconds": 121}, "restore settle"),
        ({"services": ()}, "at least one"),
        ({"services": ("kubelet", "kubelet")}, "services contain duplicates"),
        ({"processes": ("python", "python")}, "processes contain duplicates"),
        (
            {"containers": ("system/plugin", "system/plugin")},
            "containers contain duplicates",
        ),
        (
            {"device_sweep_processes": ("python", "python")},
            "sweep processes contain duplicates",
        ),
        ({"device_sweep_processes": ("unsafe;",)}, "invalid device sweep"),
        ({"containers": ("system",)}, "container selector"),
        ({"containers": ("system/plugin;exit",)}, "container selector"),
        ({"container_stop_timeout_seconds": 4}, "container stop"),
        ({"container_restore_timeout_seconds": 29}, "container restore"),
        ({"device_sweep_timeout_seconds": 121}, "device sweep timeout"),
        ({"restore_command": "relative/restore"}, "absolute"),
    ],
)
def test_quiesce_configuration_refuses_unsafe_values_without_service_calls(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    transport = ServiceRunner(active={"kubelet"})
    with pytest.raises(ValueError, match=message):
        manager(tmp_path, transport, **changes)
    assert transport.commands == []


@pytest.mark.parametrize("output", ["", "x" * 2200])
def test_quiesce_command_failure_is_bounded_and_does_not_arm_a_window(
    tmp_path: Path, output: str
) -> None:
    calls = []

    def failed(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        raise subprocess.CalledProcessError(2, command, stderr=output)

    service = manager(tmp_path, failed)
    with pytest.raises(RuntimeError, match="exited with status 2") as error:
        service.quiesce(incident_id="unit", workflow_request_id="unit-workflow")
    assert len(str(error.value)) < 2100
    assert len(calls) == 1 and calls[0][0] == "systemctl"
    assert list((tmp_path / "quiesce").glob("*.json")) == []


@pytest.mark.parametrize("defect", ["workflow", "scope", "shape"])
def test_quiesce_retry_cannot_change_the_owned_workflow_or_gpu_scope(
    tmp_path: Path, defect: str
) -> None:
    transport = ServiceRunner(active={"kubelet"})
    service = manager(tmp_path, transport)
    receipt = service.quiesce(
        incident_id="unit",
        workflow_request_id="workflow",
        target_device_paths={"/unit/gpu0"},
    )
    state_path = Path(receipt["state_path"])
    if defect == "shape":
        state_path.write_text("[]", encoding="ascii")
    transport.commands.clear()
    with pytest.raises(
        RuntimeError, match="another workflow|scope changed|invalid quiesce"
    ):
        service.quiesce(
            incident_id="unit",
            workflow_request_id="other" if defect == "workflow" else "workflow",
            target_device_paths={"/unit/gpu1"} if defect == "scope" else {"/unit/gpu0"},
        )
    assert transport.commands == []
    assert state_path.exists(), (
        "unproved retries must retain the independent restore state"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"active_services": ["unsafe;"]},
        {"active_services": None},
        {"timer_unit": None},
        {"timer_unit": "unsafe;"},
        {"container_targets": {}},
        {
            "container_targets": [
                {"selector": "system/plugin", "container_id": "invalid"}
            ]
        },
        {"container_targets": [{"selector": "system", "container_id": "a" * 64}]},
    ],
)
def test_restore_refuses_unknown_state_without_canceling_the_timer(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    transport = ServiceRunner(active={"kubelet"})
    service = manager(tmp_path, transport)
    receipt = service.quiesce(incident_id="unit", workflow_request_id="workflow")
    path = Path(receipt["state_path"])
    state = json.loads(path.read_text())
    state.update(changes)
    path.write_text(json.dumps(state), encoding="ascii")
    transport.commands.clear()
    with pytest.raises(RuntimeError, match="invalid quiesce restore state"):
        service.restore(incident_id="unit")
    assert transport.commands == []
    assert json.loads(path.read_text()) == state


@pytest.mark.parametrize("command_id", [None, "unit-command"])
@pytest.mark.parametrize("attempt", [None, 2])
def test_reset_claim_is_single_use_and_does_not_claim_that_reset_executed(
    tmp_path: Path, command_id: str | None, attempt: int | None
) -> None:
    transport = ServiceRunner(active={"kubelet"})
    service = manager(tmp_path, transport)
    receipt = service.quiesce(incident_id="unit", workflow_request_id="workflow")
    service.assert_quiesced(incident_id="unit", for_reset=False)
    service.assert_quiesced(incident_id="unit", command_id=command_id, attempt=attempt)
    before = Path(receipt["state_path"]).read_bytes()
    calls = list(transport.commands)
    with pytest.raises(RuntimeError, match="already claimed") as failed:
        service.assert_quiesced(incident_id="unit", command_id=command_id, attempt=3)
    assert "read that command's result" in str(failed.value)
    assert ("see attempt 2" in str(failed.value)) is (
        command_id is not None and attempt == 2
    )
    assert Path(receipt["state_path"]).read_bytes() == before
    assert transport.commands == calls
