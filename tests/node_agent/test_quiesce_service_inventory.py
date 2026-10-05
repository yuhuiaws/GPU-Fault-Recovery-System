"""The quiesce receipt says which configured services it found nothing to stop.

Live (gpu-eks-cluster, XID 46): the reset failed busy, the journal had no
``systemctl stop`` line, and nothing in the record said whether
``nvidia-persistenced`` and ``nvidia-fabricmanager`` had been stopped, skipped
as protected, or were simply not active on the image. ``active_services`` was
always the list of what the quiesce stopped; the receipt and the state file now
also carry ``inactive_services`` -- the configured, unprotected services that
were not running when the window opened -- and the step lifts both to its
details as ``stopped_services`` / ``inactive_services`` / ``protected_services``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.node_agent.config import EKS_MANAGED_PROTECTED_QUIESCE_SERVICES
from tests.node_agent._support import GpuServiceQuiesceManager, ServiceRunner

SERVICES = (
    "nvidia-fabricmanager",
    "nvidia-dcgm",
    "nvidia-persistenced",
    "gpu-fault-metrics-collector",
    "kubelet",
    "containerd",
)


def _manager(
    tmp_path: Path, runner: ServiceRunner, **overrides
) -> GpuServiceQuiesceManager:
    return GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=SERVICES,
        processes=(),
        device_sweep_processes=(),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
        **overrides,
    )


def _quiesce(manager: GpuServiceQuiesceManager, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        type(manager), "_device_holders", lambda _self, _targets=None: []
    )
    return manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        target_device_paths={"/dev/nvidia0"},
    )


def test_the_receipt_separates_stopped_inactive_and_protected_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(
        active={"nvidia-persistenced", "gpu-fault-metrics-collector", "kubelet"}
    )
    manager = _manager(
        tmp_path, runner, protected_services=EKS_MANAGED_PROTECTED_QUIESCE_SERVICES
    )

    receipt = _quiesce(manager, monkeypatch)

    assert receipt["active_services"] == [
        "nvidia-persistenced",
        "gpu-fault-metrics-collector",
    ], "what the quiesce stopped, in configured order"
    assert receipt["inactive_services"] == ["nvidia-fabricmanager", "nvidia-dcgm"], (
        "configured, unprotected, and not running: nothing to stop, said so"
    )
    assert receipt["protected_services"] == list(EKS_MANAGED_PROTECTED_QUIESCE_SERVICES)
    stopped = [item[2] for item in runner.commands if item[:2] == ["systemctl", "stop"]]
    assert stopped == ["gpu-fault-metrics-collector", "nvidia-persistenced"], (
        "kubelet is protected and the inactive units are never stopped"
    )
    state = json.loads(Path(receipt["state_path"]).read_text(encoding="utf-8"))
    assert state["inactive_services"] == ["nvidia-fabricmanager", "nvidia-dcgm"]
    assert state["active_services"] == receipt["active_services"]


def test_an_already_quiesced_window_keeps_its_original_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(active={"nvidia-persistenced"})
    manager = _manager(tmp_path, runner)

    first = _quiesce(manager, monkeypatch)
    runner.active.add("nvidia-fabricmanager")
    again = _quiesce(manager, monkeypatch)

    assert again["already_quiesced"] is True
    assert again["active_services"] == first["active_services"], (
        "the replay answers from the state file, not from a fresh probe"
    )
