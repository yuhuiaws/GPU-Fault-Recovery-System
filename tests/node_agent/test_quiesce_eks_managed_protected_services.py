"""An eks-managed quiesce never stops kubelet or containerd.

Live 2026-10-05 (gpu-eks-cluster DESTR-001): the HyperPod quiesce list stops
``kubelet`` first, which on a plain EKS node -- no HyperPod health agent, no
host DCGM -- took the node NotReady and left nothing to carry it through the
window. The provider reaches the Agent as ``GPU_FAULT_CLUSTER_PROVIDER``
(written by the eks-managed installer only); the manager keeps the configured
list, which is part of the Agent config digest, and excludes the protected
services at quiesce time, so the HyperPod digest and environment are
unchanged. GPU-holding containers are stopped through the runtime as before;
the restore starts only what the quiesce stopped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.node_agent.config import (
    EKS_MANAGED_PROTECTED_QUIESCE_SERVICES,
    agent_config_payload,
    protected_quiesce_services,
)
from tests.node_agent._support import (
    SECRET,
    GpuServiceQuiesceManager,
    ServiceRunner,
    WorkflowOperation,
    executor_from_environment,
    no_device_clients,
    node_action_executor,
)

SERVICES = ("kubelet", "containerd", "nvidia-fabricmanager", "nvidia-persistenced")


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


def _systemctl(runner: ServiceRunner, verb: str) -> list[str]:
    return [item[2] for item in runner.commands if item[:2] == ["systemctl", verb]]


def _quiesce(manager: GpuServiceQuiesceManager, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        type(manager), "_device_holders", lambda _self, _targets=None: []
    )
    return manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        target_device_paths={"/dev/nvidia0"},
    )


def test_protected_services_stay_up_and_only_the_stopped_ones_are_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(active=set(SERVICES))
    manager = _manager(
        tmp_path, runner, protected_services=EKS_MANAGED_PROTECTED_QUIESCE_SERVICES
    )

    receipt = _quiesce(manager, monkeypatch)

    assert receipt["quiesced"] is True, receipt
    assert receipt["active_services"] == ["nvidia-fabricmanager", "nvidia-persistenced"]
    assert receipt["protected_services"] == ["kubelet", "containerd"], receipt
    assert set(_systemctl(runner, "stop")) == {
        "nvidia-fabricmanager",
        "nvidia-persistenced",
    }
    assert {"kubelet", "containerd"} <= runner.active, "the node stays Ready"

    restored = manager.restore(incident_id="incident-a")

    assert restored["restored"] is True, restored
    assert set(_systemctl(runner, "start")) == {
        "nvidia-fabricmanager",
        "nvidia-persistenced",
    }
    assert runner.active == set(SERVICES)


def test_the_hyperpod_default_still_stops_the_whole_configured_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(active=set(SERVICES))
    manager = _manager(tmp_path, runner)

    receipt = _quiesce(manager, monkeypatch)

    assert receipt["active_services"] == list(SERVICES), receipt
    assert receipt["protected_services"] == [], receipt
    assert "kubelet" in _systemctl(runner, "stop"), "HyperPod behaviour is unchanged"


def test_the_protected_list_is_not_part_of_the_agent_config_digest(
    tmp_path: Path,
) -> None:
    payloads = []
    for index, protected in enumerate(((), EKS_MANAGED_PROTECTED_QUIESCE_SERVICES)):
        runner = ServiceRunner(active=set())
        executor = node_action_executor(
            tmp_path,
            f"digest-{index}.db",
            allowed_operations={WorkflowOperation.QUIESCE_GPU_SERVICES},
            reset_enabled=True,
            service_quiesce_enabled=True,
            quiesce_manager=_manager(
                tmp_path / str(index), runner, protected_services=protected
            ),
            runner=runner,
            device_client_finder=no_device_clients,
        )
        payloads.append(agent_config_payload(executor, "profile-v1"))
    assert payloads[0] == payloads[1], (
        "the configured list stays the pinned one; the provider is not in the digest"
    )
    assert payloads[1]["quiesce"]["services"] == SERVICES


@pytest.mark.parametrize(
    ("provider", "expected"),
    [(None, ()), ("hyperpod", ()), ("eks-managed", ("kubelet", "containerd"))],
)
def test_the_provider_is_read_from_the_agent_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    expected: tuple[str, ...],
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", SECRET)
    monkeypatch.setenv("NODE_NAME", "node-a")
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOWED_OPERATIONS", "QUIESCE_GPU_SERVICES")
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", "true")
    monkeypatch.setenv("GPU_FAULT_QUIESCE_STATE_DIR", str(tmp_path / "quiesce"))
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_DB", str(tmp_path / "actions.db"))
    if provider is None:
        monkeypatch.delenv("GPU_FAULT_CLUSTER_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_CLUSTER_PROVIDER", provider)

    manager = executor_from_environment(validate_runtime_paths=False).quiesce_manager

    assert manager is not None and manager.protected_services == expected
    assert (
        protected_quiesce_services(
            {} if provider is None else {"GPU_FAULT_CLUSTER_PROVIDER": provider}
        )
        == expected
    )
