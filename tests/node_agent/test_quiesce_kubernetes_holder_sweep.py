"""eks-managed quiesce stops a Kubernetes container that still holds the GPU.

Plain EKS nodes run no host DCGM: the GPU clients a reset must clear are
containers (our exporter and the device plugin by name; a customer DCGM
exporter or GPU feature discovery by nothing). With
``GPU_FAULT_QUIESCE_SWEEP_KUBERNETES_DEVICE_HOLDERS`` (what the eks-managed
installer sets) a holder whose cgroup is under ``kubepods`` is swept like a
whitelisted process: kubelet is already stopped, the workload was stopped
before this step, and the restore's kubelet start brings the container back.
Off -- the HyperPod default -- the holder stays and VERIFY fails closed, and
the Agent config digest does not know the switch exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.node_agent import config as agent_config
from gpu_fault.node_agent.config import agent_config_payload
from tests.node_agent._support import (
    GpuServiceQuiesceManager,
    ServiceRunner,
    WorkflowOperation,
    no_device_clients,
    node_action_executor,
)

ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ID = "fd28e1326c44ca074eeb09425a6adb010481d05a4e10bc0466bf8774a1848954"
KUBE_CGROUP = (
    "/kubepods.slice/kubepods-besteffort.slice/"
    "kubepods-besteffort-pode747090b_fef5_4687_a913_2bcab51934c6.slice/"
    f"cri-containerd-{CONTAINER_ID}.scope"
)


def _manager(
    tmp_path: Path, runner: ServiceRunner, **overrides
) -> GpuServiceQuiesceManager:
    return GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
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


def _holder(cgroup: str) -> dict:
    return {
        "pid": "4321",
        "process_name": "dcgm-exporter",
        "devices": "/dev/nvidia0",
        "cgroup_paths": [cgroup],
    }


def _quiesce(manager: GpuServiceQuiesceManager, holder: dict, monkeypatch, runner):
    alive = [holder]
    monkeypatch.setattr(
        type(manager), "_device_holders", lambda _self, _targets=None: list(alive)
    )
    original = manager._run

    def runner_with_kill(cmd, *, check, timeout=30):
        if cmd[:2] == ["kill", "--signal"]:
            alive.clear()
        return original(cmd, check=check, timeout=timeout)

    monkeypatch.setattr(manager, "_run", runner_with_kill)
    receipt = manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        target_device_paths={"/dev/nvidia0"},
    )
    kills = [item for item in runner.commands if item[:2] == ["kill", "--signal"]]
    return receipt, kills


def test_the_switch_sweeps_a_kubernetes_holder_of_the_target_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(active={"kubelet"})
    manager = _manager(tmp_path, runner, sweep_kubernetes_device_holders=True)

    receipt, kills = _quiesce(manager, _holder(KUBE_CGROUP), monkeypatch, runner)

    assert kills == [["kill", "--signal", "TERM", "4321"]]
    assert receipt["unswept_device_holders"] == []
    (swept,) = receipt["swept_device_holders"]
    assert swept["reason"] == "kubernetes_container_on_quarantined_node"
    assert swept["kubernetes"] is True
    assert swept["container_id"] == CONTAINER_ID
    assert swept["pod_uid"] == "e747090b-fef5-4687-a913-2bcab51934c6"
    assert swept["signal"] == "TERM"
    stop_index = runner.commands.index(["systemctl", "stop", "kubelet"])
    assert stop_index < runner.commands.index(kills[0]), (
        "kubelet is down before the container is signalled, so nothing restarts it"
    )


def test_without_the_switch_the_kubernetes_holder_is_left_and_attributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(active={"kubelet"})
    manager = _manager(tmp_path, runner)

    receipt, kills = _quiesce(manager, _holder(KUBE_CGROUP), monkeypatch, runner)

    assert kills == [], "the HyperPod default never signals an unknown holder"
    assert receipt["swept_device_holders"] == []
    (skipped,) = receipt["unswept_device_holders"]
    assert skipped["reason"] == "outside_workload_cgroup_and_process_whitelist"
    assert skipped["kubernetes"] is True and skipped["container_id"] == CONTAINER_ID


def test_the_switch_never_reaches_a_host_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ServiceRunner(active={"kubelet"})
    manager = _manager(tmp_path, runner, sweep_kubernetes_device_holders=True)

    receipt, kills = _quiesce(
        manager, _holder("/system.slice/customer-monitor.service"), monkeypatch, runner
    )

    assert kills == []
    (skipped,) = receipt["unswept_device_holders"]
    assert skipped["kubernetes"] is False
    assert "container_id" not in skipped


def test_the_switch_is_not_part_of_the_agent_config_digest(tmp_path: Path) -> None:
    payloads = []
    for index, enabled in enumerate((False, True)):
        runner = ServiceRunner(active=set())
        executor = node_action_executor(
            tmp_path,
            f"digest-{index}.db",
            allowed_operations={WorkflowOperation.QUIESCE_GPU_SERVICES},
            reset_enabled=True,
            service_quiesce_enabled=True,
            quiesce_manager=_manager(
                tmp_path / str(index), runner, sweep_kubernetes_device_holders=enabled
            ),
            runner=runner,
            device_client_finder=no_device_clients,
        )
        payloads.append(agent_config_payload(executor, "profile-v1"))
    assert payloads[0] == payloads[1], (
        "a HyperPod agent's digest must not move because eks-managed agents sweep"
    )


def test_the_switch_is_read_from_the_agent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", "true")
    monkeypatch.delenv(
        "GPU_FAULT_QUIESCE_SWEEP_KUBERNETES_DEVICE_HOLDERS", raising=False
    )
    _enabled, default = agent_config._quiesce_from_environment("/proc", set())
    assert default is not None and default.sweep_kubernetes_device_holders is False
    monkeypatch.setenv("GPU_FAULT_QUIESCE_SWEEP_KUBERNETES_DEVICE_HOLDERS", "true")
    _enabled, configured = agent_config._quiesce_from_environment("/proc", set())
    assert configured is not None and configured.sweep_kubernetes_device_holders is True


def test_only_the_eks_managed_installer_job_asks_for_the_sweep() -> None:
    job = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text(
        encoding="utf-8"
    )
    installer = (ROOT / "deploy/node/install-gpu-fault-collector.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "\\${CLUSTER_PROVIDER}" == "eks-managed" ]]; then' in job
    assert "quiesce_args=(--quiesce-sweep-kubernetes-device-holders)" in job
    assert '"\\${quiesce_args[@]}"' in job, "the installer receives the provider's args"
    assert "--quiesce-sweep-kubernetes-device-holders)" in installer
    assert (
        'write_env GPU_FAULT_QUIESCE_SWEEP_KUBERNETES_DEVICE_HOLDERS "true"'
        in installer
    )
    assert 'QUIESCE_SWEEP_KUBERNETES_DEVICE_HOLDERS="false"' in installer, (
        "off by default: a HyperPod agent environment is unchanged"
    )
