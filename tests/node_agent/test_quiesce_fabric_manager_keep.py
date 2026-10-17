"""A per-GPU reset window keeps Fabric Manager running (batch 128).

Stopping ``nvidia-fabricmanager`` loses the fabric state, and once it runs
again every GPU of the NVLink fabric needs a reset (NVIDIA Fabric Manager user
guide). The quiesce used to stop it in every window, so after a single-GPU
reset the new instance knew that one GPU: live (p5en, H200 HGX) the Fabric
Manager log reported "All GPUs in the partition need to be reset to recover"
and every multicast team with another GPU failed (NVLS gone for the node).

A QUIESCE that carries ``fabric_manager=keep`` leaves a running Fabric Manager
alone: not stopped, recorded in ``kept_services``, and never started again by
the restore step, the fail-safe timer or the boot reconcile. Any other value
is refused before anything stops. A full GPU + NVSwitch reset (or a driver or
firmware remediation) that meets such a window -- a preempting successor
adopted it -- is refused before it touches the node: those need Fabric
Manager stopped, and the ladder's reboot re-registers the fabric.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.node_agent.quiesce import (
    FABRIC_MANAGER_KEPT_REASON,
    fabric_manager_kept,
    refuse_kept_fabric_manager,
)

from ._support import (
    CompletedProcess,
    GpuServiceQuiesceManager,
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
    quiesce_manager,
)

FM = "nvidia-fabricmanager"
KEEP = {"fabric_manager": "keep"}
ACTIVE = {FM, "nvidia-dcgm", "kubelet"}


class FabricNode(ServiceRunner):
    """systemctl plus the one-GPU inventory a full reset reads back."""

    def __call__(self, value, **kwargs):
        if value == ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]:
            self.commands.append(value)
            return CompletedProcess(value, 0, stdout="GPU-a\n", stderr="")
        return super().__call__(value, **kwargs)

    def fm_calls(self) -> list[list[str]]:
        return [
            item
            for item in self.commands
            if item[:1] == ["systemctl"]
            and item[-1] == FM
            and item[1] in {"stop", "start", "restart"}
        ]


def _agent(tmp_path: Path, runner: FabricNode):
    return node_action_executor(
        tmp_path,
        "fabric-keep.db",
        allowed_operations={
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        },
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=quiesce_manager(tmp_path, runner),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=lambda _: None,
    )


def _state_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "quiesce").glob("quiesce-*.json"))


def _quiesce(agent, parameters: dict | None = KEEP):
    return agent.execute(
        envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES, parameters=parameters))
    )


def _run(agent, operation: WorkflowOperation, command_id: str):
    return agent.execute(envelope(command(operation, command_id=command_id)))


def test_a_per_gpu_reset_window_neither_stops_nor_starts_fabric_manager(
    tmp_path: Path,
) -> None:
    runner = FabricNode(active=set(ACTIVE))
    agent = _agent(tmp_path, runner)

    quiesced = _quiesce(agent)
    (state_path,) = _state_files(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    reset = _run(agent, WorkflowOperation.RESET_GPU, "workflow/reset/node-a")
    restored = _run(agent, WorkflowOperation.RESTORE_GPU_SERVICES, "workflow/s/node-a")

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    assert quiesced.details["active_services"] == ["nvidia-dcgm", "kubelet"]
    assert quiesced.details["kept_services"] == [
        {"service": FM, "reason": FABRIC_MANAGER_KEPT_REASON}
    ]
    assert state["active_services"] == ["nvidia-dcgm", "kubelet"], (
        "the fail-safe timer restores what the state file names"
    )
    assert state["kept_services"][0]["service"] == FM
    assert reset.status is NodeActionStatus.SUCCEEDED, reset.error
    assert ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"] in runner.commands
    assert restored.status is NodeActionStatus.SUCCEEDED, restored.error
    assert restored.details["restored_services"] == ["nvidia-dcgm", "kubelet"]
    assert restored.details["kept_services"][0]["service"] == FM
    assert "fabric_manager_start" not in restored.details
    assert runner.fm_calls() == [], "Fabric Manager is never stopped nor started"
    assert FM in runner.active


def test_without_the_parameter_the_window_stops_and_starts_it_as_before(
    tmp_path: Path,
) -> None:
    runner = FabricNode(active=set(ACTIVE))
    agent = _agent(tmp_path, runner)

    quiesced = _quiesce(agent, None)
    _run(agent, WorkflowOperation.RESET_GPU, "workflow/reset/node-a")
    restored = _run(agent, WorkflowOperation.RESTORE_GPU_SERVICES, "workflow/s/node-a")

    assert quiesced.details["active_services"] == [FM, "nvidia-dcgm", "kubelet"]
    assert quiesced.details["kept_services"] == []
    assert restored.details["restored_services"] == [FM, "nvidia-dcgm", "kubelet"]
    assert runner.fm_calls() == [["systemctl", "stop", FM], ["systemctl", "start", FM]]


@pytest.mark.parametrize("value", ["KEEP", "stop", True, 1, {"keep": True}, ""])
def test_any_other_value_is_refused_before_anything_stops(
    tmp_path: Path, value
) -> None:
    runner = FabricNode(active=set(ACTIVE))
    agent = _agent(tmp_path, runner)

    result = _quiesce(agent, {"fabric_manager": value})

    assert result.status is NodeActionStatus.FAILED
    assert "fabric_manager must be 'keep' or absent" in str(result.error)
    assert result.retryable is False
    assert _state_files(tmp_path) == [], "no window was opened"
    assert not [item for item in runner.commands if item[:2] == ["systemctl", "stop"]]


def test_the_documented_values() -> None:
    assert fabric_manager_kept({}) is False
    assert fabric_manager_kept({"fabric_manager": None}) is False
    assert fabric_manager_kept({"fabric_manager": "keep"}) is True
    with pytest.raises(RuntimeError, match="must be 'keep' or absent"):
        fabric_manager_kept({"fabric_manager": "restart"})


def test_a_fabric_manager_that_is_not_running_is_nothing_to_keep(
    tmp_path: Path,
) -> None:
    runner = FabricNode(active={"nvidia-dcgm", "kubelet"})

    quiesced = _quiesce(_agent(tmp_path, runner))

    assert quiesced.details["inactive_services"] == [FM]
    assert quiesced.details["kept_services"] == []


def test_the_fail_safe_restore_leaves_a_kept_fabric_manager_alone(
    tmp_path: Path,
) -> None:
    runner = FabricNode(active=set(ACTIVE))
    _quiesce(_agent(tmp_path, runner))
    (state_path,) = _state_files(tmp_path)
    runner.commands.clear()

    # What the transient timer runs: gpu-fault-restore-gpu-services --state-file
    restored = GpuServiceQuiesceManager.restore_state_file(state_path, runner=runner)

    assert restored["restored_services"] == ["nvidia-dcgm", "kubelet"]
    assert runner.fm_calls() == []
    assert _state_files(tmp_path) == []


def test_the_boot_reconcile_leaves_a_kept_fabric_manager_alone(tmp_path: Path) -> None:
    runner = FabricNode(active=set(ACTIVE))
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=(FM, "nvidia-dcgm", "kubelet"),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
        boot_id_reader=lambda: "boot-1",
    )
    manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        keep_fabric_manager=True,
    )
    runner.commands.clear()
    manager.boot_id_reader = lambda: "boot-2"

    report = manager.reconcile_after_boot()

    assert [item["incident_id"] for item in report["restored"]] == ["incident-a"]
    started = [
        item[2] for item in runner.commands if item[:2] == ["systemctl", "start"]
    ]
    assert started == ["nvidia-dcgm", "kubelet"], (
        "the boot started Fabric Manager; the reconcile does not start it again"
    )


def test_a_re_sent_quiesce_answers_the_window_it_opened(tmp_path: Path) -> None:
    runner = FabricNode(active=set(ACTIVE))
    manager = quiesce_manager(tmp_path, runner)
    first = manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        keep_fabric_manager=True,
    )

    again = manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        keep_fabric_manager=True,
    )

    assert again["already_quiesced"] is True
    assert again["active_services"] == first["active_services"]
    assert again["kept_services"] == first["kept_services"]
    assert runner.fm_calls() == []


def test_a_full_reset_in_a_kept_window_is_refused_before_it_touches_the_node(
    tmp_path: Path,
) -> None:
    """A preempting successor adopted the per-GPU window. NVIDIA's order needs
    Fabric Manager down for ``nvidia-smi -r``; the Agent refuses instead of
    resetting under it, the window's single reset stays unclaimed and its
    restore runs as planned (the ladder's reboot re-registers the fabric)."""

    runner = FabricNode(active=set(ACTIVE))
    agent = _agent(tmp_path, runner)
    _quiesce(agent)

    refused = _run(
        agent, WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, "workflow/r/node-a"
    )
    (state_path,) = _state_files(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    restored = _run(agent, WorkflowOperation.RESTORE_GPU_SERVICES, "workflow/s/node-a")

    assert refused.status is NodeActionStatus.FAILED
    assert "needs nvidia-fabricmanager stopped" in str(refused.error)
    assert refused.retryable is False
    assert state["reset_issued"] is None, "the window's one reset is not spent"
    assert not [item for item in runner.commands if "--gpu-reset" in item]
    assert restored.status is NodeActionStatus.SUCCEEDED, restored.error
    assert runner.fm_calls() == [], "Fabric Manager was never stopped under it"


def test_a_remediation_in_a_kept_window_is_refused(tmp_path: Path) -> None:
    runner = FabricNode(active=set(ACTIVE))
    manager = quiesce_manager(tmp_path, runner)
    manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        keep_fabric_manager=True,
    )

    with pytest.raises(RuntimeError, match="driver/firmware remediation needs"):
        refuse_kept_fabric_manager(manager, "incident-a", "driver/firmware remediation")


def test_a_window_that_stopped_fabric_manager_lets_them_run(tmp_path: Path) -> None:
    runner = FabricNode(active=set(ACTIVE))
    manager = quiesce_manager(tmp_path, runner)
    refuse_kept_fabric_manager(manager, "incident-a", "x")  # no window: no refusal
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")

    refuse_kept_fabric_manager(manager, "incident-a", "x")
    refuse_kept_fabric_manager(object(), "incident-a", "x")  # type: ignore[arg-type]
    (tmp_path / "agent").mkdir()
    agent = _agent(tmp_path / "agent", FabricNode(active=set(ACTIVE)))
    _quiesce(agent, None)
    reset = _run(
        agent, WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, "workflow/r/node-a"
    )
    assert reset.status is NodeActionStatus.SUCCEEDED, reset.error
