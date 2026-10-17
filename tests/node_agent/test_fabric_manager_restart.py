"""A full GPU + NVSwitch reset brings a down Fabric Manager back (batch 127).

NVIDIA's H100-and-later procedure is: stop the GPU services and Fabric Manager,
``nvidia-smi -r``, start Fabric Manager. The quiesce only stops (and the restore
only starts) what was running, so a Fabric Manager that had FAILED at boot --
live, plain EKS p5en, "failed to configure all the available GPUs or
NVSwitches" -- stayed down through the reset and every GPU kept waiting for a
fabric manager that was not there. After a restore of a window whose claimed
reset was the full fabric reset, an enabled Fabric Manager that is not running
is started; a failure to start it is recorded, never a restore failure
(VALIDATE_GPU judges the fabric and climbs the ladder).
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import TimeoutExpired

import pytest

from gpu_fault.node_agent.fabric_manager_restart import (
    FULL_FABRIC_RESET_SCOPE,
    start_fabric_manager_after_reset,
)
from gpu_fault.node_agent.quiesce import claimed_reset_scope, record_reset_scope

from ._support import (
    CompletedProcess,
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


class FabricRunner(ServiceRunner):
    """systemctl and nvidia-smi as a one-GPU NVSwitch node answers them."""

    def __init__(
        self,
        *,
        active: set[str],
        failed: set[str] = frozenset(),
        enabled: str = "enabled",
        fail_start: str | None = None,
        start_times_out: bool = False,
    ) -> None:
        super().__init__(active=active, fail_start=fail_start)
        self.failed = set(failed)
        self.enabled = enabled
        self.start_times_out = start_times_out

    def __call__(self, value, **kwargs):
        if value == ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]:
            self.commands.append(value)
            return CompletedProcess(value, 0, stdout="GPU-a\n", stderr="")
        if value[:2] == ["systemctl", "is-active"] and value[2] != "--quiet":
            self.commands.append(value)
            unit = value[2]
            state = (
                "active"
                if unit in self.active
                else "failed"
                if unit in self.failed
                else "inactive"
            )
            return CompletedProcess(value, 0 if state == "active" else 3, state + "\n")
        if value[:2] == ["systemctl", "is-enabled"]:
            self.commands.append(value)
            return CompletedProcess(value, 0, stdout=self.enabled + "\n", stderr="")
        if value[:2] == ["systemctl", "reset-failed"]:
            self.commands.append(value)
            self.failed.discard(value[2])
            return CompletedProcess(value, 0, stdout="", stderr="")
        if value == ["systemctl", "start", FM] and self.start_times_out:
            self.commands.append(value)
            raise TimeoutExpired(value, 180)
        completed = super().__call__(value, **kwargs)
        if value == ["systemctl", "start", FM] and completed.returncode:
            self.failed.add(FM)
        return completed


def _agent(tmp_path, runner: FabricRunner):
    return node_action_executor(
        tmp_path,
        "fabric-actions.db",
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


def _window(agent, reset: WorkflowOperation):
    quiesced = agent.execute(envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES)))
    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    reset_result = agent.execute(
        envelope(command(reset, command_id="workflow/r/node-a"))
    )
    assert reset_result.status is NodeActionStatus.SUCCEEDED, reset_result.error
    return agent.execute(
        envelope(
            command(
                WorkflowOperation.RESTORE_GPU_SERVICES, command_id="workflow/s/node-a"
            )
        )
    )


def _starts(runner: FabricRunner) -> list[list[str]]:
    return [item for item in runner.commands if item == ["systemctl", "start", FM]]


def test_a_full_fabric_reset_starts_the_fabric_manager_that_had_failed(
    tmp_path,
) -> None:
    runner = FabricRunner(active={"nvidia-dcgm", "kubelet"}, failed={FM})

    restored = _window(
        _agent(tmp_path, runner), WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    )

    assert restored.status is NodeActionStatus.SUCCEEDED, restored.error
    assert restored.details["restored_services"] == ["nvidia-dcgm", "kubelet"], (
        "the quiesce never stopped the failed unit, so the restore does not own it"
    )
    assert restored.details["fabric_manager_start"] == {
        "unit": FM,
        "state_before": "failed",
        "enabled": "enabled",
        "attempted": True,
        "returncode": 0,
        "state_after": "active",
        "active": True,
    }
    reset_at = runner.commands.index(["nvidia-smi", "--gpu-reset"])
    reset_failed_at = runner.commands.index(["systemctl", "reset-failed", FM])
    start_at = runner.commands.index(["systemctl", "start", FM])
    assert reset_at < reset_failed_at < start_at, (
        "Fabric Manager starts after the reset, the NVIDIA order"
    )


def test_a_fabric_manager_the_restore_brought_back_is_not_started_twice(
    tmp_path,
) -> None:
    runner = FabricRunner(active={FM, "nvidia-dcgm", "kubelet"})

    restored = _window(
        _agent(tmp_path, runner), WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    )

    assert restored.status is NodeActionStatus.SUCCEEDED, restored.error
    assert "fabric_manager_start" not in restored.details
    assert len(_starts(runner)) == 1, "the restore started it, once"


def test_a_single_gpu_reset_leaves_a_failed_fabric_manager_alone(tmp_path) -> None:
    runner = FabricRunner(active={"nvidia-dcgm", "kubelet"}, failed={FM})

    restored = _window(_agent(tmp_path, runner), WorkflowOperation.RESET_GPU)

    assert restored.status is NodeActionStatus.SUCCEEDED, restored.error
    assert "fabric_manager_start" not in restored.details
    assert _starts(runner) == [], "only the full fabric reset re-registers the fabric"


def test_a_fabric_manager_that_fails_again_is_recorded_not_raised(tmp_path) -> None:
    runner = FabricRunner(active={"nvidia-dcgm", "kubelet"}, failed={FM}, fail_start=FM)

    restored = _window(
        _agent(tmp_path, runner), WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    )

    assert restored.status is NodeActionStatus.SUCCEEDED, (
        "every other service is back; VALIDATE_GPU judges the fabric"
    )
    record = restored.details["fabric_manager_start"]
    assert (record["attempted"], record["returncode"], record["active"]) == (
        True,
        1,
        False,
    )
    assert record["state_after"] == "failed"
    assert "start failed" in record["output_tail"]


@pytest.mark.parametrize("enabled", ["disabled", "masked", "unknown"])
def test_an_operator_s_disabled_or_masked_unit_stays_down(enabled) -> None:
    runner = FabricRunner(active=set(), failed={FM}, enabled=enabled)

    record = start_fabric_manager_after_reset(runner, ["kubelet"])

    assert record == {
        "fabric_manager_start": {
            "unit": FM,
            "state_before": "failed",
            "enabled": enabled,
            "attempted": False,
        }
    }
    assert _starts(runner) == []


def test_a_running_fabric_manager_is_left_alone() -> None:
    runner = FabricRunner(active={FM})

    assert start_fabric_manager_after_reset(runner, []) == {}
    assert _starts(runner) == []


def test_a_start_that_times_out_is_recorded() -> None:
    runner = FabricRunner(active=set(), failed={FM}, start_times_out=True)

    record = start_fabric_manager_after_reset(runner, [])["fabric_manager_start"]

    assert record["attempted"] is True and record["active"] is False
    assert record["error"].startswith("TimeoutExpired"), record


def test_the_reset_scope_rides_on_the_claim_and_leaves_with_the_window(
    tmp_path,
) -> None:
    runner = FabricRunner(active={"kubelet"})
    manager = quiesce_manager(tmp_path, runner)
    receipt = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )
    assert claimed_reset_scope(manager, "incident-a") is None, "nothing claimed yet"

    record_reset_scope(manager, "incident-a", FULL_FABRIC_RESET_SCOPE)
    assert claimed_reset_scope(manager, "incident-a") is None, (
        "a scope is only ever noted on a claim"
    )
    manager.assert_quiesced(incident_id="incident-a", command_id="workflow/r/node-a")
    record_reset_scope(manager, "incident-a", FULL_FABRIC_RESET_SCOPE)

    assert claimed_reset_scope(manager, "incident-a") == FULL_FABRIC_RESET_SCOPE
    state = json.loads(Path(receipt["state_path"]).read_text(encoding="utf-8"))
    assert state["reset_issued"]["command_id"] == "workflow/r/node-a", (
        "the claim keeps what it recorded"
    )
    manager.restore(incident_id="incident-a")
    assert claimed_reset_scope(manager, "incident-a") is None


def test_an_unreadable_window_or_a_stand_in_manager_has_no_scope(tmp_path) -> None:
    runner = FabricRunner(active=set())
    manager = quiesce_manager(tmp_path, runner)
    receipt = manager.quiesce(
        incident_id="incident-a", workflow_request_id="workflow-a"
    )
    Path(receipt["state_path"]).write_text("[]", encoding="utf-8")

    assert claimed_reset_scope(manager, "incident-a") is None
    record_reset_scope(manager, "incident-a", FULL_FABRIC_RESET_SCOPE)  # no raise
    assert claimed_reset_scope(object(), "incident-a") is None  # type: ignore[arg-type]
    record_reset_scope(object(), "incident-a", "x")  # type: ignore[arg-type]
