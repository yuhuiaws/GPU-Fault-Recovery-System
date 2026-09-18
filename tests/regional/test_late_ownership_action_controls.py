from __future__ import annotations

from subprocess import CompletedProcess

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError, dispatch
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent.protocol import NodeActionStatus
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from tests.execution import test_cluster_executor_batching as batches
from tests.node_agent._support import (
    FakeRunner,
    Quiesced,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)


@pytest.mark.parametrize(
    "operation,targets,reason",
    [
        (WorkflowOperation.RESET_GPU, [], "explicit GPU UUID"),
        (WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, [], "unique explicit"),
        (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            ["GPU-a", "GPU-a"],
            "unique explicit",
        ),
        (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            ["foreign"],
            "does not match local",
        ),
    ],
)
def test_invalid_reset_targets_never_reach_fake_physical_runner(
    tmp_path, operation, targets, reason
):
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "target-refusal.db",
        allowed_operations={operation},
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )
    result = agent.execute(envelope(command(operation, gpu_uuids=targets)))
    assert result.status is NodeActionStatus.FAILED and not result.retryable
    assert reason in result.error
    assert not any("--gpu-reset" in argv for argv in runner.commands), (
        "invalid reset targets must not reach the physical runner"
    )


def test_full_reset_inventory_change_is_not_reported_as_success(tmp_path):
    class Runner(FakeRunner):
        changed = False

        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if "--query-gpu=uuid" in argv and self.changed:
                return CompletedProcess(argv, 0, "GPU-other\n", "")
            if "--gpu-reset" in argv:
                self.changed = True
            return result

    runner = Runner()
    operation = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    agent = node_action_executor(
        tmp_path,
        "changed-inventory.db",
        allowed_operations={operation},
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )
    result = agent.execute(envelope(command(operation)))
    assert result.status is NodeActionStatus.FAILED and not result.retryable
    assert "inventory changed after full fabric reset" in result.error
    assert len([argv for argv in runner.commands if "--gpu-reset" in argv]) == 1


def test_batched_fleet_hold_keeps_prior_success_and_does_not_call_later_adapter(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claim.json")
    )
    prior = {"1": {"status": "SUCCEEDED", "details": batches.quiesce_details()}}
    command_value = batches.compound_command({BATCHED_RESULTS_KEY: prior})
    client = batches.FakeClient(command_value)
    adapter = batches.FakeNodeAdapter(batches.all_succeed())
    executor = batches.executor(client, adapter)
    executor.fleet_registry = object()
    monkeypatch.setattr(
        dispatch, "fleet_preflight_reason", lambda *args: "release fence is closed"
    )
    result = executor.dispatch.execute(command_value)
    assert result.status is RemoteCommandStatus.WAITING
    assert result.details["fleet_preflight_blocked"] is True
    assert result.details["reason"] == "release fence is closed"
    assert result.details[BATCHED_RESULTS_KEY]["1"] == prior["1"]
    assert all(
        context.step.operation is not WorkflowOperation.RESET_GPU
        for context in adapter.contexts
    ), "the fleet hold must prevent dispatch of the later reset"
    assert "3" not in result.details[BATCHED_RESULTS_KEY]


def test_batched_resume_deduplicates_completed_operations_and_ignores_future_waits(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claim.json")
    )
    prior = {
        "1": {"status": "SUCCEEDED", "details": batches.quiesce_details()},
        "3": {
            "status": "WAITING",
            "details": {"node_action_command_id": "accepted-original"},
        },
    }
    command_value = batches.compound_command({BATCHED_RESULTS_KEY: prior})
    workflow = command_value.workflow.model_copy(
        update={"completed_operations": [WorkflowOperation.QUIESCE_GPU_SERVICES]}
    )
    command_value = command_value.model_copy(update={"workflow": workflow})
    client = batches.FakeClient(command_value)
    adapter = batches.FakeNodeAdapter(batches.all_succeed())
    result = batches.executor(client, adapter).dispatch.execute(command_value)
    assert result.status is RemoteCommandStatus.SUCCEEDED
    assert [context.step_index for context in adapter.contexts] == [2, 3, 4]
    first, reset, restored = adapter.contexts
    assert (
        first.workflow.completed_operations.count(
            WorkflowOperation.QUIESCE_GPU_SERVICES
        )
        == 1
    )
    assert not any(
        record.step_index == 3 and record.status is WorkflowStepStatus.WAITING
        for record in first.workflow.step_executions
    ), "the first batch must not record a wait for an unreached step"
    assert any(
        record.step_index == 3 and record.status is WorkflowStepStatus.WAITING
        for record in reset.workflow.step_executions
    ), "the reset batch must retain the later waiting step"
    assert (
        restored.workflow.completed_operations.count(WorkflowOperation.RESET_GPU) == 1
    )


def test_dispatch_requires_claim_lease_before_touching_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claim.json")
    )
    command_value = batches.compound_command().model_copy(update={"lease_token": None})
    adapter = batches.FakeNodeAdapter(batches.all_succeed())
    executor = batches.executor(batches.FakeClient(command_value), adapter)
    with pytest.raises(ClusterExecutorError, match="no lease token"):
        executor.dispatch.execute(command_value)
    assert adapter.contexts == []
