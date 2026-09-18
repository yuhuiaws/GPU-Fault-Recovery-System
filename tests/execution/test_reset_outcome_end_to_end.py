from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.models import (
    BlockedKind,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    workflow_is_open,
)
from gpu_fault.node_agent import (
    NodeActionExecutor,
    NodeActionLedger,
    NodeActionResult,
    SignedNodeAction,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.node_agent._support import SECRET, FakeRunner, no_device_clients

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
GPUS = ["GPU-a", "GPU-b", "GPU-c"]


class ResetRunner(FakeRunner):
    def __init__(self, *, fabric: bool, timeout: bool) -> None:
        super().__init__(
            reset_timeout_seconds=(180 if fabric else 120) if timeout else None,
            reset_timeout_gpu_uuid=None if fabric else "GPU-b",
        )
        self.fail_cleanly = not timeout

    def __call__(self, command: list[str], **kwargs: Any) -> CompletedProcess[str]:
        if "--query-gpu=uuid" in command:
            self.commands.append(command)
            return CompletedProcess(command, 0, stdout="\n".join(GPUS), stderr="")
        if self.fail_cleanly and "--gpu-reset" in command and "GPU-b" in command:
            self.commands.append(command)
            raise CalledProcessError(1, command, stderr="reset failed cleanly")
        return super().__call__(command, **kwargs)


class ClaimedWindow:
    def __init__(self) -> None:
        self.commands: list[str | None] = []

    def assert_quiesced(self, *, incident_id: str, command_id: str | None, **_: Any):
        assert incident_id == "incident-active"
        self.commands.append(command_id)


def reset_flow(
    tmp_path: Path, operation: WorkflowOperation, *, dag: bool, timeout: bool
):
    h = FlowHarness([QUIESCE, operation, RESTORE])
    runner = ResetRunner(
        fabric=operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, timeout=timeout
    )
    window = ClaimedWindow()
    agent = NodeActionExecutor(
        secret=SECRET,
        node_ids={"node-a"},
        allowed_operations={operation},
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=window,
        ledger=NodeActionLedger(str(tmp_path / "actions.db")),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {
            gpu: f"/dev/nvidia{index}" for index, gpu in enumerate(GPUS)
        },
        sleep=lambda _: None,
    )
    sent: list[SignedNodeAction] = []
    results: list[NodeActionResult] = []

    def send(_endpoint: str, envelope: SignedNodeAction) -> NodeActionResult:
        sent.append(envelope)
        result = agent.execute(envelope)
        results.append(result)
        return NodeActionResult.model_validate_json(result.model_dump_json())

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, SECRET, sender=send
    )
    h.executor.adapters.insert(0, adapter)
    h.amend(
        dag_enabled=dag,
        official_steps=[
            step.model_copy(
                update={
                    "execution_owner": adapter.owner
                    if step.operation is operation
                    else step.execution_owner,
                    "gpu_uuids": GPUS,
                    "branch_id": "branch:node-a" if dag else None,
                    "branch_node_ids": ["node-a"] if dag else [],
                    "depends_on_step_indexes": [index - 1] if dag and index else [],
                }
            )
            for index, step in enumerate(h.workflow.official_steps)
        ],
    )

    def forbidden_rung(*_args: Any):
        raise AssertionError("an unknown reset must not compile a hardware rung")

    if dag:
        h.executor.branch_escalator = BranchEscalator(
            DagBrancher(RecoveryArbiter()), forbidden_rung
        )
    return h, agent, runner, window, sent, results


@pytest.mark.parametrize("dag", [False, True], ids=["sequential", "dag"])
@pytest.mark.parametrize(
    "operation",
    [WorkflowOperation.RESET_GPU, WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES],
)
def test_real_reset_timeout_remains_unknown_through_workflow_and_both_ladders(
    tmp_path: Path, operation: WorkflowOperation, dag: bool
) -> None:
    h, agent, runner, window, sent, results = reset_flow(
        tmp_path, operation, dag=dag, timeout=True
    )

    assert h.execute().status is WorkflowStatus.BLOCKED
    saved = h.store.get_workflow(h.workflow.request_id)
    failure = next(
        item for item in saved.step_executions if item.operation is operation
    )
    assert failure.status is WorkflowStepStatus.FAILED
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "an unknown reset must retain node occupancy for operator reconciliation"
    )
    assert failure.details["outcome_unknown"] is True
    assert failure.details["manual_confirmation_required"] is True
    assert failure.details["failed_nodes"] == ["node-a"]
    assert "outcome unknown" in failure.details["node_failures"]["node-a"][0]
    expected = (
        {
            "reset_completed": ["GPU-a"],
            "reset_outcome_unknown": ["GPU-b"],
            "reset_failed": [],
            "reset_not_attempted": ["GPU-c"],
        }
        if operation is WorkflowOperation.RESET_GPU
        else {
            "reset_completed": [],
            "reset_outcome_unknown": GPUS,
            "reset_failed": [],
            "reset_not_attempted": [],
        }
    )
    for key, value in expected.items():
        assert failure.details[key] == value
        assert results[0].details[key] == value
    assert not results[0].retryable, "a timed-out reset must not be resubmitted"
    assert [context.step.operation for context in h.adapter.calls] == [QUIESCE]
    assert saved.branch_escalation_counts == {}
    assert RESTORE not in saved.completed_operations
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None
    assert classification[:3] == (
        "manual_confirmation_required",
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
    )
    assert h.execute().status is WorkflowStatus.BLOCKED
    assert len(sent) == 1
    assert agent.execute(sent[0]) == results[0], "ledger replay must not reset again"
    assert window.commands == [sent[0].command.command_id]
    assert [command for command in runner.commands if "--gpu-reset" in command] == (
        [["nvidia-smi", "--gpu-reset", "-i", gpu] for gpu in GPUS[:2]]
        if operation is WorkflowOperation.RESET_GPU
        else [["nvidia-smi", "--gpu-reset"]]
    )


def test_confirmed_partial_reset_failure_keeps_progress_and_can_restore(tmp_path: Path):
    h, _agent, _runner, _window, _sent, results = reset_flow(
        tmp_path, WorkflowOperation.RESET_GPU, dag=False, timeout=False
    )
    assert h.execute().status is WorkflowStatus.FAILED
    saved = h.store.get_workflow(h.workflow.request_id)
    failure = next(
        item
        for item in saved.step_executions
        if item.operation is WorkflowOperation.RESET_GPU
    )
    for key, value in {
        "reset_completed": ["GPU-a"],
        "reset_outcome_unknown": [],
        "reset_failed": ["GPU-b"],
        "reset_not_attempted": ["GPU-c"],
    }.items():
        assert results[0].details[key] == value
        assert failure.details[key] == value
    assert "outcome_unknown" not in failure.details
    assert [context.step.operation for context in h.adapter.calls] == [QUIESCE, RESTORE]
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None
    assert classification[1] is RecoveryAction.REBOOT_NODE
