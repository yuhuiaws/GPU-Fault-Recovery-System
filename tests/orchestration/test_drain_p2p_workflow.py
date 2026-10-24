from datetime import datetime, timezone

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.gpu_fabric_window import fabric_manager_policy
from gpu_fault.models import (
    RecoveryAction,
    WorkflowStepSpec,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.coordinator import (
    OFFICIAL_WORKFLOW_OPERATION,
    OPERATION_CAPABILITY,
    PRE_ACTION_OPERATION,
    PRE_ACTION_ORDER,
)
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.workflow_builder import WorkflowBuilder
from gpu_fault.policy import DynamicRecoveryAction, GpuFaultPolicyEngine, XidEvent
from tests._builders import build_store, workflow_request, workflow_step_execution
from tests.execution.test_node_action_transport_retry import (
    ENDPOINT,
    SECRET,
    step_context,
)


@pytest.mark.parametrize("active", [True, False])
def test_drain_p2p_is_a_fenced_complete_recovery_workflow(active):
    builder = WorkflowBuilder(
        build_store(),
        target_driver_branch=None,
        target_firmware_version=None,
        sxid_driver_remediation_codes=set(),
        sxid_firmware_update_codes=set(),
        operation_capability=OPERATION_CAPABILITY,
        official_workflow_operation=OFFICIAL_WORKFLOW_OPERATION,
        pre_action_order=PRE_ACTION_ORDER,
        pre_action_operation=PRE_ACTION_OPERATION,
    )
    event = XidEvent(
        event_id="drain-p2p",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=datetime.now(timezone.utc),
        xid=154,
        product="H100",
        gpu_uuid="GPU-a",
        xid_154_action=DynamicRecoveryAction.DRAIN_P2P,
        workload_state=WorkloadState.ACTIVE if active else WorkloadState.IDLE,
        affected_workload_ids=["job-a"] if active else [],
    )
    decision = GpuFaultPolicyEngine().evaluate_xid(event)
    operations = builder.official_operations(event, decision)
    assert operations == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        *([Op.STOP_WORKLOADS] if active else []),
        Op.QUIESCE_GPU_SERVICES,
        Op.VERIFY_NO_GPU_CLIENTS,
        Op.RECOVER_DRAIN_P2P,
        Op.RESTORE_GPU_SERVICES,
        Op.VALIDATE_GPU,
        Op.VALIDATE_FABRIC,
        Op.RESTORE_SCHEDULING,
        *([Op.RESTART_WORKLOAD] if active else []),
    ]
    assert builder.workflow_evidence_errors(event, operations) == []
    missing_identity = event.model_copy(update={"gpu_uuid": None})
    assert builder.workflow_evidence_errors(missing_identity, operations), (
        "Drain P2P recovery without a GPU UUID must be blocked"
    )
    steps = [
        WorkflowStepSpec(operation=op, execution_owner="test", node_ids=["node-a"])
        for op in operations
    ]
    assert (
        fabric_manager_policy(steps, operations.index(Op.QUIESCE_GPU_SERVICES))
        == "keep"
    )


def test_recovery_reboot_request_and_unavailable_evidence_have_distinct_actions():
    for details, action in [
        ({"gpu_recovery_reboot_required": True}, RecoveryAction.REBOOT_NODE),
        ({"reset_failed": ["GPU-a"]}, RecoveryAction.REBOOT_NODE),
        ({}, RecoveryAction.ESCALATE_OPERATOR),
        (
            {"outcome_unknown": True, "manual_confirmation_required": True},
            RecoveryAction.ESCALATE_OPERATOR,
        ),
    ]:

        def sender(_endpoint, envelope):
            return NodeActionResult(
                command_id=envelope.command.command_id,
                operation=Op.RECOVER_DRAIN_P2P,
                status=NodeActionStatus.FAILED,
                error="driver recovery did not complete",
                details=details,
            )

        adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET, sender=sender)
        outcome = adapter.execute(step_context(adapter, Op.RECOVER_DRAIN_P2P))
        assert outcome.status is WorkflowStepStatus.FAILED, outcome
        workflow = workflow_request(
            "recovery-workflow",
            "incident",
            official_steps=[
                WorkflowStepSpec(
                    operation=Op.RECOVER_DRAIN_P2P,
                    execution_owner="node-action",
                    node_ids=["node-a"],
                    gpu_uuids=["GPU-a"],
                )
            ],
            step_executions=[
                workflow_step_execution(
                    0,
                    Op.RECOVER_DRAIN_P2P,
                    status=WorkflowStepStatus.FAILED,
                    details=outcome.details,
                )
            ],
        )
        classification = HardwareEscalationService.classify(workflow)
        assert classification is not None
        assert classification[1] is action


def test_driver_reboot_request_uses_the_existing_branch_ladder():
    def compile_steps(_workflow, operations, node_id, gpu_uuids):
        return [
            WorkflowStepSpec(
                operation=op,
                execution_owner="test",
                node_ids=[node_id],
                gpu_uuids=list(gpu_uuids),
            )
            for op in operations
        ]

    steps = compile_steps(None, [Op.RECOVER_DRAIN_P2P], "node-a", ["GPU-a"])
    workflow = workflow_request("drain-reboot", "incident", official_steps=steps)
    escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), compile_steps)

    result = escalator.escalate_branch(
        workflow,
        0,
        "driver requested reboot",
        details={"gpu_recovery_reboot_required": True},
    )

    assert result is not None
    assert result.next_operation is Op.RESTART_NODE
    assert any(
        step.operation is Op.RESTART_NODE for step in result.workflow.official_steps
    ), result.workflow.official_steps


def test_failed_validation_after_p2p_recovery_uses_reboot_ladder():
    workflow = workflow_request(
        "recovery-validation",
        "incident",
        official_steps=[
            WorkflowStepSpec(
                operation=op,
                execution_owner="test",
                node_ids=["node-a"],
                gpu_uuids=["GPU-a"],
            )
            for op in (Op.RECOVER_DRAIN_P2P, Op.VALIDATE_GPU)
        ],
        step_executions=[
            workflow_step_execution(
                0, Op.RECOVER_DRAIN_P2P, status=WorkflowStepStatus.SUCCEEDED
            ),
            workflow_step_execution(
                1, Op.VALIDATE_GPU, status=WorkflowStepStatus.FAILED
            ),
        ],
        completed_step_indexes=[0],
    )
    classification = HardwareEscalationService.classify(workflow)

    assert classification is not None
    assert classification[1] is RecoveryAction.REBOOT_NODE
