"""Two DAG branches refused at the Agent's boundary restore both nodes and hand off.

Live (gpu-eks-cluster, DESTR-015): two node branches each quiesced their own
node; both RESET_GPU steps were refused OWNERSHIP_PERMIT_EXPIRED with the
Agent's attestation (not-applied, nothing in flight). The one
RESTORE_GPU_SERVICES the record appended for both nodes was judged from the
latest quiesce alone and FAILED before any executor claimed a node action --
"quiesce maintenance evidence is missing agent generation" for the other
branch's node -- so the services stayed stopped on both nodes and the ladder,
reading a failed restore rather than a pure refusal, quarantined both via the
support-after hand-off. The compensation now reads each node's generation from
its own quiesce, the restore runs on both nodes, and the hand-off plans
evidence and the ticket only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.app import default_simulated_profile
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.node_action_uncertainty import COMPENSATION_RESTORE_KEY
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import (
    NODE_ACTION_REFUSED_STAGE,
    HardwareEscalationService,
)
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from tests._builders import active_workflow_executor, workflow_step
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.regional._cov95_runtime_adapter import fleet_adapter

OP = WorkflowOperation
QUIESCE, RESET, RESTORE = OP.QUIESCE_GPU_SERVICES, OP.RESET_GPU, OP.RESTORE_GPU_SERVICES
NODES = ["node-a", "node-b"]


def _refusal(node: str) -> dict[str, Any]:
    """The Agent's attested refusal of the live record, per node."""

    return {
        "reason": "OWNERSHIP_PERMIT_EXPIRED",
        "safety_rejection": True,
        "manual_confirmation_required": True,
        "node_action_outcome_known": True,
        "agent_queue_ownership_checked": True,
        "ownership_check_boundary": "AGENT_PRE_SPAWN",
        "ownership_permit_timing": {
            "pre_permit_seconds": 14.6,
            "permit_wait_seconds": 1.8,
            "post_permit_seconds": 8.1,
            "permit_lifetime_seconds": 3.0,
        },
        "gpu_reset_attempts_spawned": 1,
        "reset_verdict": "not-applied",
        "reset_completed": [],
        "reset_outcome_unknown": [],
        "reset_failed": [f"GPU-{node}"],
        "reset_not_attempted": [],
        "failed_nodes": [node],
        "node_failures": {
            node: [
                "node action refused before acting: OwnershipRefused: OWNERSHIP_PERMIT_EXPIRED"
            ]
        },
    }


class _StepBuilder:
    def compile_steps(
        self, operations, profile, node_ids, gpu_uuids, workload_ids=None
    ):
        return (
            [
                workflow_step(
                    operation,
                    node_ids=list(node_ids),
                    gpu_uuids=list(gpu_uuids),
                    workload_ids=list(workload_ids or []),
                )
                for operation in operations
            ],
            [],
        )


def _two_branch_flow() -> tuple[FlowHarness, Any, list[SignedNodeAction]]:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    sent: list[SignedNodeAction] = []

    def send(_endpoint: str, envelope: SignedNodeAction) -> NodeActionResult:
        sent.append(envelope)
        return NodeActionResult(
            command_id=envelope.command.command_id,
            operation=envelope.command.operation,
            status=NodeActionStatus.SUCCEEDED,
            details={"restored": True},
        )

    fleet = fleet_adapter(sender=send)
    expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()

    def scripted(context: WorkflowStepContext) -> None:
        (node,) = context.step.node_ids
        h.adapter.outcomes[QUIESCE] = WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "agent_generations": {node: 1},
                "maintenance_window_expires_at": expires,
            },
        )
        h.adapter.outcomes[RESET] = WorkflowStepOutcome.failed(
            f"node agent {node}: ResetProgressError: OWNERSHIP_PERMIT_EXPIRED",
            details=_refusal(node),
        )

    h.adapter.before = scripted
    base = h.workflow.official_steps
    steps = []
    for offset, node in enumerate(NODES):
        first = 3 * offset
        branch = f"branch:{node}"
        steps.extend(
            [
                base[0].model_copy(
                    update={
                        "node_ids": [node],
                        "gpu_uuids": [f"GPU-{node}"],
                        "branch_id": branch,
                        "branch_node_ids": [node],
                    }
                ),
                base[1].model_copy(
                    update={
                        "node_ids": [node],
                        "gpu_uuids": [f"GPU-{node}"],
                        "branch_id": branch,
                        "branch_node_ids": [node],
                        "depends_on_step_indexes": [first],
                    }
                ),
                base[2].model_copy(
                    update={
                        "execution_owner": fleet.owner,
                        "node_ids": [node],
                        "gpu_uuids": [f"GPU-{node}"],
                        "branch_id": branch,
                        "branch_node_ids": [node],
                        "depends_on_step_indexes": [first + 1],
                    }
                ),
            ]
        )
    h.store.save_profile(default_simulated_profile())
    h.amend(
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        runtime_profile_version="simulated-v1",
    )
    incident = h.store.get_incident(h.incident.incident_id)
    h.store.save_incident(
        incident.model_copy(
            update={"node_ids": NODES, "gpu_uuids": [f"GPU-{n}" for n in NODES]}
        ),
        expected=incident,
    )
    h.incident = h.store.get_incident(h.incident.incident_id)
    h.executor = active_workflow_executor(
        h.store, [fleet, h.adapter], [QUIESCE, RESET, RESTORE]
    )

    def forbidden_rung(*_args: Any) -> Any:
        raise AssertionError("a refused reset must not compile a hardware rung")

    h.executor.branch_escalator = BranchEscalator(
        DagBrancher(RecoveryArbiter()), forbidden_rung
    )
    return h, fleet, sent


def test_two_refused_branches_restore_both_nodes_and_hand_off_without_quarantine() -> (
    None
):
    h, fleet, sent = _two_branch_flow()

    result = h.execute()

    assert result.status is WorkflowStatus.FAILED, result
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.blocked_kind is None
    assert sorted(saved.exhausted_branch_ids) == ["branch:node-a", "branch:node-b"]
    assert saved.branch_escalation_counts == {}
    compensation = [
        step
        for step in saved.official_steps
        if step.operation is RESTORE and step.parameters.get(COMPENSATION_RESTORE_KEY)
    ]
    assert len(compensation) == 1, "one restore is appended for the two quiesced nodes"
    assert compensation[0].node_ids == NODES
    assert compensation[0].parameters[COMPENSATION_RESTORE_KEY] == [0, 3]
    assert compensation[0].execution_owner == fleet.owner
    restore = [
        item
        for item in saved.step_executions
        if item.operation is RESTORE and item.status is WorkflowStepStatus.SUCCEEDED
    ]
    assert restore, [
        (item.operation.value, item.status.value, item.error)
        for item in saved.step_executions
    ]
    assert sorted(envelope.command.node_id for envelope in sent) == NODES, (
        "the restore reached both agents"
    )
    assert all(envelope.command.operation is RESTORE for envelope in sent), (
        "nothing but the restore reached the agents: no reboot, no reset retry"
    )
    assert RESTORE in saved.completed_operations
    occupying = NodeConflictService(
        h.store, RecoveryArbiter()
    ).active_node_exclusive_workflow(h.incident.cluster_id, set(NODES))
    assert occupying is None, "no parked record holds either node"
    classified = HardwareEscalationService.classify(saved)
    assert classified is not None
    stage, action, operation, flagged = classified
    assert stage == NODE_ACTION_REFUSED_STAGE
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is OP.ESCALATE_SUPPORT
    assert sorted(item.step_index for item in flagged) == [1, 4]
    incident = h.store.get_incident(h.incident.incident_id)
    assert incident.state is IncidentState.ESCALATED
    escalation = HardwareEscalationService(h.store, _StepBuilder()).escalate(saved)
    assert escalation is not None
    _successor_incident, successor = escalation
    assert [step.operation for step in successor.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.ESCALATE_SUPPORT,
    ], "no MARK_UNSCHEDULABLE, no QUARANTINE, no reboot for nodes nothing was done to"
