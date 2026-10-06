"""A node action the Node Agent refused without touching the GPU climbs no ladder.

Live (gpu-eks-cluster, DESTR-001 attempt 6): the reset's final ownership
permit expired under the pre-spawn reads, the agent refused
OWNERSHIP_PERMIT_EXPIRED with reset_verdict not-applied and
node_action_outcome_known, the compensation restore SUCCEEDED and the chain
ended FAILED -- then the escalation ladder opened a drain successor that
quarantined and cordoned a healthy node nothing had been done to. A refusal
that applied nothing is the executor rejection's sibling: no reboot rung, no
drain, no ticket; the incident is the operator's as ESCALATED with the chain's
own cordon still on the node, and the restore disposition puts it back.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import (
    NODE_ACTION_REFUSED_STAGE,
    HardwareEscalationService,
    node_action_refused_before_acting,
)
from tests._builders import (
    copy_model,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
NOT_ATTEMPTED: dict[str, Any] = {
    "reason": "OWNERSHIP_PERMIT_EXPIRED",
    "safety_rejection": True,
    "manual_confirmation_required": True,
    "node_action_not_started": True,
    "node_action_outcome_known": True,
    "reset_verdict": "not-attempted",
    "reset_completed": [],
    "reset_failed": [],
    "reset_not_attempted": ["GPU-a"],
    "ownership_permit_timing": {"post_permit_seconds": 22.1},
}
NOT_APPLIED: dict[str, Any] = {
    "reason": "OWNERSHIP_FINAL_CLIENTS_CHANGED",
    "safety_rejection": True,
    "manual_confirmation_required": True,
    "node_action_outcome_known": True,
    "gpu_reset_attempts_spawned": 1,
    "reset_verdict": "not-applied",
    "reset_completed": [],
    "reset_failed": ["GPU-a"],
    "reset_not_attempted": [],
}
BUSY: dict[str, Any] = {
    "node_action_outcome_known": True,
    "gpu_client_holders": {"device_holders": []},
    "gpu_reset_busy_attempts": 3,
    "reset_failed": ["GPU-a"],
}


def _compile_steps(workflow: Any, operations: Any, node: str, gpus: Any) -> list[Any]:
    return [
        workflow_step(operation, node_ids=[node], gpu_uuids=list(gpus))
        for operation in operations
    ]


def _reset_chain(details: dict[str, Any]) -> tuple[Any, int]:
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        workflow_step(
            OP.QUIESCE_GPU_SERVICES, node_ids=["node-a"], gpu_uuids=["GPU-a"]
        ),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        workflow_step(
            OP.RESTORE_GPU_SERVICES, node_ids=["node-a"], gpu_uuids=["GPU-a"]
        ),
        workflow_step(OP.VALIDATE_GPU, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
    ]
    workflow = workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[OP.MARK_UNSCHEDULABLE, OP.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES),
            workflow_step_execution(
                2,
                OP.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error=str(details.get("reason") or "In use by another client"),
                details=details,
            ),
        ],
    )
    return workflow, 2


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (NOT_ATTEMPTED, True),
        (NOT_APPLIED, True),
        (BUSY, False),
        ({**NOT_APPLIED, "outcome_unknown": True}, False),
        ({**NOT_ATTEMPTED, "reset_completed": ["GPU-a"]}, False),
        ({**NOT_ATTEMPTED, "safety_rejection": False}, False),
        ({"safety_rejection": True}, False),
        (None, False),
    ],
    ids=[
        "not-attempted",
        "not-applied",
        "busy-after-retries",
        "unknown",
        "something-completed",
        "not-a-safety-refusal",
        "unattested",
        "no-details",
    ],
)
def test_the_predicate_needs_a_safety_refusal_the_agent_attested(
    details: dict[str, Any] | None, expected: bool
) -> None:
    assert node_action_refused_before_acting(details) is expected


@pytest.mark.parametrize(
    "details", [NOT_ATTEMPTED, NOT_APPLIED], ids=["not-attempted", "not-applied"]
)
def test_the_branch_ladder_exhausts_a_refused_reset_instead_of_rebooting(
    details: dict[str, Any],
) -> None:
    workflow, failed_index = _reset_chain(details)
    escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), _compile_steps)

    escalation = escalator.escalate_branch(
        workflow, failed_index, str(details["reason"]), details=details
    )

    assert escalation is not None
    assert escalation.outcome == "exhausted", escalation.reason
    assert "refused the command at its safety boundary" in escalation.reason
    assert "nothing was applied" in escalation.reason
    assert OP.RESTART_NODE not in {
        step.operation for step in escalation.workflow.official_steps
    }, "no reboot rung for a node the refusal never touched"


@pytest.mark.parametrize(
    "details", [NOT_ATTEMPTED, NOT_APPLIED], ids=["not-attempted", "not-applied"]
)
def test_the_whole_workflow_classifier_hands_off_without_a_hardware_rung(
    details: dict[str, Any],
) -> None:
    workflow, _ = _reset_chain(details)

    classified = HardwareEscalationService.classify(
        copy_model(workflow, status=WorkflowStatus.FAILED)
    )

    assert classified is not None, "the operator is told"
    stage, action, operation, flagged = classified
    assert stage == NODE_ACTION_REFUSED_STAGE
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is OP.ESCALATE_SUPPORT
    assert [item.operation for item in flagged] == [OP.RESET_GPU]


def test_a_busy_reset_and_a_real_failure_still_climb() -> None:
    for details in (BUSY, {"reset_failed": ["GPU-a"]}):
        workflow, failed_index = _reset_chain(details)
        escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), _compile_steps)
        escalation = escalator.escalate_branch(
            workflow, failed_index, "In use by another client", details=details
        )
        assert escalation is not None and escalation.outcome == "escalated", details
        classified = HardwareEscalationService.classify(
            copy_model(workflow, status=WorkflowStatus.FAILED)
        )
        assert classified is not None, details


class _StepBuilder:
    """Compiles one step per operation with the requested scope."""

    def compile_steps(
        self,
        operations: Any,
        profile: Any,
        node_ids: Any,
        gpu_uuids: Any,
        workload_ids: Any = None,
    ) -> tuple[list[Any], list[Any]]:
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


@pytest.mark.parametrize(
    "details", [NOT_ATTEMPTED, NOT_APPLIED], ids=["not-attempted", "not-applied"]
)
def test_the_hand_off_plans_evidence_and_the_ticket_but_no_quarantine(
    details: dict[str, Any],
) -> None:
    """Live (gpu-eks-cluster): the successor of the refused reset quarantined
    and cordoned a healthy node. The hand-off now freezes evidence and opens
    the ticket; the chain's own cordon is what stays on the node."""

    from gpu_fault.app import default_simulated_profile
    from gpu_fault.models import IncidentState
    from tests._builders import build_store, fault_incident

    store = build_store()
    store.save_profile(default_simulated_profile())
    workflow, _ = _reset_chain(details)
    workflow = copy_model(
        workflow, status=WorkflowStatus.FAILED, runtime_profile_version="simulated-v1"
    )
    incident = fault_incident(
        workflow.incident_id,
        "event-reset",
        event_type="XID",
        state=IncidentState.ESCALATED,
        workflow_request_id=workflow.request_id,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        fencing_token=1,
    )
    store.save_incident_and_workflow(incident, workflow)

    escalation = HardwareEscalationService(store, _StepBuilder()).escalate(workflow)

    assert escalation is not None, "the operator is handed the refusal"
    successor_incident, successor = escalation
    assert successor_incident.event_id == f"support-after-{workflow.request_id}"
    assert [step.operation for step in successor.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.ESCALATE_SUPPORT,
    ], "no MARK_UNSCHEDULABLE, no QUARANTINE, no reboot: nothing was done to the node"
