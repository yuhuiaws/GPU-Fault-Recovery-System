"""A validation the product's own quiesce taint starved climbs no ladder.

Live 2026-10-07 (gpu-eks-cluster DESTR-025, attempt 9): after the in-place
RESET_GPU -> RESTART_NODE rung the rebooted node still carried the eks-managed
``gpu-fault.io/quiesce`` NoExecute taint (the retired RESTORE_GPU_SERVICES
would have lifted it), the exporter could not schedule, VALIDATE_GPU waited out
its 600 s cap with ``quiesce_taint_present`` as its only word, the branch
ladder appended a REPLACE_NODE that no spare could serve, and the exhausted
record's support hand-off quarantined a node nothing had been learned about.
The spec authorises no replace at any point of that case. Such a validation is
"could not look", not "looked and found a fault": neither ladder climbs on
it, and the hand-off plans evidence and the ticket only.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import (
    VALIDATION_UNOBSERVED_STAGE,
    HardwareEscalationService,
    validation_unobserved_failure,
)
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
NODE = "node-a"
RESTORE_EXIT = (
    "the node still carries the gpu-fault quiesce taint (NoExecute) of wf-reset; "
    "the exporter cannot run until it is removed -- restore the owner incident "
    "(gpu-fault-admin submit-remediation --disposition restore)"
)
UNOBSERVED: dict[str, Any] = {
    "node_pending": {
        NODE: {
            "missing_or_stale_collectors": ["GPU_METRICS"],
            "quiesce_taint_present": True,
            "quiesce_taint_holders": {"wf-reset": "incident-00000000"},
            "reason": RESTORE_EXIT,
        }
    },
    "pending_nodes": [NODE],
    "failed_nodes_observed": [],
    "node_failures_observed": {},
    "step_waiting_seconds": 603,
    "step_waiting_timeout_seconds": 600,
}


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (UNOBSERVED, True),
        ({**UNOBSERVED, "failed_nodes_observed": [NODE]}, False),
        ({**UNOBSERVED, "node_failures_observed": {NODE: ["xid"]}}, False),
        ({**UNOBSERVED, "node_failures": {NODE: ["dcgm"]}}, False),
        (
            {
                **UNOBSERVED,
                "node_pending": {
                    NODE: {"missing_or_stale_collectors": ["GPU_METRICS"]}
                },
            },
            False,
        ),
        (
            {
                **UNOBSERVED,
                "node_pending": {
                    NODE: UNOBSERVED["node_pending"][NODE],
                    "node-b": {"missing_or_stale_collectors": ["GPU_METRICS"]},
                },
            },
            False,
        ),
        ({**UNOBSERVED, "node_pending": {}}, False),
        ({"step_waiting_timeout_seconds": 600}, False),
        (None, False),
    ],
    ids=[
        "starved-by-own-taint",
        "a-node-was-observed-failing",
        "node-failures-observed",
        "node-failures",
        "pending-for-another-reason",
        "one-node-pending-for-another-reason",
        "nothing-pending",
        "no-pending-evidence",
        "no-details",
    ],
)
def test_the_predicate_needs_every_pending_node_behind_the_products_taint(
    details: dict[str, Any] | None, expected: bool
) -> None:
    assert validation_unobserved_failure(details) is expected


def _compile_steps(workflow: Any, operations: Any, node: str, gpus: Any) -> list[Any]:
    return [
        workflow_step(operation, node_ids=[node], gpu_uuids=list(gpus))
        for operation in operations
    ]


def _rebooted_chain(validation_details: dict[str, Any]) -> tuple[Any, int]:
    """QUIESCE ok, RESET failed, RESTART_NODE ok, VALIDATE_GPU failed: the flat
    DESTR-025 record after its reboot rung, as the whole-workflow path sees it."""

    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[NODE], gpu_uuids=["GPU-a"]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[NODE], gpu_uuids=["GPU-a"]),
        workflow_step(OP.RESET_GPU, node_ids=[NODE], gpu_uuids=["GPU-a"]),
        workflow_step(OP.RESTART_NODE, node_ids=[NODE], gpu_uuids=["GPU-a"]),
        workflow_step(OP.VALIDATE_GPU, node_ids=[NODE], gpu_uuids=["GPU-a"]),
        workflow_step(OP.RESTORE_SCHEDULING, node_ids=[NODE], gpu_uuids=["GPU-a"]),
    ]
    workflow = workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=[0, 1, 3],
        completed_operations=[
            OP.MARK_UNSCHEDULABLE,
            OP.QUIESCE_GPU_SERVICES,
            OP.RESTART_NODE,
        ],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(
                1, OP.QUIESCE_GPU_SERVICES, details={"quiesce_taint_nodes": [NODE]}
            ),
            workflow_step_execution(
                2,
                OP.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="GpuClientsActive: GPU device clients are still active",
                details={"gpu_client_holders": {"device_holders": ["80007"]}},
            ),
            workflow_step_execution(3, OP.RESTART_NODE, details={"action": "REBOOT"}),
            workflow_step_execution(
                4,
                OP.VALIDATE_GPU,
                WorkflowStepStatus.FAILED,
                error="step 4/VALIDATE_GPU stayed non-terminal for 603s",
                details=validation_details,
            ),
        ],
    )
    return workflow, 4


def test_the_whole_workflow_classifier_hands_off_without_a_replace_rung() -> None:
    workflow, _ = _rebooted_chain(UNOBSERVED)

    classified = HardwareEscalationService.classify(
        copy_model(workflow, status=WorkflowStatus.FAILED)
    )

    assert classified is not None, "the operator is told"
    stage, action, operation, flagged = classified
    assert stage == VALIDATION_UNOBSERVED_STAGE
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is OP.ESCALATE_SUPPORT
    assert [item.operation for item in flagged] == [OP.VALIDATE_GPU], (
        "the earlier busy reset was answered by the reboot; it is not re-climbed"
    )


def test_a_validation_that_observed_a_failure_still_climbs_to_replace() -> None:
    workflow, _ = _rebooted_chain(
        {
            **UNOBSERVED,
            "failed_nodes_observed": [NODE],
            "node_failures_observed": {NODE: ["xid_79"]},
        }
    )

    classified = HardwareEscalationService.classify(
        copy_model(workflow, status=WorkflowStatus.FAILED)
    )

    assert classified is not None
    stage, action, operation, _flagged = classified
    assert stage != VALIDATION_UNOBSERVED_STAGE, stage
    assert action in {RecoveryAction.REBOOT_NODE, RecoveryAction.REPLACE_NODE}, (
        "a validation that looked and found a fault keeps a hardware rung"
    )
    assert operation in {OP.RESTART_NODE, OP.REPLACE_NODE}, operation


def test_the_flat_record_is_not_adopted_onto_the_ladder_for_an_unobserved_validation() -> (
    None
):
    workflow, failed_index = _rebooted_chain(UNOBSERVED)

    escalation = BranchEscalator(
        DagBrancher(RecoveryArbiter()), _compile_steps
    ).escalate_branch(
        workflow, failed_index, "stayed non-terminal for 603s", details=UNOBSERVED
    )

    assert escalation is None, (
        "no rung fits; the whole-workflow path hands the record to an operator"
    )


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


def test_the_hand_off_plans_evidence_and_the_ticket_and_names_the_restore_exit() -> (
    None
):
    store = build_store()
    store.save_profile(default_simulated_profile())
    workflow, _ = _rebooted_chain(UNOBSERVED)
    workflow = copy_model(
        workflow, status=WorkflowStatus.FAILED, runtime_profile_version="simulated-v1"
    )
    incident = fault_incident(
        workflow.incident_id,
        "event-reset",
        event_type="XID",
        state=IncidentState.ESCALATED,
        workflow_request_id=workflow.request_id,
        node_ids=[NODE],
        gpu_uuids=["GPU-a"],
        fencing_token=1,
    )
    store.save_incident_and_workflow(incident, workflow)

    escalation = HardwareEscalationService(store, _StepBuilder()).escalate(workflow)

    assert escalation is not None, "the operator is handed the record"
    successor_incident, successor = escalation
    assert successor_incident.event_id == f"support-after-{workflow.request_id}"
    assert [step.operation for step in successor.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.ESCALATE_SUPPORT,
    ], "no MARK_UNSCHEDULABLE, no QUARANTINE, no REPLACE_NODE: nothing was learned"
    assert successor_incident.effective_action is RecoveryAction.ESCALATE_OPERATOR
    assert any(
        VALIDATION_UNOBSERVED_STAGE in reason and RESTORE_EXIT in reason
        for reason in successor_incident.reasons
    ), successor_incident.reasons
