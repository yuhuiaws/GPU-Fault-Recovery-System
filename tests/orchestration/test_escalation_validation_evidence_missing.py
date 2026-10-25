"""A validation a bound failed with no node observed failing fires no requested
escalation.

Live 2026-10-25 (plain EKS, GF-GENERAL-LIVE-013 ``link_inactive``): the EFA
diagnostic chain (FREEZE_EVIDENCE -> VALIDATE_FABRIC) carries
``failure_escalation_action=REBOOT_NODE`` on its validation. With the node's
collectors dark the step waited out the 600 s per-step cap with
``missing_or_stale_collectors`` as its only word, and the whole-workflow
classifier applied the plan's request to the cap's failure: RESTART_NODE on a
node nothing had been learned about. The request is for a validation that
looked and found the fault still there -- a verdict (``failed_nodes``), or a
node observed failing while the step waited for its siblings
(``node_failures_observed``) -- never for one the cap or the workflow deadline
failed before it could look. Such a record is handed to an operator with the
missing evidence named, the way the quiesce-taint case already is
(``validation_unobserved_failure``), and the designed escalation stays for a
validation that observed the failure.
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
    VALIDATION_EVIDENCE_MISSING_STAGE,
    VALIDATION_UNOBSERVED_STAGE,
    HardwareEscalationService,
)
from gpu_fault.orchestration.validation_evidence import (
    missing_evidence_reason,
    validation_without_observed_failure,
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
REBOOT = {"failure_escalation_action": RecoveryAction.REBOOT_NODE.value}
INVENTORY = {
    "inventory_requirements_by_node": {
        NODE: {"metrics": {"efa_inventory_active_count": 16}}
    }
}
MISMATCH = "efa_inventory_active_count=15,expected=16"
CAP = {"step_waiting_seconds": 603, "step_waiting_timeout_seconds": 600}
#: The cap's failure of a validation whose node's collectors stayed dark.
DARK: dict[str, Any] = {
    "node_pending": {
        NODE: {"missing_or_stale_collectors": ["GPU_METRICS", "HOST_TELEMETRY"]}
    },
    "pending_nodes": [NODE],
    "failed_nodes_observed": [],
    "node_failures_observed": {},
    **CAP,
}
#: The cap's failure of a validation that never saw a fresh inventory sample.
NO_FRESH_INVENTORY: dict[str, Any] = {
    **DARK,
    "node_pending": {
        NODE: {
            "missing_fresh_inventory_metric": ["efa_inventory_active_count"],
            "observed_after": ["2026-10-25T12:00:30+00:00"],
        }
    },
}
#: The cap's failure of a validation that had already seen the node failing.
OBSERVED: dict[str, Any] = {
    **DARK,
    "failed_nodes_observed": [NODE],
    "node_failures_observed": {NODE: [MISMATCH]},
}
#: The adapter's own verdict on a fresh reading.
VERDICT: dict[str, Any] = {
    "failed_nodes": [NODE],
    "node_failures": {NODE: [MISMATCH]},
    "validation": "VALIDATE_FABRIC",
    "inventory_evidence": {
        NODE: {
            "efa_inventory_active_count": {
                "value": 15,
                "expected": 16,
                "observed_at": "2026-10-25T12:01:00+00:00",
            }
        }
    },
}
#: The workflow deadline's failure of a validation that had no verdict yet.
DEADLINE: dict[str, Any] = {
    "workflow_execution_deadline": "2026-10-25T12:30:00+00:00",
    "workflow_deadline_overdue_seconds": 4,
    "workflow_deadline_remote_command_cancellation": {"cancelled": 0},
    "workflow_lifetime_exceeded": False,
}
TAINT_EXIT = (
    "the node still carries the gpu-fault quiesce taint (NoExecute) of wf-reset; "
    "the exporter cannot run until it is removed -- restore the owner incident "
    "(gpu-fault-admin submit-remediation --disposition restore)"
)
#: The quiesce-taint precedent: every pending node behind the product's taint.
TAINTED: dict[str, Any] = {
    **DARK,
    "node_pending": {
        NODE: {
            "missing_or_stale_collectors": ["GPU_METRICS"],
            "quiesce_taint_present": True,
            "quiesce_taint_holders": {"wf-reset": "incident-00000000"},
            "reason": TAINT_EXIT,
        }
    },
}


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (DARK, True),
        (NO_FRESH_INVENTORY, True),
        (DEADLINE, True),
        (TAINTED, True),
        ({"step_waiting_timeout_seconds": 600}, True),
        (OBSERVED, False),
        ({**DARK, "failed_nodes_observed": [NODE]}, False),
        (VERDICT, False),
        ({**DARK, "node_failures": {NODE: [MISMATCH]}}, False),
        ({}, False),
        (None, False),
    ],
    ids=[
        "cap-with-dark-collectors",
        "cap-without-a-fresh-inventory-sample",
        "deadline-before-a-verdict",
        "cap-behind-the-quiesce-taint",
        "cap-without-a-wait-record",
        "a-node-was-observed-failing",
        "failed-nodes-observed",
        "the-adapter-gave-a-verdict",
        "node-failures-on-the-record",
        "no-details",
        "none",
    ],
)
def test_the_predicate_needs_a_bound_or_a_wait_and_no_observed_failure(
    details: dict[str, Any] | None, expected: bool
) -> None:
    assert validation_without_observed_failure(details) is expected


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            DARK["node_pending"][NODE],
            "missing_or_stale_collectors: GPU_METRICS, HOST_TELEMETRY",
        ),
        (
            NO_FRESH_INVENTORY["node_pending"][NODE],
            "missing_fresh_inventory_metric: efa_inventory_active_count; "
            "observed_after: 2026-10-25T12:00:30+00:00",
        ),
        (TAINTED["node_pending"][NODE], TAINT_EXIT),
        (
            {
                "missing_recent_metrics": ["gpu_temperature_c"],
                "missing_recent_metrics_by_gpu": {"GPU-b": ["gpu_temperature_c"]},
                "fabric_manager_unit": "activating",
                "step_waiting_slow": True,
            },
            "fabric_manager_unit: activating; missing_recent_metrics: "
            "gpu_temperature_c; missing_recent_metrics_by_gpu: GPU-b; "
            "step_waiting_slow",
        ),
        ({}, None),
        ({"quiesce_taint_present": False, "holders": None}, None),
    ],
    ids=[
        "dark-collectors",
        "no-fresh-inventory-sample-with-its-window",
        "the-adapter-wrote-the-reason",
        "every-kind-of-value",
        "empty",
        "nothing-to-say",
    ],
)
def test_the_pending_record_is_named_for_the_operator(
    record: dict[str, Any], expected: str | None
) -> None:
    assert missing_evidence_reason(record) == expected


def _diagnostic_chain(validation_details: dict[str, Any], **step_values: Any) -> Any:
    """The EFA diagnostic chain (RUN_DIAGNOSTICS for LINK_INACTIVE) after its
    FREEZE_EVIDENCE completed and its VALIDATE_FABRIC FAILED."""

    steps = [
        workflow_step(OP.FREEZE_EVIDENCE, node_ids=[NODE]),
        workflow_step(
            OP.VALIDATE_FABRIC,
            node_ids=[NODE],
            parameters={**REBOOT, **INVENTORY},
            **step_values,
        ),
    ]
    return workflow_request(
        "wf-efa-diagnostic",
        "inc-efa",
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[OP.FREEZE_EVIDENCE],
        step_executions=[
            workflow_step_execution(0, OP.FREEZE_EVIDENCE),
            workflow_step_execution(
                1,
                OP.VALIDATE_FABRIC,
                WorkflowStepStatus.FAILED,
                error="step 1/VALIDATE_FABRIC stayed non-terminal for 603s",
                details=validation_details,
            ),
        ],
    )


def _plugin_chain(
    plugin_status: WorkflowStepStatus, validation_details: dict[str, Any] | None
) -> Any:
    """The EFA device-plugin chain: RESTART_EFA_DEVICE_PLUGIN ->
    TRIGGER_HEALTH_SNAPSHOT -> VALIDATE_FABRIC, the plugin step and the
    validation both asking for REBOOT_NODE on failure (``health.py``)."""

    steps = [
        workflow_step(
            OP.RESTART_EFA_DEVICE_PLUGIN,
            node_ids=[NODE],
            parameters={"expected_count": 16, **REBOOT},
        ),
        workflow_step(OP.TRIGGER_HEALTH_SNAPSHOT, node_ids=[NODE]),
        workflow_step(
            OP.VALIDATE_FABRIC, node_ids=[NODE], parameters={**REBOOT, **INVENTORY}
        ),
    ]
    executions = [
        workflow_step_execution(
            0,
            OP.RESTART_EFA_DEVICE_PLUGIN,
            plugin_status,
            error=(
                None
                if plugin_status is WorkflowStepStatus.SUCCEEDED
                else "device plugin pod did not become Ready"
            ),
        )
    ]
    completed: list[int] = []
    if validation_details is not None:
        completed = [0, 1]
        executions.extend(
            [
                workflow_step_execution(1, OP.TRIGGER_HEALTH_SNAPSHOT),
                workflow_step_execution(
                    2,
                    OP.VALIDATE_FABRIC,
                    WorkflowStepStatus.FAILED,
                    error="step 2/VALIDATE_FABRIC stayed non-terminal for 603s",
                    details=validation_details,
                ),
            ]
        )
    return workflow_request(
        "wf-efa-plugin",
        "inc-efa-plugin",
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=completed,
        completed_operations=[steps[index].operation for index in completed],
        step_executions=executions,
    )


@pytest.mark.parametrize(
    "workflow",
    [
        _diagnostic_chain(DARK),
        _diagnostic_chain(NO_FRESH_INVENTORY),
        _diagnostic_chain(DEADLINE),
        _plugin_chain(WorkflowStepStatus.SUCCEEDED, DARK),
    ],
    ids=[
        "diagnostic-chain-dark-collectors",
        "diagnostic-chain-no-fresh-inventory-sample",
        "diagnostic-chain-deadline",
        "plugin-chain-dark-collectors",
    ],
)
def test_a_validation_that_observed_no_failure_fires_no_requested_reboot(
    workflow: Any,
) -> None:
    classified = HardwareEscalationService.classify(workflow)

    assert classified is not None, "the operator is told; a silent FAILED is not"
    stage, action, operation, flagged = classified
    assert stage == VALIDATION_EVIDENCE_MISSING_STAGE
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is OP.ESCALATE_SUPPORT
    assert [item.operation for item in flagged] == [OP.VALIDATE_FABRIC]


@pytest.mark.parametrize(
    "details",
    [OBSERVED, VERDICT],
    ids=["cap-after-the-node-was-seen-failing", "the-adapter-verdict"],
)
def test_a_validation_that_observed_the_failure_keeps_the_designed_reboot(
    details: dict[str, Any],
) -> None:
    classified = HardwareEscalationService.classify(_diagnostic_chain(details))

    assert classified is not None, "a fresh inactive link keeps the reboot rung"
    stage, action, operation, flagged = classified
    assert stage == "efa_recovery"
    assert action is RecoveryAction.REBOOT_NODE
    assert operation is OP.RESTART_NODE
    assert [item.operation for item in flagged] == [OP.VALIDATE_FABRIC]


def test_the_quiesce_taint_precedent_is_judged_first_and_unchanged() -> None:
    classified = HardwareEscalationService.classify(_diagnostic_chain(TAINTED))

    assert classified is not None, "the operator is told"
    stage, action, operation, _flagged = classified
    assert stage == VALIDATION_UNOBSERVED_STAGE
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is OP.ESCALATE_SUPPORT


def test_a_failed_node_action_keeps_its_requested_reboot() -> None:
    """The request is withheld for a validation without evidence only: a
    plugin restart that itself failed is a definite action failure."""

    classified = HardwareEscalationService.classify(
        _plugin_chain(WorkflowStepStatus.FAILED, None)
    )

    assert classified is not None, "a failed plugin restart climbs as designed"
    stage, action, operation, flagged = classified
    assert stage == "efa_recovery"
    assert action is RecoveryAction.REBOOT_NODE
    assert operation is OP.RESTART_NODE
    assert [item.operation for item in flagged] == [OP.RESTART_EFA_DEVICE_PLUGIN]


def test_the_branch_escalator_adopts_no_rung_for_the_withheld_validation() -> None:
    workflow = _diagnostic_chain(DARK)

    escalation = BranchEscalator(
        DagBrancher(RecoveryArbiter()),
        lambda workflow, operations, node, gpus: [
            workflow_step(operation, node_ids=[node]) for operation in operations
        ],
    ).escalate_branch(
        copy_model(workflow, status=WorkflowStatus.RUNNING),
        1,
        "stayed non-terminal for 603s",
        details=DARK,
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


def test_the_hand_off_plans_evidence_and_the_ticket_and_names_the_missing_evidence() -> (
    None
):
    store = build_store()
    store.save_profile(default_simulated_profile())
    workflow = copy_model(
        _diagnostic_chain(DARK), runtime_profile_version="simulated-v1"
    )
    incident = fault_incident(
        workflow.incident_id,
        "event-efa",
        event_type="HOST_HEALTH",
        state=IncidentState.ESCALATED,
        workflow_request_id=workflow.request_id,
        node_ids=[NODE],
        fencing_token=1,
    )
    store.save_incident_and_workflow(incident, workflow)

    escalation = HardwareEscalationService(store, _StepBuilder()).escalate(workflow)

    assert escalation is not None, "the operator is handed the record"
    successor_incident, successor = escalation
    assert successor_incident.event_id == f"support-after-{workflow.request_id}"
    assert successor_incident.effective_action is RecoveryAction.ESCALATE_OPERATOR
    assert [step.operation for step in successor.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.ESCALATE_SUPPORT,
    ], "no MARK_UNSCHEDULABLE, no QUARANTINE, no RESTART_NODE: nothing was learned"
    assert successor.status is WorkflowStatus.PENDING, successor.blocked_reasons
    assert any(
        VALIDATION_EVIDENCE_MISSING_STAGE in reason
        and "missing_or_stale_collectors: GPU_METRICS, HOST_TELEMETRY" in reason
        for reason in successor_incident.reasons
    ), successor_incident.reasons
    assert any(
        "REBOOT_NODE" in reason and "withheld" in reason and "per-step cap" in reason
        for reason in successor_incident.reasons
    ), successor_incident.reasons
