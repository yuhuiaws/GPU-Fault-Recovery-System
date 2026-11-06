"""The support hand-off mail says what the remediation did to the node.

The ESCALATE_SUPPORT step's HARDWARE_ESCALATION mail
(``adapters/support_escalation.py``) said "hardware taken out of service" and
"The faulty nodes stay unschedulable and quarantined" for every hand-off -- also
for the node-health ESCALATE_OPERATOR WARNING plan (FREEZE_EVIDENCE ->
ESCALATE_SUPPORT: ECC disabled, an invalid InfoROM, a downtrain a reset did not
retrain, a pending memory repair), which cordons nothing, and its step recorded
``hardware_disposition=OFFLINE_QUARANTINED``. The subject, the text, the case
draft and the disposition now follow the hand-off's records: the support
workflow and the workflows it escalated from. MARK_UNSCHEDULABLE and QUARANTINE
are one Kubernetes patch (cordon + ``gpu-fault.io/quarantined`` taint), so
either one that completed holds its nodes; a completed RESTORE_SCHEDULING
releases them. The chain terminator's copy of the mail reads the same records.
"""

from __future__ import annotations

from gpu_fault.adapters.support_escalation import SupportEscalationAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.notifications.common import HARDWARE_ESCALATION_TEMPLATE_VERSION
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import finding, memory_store
from tests.orchestration._cov95_runtime_builder import builder

OP = WorkflowOperation
HELD_LINE = "The faulty nodes stay unschedulable and quarantined"
NOT_HELD_LINE = "not cordoned or quarantined by this remediation"
SOURCE = "workflow-reset-unit"


def executed(workflow: WorkflowRequest, upto: int) -> WorkflowRequest:
    """``workflow`` with its first ``upto`` steps SUCCEEDED, as the executor records them."""

    steps = workflow.official_steps[:upto]
    return workflow.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "completed_step_indexes": list(range(upto)),
            "completed_operations": [step.operation for step in steps],
            "step_executions": [
                workflow_step_execution(index, step.operation, phase="official")
                for index, step in enumerate(steps)
            ],
        }
    )


def hand_off(store, workflow: WorkflowRequest, incident):
    """Run the workflow's ESCALATE_SUPPORT step; return its outcome and mail."""

    index = next(
        position
        for position, step in enumerate(workflow.official_steps)
        if step.operation is OP.ESCALATE_SUPPORT
    )
    step = workflow.official_steps[index]
    outcome = SupportEscalationAdapter(store).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=index,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key=f"{workflow.request_id}/{index}/ESCALATE_SUPPORT",
        )
    )
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    [notification] = store.list_notifications()
    return outcome, notification


def operator_hand_off(severity: str, metric: str, action: str | None):
    """The node-health ESCALATE_OPERATOR plan, compiled by the real family."""

    store = memory_store()
    service, _absorbed = health_service(store)
    incident, workflow = service.ingest(
        finding(
            f"operator-{metric}",
            severity=severity,
            reason=f"unit {metric}",
            recommended_action=RecoveryAction.ESCALATE_OPERATOR,
            official_action=action,
            metric_name=metric,
            gpu_uuids=["GPU-a"],
            node_id="node-a",
        )
    )
    return store, incident, workflow


def test_a_hand_off_that_isolated_nothing_says_the_node_stays_in_service():
    store, incident, workflow = operator_hand_off(
        "warning", "ecc_mode_current", "ENABLE_ECC_AND_REBOOT"
    )
    assert [step.operation for step in workflow.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.ESCALATE_SUPPORT,
    ]

    outcome, mail = hand_off(store, executed(workflow, 1), incident)

    assert HELD_LINE not in mail.body_text, (
        "nothing cordoned or quarantined the node: the mail must not say so"
    )
    assert "hardware taken out of service" not in mail.body_text, mail.body_text
    assert NOT_HELD_LINE in mail.body_text, mail.body_text
    assert mail.subject.startswith("[GPU FAULT][support escalation] "), mail.subject
    assert "Disposition: NOT_ISOLATED" in mail.support_case_draft
    assert outcome.details["hardware_disposition"] == "NOT_ISOLATED", outcome
    assert outcome.details["held_nodes"] == []
    assert f"Template: {HARDWARE_ESCALATION_TEMPLATE_VERSION}" in mail.body_text


def test_a_critical_operator_hand_off_holds_the_node_it_cordoned():
    """MARK_UNSCHEDULABLE is the same patch as QUARANTINE: the node is cordoned
    and carries the incident's quarantine taint."""

    store, incident, workflow = operator_hand_off(
        "critical", "gpu_recovery_action", "RECOVER_IMEX_DOMAIN"
    )
    assert [step.operation for step in workflow.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.MARK_UNSCHEDULABLE,
        OP.ESCALATE_SUPPORT,
    ]

    outcome, mail = hand_off(store, executed(workflow, 2), incident)

    assert HELD_LINE in mail.body_text, mail.body_text
    assert mail.body_text.startswith(
        "GPU node automatic recovery failed: hardware taken out of service"
    ), mail.body_text
    assert mail.subject.startswith("[GPU FAULT][hardware out of service] "), (
        mail.subject
    )
    assert "Disposition: HARDWARE_OFFLINE" in mail.support_case_draft
    assert outcome.details["hardware_disposition"] == "OFFLINE_QUARANTINED"
    assert outcome.details["held_nodes"] == ["node-a"]


def _source_chain(store, *, restored: bool) -> None:
    """A reset chain that cordoned node-a and failed at RESET_GPU, or that
    restored its scheduling before the failure that handed it over."""

    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE),
        workflow_step(OP.RESET_GPU),
        workflow_step(OP.RESTORE_SCHEDULING),
        workflow_step(OP.RESTART_WORKLOAD),
    ]
    executions = [workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official")]
    if restored:
        executions += [
            workflow_step_execution(1, OP.RESET_GPU, phase="official"),
            workflow_step_execution(2, OP.RESTORE_SCHEDULING, phase="official"),
            workflow_step_execution(
                3, OP.RESTART_WORKLOAD, WorkflowStepStatus.FAILED, phase="official"
            ),
        ]
    else:
        executions.append(
            workflow_step_execution(
                1, OP.RESET_GPU, WorkflowStepStatus.FAILED, phase="official"
            )
        )
    store.save_incident(
        fault_incident(
            "incident-reset-unit",
            "event-reset-unit",
            state=IncidentState.QUARANTINED,
            workflow_request_id=SOURCE,
        )
    )
    store.save_workflow(
        workflow_request(
            SOURCE,
            "incident-reset-unit",
            status=WorkflowStatus.FAILED,
            official_steps=steps,
            completed_step_indexes=[
                item.step_index
                for item in executions
                if item.status is WorkflowStepStatus.SUCCEEDED
            ],
            step_executions=executions,
        )
    )


def _evidence_only_successor():
    """``support-after-<source>`` planning FREEZE_EVIDENCE -> ESCALATE_SUPPORT
    only (a refused node action, an unobserved validation): it keeps whatever
    the predecessor left on the node."""

    workflow = workflow_request(
        f"workflow-support-after-{SOURCE}",
        f"inc-support-after-{SOURCE}",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(OP.FREEZE_EVIDENCE),
            workflow_step(OP.ESCALATE_SUPPORT, "gpu-fault-support-escalation"),
        ],
    )
    incident = fault_incident(
        f"inc-support-after-{SOURCE}",
        f"support-after-{SOURCE}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
    )
    return executed(workflow, 1), incident


def test_an_evidence_only_successor_reports_the_hold_its_predecessor_left():
    store = build_store()
    _source_chain(store, restored=False)
    workflow, incident = _evidence_only_successor()

    outcome, mail = hand_off(store, workflow, incident)

    assert HELD_LINE in mail.body_text, (
        "the predecessor's cordon is still on the node: the lineage says so"
    )
    assert outcome.details["held_nodes"] == ["node-a"]


def test_a_hold_the_chain_released_is_not_reported_as_quarantined():
    store = build_store()
    _source_chain(store, restored=True)
    workflow, incident = _evidence_only_successor()

    outcome, mail = hand_off(store, workflow, incident)

    assert HELD_LINE not in mail.body_text, mail.body_text
    assert NOT_HELD_LINE in mail.body_text
    assert outcome.details["hardware_disposition"] == "NOT_ISOLATED"


def test_a_partial_isolation_names_each_node_by_its_state():
    """A two-node cordon refused after the first node keeps that node isolated
    (``isolated_nodes``) and never touched the second."""

    store = build_store()
    nodes = ["node-a", "node-b"]
    workflow = workflow_request(
        "workflow-two-nodes",
        "incident-two-nodes",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(OP.FREEZE_EVIDENCE, node_ids=nodes),
            workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=nodes),
            workflow_step(OP.ESCALATE_SUPPORT, node_ids=nodes),
        ],
        step_executions=[
            workflow_step_execution(0, OP.FREEZE_EVIDENCE, phase="official"),
            workflow_step_execution(
                1,
                OP.MARK_UNSCHEDULABLE,
                WorkflowStepStatus.FAILED,
                phase="official",
                details={"safety_rejection": True, "isolated_nodes": ["node-a"]},
            ),
        ],
        completed_step_indexes=[0],
    )
    incident = fault_incident(
        "incident-two-nodes",
        "event-two-nodes",
        node_ids=nodes,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
    )

    outcome, mail = hand_off(store, workflow, incident)

    assert "- Unschedulable and quarantined, scheduling not restored: node-a" in (
        mail.body_text
    ), mail.body_text
    assert "- Not cordoned or quarantined by this remediation: node-b" in (
        mail.body_text
    ), mail.body_text
    assert mail.subject.startswith("[GPU FAULT][hardware out of service] "), (
        mail.subject
    )
    assert outcome.details["hardware_disposition"] == "PARTIALLY_QUARANTINED"
    assert outcome.details["held_nodes"] == ["node-a"]
    assert "Disposition: HARDWARE_OFFLINE (node-a); NOT_ISOLATED (node-b)" in (
        mail.support_case_draft
    )


def test_the_chain_terminator_mail_reads_the_same_records():
    """A support hand-off whose own workflow failed is handed over by the
    terminator with the same mail; a cordon that was refused on the predecessor
    and on the hand-off itself holds nothing."""

    store = build_store()
    refused = {"safety_rejection": True, "isolated_nodes": []}
    store.save_incident(
        fault_incident(
            "incident-refused-unit",
            "event-refused-unit",
            state=IncidentState.ESCALATED,
            workflow_request_id=SOURCE,
        )
    )
    store.save_workflow(
        workflow_request(
            SOURCE,
            "incident-refused-unit",
            status=WorkflowStatus.FAILED,
            official_steps=[workflow_step(OP.MARK_UNSCHEDULABLE)],
            step_executions=[
                workflow_step_execution(
                    0,
                    OP.MARK_UNSCHEDULABLE,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details=refused,
                )
            ],
        )
    )
    support = workflow_request(
        f"workflow-support-after-{SOURCE}",
        f"inc-support-after-{SOURCE}",
        status=WorkflowStatus.FAILED,
        official_steps=[
            workflow_step(OP.FREEZE_EVIDENCE),
            workflow_step(OP.ESCALATE_SUPPORT),
        ],
        step_executions=[
            workflow_step_execution(
                0, OP.FREEZE_EVIDENCE, WorkflowStepStatus.FAILED, phase="official"
            )
        ],
    )
    store.save_incident(
        fault_incident(
            support.incident_id,
            f"support-after-{SOURCE}",
            state=IncidentState.ESCALATED,
            workflow_request_id=support.request_id,
        )
    )
    store.save_workflow(support)

    assert HardwareEscalationService(store, builder()).escalate(support) is None
    [mail] = store.list_notifications()

    assert HELD_LINE not in mail.body_text, mail.body_text
    assert NOT_HELD_LINE in mail.body_text, mail.body_text
    assert mail.subject.startswith("[GPU FAULT][support escalation] "), mail.subject
