"""The hand-off email names the job a failed remediation left suspended.

Batch 139, user decision G7: the operator's validated restore does not restart
a training job the remediation stopped; the job's owner does. The email that
ends the automated remediation -- the support hand-off's HARDWARE_ESCALATION
mail (``adapters/support_escalation.py``) -- therefore names every workload a
STOP of the escalation chain suspended and nothing restarted, with the same
line the ``restore`` plan prints (``suspended_workloads``).
"""

from __future__ import annotations

from gpu_fault.adapters.support_escalation import SupportEscalationAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY
from tests._builders import (
    build_store,
    contained_stop_receipt,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
WORKLOAD = "training/pytorchjob/train"
SOURCE = "workflow-reset-b139"
LINE = (
    "workload training/pytorchjob/train stays suspended (stopped by workflow "
    f"{SOURCE}): the restore does not restart it; its owner must restart it"
)


def _source_chain(store, *, restarted: bool = False) -> None:
    """A reset chain that stopped the job, then failed at RESET_GPU."""

    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(OP.STOP_WORKLOADS, node_ids=["node-a"], workload_ids=[WORKLOAD]),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"]),
        workflow_step(
            OP.RESTART_WORKLOAD, node_ids=["node-a"], workload_ids=[WORKLOAD]
        ),
    ]
    executions = [
        workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
        workflow_step_execution(
            1,
            OP.STOP_WORKLOADS,
            phase="official",
            details={
                STOP_OWNERSHIP_RECEIPT_KEY: contained_stop_receipt(
                    workflow_id=SOURCE,
                    workloads=[
                        {
                            "workload_id": WORKLOAD,
                            "namespace": "training",
                            "kind": "pytorchjob",
                            "name": "train",
                            "uid": "train-uid",
                            "attempt_id": "attempt-1",
                            "owners": [],
                        }
                    ],
                )
            },
        ),
        workflow_step_execution(
            2, OP.RESET_GPU, WorkflowStepStatus.FAILED, phase="official"
        ),
    ]
    if restarted:
        executions.append(
            workflow_step_execution(3, OP.RESTART_WORKLOAD, phase="official")
        )
    store.save_incident(
        fault_incident(
            "incident-reset-b139",
            "event-reset-b139",
            node_ids=["node-a"],
            state=IncidentState.QUARANTINED,
            workflow_request_id=SOURCE,
        )
    )
    store.save_workflow(
        workflow_request(
            SOURCE,
            "incident-reset-b139",
            status=WorkflowStatus.FAILED,
            official_steps=steps,
            completed_step_indexes=[0, 1],
            step_executions=executions,
        )
    )


def _hand_off(store):
    step = workflow_step(
        OP.ESCALATE_SUPPORT,
        "gpu-fault-support-escalation",
        node_ids=["node-a"],
        workload_ids=[WORKLOAD],
    )
    incident = fault_incident(
        f"inc-support-after-{SOURCE}",
        f"support-after-{SOURCE}",
        node_ids=["node-a"],
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=f"workflow-support-after-{SOURCE}",
    )
    workflow = workflow_request(
        f"workflow-support-after-{SOURCE}",
        incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_steps=[step],
    )
    outcome = SupportEscalationAdapter(store).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key=f"{workflow.request_id}/0/ESCALATE_SUPPORT",
        )
    )
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    [notification] = store.list_notifications()
    return notification


def test_the_hand_off_email_names_the_job_the_failed_chain_left_suspended() -> None:
    store = build_store()
    _source_chain(store)

    notification = _hand_off(store)

    assert LINE in notification.body_text
    assert LINE in notification.support_case_draft
    assert "Template: hardware-escalation-zh-v3" in notification.body_text


def test_the_hand_off_email_says_none_when_the_chain_restarted_the_job() -> None:
    store = build_store()
    _source_chain(store, restarted=True)

    notification = _hand_off(store)

    assert "stays suspended" not in notification.body_text
    assert "did not restart it" not in notification.body_text
    assert "their owner must):\n  - NONE" in notification.body_text
