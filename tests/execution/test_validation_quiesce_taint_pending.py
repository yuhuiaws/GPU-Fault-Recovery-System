"""A GPU validation starved by an orphaned quiesce taint names the taint.

Live (gpu-eks-cluster, XID 46): the reset chain's RESTORE_GPU_SERVICES
compensation was refused by the executor, its ``gpu-fault.io/quiesce``
NoExecute taint stayed on ip-10-0-41-164 and evicted the exporter, and the
reboot successor's VALIDATE_GPU "stayed non-terminal for 602s" with
``missing_recent_metrics`` as its only word. The validation adapter has no
cluster client, but the step records do say who tainted what: the pending
details now carry ``quiesce_taint_present`` with the holding workflow(s) and
a reason that names the operator exit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.adapters import GpuValidationAdapter, quarantine_taint_value
from gpu_fault.adapters.gpu_validation import quiesce_taint_pending
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
NODE = "ip-10-0-41-164"
RESET_INCIDENT = "inc-xid46"
RESET_WORKFLOW = "wf-reset-3fce97ba"


class NoMetrics:
    def __init__(self, store) -> None:
        self.store = store

    def latest(self, cluster_id, node_id):
        return []

    def findings(self, cluster_id, node_id):
        return []


def _reset_chain(store, *, restored: bool) -> None:
    executions = [
        workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
        workflow_step_execution(
            1, OP.QUIESCE_GPU_SERVICES, details={"quiesce_taint_nodes": [NODE]}
        ),
        workflow_step_execution(
            2, OP.RESET_GPU, WorkflowStepStatus.FAILED, error="In use by another client"
        ),
    ]
    if restored:
        executions.append(
            workflow_step_execution(
                3,
                OP.RESTORE_GPU_SERVICES,
                details={"quiesce_taint_removed_nodes": [NODE]},
            )
        )
    store.save_incident(
        fault_incident(
            RESET_INCIDENT,
            "event-xid46",
            state=IncidentState.QUARANTINED,
            workflow_request_id=RESET_WORKFLOW,
            node_ids=[NODE],
            fencing_token=1,
        )
    )
    store.save_workflow(
        workflow_request(
            RESET_WORKFLOW,
            RESET_INCIDENT,
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            official_steps=[
                workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[NODE]),
                workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[NODE]),
                workflow_step(OP.RESET_GPU, node_ids=[NODE]),
                workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=[NODE]),
            ],
            step_executions=executions,
        )
    )


def _successor_context(store) -> WorkflowStepContext:
    step = workflow_step(
        OP.VALIDATE_GPU, "gpu-fault-validation-adapter", node_ids=[NODE]
    )
    workflow = workflow_request(
        f"workflow-reboot-after-{RESET_WORKFLOW}",
        f"inc-reboot-after-{RESET_WORKFLOW}",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(OP.RESTART_NODE, node_ids=[NODE]), step],
        completed_step_indexes=[0],
        step_executions=[workflow_step_execution(0, OP.RESTART_NODE)],
    )
    incident = fault_incident(
        f"inc-reboot-after-{RESET_WORKFLOW}",
        f"reboot-after-{RESET_WORKFLOW}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
        node_ids=[NODE],
        fencing_token=1,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="reboot-after/validate",
    )


def test_the_successors_validation_names_the_predecessors_orphaned_taint() -> None:
    store = build_store()
    _reset_chain(store, restored=False)
    context = _successor_context(store)

    outcome = GpuValidationAdapter(NoMetrics(store), store=store).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    pending = outcome.details["node_pending"][NODE]
    assert pending["quiesce_taint_present"] is True
    assert pending["quiesce_taint_holders"] == {
        RESET_WORKFLOW: quarantine_taint_value(RESET_INCIDENT)
    }, "the workflow that tainted the node and the value it wrote"
    assert "gpu-fault quiesce taint" in pending["reason"]
    assert "submit-remediation --disposition restore" in pending["reason"]
    assert "missing_or_stale_collectors" in pending, (
        "the wait itself is unchanged; only its reason is named"
    )


def test_a_restored_predecessor_leaves_no_taint_to_name() -> None:
    store = build_store()
    _reset_chain(store, restored=True)
    context = _successor_context(store)

    outcome = GpuValidationAdapter(NoMetrics(store), store=store).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    pending = outcome.details["node_pending"][NODE]
    assert "quiesce_taint_present" not in pending
    assert "reason" not in pending


def test_the_lineage_is_followed_through_queued_predecessors_too() -> None:
    store = build_store()
    _reset_chain(store, restored=False)
    context = _successor_context(store)
    queued = context.workflow.model_copy(
        update={"predecessor_workflow_id": RESET_WORKFLOW}
    )
    detached = context.incident.model_copy(update={"event_id": "event-unrelated"})
    store.save_incident(detached)
    store.save_workflow(queued)
    context = WorkflowStepContext(
        workflow=queued,
        incident=detached,
        step=context.step,
        step_index=context.step_index,
        request=context.request,
        idempotency_key=context.idempotency_key,
    )

    pending = quiesce_taint_pending(store, context, NODE)

    assert pending["quiesce_taint_holders"] == {
        RESET_WORKFLOW: quarantine_taint_value(RESET_INCIDENT)
    }
    assert quiesce_taint_pending(store, context, "ip-10-0-41-9") == {}, (
        "another node of the same lineage carries nothing"
    )


def test_an_unrelated_workflow_reports_no_taint() -> None:
    store = build_store()
    _reset_chain(store, restored=False)
    context = _successor_context(store)
    stranger = context.incident.model_copy(update={"event_id": "event-unrelated"})
    store.save_incident(stranger)
    context = WorkflowStepContext(
        workflow=context.workflow,
        incident=stranger,
        step=context.step,
        step_index=context.step_index,
        request=context.request,
        idempotency_key=context.idempotency_key,
    )

    assert quiesce_taint_pending(store, context, NODE) == {}


def _unused() -> datetime:
    return datetime.now(timezone.utc)
