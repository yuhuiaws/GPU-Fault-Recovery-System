"""The application wires ``IncidentClosureService.on_terminal`` into the
executor's terminal funnel, so a validated restore that lands through the real
``execute`` path closes the ESCALATED incident waiting on the same node.

Why the hook layer and not the escalation service: ``HardwareEscalationService``
only ever sees FAILED workflows (the dispatcher's failure handler); the restore
that frees a node ends SUCCEEDED, and the executor's ``_save_terminal`` is the
one funnel every terminal write -- ``execute``, ``terminalize_claimed``, the
step-boundary preemption -- lands through (F-C9 / ARCH-B3).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.execution.models import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.execution._support import FakeAdapter
from tests.orchestration._incident_closure_support import _escalated_reset

VALIDATE = WorkflowOperation.VALIDATE_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _context(store, adapter) -> ApplicationContext:
    return ApplicationContext(
        store=store,
        production_executor_config=ProductionExecutorConfig(
            enabled=True,
            executor_id="executor-a",
            allowed_operations=frozenset({VALIDATE, RESTORE}),
        ),
        production_adapters=[adapter],
        execution_token="x" * 32,
    )


def _pending_restore(store, incident_id: str = "inc-support"):
    workflow = workflow_request(
        f"wf-restore-{incident_id}",
        incident_id,
        status=WorkflowStatus.PENDING,
        runtime_profile_version="active-v1",
        official_action="RESTORE_SCHEDULING",
        official_steps=[workflow_step(VALIDATE), workflow_step(RESTORE)],
        created_at=NOW - timedelta(minutes=1),
        updated_at=NOW - timedelta(minutes=1),
    )
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        workflow_request_id=workflow.request_id,
        created_at=NOW - timedelta(minutes=1),
        updated_at=NOW - timedelta(minutes=1),
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def test_the_context_registers_the_closure_hook_on_its_executor() -> None:
    context = _context(build_store(), FakeAdapter({}))

    assert context.incident_closure.on_terminal in context.workflow_executor.on_terminal


def test_a_restore_executed_by_the_context_closes_the_escalated_sibling() -> None:
    store = build_store()
    reset, _ = _escalated_reset(store)
    _, restore = _pending_restore(store)
    context = _context(
        store,
        FakeAdapter(
            {
                VALIDATE: WorkflowStepOutcome.succeeded(),
                RESTORE: WorkflowStepOutcome.succeeded(),
            }
        ),
    )

    result = context.workflow_executor.execute(
        restore.request_id, WorkflowExecutionRequest(expected_fencing_token=3)
    )

    assert result.status is WorkflowStatus.SUCCEEDED, result
    closed = store.get_incident(reset.incident_id)
    assert closed.state is IncidentState.RECOVERED
    assert closed.reasons[-1] == "node restored via incident inc-support"
    assert context.incident_closure.auto_closed_by_restore_total == 1
    assert store.get_incident("inc-support").state is IncidentState.RECOVERED


def test_a_failed_restore_leaves_the_escalated_sibling_alone() -> None:
    store = build_store()
    reset, _ = _escalated_reset(store)
    _, restore = _pending_restore(store)
    context = _context(
        store,
        FakeAdapter(
            {
                VALIDATE: WorkflowStepOutcome.failed("validation refused"),
                RESTORE: WorkflowStepOutcome.succeeded(),
            }
        ),
    )

    result = context.workflow_executor.execute(
        restore.request_id, WorkflowExecutionRequest(expected_fencing_token=3)
    )

    assert result.status is WorkflowStatus.FAILED, result
    assert store.get_incident(reset.incident_id).state is IncidentState.ESCALATED
    assert context.incident_closure.auto_closed_by_restore_total == 0
