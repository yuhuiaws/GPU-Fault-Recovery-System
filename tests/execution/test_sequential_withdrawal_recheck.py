"""A sequential workflow re-reads the job withdrawal before every step.

Control-plane review 2026-09-08, D-2 (F-N1 §7). The DAG loop re-applies
``_apply_workload_withdrawal`` on every pass; the sequential loop applied it
once, before its first step. A user stop that landed while step *i* was
running still let steps *i+1..n* -- the job restart included -- be handed to
their adapters in the same tick: the restart command was minted and claimed by
the data plane, and the loop then wrote the workflow SUPERSEDED with nobody
left to collect that restart.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepSpec,
)
from tests._builders import active_workflow_executor, build_store, execute_workflow
from tests.execution._support import workflow_state

VALIDATE = WorkflowOperation.VALIDATE_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD


class _StoppingAdapter:
    """Succeeds every step; the job owner stops the job while step 0 runs."""

    def __init__(self, store) -> None:
        self.store = store
        self.calls: list[str] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append(context.idempotency_key)
        if context.step_index == 0:
            self.store.amend_workflow(
                context.workflow.request_id,
                {
                    "workload_withdrawn_at": datetime.now(timezone.utc),
                    "workload_withdrawn_reason": "user stop",
                },
            )
        return WorkflowStepOutcome.succeeded()


def test_a_stop_landing_mid_run_keeps_the_later_steps_from_starting() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [VALIDATE, RESTORE, RESTART_JOB])
    adapter = _StoppingAdapter(store)
    executor = active_workflow_executor(
        store, [adapter], {VALIDATE, RESTORE, RESTART_JOB}
    )

    result = execute_workflow(executor, workflow.request_id)

    assert adapter.calls == [f"{workflow.request_id}/0/VALIDATE_GPU"], adapter.calls
    assert result.status is WorkflowStatus.SUPERSEDED
    saved = store.get_workflow(workflow.request_id)
    assert saved.status is WorkflowStatus.SUPERSEDED
    assert sorted(saved.superseded_step_indexes) == [1, 2]
    assert saved.completed_step_indexes == [0]
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED
