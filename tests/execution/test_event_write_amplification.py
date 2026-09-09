"""A WAITING workflow does not rewrite its row twice, and its audit trail does
not fill with events that say nothing new.

Control-plane review 2026-09-08, D-10. Every redispatch of a WAITING step --
one per 5 s tick -- appended a CLAIM event and a STEP_ATTEMPT event and wrote
the whole row twice, so a ten-minute wait produced ~240 events and ~240 row
writes (each with TOAST and six partial-index updates), and the 500-event cap
(``WORKFLOW_EVENTS_LIMIT``) evicted the early STEP_ATTEMPT and PLAN_REWRITE
history in about twenty minutes. Every claim also re-reserved the job restart
budget.

Now: a CLAIM event only when the claim opened a new execution epoch; a
STEP_ATTEMPT only when the attempt says something the previous one did not
(status, reason, error, operation id, or any allow-listed detail other than the
ever-growing ``step_waiting_seconds``); and a RESTART_WORKLOAD step that already
has a record is not re-reserved.
"""

from __future__ import annotations

from gpu_fault.execution import step_bounds
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    execute_workflow,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
RESET = WorkflowOperation.RESET_GPU
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD


def _events(workflow, kind):
    return [event for event in workflow.events if event.kind is kind]


class _CountingStore:
    def __init__(self, store) -> None:
        self._store = store
        self.reservations = 0

    def reserve_job_restart(self, *args, **kwargs):
        self.reservations += 1
        return self._store.reserve_job_restart(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


def test_a_redispatched_waiting_workflow_records_one_claim_and_one_attempt() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    adapter = FakeAdapter(
        {
            FREEZE: WorkflowStepOutcome.waiting(
                operation_id="remote/cmd-0",
                details={"remote_status": "LEASED", "remote_command_id": "cmd-0"},
            )
        }
    )
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    for _ in range(4):
        assert execute_workflow(executor, workflow.request_id).status is (
            WorkflowStatus.RUNNING
        )

    saved = store.get_workflow(workflow.request_id)
    assert len(_events(saved, WorkflowEventKind.CLAIM)) == 1, saved.events
    attempts = _events(saved, WorkflowEventKind.STEP_ATTEMPT)
    assert len(attempts) == 1, [event.model_dump() for event in attempts]
    assert attempts[0].code == WorkflowEventCode.STEP_WAITING
    # The record itself still says how long the step has waited.
    assert saved.step_executions[-1].details["step_waiting_seconds"] >= 0


def test_a_new_epoch_records_a_new_claim() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    adapter = FakeAdapter({FREEZE: WorkflowStepOutcome.waiting(operation_id="op-0")})
    first = active_workflow_executor(
        store, [adapter], {FREEZE}, executor_id="executor-a"
    )
    execute_workflow(first, workflow.request_id)
    # The first holder's lease lapses; another executor takes the row.
    current = store.get_workflow(workflow.request_id)
    store.save_workflow(
        current.model_copy(update={"execution_lease_expires_at": current.created_at}),
        expected=current,
    )
    second = active_workflow_executor(
        store, [adapter], {FREEZE}, executor_id="executor-b"
    )
    execute_workflow(second, workflow.request_id)

    claims = _events(store.get_workflow(workflow.request_id), WorkflowEventKind.CLAIM)
    assert [event.actor for event in claims] == ["executor-a", "executor-b"]


def test_an_attempt_that_says_something_new_is_recorded() -> None:
    workflow = workflow_request(
        "wf", "inc", official_steps=[workflow_step(RESET, node_ids=["node-b"])]
    )
    step = workflow.official_steps[0]
    waits = (
        {"reason": "NODE_BUSY", "step_waiting_seconds": 0},
        {"reason": "NODE_BUSY", "step_waiting_seconds": 30},  # only the clock moved
        {"reason": "NODE_BUSY", "step_waiting_seconds": 60, "step_waiting_slow": True},
        {"reason": "AGENT_LAG", "step_waiting_seconds": 90},
    )
    for details in waits:
        workflow = step_bounds.record_attempt(
            workflow,
            step,
            0,
            WorkflowStepOutcome.waiting(operation_id="cmd-1", details=dict(details)),
        )
    workflow = step_bounds.record_attempt(
        workflow, step, 0, WorkflowStepOutcome.succeeded(operation_id="cmd-1")
    )

    attempts = _events(workflow, WorkflowEventKind.STEP_ATTEMPT)
    assert [event.details.get("reason") for event in attempts] == [
        "NODE_BUSY",
        "NODE_BUSY",
        "AGENT_LAG",
        None,
    ]
    assert [event.details["attempt"] for event in attempts] == [1, 2, 3, 4]
    assert attempts[1].details["step_waiting_slow"] is True
    assert attempts[-1].code == WorkflowEventCode.STEP_SUCCEEDED
    assert len(workflow.step_executions) == 1


def test_a_waiting_restart_is_reserved_once_across_ticks() -> None:
    store = build_store()
    counting = _CountingStore(store)
    _, workflow = workflow_state(counting, [RESTART_JOB])
    adapter = FakeAdapter(
        {RESTART_JOB: WorkflowStepOutcome.waiting(operation_id="restart-1")}
    )
    executor = active_workflow_executor(counting, [adapter], {RESTART_JOB})

    for _ in range(3):
        execute_workflow(executor, workflow.request_id)

    assert counting.reservations == 1, counting.reservations
    state = store.get_restart_budget("cluster-a", "training-job")
    assert state.restart_count == 1
