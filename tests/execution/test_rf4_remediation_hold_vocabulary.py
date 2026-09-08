"""The two "node under repair" waits speak one vocabulary (RF-4, review item 4).

The dispatcher holds a PENDING job workflow whose nodes another workflow is
repairing (F-N1 §8); the restart adapter holds an ``after_incident`` restart
until the incident that owns its nodes is RECOVERED (F-G5). Their product
semantics differ on purpose and stay apart. What they share now: the reason
codes in ``details["reason"]`` (``NODE_UNDER_REMEDIATION`` while holding,
``NODE_REMEDIATION_TIMEOUT`` when giving up, ``INCIDENT_NOT_RECOVERABLE`` when
the wait can never end), one ``HOLD`` audit event per (workflow, reason,
remediation workflow) rather than one per poll, a ``PLAN_REWRITE`` event for
the dispatcher's give-up, and ``remediation_workflow_id`` in both so an
operator can follow either wait to the same object.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteIncidentOwnershipReport
from tests._builders import (
    active_workflow_executor,
    build_store,
    execute_workflow,
    fault_incident,
)
from tests.completion.test_restart_execution_premise import (
    BatchApi,
    UnusedApi,
    adapter,
    restart_context,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state
from tests.execution.test_node_busy_wait import (
    RESTART_JOB,
    _busy_node,
    _dispatcher,
    _job_workflow,
)

HOLDING = "NODE_UNDER_REMEDIATION"
TIMEOUT = "NODE_REMEDIATION_TIMEOUT"
NEVER = "INCIDENT_NOT_RECOVERABLE"
PREMISE = {"requires_incident_state": "RECOVERED", "incident_id": "inc-source"}


def _events(workflow: WorkflowRequest, kind: WorkflowEventKind):
    return [event for event in workflow.events if event.kind is kind]


# --- the dispatcher's hold -------------------------------------------------


def test_the_dispatcher_hold_records_one_hold_event_however_often_it_polls():
    store = build_store()
    _busy_node(store)
    _job_workflow(store, created_at=datetime.now(timezone.utc))
    dispatcher, _ = _dispatcher(store)

    for _ in range(3):
        dispatcher.run_once()

    held = store.get_workflow("wf-job")
    assert held.status is WorkflowStatus.PENDING
    holds = _events(held, WorkflowEventKind.HOLD)
    assert len(holds) == 1, [event.model_dump() for event in held.events]
    assert holds[0].details["reason"] == HOLDING
    assert holds[0].details["remediation_workflow_id"] == "wf-node"
    assert holds[0].details["node_ids"] == ["node-a"]
    assert holds[0].actor == "dispatcher"


def test_the_dispatcher_give_up_records_a_plan_rewrite_with_the_timeout_code():
    store = build_store()
    _busy_node(store)
    long_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    _job_workflow(store, created_at=long_ago, held_since=long_ago)
    dispatcher, _ = _dispatcher(store)

    dispatcher.run_once()

    rewritten = store.get_workflow("wf-job")
    assert rewritten.status is WorkflowStatus.PENDING, "the stop still has to run"
    rewrites = _events(rewritten, WorkflowEventKind.PLAN_REWRITE)
    assert len(rewrites) == 1, [event.model_dump() for event in rewritten.events]
    assert rewrites[0].details["reason"] == TIMEOUT
    assert rewrites[0].details["remediation_workflow_id"] == "wf-node"
    # The original plan was STOP / RESET / RESTART; only the STOP is kept.
    assert rewrites[0].details["kept_step_indexes"] == [0]
    assert rewrites[0].details["superseded_step_indexes"] == [1, 2]
    assert TIMEOUT in (rewritten.terminal_failure_reason or "")
    assert "node-a" in (rewritten.terminal_failure_reason or "")

    dispatcher.run_once()

    ended = store.get_workflow("wf-job")
    assert ended.status is WorkflowStatus.FAILED
    terminal = _events(ended, WorkflowEventKind.TERMINAL)
    assert len(terminal) == 1, [event.model_dump() for event in ended.events]
    assert TIMEOUT in (terminal[0].reason or "")


# --- the restart adapter's premise wait -------------------------------------


def test_the_premise_wait_carries_the_shared_code_and_the_remediation_id():
    store = build_store()
    store.save_incident(
        fault_incident(
            "inc-source",
            "event-source",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="wf-node-repair",
        )
    )
    batch = BatchApi()

    waiting = adapter(batch, store).execute(restart_context(PREMISE))

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["reason"] == HOLDING
    assert waiting.details["premise_reason"] == "INCIDENT_NOT_RECOVERED"
    assert waiting.details["remediation_workflow_id"] == "wf-node-repair"
    assert batch.created == {}


def test_the_storeless_premise_wait_reads_the_remediation_id_from_the_provider():
    class Provider:
        def incident_ownership(self, incident_id: str) -> RemoteIncidentOwnershipReport:
            return RemoteIncidentOwnershipReport(
                incident_id=incident_id,
                known=True,
                incident_state="SAFETY_PENDING",
                workflow_request_id="wf-remote-repair",
            )

    subject = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=BatchApi(),
        custom_api=UnusedApi(),
        store=None,
        ownership_provider=Provider(),
    )

    waiting = subject.execute(restart_context(PREMISE))

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["reason"] == HOLDING
    assert waiting.details["remediation_workflow_id"] == "wf-remote-repair"


def test_the_premise_failure_keeps_its_never_recoverable_code():
    store = build_store()
    store.save_incident(
        fault_incident(
            "inc-source",
            "event-source",
            state=IncidentState.ESCALATED,
            workflow_request_id="wf-node-repair",
        )
    )

    outcome = adapter(BatchApi(), store).execute(restart_context(PREMISE))

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == NEVER
    assert outcome.details["remediation_workflow_id"] == "wf-node-repair"


# --- the executor turns the adapter's hold into the same HOLD event ---------


def test_the_executor_records_one_hold_event_for_a_repeated_premise_wait():
    store = build_store()
    _, workflow = workflow_state(store, [RESTART_JOB])
    holding = FakeAdapter(
        {
            RESTART_JOB: WorkflowStepOutcome.waiting(
                operation_id="restart-hold",
                details={
                    "reason": HOLDING,
                    "premise_reason": "INCIDENT_NOT_RECOVERED",
                    "remediation_workflow_id": "wf-node-repair",
                },
            )
        }
    )
    executor = active_workflow_executor(store, [holding], [RESTART_JOB])

    first = execute_workflow(executor, workflow.request_id)
    second = execute_workflow(executor, workflow.request_id)

    assert first.waiting_step_index == 0
    assert second.waiting_step_index == 0
    saved = store.get_workflow(workflow.request_id)
    holds = _events(saved, WorkflowEventKind.HOLD)
    assert len(holds) == 1, [event.model_dump() for event in saved.events]
    assert holds[0].details["reason"] == HOLDING
    assert holds[0].details["remediation_workflow_id"] == "wf-node-repair"
    assert holds[0].step_index == 0
    assert holds[0].operation is RESTART_JOB
    assert holds[0].actor == executor.config.executor_id


def test_a_hold_on_a_different_remediation_is_a_new_event():
    store = build_store()
    _, workflow = workflow_state(store, [RESTART_JOB])

    def _holding(remediation_id: str) -> FakeAdapter:
        return FakeAdapter(
            {
                RESTART_JOB: WorkflowStepOutcome.waiting(
                    operation_id="restart-hold",
                    details={
                        "reason": HOLDING,
                        "remediation_workflow_id": remediation_id,
                    },
                )
            }
        )

    execute_workflow(
        active_workflow_executor(store, [_holding("wf-first")], [RESTART_JOB]),
        workflow.request_id,
    )
    execute_workflow(
        active_workflow_executor(store, [_holding("wf-second")], [RESTART_JOB]),
        workflow.request_id,
    )

    saved = store.get_workflow(workflow.request_id)
    holds = _events(saved, WorkflowEventKind.HOLD)
    assert [event.details["remediation_workflow_id"] for event in holds] == [
        "wf-first",
        "wf-second",
    ]
    # Sanity: the workflow is still the waiting row, not a terminal one.
    assert saved.status is WorkflowStatus.RUNNING
