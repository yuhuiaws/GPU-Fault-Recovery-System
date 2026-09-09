"""The rule A window opens when the dispatcher first sees the busy node.

Control-plane review 2026-09-08, D-11 (F-N1 §8). ``_eligible`` measured the
job workflow's wait from ``created_at``, but its ``not_before`` (aggregation
window) and open-predecessor filters run *before* the busy check and
``continue`` -- none of that time was a wait on the busy node. A job workflow
gated for longer than the window was therefore stopped and failed on its very
first look, and its owner saw "waited 240 s" for a wait of zero.

The window now opens at the first HOLD the dispatcher recorded for the row;
the first look always holds. ``dispatch_eligible_at`` would not do: a
predecessor gate leaves it equal to ``created_at``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution import test_node_busy_wait as node_busy
from tests.execution._support import (
    RESTART_PARAMETERS,
    FakeAdapter,
    WorkflowStepOutcome,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
WINDOW = 300


def _gated_job_workflow(store, *, created_at: datetime, **gate) -> None:
    incident = fault_incident(
        "inc-job",
        "event-job",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-job",
        node_ids=["node-a", "node-b"],
        job_id="train-1",
        attempt_id="train-1-a1",
        created_at=created_at,
        updated_at=created_at,
        fencing_token=3,
    )
    workflow = workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(
                STOP, node_ids=["node-a", "node-b"], workload_ids=["training/job/t"]
            ),
            workflow_step(RESET, node_ids=["node-a"]),
            workflow_step(
                RESTART_JOB,
                node_ids=["node-a", "node-b"],
                parameters=dict(RESTART_PARAMETERS),
            ),
        ],
        created_at=created_at,
        updated_at=created_at,
        **gate,
    )
    store.save_incident_and_workflow(incident, workflow)


def _dispatcher(store):
    adapter = FakeAdapter(
        {
            STOP: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.succeeded(),
            RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = active_workflow_executor(store, [adapter], {STOP, RESET, RESTART_JOB})
    return (
        WorkflowDispatcher(
            store,
            executor,
            WorkflowDispatcherConfig(
                enabled=True,
                batch_size=10,
                max_workers=1,
                node_busy_wait_seconds=WINDOW,
            ),
        ),
        adapter,
    )


def _rewind_holds(store, request_id: str, by: timedelta) -> None:
    current = store.get_workflow(request_id)
    store.amend_workflow(
        request_id,
        {
            "events": [
                event.model_copy(update={"at": event.at - by})
                if event.kind is WorkflowEventKind.HOLD
                else event
                for event in current.events
            ]
        },
    )


def test_a_row_released_from_its_aggregation_window_starts_the_wait_at_zero():
    store = build_store()
    node_busy._busy_node(store)
    now = datetime.now(timezone.utc)
    # Created long before the window, but only dispatchable since a moment ago.
    _gated_job_workflow(
        store,
        created_at=now - timedelta(minutes=20),
        not_before=now - timedelta(seconds=1),
    )
    dispatcher, adapter = _dispatcher(store)

    report = dispatcher.run_once()

    assert report.filtered.get("node_busy") == 1, report.filtered
    assert report.filtered.get("node_busy_timeout") is None
    assert adapter.calls == []
    saved = store.get_workflow("wf-job")
    assert saved.status is WorkflowStatus.PENDING
    assert saved.terminal_failure_reason is None
    holds = [event for event in saved.events if event.kind is WorkflowEventKind.HOLD]
    assert len(holds) == 1
    held_since = datetime.fromisoformat(holds[0].details["held_since"])
    assert now - timedelta(seconds=5) <= held_since <= now + timedelta(seconds=60)


def test_a_row_released_from_behind_a_predecessor_starts_the_wait_at_zero():
    store = build_store()
    node_busy._busy_node(store)
    now = datetime.now(timezone.utc)
    predecessor = workflow_request(
        "wf-before",
        "inc-before",
        status=WorkflowStatus.SUCCEEDED,
        official_steps=[workflow_step(RESET)],
    )
    store.save_incident_and_workflow(
        fault_incident(
            "inc-before",
            "event-before",
            state=IncidentState.RECOVERED,
            workflow_request_id="wf-before",
        ),
        predecessor,
    )
    _gated_job_workflow(
        store,
        created_at=now - timedelta(minutes=20),
        predecessor_workflow_id="wf-before",
    )
    dispatcher, adapter = _dispatcher(store)

    report = dispatcher.run_once()

    assert report.filtered.get("node_busy") == 1, report.filtered
    assert adapter.calls == []
    assert store.get_workflow("wf-job").status is WorkflowStatus.PENDING


def test_the_window_is_measured_from_the_first_hold():
    store = build_store()
    node_busy._busy_node(store)
    _gated_job_workflow(store, created_at=datetime.now(timezone.utc))
    dispatcher, adapter = _dispatcher(store)

    assert dispatcher.run_once().filtered.get("node_busy") == 1
    _rewind_holds(store, "wf-job", timedelta(seconds=WINDOW + 60))
    gave_up = dispatcher.run_once()
    dispatcher.run_once()

    assert gave_up.filtered.get("node_busy_timeout") == 1, gave_up.filtered
    saved = store.get_workflow("wf-job")
    assert saved.status is WorkflowStatus.FAILED
    assert adapter.calls == ["wf-job/0/STOP_WORKLOADS"]
    assert dispatcher.node_busy_timeouts_total == 1
