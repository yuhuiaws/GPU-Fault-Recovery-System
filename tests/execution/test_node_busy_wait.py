"""A job workflow whose nodes are still under another remediation waits a
bounded time, then fails by stopping the job rather than by racing the
node workflow.

F-N1 §8 (docs/review/F-N1-设计与实施计划.md). Serialization behind an
in-flight node-exclusive workflow existed, but with no bound (the 2026-09-04
five-hour starvation) and no signal to the data plane. Now the wait is
capped; at the cap the job workflow is rewritten to a single STOP_WORKLOADS
(the notification the job owner can see) and ends FAILED / ESCALATED.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import (
    RESTART_PARAMETERS,
    FakeAdapter,
    WorkflowStepOutcome,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD


def _busy_node(store) -> None:
    incident = fault_incident(
        "inc-node",
        "event-node",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-node",
        node_ids=["node-a"],
    )
    workflow = workflow_request(
        "wf-node",
        "inc-node",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(REBOOT, node_ids=["node-a"])],
        execution_owner_id="executor-elsewhere",
        execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    store.save_incident_and_workflow(incident, workflow)


def _job_workflow(store, *, created_at: datetime) -> None:
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
        fencing_token=3,  # the builders' workflow default; the executor checks they match
    )
    workflow = workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(
                STOP,
                node_ids=["node-a", "node-b"],
                workload_ids=["training/job/train-1"],
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
    dispatcher = WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, node_busy_wait_seconds=300
        ),
    )
    return dispatcher, adapter


def test_a_job_workflow_waits_while_one_of_its_nodes_is_being_repaired():
    store = build_store()
    _busy_node(store)
    _job_workflow(store, created_at=datetime.now(timezone.utc))
    dispatcher, adapter = _dispatcher(store)

    report = dispatcher.run_once()

    assert adapter.calls == []
    assert report.filtered.get("node_busy") == 1
    assert store.get_workflow("wf-job").status is WorkflowStatus.PENDING
    assert store.get_workflow("wf-node").status is WorkflowStatus.RUNNING


def test_past_the_wait_the_job_workflow_stops_the_job_and_fails():
    store = build_store()
    _busy_node(store)
    _job_workflow(store, created_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    dispatcher, adapter = _dispatcher(store)

    dispatcher.run_once()  # rewrites the plan to a stop-only failure
    dispatcher.run_once()  # executes it

    saved = store.get_workflow("wf-job")
    assert adapter.calls == ["wf-job/0/STOP_WORKLOADS"]
    assert saved.status is WorkflowStatus.FAILED
    assert "node-a" in (saved.terminal_failure_reason or "")
    assert store.get_incident("inc-job").state is IncidentState.ESCALATED
    assert store.get_workflow("wf-node").status is WorkflowStatus.RUNNING
    assert dispatcher.node_busy_timeouts_total == 1
