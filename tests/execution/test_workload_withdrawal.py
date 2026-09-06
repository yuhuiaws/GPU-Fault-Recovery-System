"""When the training job a workflow is repairing is stopped by someone
else, the workflow winds down: in-flight actions finish, nodes already
touched are released, nothing new starts, the job is not restarted.

F-N1 §7 (docs/review/F-N1-设计与实施计划.md). The completion service used
to record "user stop, no automatic restart" and stop there; the job workflow
kept repairing every branch, escalating in place, and finally un-suspended
a job its owner had just stopped -- or failed the restart of a deleted job
and escalated hardware that was fine.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import (
    RESTART_PARAMETERS,
    FakeAdapter,
    WorkflowStepOutcome,
    workflow_state,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

STOP = WorkflowOperation.STOP_WORKLOADS
CORDON = WorkflowOperation.MARK_UNSCHEDULABLE
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "withdraw.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def test_amend_workflow_bumps_the_merge_revision_so_leased_writers_notice(store):
    incident = fault_incident("inc-w", "event-w", workflow_request_id="wf-w")
    workflow = workflow_request(
        "wf-w",
        "inc-w",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(RESET)],
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident_and_workflow(incident, workflow)

    amended = store.amend_workflow(
        "wf-w", {"workload_withdrawn_at": NOW, "workload_withdrawn_reason": "user stop"}
    )

    assert amended.workload_withdrawn_at == NOW
    assert amended.workload_withdrawn_reason == "user stop"
    assert amended.merge_revision == workflow.merge_revision + 1
    assert store.get_workflow("wf-w").workload_withdrawn_at == NOW
    with pytest.raises(NotFoundError):
        store.amend_workflow("wf-missing", {"workload_withdrawn_reason": "x"})


def test_a_user_stop_withdraws_the_workflow_that_owns_the_job(context, failed_event):
    incident = fault_incident(
        "inc-job",
        "event-job",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-job",
        job_id=failed_event.job_id,
        attempt_id=failed_event.attempt_id,
    )
    workflow = workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(
                STOP, workload_ids=["training/pytorchjob/distributed-training"]
            ),
            workflow_step(RESET, node_ids=["node-a"]),
            workflow_step(RESTART_JOB, parameters=dict(RESTART_PARAMETERS)),
        ],
    )
    context.store.save_incident_and_workflow(incident, workflow)

    decision = context.completion.handle_terminal(
        copy_model(failed_event, terminal_status=TerminalStatus.STOPPED)
    )

    assert decision.status is DecisionStatus.NO_ACTION
    withdrawn = context.store.get_workflow("wf-job")
    assert withdrawn.workload_withdrawn_at is not None
    assert "stop" in (withdrawn.workload_withdrawn_reason or "")


def test_a_withdrawn_dag_finishes_in_flight_work_releases_nodes_and_stops():
    store = build_store()
    incident, workflow = workflow_state(
        store, [STOP, CORDON, RESET, RESTORE, RESET, RESTART_JOB]
    )
    steps = [
        copy_model(
            workflow.official_steps[0],
            branch_id="shared",
            node_ids=["node-b", "node-c"],
        ),
        copy_model(
            workflow.official_steps[1],
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[2],
            node_ids=["node-b"],
            depends_on_step_indexes=[1],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[3],
            node_ids=["node-b"],
            depends_on_step_indexes=[2],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[4],
            node_ids=["node-c"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-c",
        ),
        copy_model(
            workflow.official_steps[5],
            depends_on_step_indexes=[3, 4],
            branch_id="join",
            node_ids=["node-b", "node-c"],
            parameters=dict(RESTART_PARAMETERS),
        ),
    ]
    store.save_workflow(
        copy_model(
            workflow,
            dag_enabled=True,
            official_steps=steps,
            completed_step_indexes=[0, 1],
            completed_operations=[STOP, CORDON],
            step_executions=[
                workflow_step_execution(0, STOP),
                workflow_step_execution(1, CORDON),
                # node-b's reset is executing on the node right now.
                workflow_step_execution(
                    2,
                    RESET,
                    WorkflowStepStatus.WAITING,
                    adapter_operation_id="remote/reset-b",
                ),
            ],
            workload_withdrawn_at=datetime.now(timezone.utc),
            workload_withdrawn_reason="controller-initiated or user stop",
        )
    )
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.succeeded(),
            RESTORE: WorkflowStepOutcome.succeeded(),
            RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = active_workflow_executor(
        store, [adapter], {STOP, CORDON, RESET, RESTORE, RESTART_JOB}
    )

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    saved = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.SUPERSEDED
    assert saved.status is WorkflowStatus.SUPERSEDED
    # The in-flight reset finished and node-b was released; node-c's reset
    # never started and the job was not restarted.
    assert adapter.calls == [
        "workflow-active/2/RESET_GPU",
        "workflow-active/3/RESTORE_SCHEDULING",
    ]
    assert {4, 5} <= set(saved.superseded_step_indexes)
    assert "stop" in (saved.preemption_reason or "")
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED


def test_events_for_a_withdrawn_workflow_are_recorded_only():
    merger = WorkflowMergeService(
        RecoveryArbiter(),
        DagBrancher(RecoveryArbiter()),
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(),
        workflow_resource_claims_by_node=lambda _workflow: {},
    )
    existing = workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(RESET, node_ids=["node-a"])],
        workload_withdrawn_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    candidate = workflow_request(
        "wf-new", "inc-new", official_steps=[workflow_step(RESET, node_ids=["node-a"])]
    )

    assert (
        merger.disposition(existing, candidate, "node-a", set()) == "ABSORB_RECORD_ONLY"
    )
