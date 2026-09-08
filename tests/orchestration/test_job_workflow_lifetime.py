"""Every workflow has a lifetime; at the deadline it ends, escalates to an
operator, and later events for the same node or job are recorded, not
re-planned.

F-N1 v2 (docs/review/F-N1-设计与实施计划.md §0). A remediation that is still
open an hour after it started is itself the incident. The old bounds were a
30-minute execution deadline stamped at the first claim (re-stamped by
nothing, reaped as a plain FAILED that spawned the next reboot/replace
generation) and, for job workflows, "the join has started". Now the deadline
is stamped at the first claim by kind, inherited along the escalation chain,
caps the execution deadline, fails the workflow with a distinct marker that
the branch escalator and the hardware escalation both honour, and turns
later same-scope events into incident records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.execution.restart_budget_preflight import claim_deadlines
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    lifetime_exceeded,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state

ALL_NODES = ["node-a", "node-b", "node-c"]
CREATED = datetime(2026, 9, 6, 8, 0, tzinfo=timezone.utc)
RESET = WorkflowOperation.RESET_GPU


def _job_workflow(node_id: str, request_id: str, **updates) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "incident-dag",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="RESET_GPU",
        created_at=CREATED,
        updated_at=CREATED,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "owner",
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
            ),
            workflow_step(RESET, "owner", node_ids=[node_id]),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING, "owner", node_ids=[node_id]
            ),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                "owner",
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
            ),
        ],
        **updates,
    )


def _merger(brancher: DagBrancher | None = None) -> WorkflowMergeService:
    return WorkflowMergeService(
        RecoveryArbiter(),
        brancher or DagBrancher(RecoveryArbiter()),
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(),
        workflow_resource_claims_by_node=lambda _workflow: {},
    )


# ---------------------------------------------------------------- stamping


def test_the_first_claim_stamps_the_lifetime_by_kind_and_caps_the_execution_deadline():
    now = CREATED + timedelta(minutes=1)
    job = copy_model(_job_workflow("node-a", "wf-job"), dag_enabled=True)
    node = workflow_request(
        "wf-node",
        "incident-node",
        official_steps=[workflow_step(RESET, node_ids=["node-a"])],
        created_at=CREATED,
        updated_at=CREATED,
    )

    job_execution, job_lifetime = claim_deadlines(
        job,
        now,
        timeout_seconds=1800,
        job_lifetime_seconds=3600,
        node_lifetime_seconds=900,
    )
    node_execution, node_lifetime = claim_deadlines(
        node,
        now,
        timeout_seconds=1800,
        job_lifetime_seconds=3600,
        node_lifetime_seconds=900,
    )

    assert job_lifetime == now + timedelta(hours=1)
    assert job_execution == now + timedelta(seconds=1800)
    assert node_lifetime == now + timedelta(minutes=15)
    # A 30-minute timeout past a 15-minute lifetime is capped by the lifetime.
    assert node_execution == node_lifetime
    # A stamped lifetime is kept, not moved, on later claims.
    later = copy_model(
        job, lifetime_deadline_at=job_lifetime, execution_deadline=job_execution
    )
    assert (
        claim_deadlines(
            later,
            now + timedelta(minutes=10),
            timeout_seconds=1800,
            job_lifetime_seconds=60,
            node_lifetime_seconds=60,
        )[1]
        == job_lifetime
    )


def test_lifetime_is_exceeded_only_after_the_deadline():
    workflow = _job_workflow("node-a", "wf")
    assert lifetime_exceeded(workflow, now=CREATED) is False  # unstamped: never
    stamped = copy_model(workflow, lifetime_deadline_at=CREATED + timedelta(hours=1))
    assert lifetime_exceeded(stamped, now=CREATED + timedelta(minutes=59)) is False
    assert lifetime_exceeded(stamped, now=CREATED + timedelta(hours=1)) is True


# ---------------------------------------------------------------- at the deadline


def test_a_job_workflow_past_its_lifetime_fails_without_escalating_a_branch():
    store = build_store()
    _, workflow = workflow_state(
        store,
        [WorkflowOperation.STOP_WORKLOADS, RESET, WorkflowOperation.RESTART_WORKLOAD],
    )
    steps = [
        copy_model(workflow.official_steps[0], branch_id="shared"),
        copy_model(
            workflow.official_steps[1],
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[2], depends_on_step_indexes=[1], branch_id="join"
        ),
    ]
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    store.save_workflow(
        copy_model(
            workflow,
            dag_enabled=True,
            official_steps=steps,
            completed_step_indexes=[0],
            completed_operations=[WorkflowOperation.STOP_WORKLOADS],
            lifetime_deadline_at=past,
        )
    )
    cancelled: list[str] = []
    original = store.cancel_remote_commands_for_workflow

    def spy(workflow_request_id, *, reason):
        cancelled.append(reason)
        return original(workflow_request_id, reason=reason)

    store.cancel_remote_commands_for_workflow = spy
    escalations: list[int] = []

    class _RecordingEscalator(BranchEscalator):
        def escalate_branch(self, workflow, failed_index, error, **kwargs):
            escalations.append(failed_index)
            return super().escalate_branch(workflow, failed_index, error, **kwargs)

    adapter = FakeAdapter({RESET: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(
        store, [adapter], {RESET, WorkflowOperation.RESTART_WORKLOAD}
    )
    executor.branch_escalator = _RecordingEscalator(
        DagBrancher(RecoveryArbiter()), lambda *_args: [], max_rungs=2
    )

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    saved = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []  # nothing new was started
    assert escalations == []  # the branch ladder was not consulted
    assert cancelled and "lifetime" in cancelled[0]
    failed = [
        item
        for item in saved.step_executions
        if item.status is WorkflowStepStatus.FAILED
    ]
    assert failed and failed[0].details.get("workflow_lifetime_exceeded") is True
    assert executor.lifetime_exceeded_total == 1
    assert store.get_incident(saved.incident_id).state is IncidentState.ESCALATED


def test_a_single_node_workflow_past_its_lifetime_escalates_only_to_support():
    store = build_store()
    incident, workflow = workflow_state(store, [RESET])
    store.save_workflow(
        copy_model(
            workflow,
            lifetime_deadline_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
    )
    adapter = FakeAdapter({RESET: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(store, [adapter], {RESET})

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    saved = store.get_workflow(workflow.request_id)
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None
    failed_stage, next_action, next_operation, _ = classification
    assert failed_stage == "lifetime_exceeded"
    assert next_operation is WorkflowOperation.ESCALATE_SUPPORT


def test_the_quiesce_undo_still_runs_after_the_lifetime_and_counts_once():
    store = build_store()
    _, workflow = workflow_state(
        store,
        [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            RESET,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        ],
    )
    store.save_workflow(
        copy_model(
            workflow,
            completed_step_indexes=[0],
            completed_operations=[WorkflowOperation.QUIESCE_GPU_SERVICES],
            lifetime_deadline_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
    )
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.succeeded(),
            WorkflowOperation.RESTORE_GPU_SERVICES: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = active_workflow_executor(
        store, [adapter], {RESET, WorkflowOperation.RESTORE_GPU_SERVICES}
    )

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert result.status is WorkflowStatus.FAILED
    # The reset never started; the services the workflow stopped were restored.
    assert adapter.calls == ["workflow-active/2/RESTORE_GPU_SERVICES"]
    assert executor.lifetime_exceeded_total == 1


# ---------------------------------------------------------------- after the deadline


def test_events_after_the_lifetime_are_recorded_on_the_incident_not_replanned():
    merger = _merger()
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    running = copy_model(
        _job_workflow("node-a", "wf-job"),
        status=WorkflowStatus.RUNNING,
        dag_enabled=True,
        lifetime_deadline_at=past,
    )
    failed = copy_model(running, status=WorkflowStatus.FAILED)
    incident = fault_incident(
        "incident-dag",
        "event-dag",
        state=IncidentState.ESCALATED,
        workflow_request_id="wf-job",
    )

    still_running = merger.disposition(
        running, _job_workflow("node-c", "wf-c"), "node-c", set()
    )
    assert still_running == "ABSORB_RECORD_ONLY"
    # A terminal record whose lifetime passed stays the merge target while the
    # incident is with an operator, so the event lands on that incident.
    kept_incident, kept_workflow = NodeConflictService.reopen_if_terminal(
        incident, failed
    )
    assert kept_workflow is failed and kept_incident is incident
    assert (
        merger.disposition(failed, _job_workflow("node-c", "wf-c"), "node-c", set())
        == "ABSORB_RECORD_ONLY"
    )
    assert merger.lifetime_record_only_total == 2
    # Recovered means closed: the next event is a new fault.
    recovered = copy_model(incident, state=IncidentState.RECOVERED)
    assert NodeConflictService.reopen_if_terminal(recovered, failed) == (None, None)


def test_a_successor_and_a_replacement_inherit_the_lifetime():
    lifetime = CREATED + timedelta(hours=1)
    existing = copy_model(
        _job_workflow("node-a", "wf-job"), lifetime_deadline_at=lifetime
    )
    merger = _merger()

    successor = merger.prepare_preempting_successor(
        existing,
        copy_model(
            workflow_request(
                "wf-next",
                "incident-dag",
                official_steps=[
                    workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-a"])
                ],
            ),
            predecessor_workflow_id=existing.request_id,
        ),
    )

    assert successor.lifetime_deadline_at == lifetime
