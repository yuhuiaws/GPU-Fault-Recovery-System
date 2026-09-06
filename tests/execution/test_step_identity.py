"""``step_index`` alone is not a step's identity: the phase's operation is too.

FINAL-建议汇总 F-C2 (P0-62D). Safety steps and official steps are two lists
that share indexes, and both record into the same ``step_executions``. Three
consumers keyed on the index alone: recording a safety attempt at index 0
deleted the official step's record at index 0, the reservation release read
"index 0 was attempted" off a safety record and kept the budget, and the
quiesce handoff dropped whatever else sat at the quiesce step's index.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution import restart_budget_preflight, step_bounds
from gpu_fault.execution.executor import WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 5, 18, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
JOB = "job-a"


def test_recording_a_safety_attempt_keeps_the_official_record_at_that_index():
    workflow = workflow_request(
        "wf-phase",
        "inc-phase",
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        safety_steps=[workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE)],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.FREEZE_EVIDENCE, WorkflowStepStatus.SUCCEEDED
            )
        ],
    )

    recorded = step_bounds.record_attempt(
        workflow, workflow.safety_steps[0], 0, WorkflowStepOutcome.succeeded()
    )

    assert sorted(
        (item.step_index, item.operation.value) for item in recorded.step_executions
    ) == [(0, "FREEZE_EVIDENCE"), (0, "MARK_UNSCHEDULABLE")]


def test_release_ignores_a_same_index_execution_of_another_operation():
    store = build_store()
    workflow = workflow_request(
        "wf-release",
        "inc-release",
        official_steps=[
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                parameters={"cluster_id": CLUSTER, "job_id": JOB},
            )
        ],
        safety_steps=[workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE)],
        # The safety phase ran step 0; the official RESTART_WORKLOAD never did.
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowStepStatus.SUCCEEDED
            )
        ],
    )
    reservation = restart_budget_preflight._reservation_id(workflow, 0)
    store.reserve_job_restart(CLUSTER, JOB, 1, reservation)

    restart_budget_preflight.release_unattempted_restart_reservations(store, workflow)

    # Budget is 1: this second reservation only fits if the first was released.
    _, accepted = store.reserve_job_restart(CLUSTER, JOB, 1, "the-next-real-restart")
    assert accepted is True


def test_quiesce_handoff_keeps_other_operations_at_the_quiesce_index():
    store = build_store()
    executor = active_workflow_executor(store, [], [])
    incident = fault_incident(
        "inc-handoff",
        "event-handoff",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-successor",
        fencing_token=1,
    )
    predecessor = workflow_request(
        "wf-predecessor",
        "inc-handoff",
        status=WorkflowStatus.SUPERSEDED,
        fencing_token=1,
        preempted_by_workflow_id="wf-successor",
        official_steps=[
            workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=["node-a"])
        ],
        completed_step_indexes=[0],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                WorkflowStepStatus.SUCCEEDED,
                adapter_operation_id="quiesce-1",
            )
        ],
    )
    successor = workflow_request(
        "wf-successor",
        "inc-handoff",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        predecessor_workflow_id="wf-predecessor",
        official_steps=[
            workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=["node-a"]),
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"]),
        ],
        safety_steps=[workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE)],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowStepStatus.SUCCEEDED
            )
        ],
    )
    store.save_incident_and_workflow(incident, successor)
    store.save_workflow(predecessor)

    adopted = executor._adopt_quiesce_handoff_from_predecessor(successor, incident)

    assert sorted(
        (item.step_index, item.operation.value) for item in adopted.step_executions
    ) == [(0, "MARK_UNSCHEDULABLE"), (0, "QUIESCE_GPU_SERVICES")]


# --- F-C2 identity key = (phase, step_index, operation) -----------------------


def test_the_same_operation_in_both_phases_keeps_both_records():
    """FREEZE_EVIDENCE sits at index 0 of both lists: the safety attempt must
    not erase the official record, which shares index *and* operation."""

    workflow = workflow_request(
        "wf-both",
        "inc-both",
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        safety_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowStepStatus.SUCCEEDED,
                phase="official",
            )
        ],
    )
    safety = workflow.model_copy(update={"safety_only": True})

    recorded = step_bounds.record_attempt(
        safety, safety.safety_steps[0], 0, WorkflowStepOutcome.failed("boom")
    )

    assert sorted(
        (item.phase, item.status.value) for item in recorded.step_executions
    ) == [("official", "SUCCEEDED"), ("safety", "FAILED")]


def test_a_record_written_before_the_phase_existed_matches_either_phase():
    """Dual read: a legacy record (no phase) is *this* step's record for
    whichever phase asks, so it is replaced rather than duplicated."""

    workflow = workflow_request(
        "wf-legacy",
        "inc-legacy",
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        safety_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.FREEZE_EVIDENCE, WorkflowStepStatus.WAITING
            )
        ],
    )
    assert workflow.step_executions[0].phase is None

    recorded = step_bounds.record_attempt(
        workflow, workflow.official_steps[0], 0, WorkflowStepOutcome.succeeded()
    )

    assert [(item.phase, item.status.value) for item in recorded.step_executions] == [
        ("official", "SUCCEEDED")
    ]
    assert (
        step_bounds.previous_execution(workflow, workflow.safety_steps[0], 0)
        is workflow.step_executions[0]
    ), "a legacy record answers for the safety phase too"


def test_previous_execution_prefers_the_asking_phases_record():
    workflow = workflow_request(
        "wf-prefer",
        "inc-prefer",
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        safety_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowStepStatus.SUCCEEDED,
                phase="official",
            ),
            workflow_step_execution(
                0,
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowStepStatus.WAITING,
                phase="safety",
            ),
        ],
    )
    safety = workflow.model_copy(update={"safety_only": True})

    official = step_bounds.previous_execution(workflow, workflow.official_steps[0], 0)
    waiting = step_bounds.previous_execution(safety, safety.safety_steps[0], 0)

    assert official is not None and official.phase == "official"
    assert waiting is not None and waiting.phase == "safety"
