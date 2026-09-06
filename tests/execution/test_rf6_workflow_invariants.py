"""The workflow invariant checker names every way a record can be malformed.

Review item 6: the executor, dispatcher and merge paths each assume a shape
for ``WorkflowRequest`` and each enforce only their own corner. The checker
puts the whole shape in one place; each violated invariant is one sentence,
so a log line or a raised error says exactly which assumption broke.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution.invariants import (
    InvariantMode,
    WorkflowInvariantError,
    check_workflow_invariants,
    workflow_invariant_violations,
)
from gpu_fault.models import (
    WORKFLOW_EVENTS_LIMIT,
    WorkflowEvent,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from tests._builders import (
    copy_model,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
RESTART = WorkflowOperation.RESTART_WORKLOAD


def _job_dag(**updates: Any) -> WorkflowRequest:
    """STOP (shared) -> {node-b RESET, node-c REBOOT} -> RESTART_WORKLOAD (join),
    mid-flight: STOP done, claimed by executor-a, lifetime stamped."""

    steps = [
        workflow_step(STOP, node_ids=["node-b", "node-c"], branch_id="shared"),
        workflow_step(
            RESET,
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        workflow_step(
            REBOOT,
            node_ids=["node-c"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-c",
        ),
        workflow_step(
            RESTART,
            node_ids=["node-b", "node-c"],
            depends_on_step_indexes=[1, 2],
            branch_id="join",
        ),
    ]
    workflow = workflow_request(
        "wf-dag",
        "incident-dag",
        status=WorkflowStatus.RUNNING,
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[STOP],
        step_executions=[workflow_step_execution(0, STOP, phase="official")],
        execution_owner_id="executor-a",
        execution_lease_expires_at=NOW + timedelta(minutes=3),
        created_at=NOW,
        updated_at=NOW,
        lifetime_deadline_at=NOW + timedelta(hours=1),
        events=[
            WorkflowEvent(kind=WorkflowEventKind.CLAIM, at=NOW),
            WorkflowEvent(kind=WorkflowEventKind.STEP_ATTEMPT, at=NOW),
        ],
    )
    return copy_model(workflow, **updates) if updates else workflow


def _with_step(workflow: WorkflowRequest, index: int, **updates: Any) -> list[Any]:
    steps = list(workflow.official_steps)
    steps[index] = copy_model(steps[index], **updates)
    return steps


def _events(*ats: datetime) -> list[WorkflowEvent]:
    return [WorkflowEvent(kind=WorkflowEventKind.STEP_ATTEMPT, at=at) for at in ats]


BASE = _job_dag()
ALL_DONE = [0, 1, 2, 3]

# (case, updates on the well-formed DAG, the phrase the single sentence carries)
VIOLATIONS: list[tuple[str, dict[str, Any], str]] = [
    (
        "completed indexes duplicated",
        {"completed_step_indexes": [0, 0]},
        "completed_step_indexes contain duplicates",
    ),
    (
        "superseded indexes duplicated",
        {"superseded_step_indexes": [2, 2]},
        "superseded_step_indexes contain duplicates",
    ),
    (
        "self dependency",
        {"official_steps": _with_step(BASE, 1, depends_on_step_indexes=[1])},
        "step 1 depends on invalid indexes: [1]",
    ),
    (
        "dependency outside the list",
        {"official_steps": _with_step(BASE, 1, depends_on_step_indexes=[9])},
        "step 1 depends on invalid indexes: [9]",
    ),
    (
        "negative dependency",
        {"official_steps": _with_step(BASE, 1, depends_on_step_indexes=[-1])},
        "step 1 depends on invalid indexes: [-1]",
    ),
    (
        "join label on a node step",
        {"official_steps": _with_step(BASE, 1, branch_id="join")},
        'DAG step 1 (RESET_GPU) carries branch_id "join"',
    ),
    (
        "exhausted branch that never existed",
        {"exhausted_branch_ids": ["branch:node-z"]},
        "exhausted_branch_ids name branches absent from official_steps",
    ),
    (
        "escalation count for a node no step touches",
        {"branch_escalation_counts": {"node-z": 1}},
        "branch_escalation_counts name nodes absent from every step",
    ),
    (
        "SUCCEEDED still leased",
        {
            "status": WorkflowStatus.SUCCEEDED,
            "completed_step_indexes": ALL_DONE,
            "execution_owner_id": None,
        },
        "SUCCEEDED workflow still holds an execution lease",
    ),
    (
        "FAILED still leased",
        {"status": WorkflowStatus.FAILED, "execution_owner_id": None},
        "FAILED workflow still holds an execution lease",
    ),
    (
        "BLOCKED still owned",
        {"status": WorkflowStatus.BLOCKED, "execution_lease_expires_at": None},
        "BLOCKED workflow still has an execution owner",
    ),
    (
        "RUNNING without an owner",
        {"execution_owner_id": None, "execution_lease_expires_at": None},
        "RUNNING workflow has no execution owner",
    ),
    (
        "preempted but neither superseded nor pending",
        {"preempted_by_workflow_id": "wf-successor"},
        "preempted_by_workflow_id is set",
    ),
    (
        "withdrawn without a reason",
        {"workload_withdrawn_at": NOW},
        "workload_withdrawn_at is set without workload_withdrawn_reason",
    ),
    (
        "lifetime before creation",
        {"lifetime_deadline_at": NOW - timedelta(seconds=1)},
        "lifetime_deadline_at precedes created_at",
    ),
    (
        "execution record outside its phase",
        {
            "step_executions": [
                workflow_step_execution(0, STOP, phase="official"),
                workflow_step_execution(7, STOP, phase="official"),
            ]
        },
        "step execution official/7/STOP_WORKLOADS points outside",
    ),
    (
        "execution record in an empty phase",
        {
            "step_executions": [
                workflow_step_execution(0, STOP, phase="official"),
                workflow_step_execution(0, STOP, phase="safety"),
            ]
        },
        "step execution safety/0/STOP_WORKLOADS points outside",
    ),
    (
        "events over the cap",
        {"events": _events(*([NOW] * (WORKFLOW_EVENTS_LIMIT + 1)))},
        f"events exceed WORKFLOW_EVENTS_LIMIT: {WORKFLOW_EVENTS_LIMIT + 1}",
    ),
    (
        "events out of order",
        {"events": _events(NOW + timedelta(seconds=5), NOW)},
        "events are not in chronological order at position 1",
    ),
]


@pytest.mark.parametrize(
    ("case", "updates", "phrase"), VIOLATIONS, ids=[case for case, _, _ in VIOLATIONS]
)
def test_each_broken_invariant_is_exactly_one_sentence(
    case: str, updates: dict[str, Any], phrase: str
) -> None:
    violations = workflow_invariant_violations(_job_dag(**updates))

    assert len(violations) == 1, f"{case}: expected one sentence, got {violations}"
    assert phrase in violations[0], f"{case}: {violations[0]!r} lacks {phrase!r}"


def test_a_well_formed_dag_built_from_the_shared_builders_passes() -> None:
    workflow = _job_dag()

    assert workflow_invariant_violations(workflow) == [], (
        "the reference DAG must be clean"
    )
    check_workflow_invariants(workflow, InvariantMode.RAISE)


def test_a_succeeded_workflow_that_still_has_an_owner_is_a_violation() -> None:
    finished = _job_dag(
        status=WorkflowStatus.SUCCEEDED,
        completed_step_indexes=ALL_DONE,
        execution_lease_expires_at=None,
    )

    violations = workflow_invariant_violations(finished)

    assert violations == ["SUCCEEDED workflow still has an execution owner"], violations


def test_equal_event_timestamps_and_legacy_unphased_records_are_allowed() -> None:
    workflow = _job_dag(
        events=_events(NOW, NOW, NOW + timedelta(seconds=1)),
        step_executions=[workflow_step_execution(0, STOP, phase=None)],
    )

    assert workflow_invariant_violations(workflow) == [], (
        "equal timestamps and a None phase are legitimate"
    )


def test_a_pending_preemption_explains_the_preempted_by_marker() -> None:
    marked = _job_dag(
        preempted_by_workflow_id="wf-successor",
        preemption_pending_by_workflow_id="wf-successor",
    )

    assert workflow_invariant_violations(marked) == [], (
        "a pending preemption legitimately carries the successor id"
    )


def test_modes_raise_log_or_ignore(caplog: pytest.LogCaptureFixture) -> None:
    broken = _job_dag(workload_withdrawn_at=NOW)

    with pytest.raises(WorkflowInvariantError, match="wf-dag"):
        check_workflow_invariants(broken, InvariantMode.RAISE)
    with caplog.at_level(logging.ERROR, logger="gpu_fault.execution.invariants"):
        check_workflow_invariants(broken, InvariantMode.LOG)
    assert any("workload_withdrawn_reason" in r.message for r in caplog.records), (
        caplog.records
    )
    caplog.clear()
    check_workflow_invariants(broken, InvariantMode.OFF)
    assert caplog.records == [], "OFF must neither raise nor log"
