"""RF-8: the structured events render as one chronological, flat timeline.

``incident.reasons`` stays as the human trail. ``workflow_timeline`` is the
machine-readable one for admin and report tooling: every event, in order,
as a flat dict with the same eight keys whatever the kind, so a report can
print or filter it without knowing which module wrote which event.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    WORKFLOW_EVENTS_LIMIT,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    record_workflow_event,
)
from gpu_fault.workflow_history import TIMELINE_KEYS, workflow_timeline
from tests._builders import workflow_request, workflow_step

T0 = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
RESET = WorkflowOperation.RESET_GPU


def _synthetic_workflow():
    workflow = workflow_request(
        "workflow-rf8",
        "incident-rf8",
        official_steps=[workflow_step(RESET, node_ids=["node-b"])],
    )
    # Recorded out of clock order on purpose: a late writer with an earlier
    # stamp (a lease renewal that raced) must still render in clock order.
    plan = [
        (T0, WorkflowEventKind.CLAIM, WorkflowEventCode.CLAIMED, None, None),
        (
            T0 + timedelta(seconds=20),
            WorkflowEventKind.STEP_ATTEMPT,
            WorkflowEventCode.STEP_WAITING,
            0,
            "WAITING",
        ),
        (
            T0 + timedelta(seconds=10),
            WorkflowEventKind.STEP_ATTEMPT,
            WorkflowEventCode.STEP_WAITING,
            0,
            "WAITING",
        ),
        (
            T0 + timedelta(seconds=30),
            WorkflowEventKind.PLAN_REWRITE,
            WorkflowEventCode.BRANCH_REPLACED,
            None,
            None,
        ),
        (
            T0 + timedelta(seconds=40),
            WorkflowEventKind.TERMINAL,
            WorkflowEventCode.TERMINALIZED,
            None,
            "FAILED",
        ),
    ]
    for at, kind, code, step_index, status in plan:
        workflow = record_workflow_event(
            workflow,
            kind,
            code=code,
            at=at,
            step_index=step_index,
            operation=RESET if step_index is not None else None,
            phase="official" if step_index is not None else None,
            status=status,
            reason=f"{code} happened",
            details={"attempt": 1} if step_index is not None else {},
        )
    return workflow


def test_the_timeline_is_chronological_and_complete():
    workflow = _synthetic_workflow()

    timeline = workflow_timeline(workflow)

    assert len(timeline) == len(workflow.events)
    stamps = [row["at"] for row in timeline]
    assert stamps == sorted(stamps), stamps
    assert all(isinstance(stamp, str) for stamp in stamps), stamps
    assert [row["code"] for row in timeline] == [
        WorkflowEventCode.CLAIMED,
        WorkflowEventCode.STEP_WAITING,
        WorkflowEventCode.STEP_WAITING,
        WorkflowEventCode.BRANCH_REPLACED,
        WorkflowEventCode.TERMINALIZED,
    ]
    assert all(tuple(row) == TIMELINE_KEYS for row in timeline), timeline
    assert set(TIMELINE_KEYS) >= {
        "at",
        "kind",
        "code",
        "step_index",
        "operation",
        "phase",
        "status",
        "reason",
    }
    step_rows = [row for row in timeline if row["step_index"] == 0]
    assert all(row["operation"] == RESET.value for row in step_rows), step_rows
    assert all(row["phase"] == "official" for row in step_rows), step_rows
    assert all(row["kind"] == "STEP_ATTEMPT" for row in step_rows), step_rows
    assert timeline[-1]["status"] == "FAILED"
    assert timeline[-1]["reason"] == "TERMINALIZED happened"


def test_the_timeline_is_plain_data():
    import json

    timeline = workflow_timeline(_synthetic_workflow())

    assert json.loads(json.dumps(timeline)) == timeline


def test_an_empty_workflow_has_an_empty_timeline():
    assert workflow_timeline(workflow_request("wf", "inc")) == []


def test_the_event_list_is_bounded_and_keeps_its_head():
    workflow = workflow_request("wf", "inc")
    for index in range(WORKFLOW_EVENTS_LIMIT + 50):
        workflow = record_workflow_event(
            workflow,
            WorkflowEventKind.STEP_ATTEMPT,
            code=WorkflowEventCode.STEP_WAITING,
            at=T0 + timedelta(seconds=index),
            details={"attempt": index + 1},
        )

    assert len(workflow.events) == WORKFLOW_EVENTS_LIMIT
    assert workflow.events[0].details["attempt"] == 1
    assert workflow.events[-1].details["attempt"] == WORKFLOW_EVENTS_LIMIT + 50
    assert len(workflow_timeline(workflow)) == WORKFLOW_EVENTS_LIMIT
