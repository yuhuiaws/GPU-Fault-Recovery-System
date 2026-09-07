"""A bounded history says where it lost the middle (I5).

``bounded_reasons`` keeps the first half of its budget and the most recent
half; ``record_workflow_event`` keeps the first fifth and the most recent rest.
Both silently dropped everything in between, so an auditor reading a full list
could not tell a quiet workflow from one whose middle had been evicted. The
bounds do not change; when something is dropped, one marker entry at the seam
says how much and (for events, which carry a clock) over what time range.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    INCIDENT_REASONS_LIMIT,
    REASONS_TRUNCATED_PREFIX,
    WORKFLOW_EVENTS_LIMIT,
    WorkflowEventKind,
    bounded_reasons,
    record_workflow_event,
)
from tests._builders import workflow_request

NOW = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)


def _markers(reasons: list[str]) -> list[str]:
    return [item for item in reasons if item.startswith(REASONS_TRUNCATED_PREFIX)]


def test_reasons_within_the_limit_carry_no_marker() -> None:
    values = [f"reason-{index}" for index in range(INCIDENT_REASONS_LIMIT)]

    assert bounded_reasons(values) == values
    assert _markers(bounded_reasons(["a", "b"], limit=3)) == []


def test_a_truncated_reason_list_marks_the_gap_exactly_once() -> None:
    values = [f"reason-{index}" for index in range(60)]

    bounded = bounded_reasons(values, limit=10)

    assert len(bounded) == 10, "the bound must hold with the marker included"
    head = 10 // 2
    assert bounded[:head] == values[:head], "the opening reasons must survive"
    markers = _markers(bounded)
    assert len(markers) == 1, f"expected one gap marker, found {markers}"
    assert bounded[head] == markers[0], "the marker must sit at the seam"
    # 60 unique reasons, 5 opening + 4 recent kept, so 51 were dropped.
    assert "51" in markers[0], (
        f"the marker does not say how many were dropped: {markers[0]}"
    )
    assert bounded[head + 1 :] == values[-4:]


def test_re_bounding_a_marked_list_keeps_one_marker_and_accumulates() -> None:
    values = [f"reason-{index}" for index in range(60)]
    bounded = bounded_reasons(values, limit=10)

    again = bounded_reasons([*bounded, "reason-new"], limit=10)

    assert len(again) == 10
    assert again[:5] == values[:5]
    assert again[-1] == "reason-new"
    markers = _markers(again)
    assert len(markers) == 1, f"a re-bound list grew a second marker: {again}"
    # The previous marker's 51 plus the one recent reason evicted this time.
    assert "52" in markers[0], f"the marker did not accumulate: {markers[0]}"


def test_events_over_the_limit_carry_one_marker_with_the_dropped_range() -> None:
    workflow = workflow_request("wf-history", "inc-history")
    total = WORKFLOW_EVENTS_LIMIT + 20
    for index in range(total):
        workflow = record_workflow_event(
            workflow,
            WorkflowEventKind.STEP_ATTEMPT,
            reason=f"attempt-{index}",
            at=NOW + timedelta(seconds=index),
        )

    events = workflow.events
    assert len(events) == WORKFLOW_EVENTS_LIMIT, "the event bound must hold"
    head = WORKFLOW_EVENTS_LIMIT // 5
    markers = [
        item for item in events if item.kind is WorkflowEventKind.HISTORY_TRUNCATED
    ]
    assert len(markers) == 1, f"expected one gap marker, found {len(markers)}"
    marker = events[head]
    assert marker.kind is WorkflowEventKind.HISTORY_TRUNCATED, (
        "the marker must sit between the opening events and the recent ones"
    )
    assert [item.reason for item in events[:head]] == [
        f"attempt-{index}" for index in range(head)
    ], "the opening events must survive"
    assert events[-1].reason == f"attempt-{total - 1}"
    # head opening events kept, LIMIT - head - 1 recent events kept.
    dropped = total - head - (WORKFLOW_EVENTS_LIMIT - head - 1)
    assert marker.details["dropped"] == dropped
    assert marker.details["dropped_from"] == (NOW + timedelta(seconds=head)).isoformat()
    assert (
        marker.details["dropped_to"]
        == (NOW + timedelta(seconds=head + dropped - 1)).isoformat()
    )
    assert marker.code == "HISTORY_TRUNCATED"


def test_events_within_the_limit_carry_no_marker() -> None:
    workflow = workflow_request("wf-quiet", "inc-quiet")
    for index in range(WORKFLOW_EVENTS_LIMIT):
        workflow = record_workflow_event(
            workflow, WorkflowEventKind.STEP_ATTEMPT, reason=f"attempt-{index}"
        )

    kinds = {item.kind for item in workflow.events}
    assert WorkflowEventKind.HISTORY_TRUNCATED not in kinds, (
        "a list that never overflowed must not claim a gap"
    )
    assert len(workflow.events) == WORKFLOW_EVENTS_LIMIT
