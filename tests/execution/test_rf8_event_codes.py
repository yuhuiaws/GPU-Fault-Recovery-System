"""RF-8: a stable machine code on every event, shared across the writers.

The free text in ``incident.reasons`` stays for people. The code is what a
report or an alert keys on, so the vocabulary is one enum in ``models`` that
every writer -- executor, dispatcher, brancher, escalator, merge -- imports
rather than spelling its own.
"""

from __future__ import annotations

from gpu_fault.models import (
    WorkflowEvent,
    WorkflowEventCode,
    WorkflowEventKind,
    record_workflow_event,
)
from tests._builders import workflow_request

REQUIRED_CODES = {
    "CLAIMED",
    "STEP_WAITING",
    "STEP_FAILED",
    "STEP_SUCCEEDED",
    "BRANCH_APPENDED",
    "BRANCH_REPLACED",
    "BRANCH_WIDENED",
    "BRANCH_ESCALATED",
    "BRANCH_EXHAUSTED",
    "PREEMPTED",
    "PREEMPTION_BOUNDARY_CLOSED",
    "NODE_UNDER_REMEDIATION",
    "NODE_REMEDIATION_TIMEOUT",
    "WORKLOAD_WITHDRAWN",
    "LIFETIME_EXCEEDED",
    "DEADLINE_EXCEEDED",
    "TERMINALIZED",
    "INVARIANT_VIOLATION",
}


def test_the_code_vocabulary_covers_every_writer():
    names = {member.name for member in WorkflowEventCode}
    assert REQUIRED_CODES <= names, sorted(REQUIRED_CODES - names)
    assert all(member.name == member.value for member in WorkflowEventCode), (
        "a code is spelled the same in Python and on the wire"
    )


def test_an_event_carries_its_code_and_round_trips_as_a_string():
    workflow = record_workflow_event(
        workflow_request("wf", "inc"),
        WorkflowEventKind.HOLD,
        code=WorkflowEventCode.NODE_UNDER_REMEDIATION,
        reason="node-b is being rebooted by workflow-x",
    )

    (event,) = workflow.events
    assert event.code == "NODE_UNDER_REMEDIATION"
    dumped = event.model_dump(mode="json")
    assert dumped["code"] == "NODE_UNDER_REMEDIATION"
    assert WorkflowEvent.model_validate(dumped) == event


def test_a_code_is_optional_so_older_rows_still_load():
    legacy = {"kind": "CLAIM", "at": "2026-09-06T09:00:00+00:00"}

    event = WorkflowEvent.model_validate(legacy)

    assert event.code is None
    assert event.kind is WorkflowEventKind.CLAIM
