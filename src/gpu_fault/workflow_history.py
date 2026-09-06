"""Read-side rendering of a workflow's audit events (RF-8).

``WorkflowRequest.events`` is the machine-readable history: bounded,
structured, never rewritten. ``incident.reasons`` stays the human one. This
module turns the former into flat rows for admin and report tooling so a
caller can print, filter or diff a workflow's life without knowing which
module wrote which event. Pure: no store, no clock.
"""

from __future__ import annotations

from gpu_fault.models import WorkflowEvent, WorkflowRequest

# Every row has exactly these keys, in this order, whatever the event kind.
TIMELINE_KEYS: tuple[str, ...] = (
    "at",
    "kind",
    "code",
    "step_index",
    "operation",
    "phase",
    "status",
    "reason",
    "dag_revision",
)


def timeline_row(event: WorkflowEvent) -> dict[str, object]:
    """One event as a flat, JSON-ready dict with ``TIMELINE_KEYS``.

    ``details`` are deliberately left out: they are per-kind and a report
    that wants them reads the event itself.
    """

    return {
        "at": event.at.isoformat(),
        "kind": event.kind.value,
        "code": event.code,
        "step_index": event.step_index,
        "operation": None if event.operation is None else event.operation.value,
        "phase": event.phase,
        "status": event.status,
        "reason": event.reason,
        "dag_revision": event.dag_revision,
    }


def workflow_timeline(workflow: WorkflowRequest) -> list[dict[str, object]]:
    """The workflow's events in clock order, one flat row each.

    Sorted rather than trusted: writers append in their own order, and a
    lease renewal that raced a step can stamp an earlier clock later. The
    sort is stable, so events with one stamp keep their append order.
    """

    return [
        timeline_row(event)
        for event in sorted(workflow.events, key=lambda event: event.at)
    ]
