"""Superseding an abandoned generation leaves an audit line on the incident
(log 61 item 5).

The workflow row records ``preemption_reason``; the incident, which is what an
operator reads first, said nothing about the record the sweep closed under it.
The reason is bounded and deduplicated, so a repeated sweep writes it once.
"""

from __future__ import annotations

from gpu_fault.models import WorkflowStatus
from tests._builders import build_store
from tests.execution.test_abandoned_generation import (
    ABANDONED,
    CURRENT,
    INCIDENT,
    dispatcher,
    scenario,
)


def _supersede_markers(store) -> list[str]:
    return [
        reason
        for reason in store.get_incident(INCIDENT).reasons
        if "abandoned generation" in reason and ABANDONED in reason
    ]


def test_the_incident_records_the_supersession_once() -> None:
    store = build_store()
    scenario(store)
    sweep = dispatcher(store)

    sweep.run_once()

    assert store.get_workflow(ABANDONED).status is WorkflowStatus.SUPERSEDED
    markers = _supersede_markers(store)
    assert len(markers) == 1, markers
    assert CURRENT in markers[0], "the audit names the workflow that replaced it"

    sweep.run_once()

    assert len(_supersede_markers(store)) == 1, "a second sweep must not repeat it"
