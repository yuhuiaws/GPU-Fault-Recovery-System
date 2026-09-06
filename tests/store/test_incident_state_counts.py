"""``incident_state_counts`` is the server-side aggregate behind the /metrics
incident-state gauge (F-L1): the ESCALATED bucket is the operator queue, so it
must be exact and must not depend on the bounded detail scan."""

from __future__ import annotations

import pytest

from gpu_fault.models import IncidentState
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "incidents.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


def test_every_state_is_reported_and_only_persisted_incidents_count(store) -> None:
    store.save_incident(fault_incident("inc-1", "evt-1", state=IncidentState.ESCALATED))
    store.save_incident(fault_incident("inc-2", "evt-2", state=IncidentState.ESCALATED))
    store.save_incident(fault_incident("inc-3", "evt-3", state=IncidentState.RECOVERED))

    counts = store.incident_state_counts()

    assert set(counts) == set(IncidentState), "every state must be present"
    assert counts[IncidentState.ESCALATED] == 2
    assert counts[IncidentState.RECOVERED] == 1
    assert counts[IncidentState.DETECTED] == 0


def test_state_transitions_move_between_buckets(store) -> None:
    incident = fault_incident("inc-1", "evt-1", state=IncidentState.DETECTED)
    store.save_incident(incident)
    store.save_incident(incident.model_copy(update={"state": IncidentState.ESCALATED}))

    counts = store.incident_state_counts()

    assert counts[IncidentState.DETECTED] == 0
    assert counts[IncidentState.ESCALATED] == 1
