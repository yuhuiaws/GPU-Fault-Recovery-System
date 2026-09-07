"""``save_incident`` never overwrites an incident that moved since it was read.

Architecture review 2026-09-07, item D1. ``save_incident`` was a blind
whole-row write on every backend and ``FaultIncident`` carries no version
field, so the coordinator's read-append-save of ``reasons`` overwrote a
concurrent state change or generation move. The guard mirrors
``save_workflow`` (store review item B): without ``expected`` the write is
refused when the row's ``fencing_token`` is already ahead of the copy -- the
generation only moves forward, so a lower token is a copy read before a
generation change -- and with ``expected`` it is a compare-and-set on the whole
payload the caller read.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.models import IncidentState
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import build_store, copy_model, fault_incident
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "incident-guard.db"))
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


def _seed(store):
    incident = fault_incident(
        "inc-g",
        "event-g",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-g",
        fencing_token=3,
        reasons=["first"],
    )
    store.save_incident(incident)
    return store.get_incident("inc-g")


def test_a_new_incident_is_inserted_and_linked(store) -> None:
    incident = fault_incident("inc-new", "event-new")

    store.save_incident(incident)

    assert store.get_incident("inc-new") == incident
    assert store.get_incident_by_event("event-new") == incident


def test_unchanged_versions_land(store) -> None:
    stored = _seed(store)

    store.save_incident(copy_model(stored, state=IncidentState.RECOVERED))

    assert store.get_incident("inc-g").state is IncidentState.RECOVERED


def test_a_copy_read_before_a_generation_change_is_refused(store) -> None:
    stored = _seed(store)
    # The generation moves forward through a plain save (the coordinator
    # compiles the new generation; fixtures do the same).
    store.save_incident(copy_model(stored, fencing_token=4))

    with pytest.raises(StaleWriteError, match="fencing_token") as raised:
        store.save_incident(copy_model(stored, reasons=["first", "late"]))

    assert "re-read" in str(raised.value), "the refusal tells the caller what to do"
    current = store.get_incident("inc-g")
    assert current.fencing_token == 4
    assert current.reasons == ["first"]


def test_a_workflow_pointer_move_is_only_caught_with_expected(store) -> None:
    """``workflow_request_id`` is re-pointed by the reconcile paths through a
    plain save, so it is not a guarded version field: a stale copy that read
    the old pointer must pass ``expected`` to be refused."""

    stored = _seed(store)
    store.save_incident(copy_model(stored, workflow_request_id="wf-successor"))

    with pytest.raises(StaleWriteError):
        store.save_incident(
            copy_model(stored, reasons=["first", "late"]), expected=stored
        )

    assert store.get_incident("inc-g").workflow_request_id == "wf-successor"


def test_expected_mismatch_is_refused(store) -> None:
    stored = _seed(store)
    # A write that keeps both version fields still changes the payload: the
    # executor recovered the incident while the coordinator appended a reason.
    store.save_incident(copy_model(stored, state=IncidentState.RECOVERED))

    with pytest.raises(StaleWriteError):
        store.save_incident(
            copy_model(stored, reasons=["first", "late"]), expected=stored
        )

    current = store.get_incident("inc-g")
    assert current.state is IncidentState.RECOVERED
    assert current.reasons == ["first"]


def test_expected_match_lands_and_keeps_the_event_link(store) -> None:
    stored = _seed(store)

    store.save_incident(copy_model(stored, reasons=["first", "late"]), expected=stored)

    current = store.get_incident("inc-g")
    assert current.reasons == ["first", "late"]
    assert store.get_incident_by_event("event-g") == current


def test_expected_on_a_missing_row_is_refused(store) -> None:
    incident = fault_incident("inc-missing", "event-missing")

    with pytest.raises(StaleWriteError):
        store.save_incident(incident, expected=incident)

    assert store.get_incident_by_event("event-missing") is None


def test_a_blind_write_cannot_move_the_generation_backwards(store) -> None:
    """A lower token than the row's is by definition a copy read before the
    generation changed; only ``expected`` may rewrite such a row on purpose."""

    stored = _seed(store)
    store.save_incident(copy_model(stored, fencing_token=5))

    with pytest.raises(StaleWriteError, match="fencing_token"):
        store.save_incident(copy_model(stored, fencing_token=4))

    assert store.get_incident("inc-g").fencing_token == 5


def test_save_then_read_round_trips_a_fresh_copy(store) -> None:
    """The ordinary idiom -- read, modify, save -- keeps working."""

    _seed(store)
    for state in (IncidentState.SAFETY_PENDING, IncidentState.RECOVERED):
        store.save_incident(copy_model(store.get_incident("inc-g"), state=state))
        assert store.get_incident("inc-g").state is state
