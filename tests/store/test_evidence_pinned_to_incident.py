"""Raw evidence an open incident still refers to survives the TTL sweep.

Architecture review 2026-09-07, item D6. ``RawEvidenceRecord`` rows expire
after 24 h (``EvidenceService.retention``) while incidents live forever, so by
the time an operator opens an ESCALATED or QUARANTINED incident the evidence
captured around its fault is gone. The evidence a record binds to an incident
is not a stored pointer: ``NodeMarker.raw_evidence_ref`` and
``NodeHealthFinding.evidence_ref`` carry collector URIs (``prometheus://``,
``nvidia-smi://``, ``journal://``), never a record id. What does bind them is
the capture itself: the record names the attempt that was running
(``attempt_ids``) and the node it came from, and the incident names its attempt
and its nodes. The sweep therefore keeps a record while a non-RECOVERED incident
of the same cluster names its attempt, or names its node and was created within
``EVIDENCE_PIN_WINDOW`` of the observation; once the incident is RECOVERED the
record expires like any other.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.evidence_pins import EVIDENCE_PIN_WINDOW
from gpu_fault.telemetry import EvidenceKind, EvidenceService
from tests._builders import build_store, copy_model, fault_incident
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime.now(timezone.utc).replace(microsecond=0)
SWEEP_AT = NOW + timedelta(hours=3)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "evidence-pins.db"))
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


def _capture(
    store,
    record_id: str,
    *,
    node_id: str = "node-a",
    attempt_ids: list[str] | None = None,
    observed_at: datetime = NOW,
) -> None:
    EvidenceService(store, retention=timedelta(hours=1)).capture(
        record_id=record_id,
        cluster_id="cluster-a",
        node_id=node_id,
        kind=EvidenceKind.GPU_METRICS,
        observed_at=observed_at,
        attempt_ids=["attempt-a"] if attempt_ids is None else attempt_ids,
        payload={"record": record_id},
    )


def _incident(
    store,
    incident_id: str = "inc-open",
    *,
    state: IncidentState = IncidentState.ESCALATED,
    node_ids: list[str] | None = None,
    attempt_id: str | None = "attempt-a",
    created_at: datetime = NOW,
    cluster_id: str = "cluster-a",
):
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        cluster_id=cluster_id,
        node_ids=["node-a"] if node_ids is None else node_ids,
        state=state,
        attempt_id=attempt_id,
        created_at=created_at,
        updated_at=created_at,
    )
    store.save_incident(incident)
    return incident


def _remaining(store) -> list[str]:
    return sorted(
        item.record_id for item in store.list_raw_evidence("cluster-a", limit=100)
    )


def test_evidence_of_an_open_incident_survives_the_sweep(store) -> None:
    _capture(store, "ev-pinned")
    _incident(store)

    deleted = store.cleanup_expired_raw_evidence(now=SWEEP_AT)

    assert deleted == 0
    assert _remaining(store) == ["ev-pinned"]


def test_the_same_evidence_expires_once_the_incident_is_recovered(store) -> None:
    _capture(store, "ev-pinned")
    incident = _incident(store)
    store.save_incident(copy_model(incident, state=IncidentState.RECOVERED))

    deleted = store.cleanup_expired_raw_evidence(now=SWEEP_AT)

    assert deleted == 1
    assert _remaining(store) == []


def test_the_attempt_reference_pins_evidence_from_another_node(store) -> None:
    """A job incident on node-b still needs what node-a saw during its attempt."""

    _capture(store, "ev-attempt", node_id="node-a", attempt_ids=["attempt-a"])
    _incident(store, node_ids=["node-b"], attempt_id="attempt-a")

    assert store.cleanup_expired_raw_evidence(now=SWEEP_AT) == 0
    assert _remaining(store) == ["ev-attempt"]


def test_the_node_reference_is_bounded_by_the_pin_window(store) -> None:
    """Node-scoped pinning would otherwise keep every record of a node with a
    long-open incident: only the window around the incident's lifetime holds."""

    _capture(store, "ev-in-window", attempt_ids=[])
    _capture(
        store,
        "ev-long-before",
        attempt_ids=[],
        observed_at=NOW - EVIDENCE_PIN_WINDOW - timedelta(minutes=5),
    )
    _incident(store, attempt_id=None)

    deleted = store.cleanup_expired_raw_evidence(now=SWEEP_AT)

    assert deleted == 1
    assert _remaining(store) == ["ev-in-window"]


def test_an_incident_elsewhere_pins_nothing(store) -> None:
    _capture(store, "ev-other-node", node_id="node-z", attempt_ids=["attempt-z"])
    _incident(store, "inc-other-cluster", cluster_id="cluster-b")
    _incident(store, "inc-other-node", node_ids=["node-a"], attempt_id="attempt-a")

    deleted = store.cleanup_expired_raw_evidence(now=SWEEP_AT)

    assert deleted == 1
    assert _remaining(store) == []


def test_unexpired_and_pinned_rows_do_not_starve_the_limit(store) -> None:
    """The bound is on rows deleted; pinned rows are skipped, not counted."""

    _capture(store, "ev-pinned")
    _incident(store)
    for index in range(3):
        _capture(store, f"ev-free-{index}", node_id="node-free", attempt_ids=[])

    assert store.cleanup_expired_raw_evidence(now=SWEEP_AT, limit=2) == 2
    assert store.cleanup_expired_raw_evidence(now=SWEEP_AT, limit=2) == 1
    assert _remaining(store) == ["ev-pinned"]
