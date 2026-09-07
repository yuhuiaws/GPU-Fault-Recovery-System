"""Retiring a marker keeps why, when and by whom it was retired (I4).

``retire_markers_for_incident`` rewrote the marker with ``active=False`` and
put the reason in a log line only, so the marker table could say a node had
been cleared but never why. The three fields are additive: an older payload
without them still loads, and every backend round-trips them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.models import MarkerScope, NodeMarker, RecoveryAction, Severity
from gpu_fault.store import SqliteStore
from tests._builders import build_store
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 7, 9, 30, tzinfo=timezone.utc)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "markers.db"))
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


def _marker(marker_id: str, incident_id: str = "inc-a") -> NodeMarker:
    return NodeMarker(
        marker_id=marker_id,
        source="test-agent",
        trusted=True,
        incident_id=incident_id,
        observed_at=NOW - timedelta(minutes=5),
        expires_at=NOW + timedelta(minutes=30),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.REBOOT_NODE,
        action_owner="simulated-runtime",
        mapping_version="test-v1",
    )


def test_retiring_records_reason_time_and_actor(store) -> None:
    store.add_marker(_marker("marker-1"))

    retired = retire_markers_for_incident(
        store,
        "inc-a",
        reason="spare passed health checks",
        retired_by="spare-health",
        now=NOW,
    )

    assert retired == 1
    (saved,) = store.list_markers_for_incident("inc-a")
    assert saved.active is False
    assert saved.retired_at == NOW, "the retirement time was not persisted"
    assert saved.retired_reason == "spare passed health checks"
    assert saved.retired_by == "spare-health"


def test_a_second_retirement_keeps_the_first_history(store) -> None:
    store.add_marker(_marker("marker-2"))
    retire_markers_for_incident(
        store, "inc-a", reason="first", retired_by="completion", now=NOW
    )

    again = retire_markers_for_incident(
        store,
        "inc-a",
        reason="second",
        retired_by="spare-health",
        now=NOW + timedelta(minutes=1),
    )

    assert again == 0, "an already retired marker must be skipped"
    (saved,) = store.list_markers_for_incident("inc-a")
    assert saved.retired_reason == "first"
    assert saved.retired_by == "completion"
    assert saved.retired_at == NOW


def test_a_live_marker_carries_no_retirement_fields(store) -> None:
    store.add_marker(_marker("marker-3"))

    (saved,) = store.list_markers_for_incident("inc-a")

    assert saved.active is True
    assert saved.retired_at is None
    assert saved.retired_reason is None
    assert saved.retired_by is None


def test_a_payload_written_before_the_fields_existed_still_loads() -> None:
    payload = json.loads(_marker("marker-old").model_dump_json())
    for field in ("retired_at", "retired_reason", "retired_by"):
        payload.pop(field, None)

    loaded = NodeMarker.model_validate(payload)

    assert loaded.active is True
    assert loaded.retired_at is None
    assert loaded.retired_reason is None
    assert loaded.retired_by is None
