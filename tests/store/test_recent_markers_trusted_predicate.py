"""``list_recent_markers_for_nodes`` reads only trusted, active markers and
orders by the stored text, so the partial GIN index applies.

Control-plane review 2026-09-08, G-9 (1). The correlation read behind every
provider event filtered ``active='true'`` but not ``trusted='true'``, so the
partial index ``gpu_fault_active_marker_nodes`` (WHERE active AND trusted) could
not serve it and the whole marker kind was scanned; ``ORDER BY
(payload->>'observed_at')::timestamptz`` matched no expression index either.
Untrusted markers are advisory input the correlator never acts on, so the
predicate is a correctness no-op and a plan change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import MarkerScope, NodeMarker, RecoveryAction, Severity
from gpu_fault.store import SqliteStore
from gpu_fault.store.postgres import control_records as postgres_control_records
from tests._builders import build_store
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        instance = SqliteStore(str(tmp_path / "markers.db"))
        try:
            yield instance
        finally:
            instance.close()
        return
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


def _marker(marker_id: str, *, trusted: bool, active: bool = True, observed_at=NOW):
    return NodeMarker(
        marker_id=marker_id,
        source="test",
        cluster_id="cluster-a",
        trusted=trusted,
        active=active,
        incident_id="inc-1",
        observed_at=observed_at,
        expires_at=observed_at + timedelta(minutes=30),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RESET_GPU,
        action_owner="test",
        mapping_version="v1",
    )


def test_untrusted_markers_are_not_returned_and_newest_come_first(store) -> None:
    store.add_marker(_marker("m-untrusted", trusted=False))
    store.add_marker(_marker("m-inactive", trusted=True, active=False))
    store.add_marker(
        _marker("m-older", trusted=True, observed_at=NOW - timedelta(minutes=2))
    )
    store.add_marker(
        _marker("m-newer", trusted=True, observed_at=NOW - timedelta(minutes=1))
    )

    found = store.list_recent_markers_for_nodes({"node-a"}, NOW - timedelta(minutes=5))

    assert [item.marker_id for item in found] == ["m-newer", "m-older"]


class _RecordingCursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def execute(self, sql: str, _params: object = None) -> None:
        self._log.append(sql)

    def fetchall(self) -> list[object]:
        return []


class _RecordingDb:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.statements)


def test_the_postgres_query_has_the_index_predicate_and_text_ordering() -> None:
    """The SQL the Postgres mixin sends must match the partial GIN index it
    relies on: trusted predicate present, no timestamptz cast, text ordering."""

    mixin = postgres_control_records.PostgresControlRecordMixin.__new__(
        postgres_control_records.PostgresControlRecordMixin
    )
    db = _RecordingDb()
    mixin._db = db  # the only collaborator the query touches
    mixin.list_recent_markers_for_nodes({"node-a"}, NOW - timedelta(minutes=5))
    [sql] = db.statements
    assert "payload->>'trusted'='true'" in sql
    assert "::timestamptz" not in sql, "a cast defeats the expression index"
    assert "ORDER BY payload->>'observed_at' DESC" in sql
