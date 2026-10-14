"""Edge-triggered, statement-level NOTIFY on the processor queue and the
telemetry spool (schema v21).

The first PENDING row of a (path, priority) key notifies; a second row for a
key that still has PENDING work does not; once the key drains, the next row
notifies again. Concurrent admission flushes on a busy key produce no
notifying transaction at all, and concurrent first flushes of an idle key
produce at most one -- the rest rely on the woken claim loop, which polls
until its streams are empty.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)

PATH = "/v1/gpu-metrics"
PRIORITY = 50
WRITERS = 12


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    _truncate()
    PostgresStore(POSTGRES_URL).close()
    yield
    _truncate()


def _connect(**kwargs):
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True, **kwargs)


def _notifications(listener, timeout: float = 1.0) -> list[dict]:
    return [
        json.loads(notification.payload)
        for notification in listener.notifies(timeout=timeout, stop_after=64)
    ]


def _enqueue(
    writer, *, path: str = PATH, priority: int = PRIORITY, rows: int = 1
) -> list[str]:
    ids = [uuid.uuid4().hex for _ in range(rows)]
    values = ", ".join(
        ["(%s, 'PENDING', 'cluster-a', %s, %s, now(), now(), %s::jsonb)"] * rows
    )
    parameters: list[object] = []
    for rid in ids:
        parameters.extend(
            (rid, f"lane-{rid[:6]}", priority, json.dumps({"path": path}))
        )
    # One statement per flush, the admission shape the statement trigger sees.
    writer.execute(
        "INSERT INTO gpu_fault_processor_queue "
        "(request_id, status, cluster_id, ordering_key, priority, created_at, "
        f"updated_at, payload) VALUES {values}",
        parameters,
    )
    return ids


def _complete(writer, ids: list[str]) -> None:
    writer.execute(
        "UPDATE gpu_fault_processor_queue SET status='COMPLETED', updated_at=now() "
        "WHERE request_id = ANY(%s)",
        (ids,),
    )


def test_the_first_pending_row_of_a_key_notifies_once_and_a_busy_key_stays_quiet() -> (
    None
):
    with _connect() as listener, _connect() as writer:
        listener.execute("LISTEN gpu_fault_processor_queue")

        first = _enqueue(writer, rows=3)
        edge = _notifications(listener)
        assert edge == [{"priority": PRIORITY, "path": PATH}], edge

        second = _enqueue(writer, rows=2)
        assert _notifications(listener, timeout=0.5) == [], (
            "a key that already has PENDING work needs no second wakeup"
        )

        _complete(writer, first)  # the second batch is still PENDING
        third = _enqueue(writer)
        assert _notifications(listener, timeout=0.5) == []

        _complete(writer, second + third)  # the key drains
        _enqueue(writer)
        assert _notifications(listener) == [{"priority": PRIORITY, "path": PATH}]


def test_another_key_and_a_row_returning_to_pending_are_their_own_edges() -> None:
    with _connect() as listener, _connect() as writer:
        listener.execute("LISTEN gpu_fault_processor_queue")
        ids = _enqueue(writer)
        _notifications(listener)
        # A different (path, priority) key: its own edge.
        _enqueue(writer, path="/v1/gpu-events/xid", priority=0)
        assert _notifications(listener) == [
            {"priority": 0, "path": "/v1/gpu-events/xid"}
        ]
        # LEASED -> PENDING (a release) on an otherwise drained key notifies.
        writer.execute(
            "UPDATE gpu_fault_processor_queue SET status='LEASED' WHERE request_id=ANY(%s)",
            (ids,),
        )
        assert _notifications(listener, timeout=0.5) == [], "a lease is not an edge"
        writer.execute(
            "UPDATE gpu_fault_processor_queue SET status='PENDING' WHERE request_id=ANY(%s)",
            (ids,),
        )
        assert _notifications(listener) == [{"priority": PRIORITY, "path": PATH}]


def test_a_leased_row_does_not_silence_the_next_row_of_its_key() -> None:
    """The fault stream's backlog probe skips rows under a live lease
    (claim_backoff.py) because the busy probe reads PENDING rows only: the
    next row of a key whose only other row is LEASED is its own edge."""

    xid = {"path": "/v1/gpu-events/xid", "priority": 10}
    with _connect() as listener, _connect() as writer:
        listener.execute("LISTEN gpu_fault_processor_queue")
        ids = _enqueue(writer, **xid)
        assert _notifications(listener) == [xid]
        writer.execute(
            "UPDATE gpu_fault_processor_queue SET status='LEASED', "
            "lease_expires_at=now() + interval '2 minutes' "
            "WHERE request_id=ANY(%s)",
            (ids,),
        )
        assert _notifications(listener, timeout=0.5) == [], "a lease is not an edge"
        _enqueue(writer, **xid)
        assert _notifications(listener) == [xid], "a LEASED row silences no edge"


def _flush_concurrently(rows_each: int) -> None:
    ready = threading.Barrier(WRITERS)
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            with _connect() as connection:
                ready.wait(timeout=30)
                with connection.transaction():
                    _enqueue(connection, rows=rows_each)
        except BaseException as exc:  # noqa: BLE001 - every failure is the verdict
            errors.append(exc)

    threads = [threading.Thread(target=writer, daemon=True) for _ in range(WRITERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == [], errors


def test_concurrent_flushes_on_a_busy_key_produce_no_notifying_transaction() -> None:
    with _connect() as listener, _connect() as writer:
        listener.execute("LISTEN gpu_fault_processor_queue")
        _enqueue(writer)  # the key is busy before the burst
        assert len(_notifications(listener)) == 1

        _flush_concurrently(rows_each=3)

        assert _notifications(listener, timeout=1.0) == [], (
            "no flush on a busy key may issue NOTIFY (COMMIT serialization)"
        )
        (pending,) = writer.execute(
            "SELECT count(*) FROM gpu_fault_processor_queue WHERE status='PENDING'"
        ).fetchone()
        assert pending == 1 + WRITERS * 3


def test_concurrent_first_flushes_of_an_idle_key_produce_at_most_one_notifier() -> None:
    with _connect() as listener:
        listener.execute("LISTEN gpu_fault_processor_queue")

        _flush_concurrently(rows_each=3)

        edges = _notifications(listener, timeout=1.0)
        assert len(edges) <= 1, edges
        assert edges == [{"priority": PRIORITY, "path": PATH}], (
            "the one winner of the try-only advisory lock announces the edge"
        )


def _spool_insert(writer, key: str, available_at: datetime, path: str = PATH) -> None:
    writer.execute(
        """
        INSERT INTO gpu_fault_telemetry_spool
            (spool_key, cluster_id, path, request_id, available_at,
             created_at, updated_at, payload)
        VALUES (%s, 'cluster-a', %s, %s, %s, now(), now(), '{}'::jsonb)
        ON CONFLICT (spool_key) DO UPDATE SET
            revision = gpu_fault_telemetry_spool.revision + 1,
            payload = excluded.payload,
            updated_at = now()
        """,
        (key, path, key, available_at),
    )


def test_spool_rows_notify_on_the_edge_of_a_claimable_path_only() -> None:
    with _connect() as listener, _connect() as writer:
        listener.execute("LISTEN gpu_fault_telemetry_spool")
        now = datetime.now(UTC)

        _spool_insert(writer, "lane-1", now - timedelta(seconds=1))
        assert _notifications(listener) == [{"path": PATH}]

        _spool_insert(writer, "lane-2", now - timedelta(seconds=1))
        assert _notifications(listener, timeout=0.5) == [], (
            "lane-1 is still claimable: the consumer is awake or about to be"
        )

        # The consumer leases both rows (available_at into the future): quiet.
        writer.execute(
            "UPDATE gpu_fault_telemetry_spool SET lease_owner='w-1', "
            "available_at=now() + interval '2 minutes'"
        )
        assert _notifications(listener, timeout=0.5) == []

        # Nothing claimable remains: the next row is an edge again ...
        _spool_insert(writer, "lane-3", now - timedelta(seconds=1))
        assert _notifications(listener) == [{"path": PATH}]
        # ... and so is abandoning a leased row while lane-3 is leased too.
        writer.execute(
            "UPDATE gpu_fault_telemetry_spool SET lease_owner='w-1', "
            "available_at=now() + interval '2 minutes' WHERE spool_key='lane-3'"
        )
        writer.execute(
            "UPDATE gpu_fault_telemetry_spool SET lease_owner=NULL, available_at=now() "
            "WHERE spool_key='lane-1'"
        )
        assert _notifications(listener) == [{"path": PATH}]
