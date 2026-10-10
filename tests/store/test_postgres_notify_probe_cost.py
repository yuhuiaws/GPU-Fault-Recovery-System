"""The wakeup trigger's "was this key idle" probe costs a few pages whatever
plan the backend has cached (schema v22).

The v21 probe was one ``NOT EXISTS`` anti-join over the transition table.
plpgsql caches that plan per backend at first use; a pooled ingress backend
first ran it while the queue was empty, the planner hashed the queue side
(estimated one row) and probed it with the transition rows, and the plan
stayed until autoanalyze: on the grown backlog a 10-row admission INSERT read
49k shared buffers in 250-320 ms, once per inserted row. The v22 function asks
once per distinct key with a parameterised EXISTS and pins the planner to an
index scan and a nested-loop anti join, so the probe stops at the first PENDING
row that is not its own.

This test lays the same trap -- plan while the table is tiny, grow it by 5,000
rows of the key without any ANALYZE -- and asserts that a 10-row INSERT in the
trapped backend reads about as many pages afterwards as before. The pins are
checked in the catalog; the block counts are recorded as test properties.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
QUEUE = "gpu_fault_processor_queue"
PATH = "/v1/collector-events/gpu-metrics"
PRIORITY = 100
BACKLOG = 5000
PLANNER_PINS = {
    "enable_seqscan=off",
    "enable_bitmapscan=off",
    "enable_hashjoin=off",
    "enable_mergejoin=off",
}


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    _truncate()
    PostgresStore(POSTGRES_URL).close()
    with _connect() as admin:
        admin.execute(f"ALTER TABLE {QUEUE} SET (autovacuum_enabled = false)")
        admin.execute(f"VACUUM ANALYZE {QUEUE}")
    try:
        yield
    finally:
        with _connect() as admin:
            admin.execute(f"ALTER TABLE {QUEUE} RESET (autovacuum_enabled)")
        _truncate()


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _enqueue(writer, rows: int) -> None:
    """One multi-row INSERT: the statement shape an admission flush fires."""

    values = ", ".join(
        ["(%s, 'PENDING', 'cluster-a', %s, %s, now(), now(), %s::jsonb)"] * rows
    )
    parameters: list[object] = []
    for _ in range(rows):
        rid = uuid.uuid4().hex
        parameters.extend(
            (rid, f"lane-{rid[:8]}", PRIORITY, json.dumps({"path": PATH, "id": rid}))
        )
    writer.execute(
        f"INSERT INTO {QUEUE} (request_id, status, cluster_id, ordering_key, "
        f"priority, created_at, updated_at, payload) VALUES {values}",
        parameters,
    )


def _queue_blocks(writer, reader) -> int:
    """Heap and index blocks the queue table has served, flushed on demand."""

    writer.execute("SELECT pg_stat_force_next_flush()")
    (blocks,) = reader.execute(
        "SELECT coalesce(heap_blks_read, 0) + coalesce(heap_blks_hit, 0) "
        "+ coalesce(idx_blks_read, 0) + coalesce(idx_blks_hit, 0) "
        "FROM pg_statio_user_tables WHERE relname = %s",
        (QUEUE,),
    ).fetchone()
    return int(blocks)


def test_the_wakeup_functions_pin_the_planner_to_index_walks() -> None:
    with _connect() as connection:
        for function in (
            "gpu_fault_processor_queue_notify_pending",
            "gpu_fault_telemetry_spool_notify_available",
        ):
            (settings,) = connection.execute(
                "SELECT proconfig FROM pg_proc WHERE proname = %s", (function,)
            ).fetchone()
            assert set(settings or ()) == PLANNER_PINS, function


def test_a_flush_on_a_grown_backlog_reads_about_as_many_pages_as_on_a_small_one(
    record_property,
) -> None:
    with _connect() as writer, _connect() as grower, _connect() as reader:
        # The trap: the writer's backend plans the trigger's statements while
        # the table is tiny (six executions, so plpgsql settles on its generic
        # plan), exactly what a pooled ingress backend does before a burst.
        for _ in range(6):
            _enqueue(writer, rows=3)
        before = _queue_blocks(writer, reader)
        _enqueue(writer, rows=10)
        small = _queue_blocks(writer, reader) - before
        # The burst arrives through other backends; nothing analyses the table.
        for _ in range(BACKLOG // 100):
            _enqueue(grower, rows=100)
        (pending,) = writer.execute(
            f"SELECT count(*) FROM {QUEUE} WHERE status = 'PENDING'"
        ).fetchone()
        assert pending >= BACKLOG
        (reltuples,) = writer.execute(
            "SELECT reltuples FROM pg_class WHERE relname = %s", (QUEUE,)
        ).fetchone()
        assert reltuples <= 0, "the planner must still believe the table is empty"
        before = _queue_blocks(writer, reader)
        _enqueue(writer, rows=10)
        grown = _queue_blocks(writer, reader) - before
    record_property("queue_blocks_small_table", small)
    record_property("queue_blocks_grown_backlog", grown)
    record_property("pending_rows", pending)
    # Ten rows touch the heap and ten indexes either way; the probe adds a
    # handful of index pages. The hashed anti-join read the whole backlog once
    # per inserted row (49k buffers for 10 rows on a 6,000-page table).
    assert grown <= 2 * small + 64, (small, grown)
