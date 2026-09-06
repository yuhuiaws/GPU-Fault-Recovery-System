"""The claim window is index-served and admits starved routine rows.

FINAL-建议汇总 F-D2 (P1-76A, P1-76C, P1-77H, P2-76F). The window took the N
best rows by *raw* priority and only then promoted routine rows that had
waited past the starvation threshold, so a backlog of fresher evidence rows
deeper than the window kept a starved routine row out of it forever. And the
window's ORDER BY had no index whose leading columns were free: the
``available`` index leads with ``status, not_before``, both consumed by OR
predicates, so every claim sorted the eligible backlog. The three-step method
of F-J3 applies: operators build ``gpu_fault_processor_queue_priority_claim``
CONCURRENTLY, the DDL declares it IF NOT EXISTS, the schema check fails closed
without it, and only then does the query rely on it.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.ddl import declared_index_names
from tests.store._postgres_processor_claim_support import (
    REQUEST_LEASE,
    _evidence,
    _telemetry,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture
def store():
    assert POSTGRES_URL is not None
    instance = PostgresStore(POSTGRES_URL)
    _truncate()
    try:
        yield instance
    finally:
        instance.close()
        _truncate()


# Planner knobs the plan assertions neutralise. The tests ask "can the index
# serve this ORDER BY?", not "does this planner, on this table size, with this
# ``random_page_cost``, prefer it?" - a 12-row or a 6000-row table under a
# different cost model legitimately sorts a sequential scan, and that choice is
# not a defect of the index. With sequential scans and every sort strategy
# priced out, a Sort node can only appear if no index yields the order.
_PLAN_ONLY_KNOBS = ("enable_seqscan", "enable_sort", "enable_incremental_sort")


def _plan(store: PostgresStore, sql: str, params) -> str:
    with store._db.cursor() as cursor:
        for knob in _PLAN_ONLY_KNOBS:
            cursor.execute(f"SET {knob}=off")
        try:
            cursor.execute("EXPLAIN " + sql, params)
            return "\n".join(row[0] for row in cursor.fetchall())
        finally:
            for knob in _PLAN_ONLY_KNOBS:
                cursor.execute(f"SET {knob}=on")


def _index_state(store: PostgresStore, name: str) -> tuple[bool, str]:
    """``(indisvalid, indexdef)`` for a named index; ``(False, "")`` if absent."""

    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.indisvalid, pg_get_indexdef(i.indexrelid)
            FROM pg_index AS i
            JOIN pg_class AS c ON c.oid = i.indexrelid
            WHERE c.relname = %s
            """,
            (name,),
        )
        row = cursor.fetchone()
    if row is None:
        return False, ""
    return bool(row[0]), str(row[1])


def test_starved_routine_row_enters_a_window_full_of_fresher_evidence(store):
    now = datetime.now(timezone.utc)
    store.claim_window_multiplier = 1
    window_limit = max(1 * store.claim_window_multiplier, 1 + 64)
    for index in range(window_limit + 20):
        store.enqueue_processor_request(_evidence(f"node-evidence-{index:03d}"))
    starved = _telemetry("node-starved").model_copy(
        update={
            "created_at": now - timedelta(seconds=45),
            "updated_at": now - timedelta(seconds=45),
        }
    )
    store.enqueue_processor_request(starved)

    claimed = store.claim_active_processor_requests(
        "pod-window",
        now=now,
        lease_duration=REQUEST_LEASE,
        limit=1,
        routine_starvation_seconds=30,
    )

    assert [item.request_id for item in claimed] == [starved.request_id]


def _seed_bulk_pending(store: PostgresStore, count: int) -> None:
    """Enough PENDING evidence rows that the planner has a real choice."""

    with store._db.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO gpu_fault_processor_queue (
                request_id, status, cluster_id, correlation_key, ordering_key,
                priority, created_at, updated_at, payload
            )
            SELECT
                'processor-bulk-' || n,
                'PENDING',
                'cluster-a',
                'node-bulk-' || n,
                'cluster-a/node-bulk-' || n,
                50,
                now() - (n || ' seconds')::interval,
                now(),
                jsonb_build_object(
                    'request_id', 'processor-bulk-' || n,
                    'path', '/v1/collector-events/host-telemetry',
                    'priority', 50,
                    'status', 'PENDING'
                )
            FROM generate_series(1, %s) AS n
            """,
            (count,),
        )
        cursor.execute("ANALYZE gpu_fault_processor_queue")


def test_claim_window_walks_the_claim_order_index_in_claim_order(store):
    """With a realistic backlog the window is an ordered index walk that stops
    at its LIMIT, not a sort of every eligible row (the shape F-D2 named)."""

    _seed_bulk_pending(store, 6000)
    sql, params = store.claim_active_processor_query(
        "pod-plan",
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=4,
    )

    plan = _plan(store, sql, params)

    assert (
        "Index Scan using gpu_fault_processor_queue_priority_claim "
        "on gpu_fault_processor_queue candidate"
    ) in plan, plan
    assert "Sort Key: candidate.priority" not in plan, plan
    assert "Seq Scan on gpu_fault_processor_queue candidate" not in plan, plan


def test_claim_window_index_is_built_and_valid(store):
    """F-J3 step two: the declared index really exists on the schema the store
    booted, is not left INVALID by an aborted CONCURRENTLY build, and covers
    exactly the window's ORDER BY under the window's status predicate. This
    holds on any planner configuration."""

    valid, definition = _index_state(store, "gpu_fault_processor_queue_priority_claim")

    assert valid, definition or "index is missing"
    assert "(priority, created_at, request_id)" in definition, definition
    assert "WHERE" in definition and "'PENDING'" in definition, definition
    assert "'LEASED'" in definition, definition


def test_claim_window_uses_the_claim_order_index_even_when_small(store):
    for index in range(12):
        store.enqueue_processor_request(_evidence(f"node-plan-{index:03d}"))
    sql, params = store.claim_active_processor_query(
        "pod-plan",
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=4,
    )

    plan = _plan(store, sql, params)

    assert "gpu_fault_processor_queue_priority_claim" in plan, plan
    assert "Seq Scan on gpu_fault_processor_queue candidate" not in plan, plan


def test_declared_indexes_include_the_claim_window_index():
    assert "gpu_fault_processor_queue_priority_claim" in declared_index_names()
    assert "gpu_fault_processor_queue_available" in declared_index_names()
    # The F-D2 twin of ``priority_claim`` is gone: two identical indexes let
    # the planner pick either, which is what made the plan assertions flap.
    assert "gpu_fault_processor_queue_claim_order" not in declared_index_names()
