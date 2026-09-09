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
    _request,
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


def _seed_bulk_pending_paths(store: PostgresStore, count_per_path: int) -> None:
    """Three routine paths, ``count_per_path`` PENDING rows each, so a path
    filter has a real backlog to walk past (B-3 / G-3)."""

    with store._db.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO gpu_fault_processor_queue (
                request_id, status, cluster_id, correlation_key, ordering_key,
                priority, created_at, updated_at, payload
            )
            SELECT
                'processor-' || path.tag || '-' || n,
                'PENDING',
                'cluster-a',
                'node-' || path.tag || '-' || n,
                'cluster-a/node-' || path.tag || '-' || n,
                100,
                now() - (n || ' seconds')::interval,
                now(),
                jsonb_build_object(
                    'request_id', 'processor-' || path.tag || '-' || n,
                    'path', path.name,
                    'priority', 100,
                    'status', 'PENDING'
                )
            FROM generate_series(1, %s) AS n
            CROSS JOIN (
                VALUES
                    ('inv', '/v1/collector-events/gpu-inventory'),
                    ('met', '/v1/collector-events/gpu-metrics'),
                    ('host', '/v1/collector-events/host-telemetry')
            ) AS path(tag, name)
            """,
            (count_per_path,),
        )
        cursor.execute("ANALYZE gpu_fault_processor_queue")


@pytest.mark.parametrize(
    "include_paths",
    [
        {"/v1/collector-events/gpu-inventory"},
        {"/v1/collector-events/gpu-inventory", "/v1/collector-events/gpu-metrics"},
    ],
    ids=["single-path", "two-paths"],
)
def test_claim_window_with_include_paths_walks_the_path_index_in_order(
    store, include_paths
):
    """Every dedicated pool claims with ``include_paths`` (7 of the 8
    sub-claims per cycle). Their window must be an ordered walk of
    ``gpu_fault_processor_queue_path_priority_claim`` per path that stops at
    the LIMIT - not a Bitmap scan of the whole path backlog followed by a
    top-N sort, whose cost grows with depth (B-3 / G-3)."""

    _seed_bulk_pending_paths(store, 2000)
    sql, params = store.claim_active_processor_query(
        "pod-plan",
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=16,
        include_paths=include_paths,
    )

    plan = _plan(store, sql, params)
    # The interlock subplans under candidate_ids legitimately probe the
    # path index with an IN list; the assertions are about the window.
    window_plan = plan.split("CTE candidate_ids")[0]

    assert "Sort Key: candidate.priority" not in window_plan, plan
    assert "Sort Key: candidate.created_at" not in window_plan, plan
    assert "Bitmap Index Scan" not in window_plan, plan
    assert (
        "Index Scan using gpu_fault_processor_queue_path_priority_claim "
        "on gpu_fault_processor_queue candidate"
    ) in window_plan, plan
    assert "Seq Scan on gpu_fault_processor_queue candidate" not in window_plan, plan
    assert "= ANY (" not in window_plan, (
        "the path filter must be an equality per sub-window, not = ANY(array)"
    )
    for path in include_paths:
        assert f"(payload ->> 'path'::text) = '{path}'::text" in window_plan, plan


def test_claim_with_include_paths_still_returns_the_oldest_rows_first(store):
    """The per-path sub-windows must not change what is claimed: the oldest
    routine rows of the named path, and only that path."""

    now = datetime.now(timezone.utc)
    metrics_ids: list[str] = []
    for index in range(6):
        created_at = now - timedelta(seconds=60 - index)
        item = _request(
            "/v1/collector-events/gpu-metrics",
            body=('{"node_id":"node-met-%d","summary":true}' % index).encode(),
        ).model_copy(update={"created_at": created_at, "updated_at": created_at})
        store.enqueue_processor_request(item)
        metrics_ids.append(item.request_id)
        store.enqueue_processor_request(_evidence(f"node-host-{index}"))

    claimed = store.claim_active_processor_requests(
        "pod-order",
        now=now,
        lease_duration=REQUEST_LEASE,
        limit=4,
        include_paths={"/v1/collector-events/gpu-metrics"},
    )

    assert [item.path for item in claimed] == ["/v1/collector-events/gpu-metrics"] * 4
    assert [item.request_id for item in claimed] == metrics_ids[:4]


def test_claim_with_exclude_paths_keeps_the_priority_index_walk(store):
    _seed_bulk_pending(store, 6000)
    sql, params = store.claim_active_processor_query(
        "pod-plan",
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=4,
        exclude_paths={
            "/v1/collector-events/gpu-inventory",
            "/v1/collector-events/gpu-metrics",
        },
    )

    plan = _plan(store, sql, params)

    assert (
        "Index Scan using gpu_fault_processor_queue_priority_claim "
        "on gpu_fault_processor_queue candidate"
    ) in plan, plan
    assert "Sort Key: candidate.priority" not in plan, plan
