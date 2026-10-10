"""The processor's hot transactions do not depend on fresh statistics.

A burst grows ``gpu_fault_processor_queue`` from 45 to 5,800 pages inside a
minute while ``pg_class`` still says 138 rows: autoanalyze is asynchronous, so
the planner keeps choosing plans that read the whole backlog for the claim
window, its correlated ``EXISTS`` probes and the completion's row locking, and
the heap is read hundreds of times a second (``LWLock:BufferContent``). The
hot transactions now start with ``SET LOCAL enable_seqscan = off``. This proves the claim touches only its window under exactly those stale
statistics (and under fresh ones), returns the same rows in the same order,
reads fewer buffers for it, and that the depth reads walk the counts primary
key. The measured buffer counts are recorded as test properties.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.planner_discipline import (
    hot_path_planner_statements,
    pin_hot_path_plans,
)
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import REQUEST_LEASE, _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
QUEUE = "gpu_fault_processor_queue"
COUNTS = "gpu_fault_processor_queue_counts"
HOST_TELEMETRY = "/v1/collector-events/host-telemetry"
GPU_TELEMETRY = "/v1/collector-events/gpu-telemetry"
BACKLOG = 5000
LIMIT = 8
WINDOW = max(LIMIT * 8, LIMIT + 64)
SHAPES = {
    "unfiltered": {},
    "include_path": {"include_paths": {HOST_TELEMETRY}},
    "exclude_paths": {"exclude_paths": {GPU_TELEMETRY}},
}


class _RollBack(Exception):
    """Raised inside a transaction block to discard its effects."""


@pytest.fixture
def store():
    assert POSTGRES_URL is not None
    _truncate()
    PostgresStore(POSTGRES_URL).close()
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()
        _truncate()


@pytest.fixture
def connection():
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connected:
        try:
            yield connected
        finally:
            connected.execute(f"ALTER TABLE {QUEUE} RESET (autovacuum_enabled)")


def _seed_backlog(connection: Any, *, fresh_statistics: bool) -> None:
    """``BACKLOG`` routine PENDING rows, oldest ``processor-bulk-5000``.

    Stale: statistics taken on the empty table, then the rows the planner
    knows nothing about -- the burst's first minute. Fresh: ANALYZE after.
    """

    _truncate()
    connection.execute(f"ALTER TABLE {QUEUE} SET (autovacuum_enabled = false)")
    connection.execute(f"ANALYZE {QUEUE}")
    at = datetime.now(timezone.utc)
    with connection.cursor() as cursor:
        with cursor.copy(
            f"COPY {QUEUE} (request_id, status, cluster_id, correlation_key, "
            "ordering_key, priority, created_at, updated_at, payload) FROM STDIN"
        ) as copy:
            for n in range(1, BACKLOG + 1):
                item = processor_request(
                    HOST_TELEMETRY,
                    body=json.dumps({"node_id": f"node-bulk-{n}"}).encode(),
                    cluster_id="cluster-a",
                ).model_copy(
                    update={
                        "request_id": f"processor-bulk-{n}",
                        "created_at": at - timedelta(seconds=n),
                        "updated_at": at,
                    }
                )
                copy.write_row(
                    (
                        item.request_id,
                        item.status.value,
                        item.cluster_id,
                        item.correlation_key,
                        item.ordering_key(),
                        item.queue_priority(),
                        item.created_at,
                        item.updated_at,
                        item.model_dump_json(),
                    )
                )
    if fresh_statistics:
        connection.execute(f"ANALYZE {QUEUE}")
    row = connection.execute(
        f"SELECT reltuples, relpages FROM pg_class WHERE relname = '{QUEUE}'"
    ).fetchone()
    if fresh_statistics:
        assert row[0] >= BACKLOG * 0.9, f"statistics must be fresh: {row}"
    else:
        assert row[0] <= 0 and row[1] <= 1, f"statistics must be stale: {row}"


def _in_transaction(connection: Any, settings: tuple[str, ...], body: Any) -> Any:
    """Run ``body(connection)`` under ``settings`` in a rolled-back transaction."""

    result: list[Any] = []
    try:
        with connection.transaction():
            for statement in settings:
                connection.execute(statement)
            result.append(body(connection))
            raise _RollBack
    except _RollBack:
        pass
    return result[0]


def _explain(
    connection: Any, sql: str, params: Any, *, settings: tuple[str, ...], analyze: bool
) -> dict:
    options = "ANALYZE, BUFFERS, FORMAT JSON" if analyze else "FORMAT JSON"

    def body(conn: Any) -> dict:
        row = conn.execute(f"EXPLAIN ({options}) {sql}", params).fetchone()
        plan: dict = row[0][0]
        return plan

    return _in_transaction(connection, settings, body)


def _nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nodes(nested)


def _scans(plan: dict, relation: str) -> set[tuple[str, str | None]]:
    return {
        (node["Node Type"], node.get("Index Name"))
        for node in _nodes(plan)
        if node.get("Relation Name") == relation and "Scan" in node.get("Node Type", "")
    }


def _shared_blocks(plan: dict) -> int:
    root = plan["Plan"]
    return int(root.get("Shared Hit Blocks", 0)) + int(
        root.get("Shared Read Blocks", 0)
    )


def _widest_queue_read(plan: dict) -> int:
    """The most rows any node reading the queue produced: the backlog (5,000)
    when the window was sorted out of a full scan, the window when it was
    walked in order."""

    return max(
        (
            int(node.get("Actual Rows", 0)) * int(node.get("Actual Loops", 1))
            for node in _nodes(plan)
            if node.get("Relation Name") == QUEUE
        ),
        default=0,
    )


def test_pin_scopes_the_planner_setting_to_the_transaction(connection: Any) -> None:
    assert hot_path_planner_statements() == ("SET LOCAL enable_seqscan = off",)
    with connection.transaction():
        pin_hot_path_plans(connection)
        assert connection.execute("SHOW enable_seqscan").fetchone() == ("off",)
    assert connection.execute("SHOW enable_seqscan").fetchone() == ("on",), (
        "SET LOCAL ends with the transaction"
    )


@pytest.mark.parametrize("statistics", ["stale", "fresh"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_claim_touches_only_its_window_and_keeps_its_answer(
    store: PostgresStore, connection: Any, shape: str, statistics: str, request
) -> None:
    _seed_backlog(connection, fresh_statistics=statistics == "fresh")
    now = datetime.now(timezone.utc)
    sql, params = store.claim_active_processor_query(
        "pod-plan", now=now, lease_duration=REQUEST_LEASE, limit=LIMIT, **SHAPES[shape]
    )
    pinned = hot_path_planner_statements()

    default_plan = _explain(connection, sql, params, settings=(), analyze=True)
    _seed_backlog(connection, fresh_statistics=statistics == "fresh")
    pinned_plan = _explain(connection, sql, params, settings=pinned, analyze=True)
    before, after = _shared_blocks(default_plan), _shared_blocks(pinned_plan)
    pinned_scans = _scans(pinned_plan, QUEUE)
    request.node.user_properties.extend(
        [
            ("default_shared_blocks", before),
            ("pinned_shared_blocks", after),
            ("default_execution_ms", round(default_plan.get("Execution Time", 0), 1)),
            ("pinned_execution_ms", round(pinned_plan.get("Execution Time", 0), 1)),
            ("default_widest_queue_read", _widest_queue_read(default_plan)),
            ("pinned_widest_queue_read", _widest_queue_read(pinned_plan)),
            ("pinned_queue_scans", json.dumps(sorted(map(str, pinned_scans)))),
        ]
    )

    assert all(kind != "Seq Scan" for kind, _ in pinned_scans), pinned_scans
    window_index = (
        "gpu_fault_processor_queue_path_priority_claim"
        if shape == "include_path"
        else "gpu_fault_processor_queue_priority_claim"
    )
    assert window_index in {name for _, name in pinned_scans}, pinned_scans
    assert all(name for kind, name in pinned_scans if kind != "Bitmap Heap Scan"), (
        f"every queue read is an index read: {pinned_scans}"
    )
    assert _widest_queue_read(pinned_plan) <= WINDOW, (
        "the window is walked in index order and stops at its LIMIT; "
        f"widest read {_widest_queue_read(pinned_plan)} rows"
    )
    assert after <= before, (before, after)

    # Same rows, same order: the default plan's answer (rolled back) against
    # the store's real claim, which pins its plan.
    _seed_backlog(connection, fresh_statistics=statistics == "fresh")

    def claim(conn: Any) -> list[str]:
        rows = conn.execute(sql, params).fetchall()
        return [row[0]["request_id"] for row in rows if row[0] is not None]

    default_rows = _in_transaction(connection, (), claim)
    claimed = store.claim_active_processor_requests(
        "pod-plan", now=now, lease_duration=REQUEST_LEASE, limit=LIMIT, **SHAPES[shape]
    )
    assert [item.request_id for item in claimed] == default_rows
    assert default_rows == [
        f"processor-bulk-{n}" for n in range(BACKLOG, BACKLOG - LIMIT, -1)
    ], "the eight oldest routine rows, oldest first"


COUNTS_SOURCE = f"SELECT cluster_id, incomplete_count FROM {COUNTS}"
PER_SCOPE_DEPTHS = f"""
    WITH counts AS ({COUNTS_SOURCE})
    SELECT cluster_id, coalesce(sum(incomplete_count), 0)
    FROM counts
    WHERE cluster_id=ANY(%s)
    GROUP BY cluster_id
"""
GLOBAL_DEPTH = f"""
    WITH counts AS ({COUNTS_SOURCE})
    SELECT coalesce(sum(incomplete_count), 0)
    FROM counts
"""


def test_depth_reads_walk_the_counts_primary_key(
    store: PostgresStore, connection: Any
) -> None:
    """The admission's per-cluster depth reads (``_processor_counter_depths``,
    issued inside the pinned admission transaction) walk the counts primary
    key instead of summing the table sequentially ~300 times a second."""

    for index in range(3):
        store.enqueue_processor_request(
            processor_request(
                HOST_TELEMETRY,
                body=json.dumps({"node_id": f"node-{index}"}).encode(),
                cluster_id=f"cluster-{index % 2}",
            )
        )
    connection.execute(f"ANALYZE {COUNTS}")
    scopes = ["cluster-0", "cluster-1", "cluster-9"]
    pinned = hot_path_planner_statements()
    plan = _explain(
        connection, PER_SCOPE_DEPTHS, (scopes,), settings=pinned, analyze=False
    )
    scans = _scans(plan, COUNTS)
    assert scans and all(kind != "Seq Scan" for kind, _ in scans), scans
    assert {name for _, name in scans} == {"gpu_fault_processor_queue_counts_pkey"}, (
        scans
    )

    def depths(conn: Any) -> tuple[dict[str, int], int]:
        per_scope = {scope: 0 for scope in scopes}
        per_scope.update(
            {
                cluster_id: int(count)
                for cluster_id, count in conn.execute(
                    PER_SCOPE_DEPTHS, (scopes,)
                ).fetchall()
            }
        )
        return per_scope, int(conn.execute(GLOBAL_DEPTH).fetchone()[0])

    expected = ({"cluster-0": 2, "cluster-1": 1, "cluster-9": 0}, 3)
    assert _in_transaction(connection, pinned, depths) == expected
    assert _in_transaction(connection, (), depths) == expected
