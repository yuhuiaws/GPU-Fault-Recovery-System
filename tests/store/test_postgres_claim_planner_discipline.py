"""The claim transactions do not depend on fresh statistics.

A burst grows ``gpu_fault_processor_queue`` from 45 to 5,800 pages inside a
minute while ``pg_class`` still says 138 rows: autoanalyze is asynchronous, so
the planner keeps choosing plans that read the whole backlog for the claim
window and its correlated ``EXISTS`` probes, and the heap is read hundreds of
times a second (``LWLock:BufferContent``). The claim transactions -- and only
those; the admission and completion transactions fire statement-level
triggers over unindexed transition tables that the same setting would degrade
-- now start with ``SET LOCAL enable_seqscan = off``. This proves the claim
touches only its window under exactly those statistics (138 rows analysed,
then 4,862 more the planner knows nothing about) and under fresh ones, and
returns the same rows in the same order. Both statistics states are produced
inside the test (explicit ``ANALYZE`` with autovacuum disabled on the table)
and verified from ``pg_class`` before anything is measured; buffer and row
counts are recorded as test properties, never compared as absolutes.

The fault stream's backlog probe runs outside any claim transaction and takes
no setting: it is written as the claim window's ordered walk, which stays an
index walk under the generic plan the plan cache may keep once psycopg has
prepared the statement (after five executions).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.processor.models import RESERVED_TIER_MAX_PRIORITY
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.planner_discipline import (
    hot_path_planner_statements,
    pin_hot_path_plans,
)
from gpu_fault.store.postgres.processor_storage import UNLEASED_FAULT_BACKLOG_SQL
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import REQUEST_LEASE, _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
QUEUE = "gpu_fault_processor_queue"
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


ANALYZED_ROWS = 138  # pg_class said reltuples 138 / relpages 45 during the burst


def _copy_rows(connection: Any, first: int, last: int, at: datetime) -> None:
    with connection.cursor() as cursor:
        with cursor.copy(
            f"COPY {QUEUE} (request_id, status, cluster_id, correlation_key, "
            "ordering_key, priority, created_at, updated_at, payload) FROM STDIN"
        ) as copy:
            for n in range(first, last + 1):
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


def _seed_backlog(connection: Any, *, statistics: str) -> None:
    """``BACKLOG`` routine PENDING rows, oldest ``processor-bulk-5000``, with
    the planner's statistics fixed by the test itself.

    ``stale``: the ``ANALYZED_ROWS`` oldest rows are analysed, then the rest
    arrive without an ANALYZE -- the burst's first minute, ``reltuples`` still
    at 138. ``fresh``: every row analysed. Autovacuum is switched off for the
    table so no autoanalyze can move the statistics under the measurement, and
    ``pg_class`` is checked before anything is measured. (An ANALYZE of the
    empty table is not a stale state: it leaves no column statistics at all on
    a database that never had any, and the planner's choice among index paths
    is then arbitrary -- that is the state of a brand-new table before its first
    autoanalyze, not of a queue under load.)
    """

    _truncate()
    connection.execute(f"ALTER TABLE {QUEUE} SET (autovacuum_enabled = false)")
    at = datetime.now(timezone.utc)
    if statistics == "stale":
        _copy_rows(connection, BACKLOG - ANALYZED_ROWS + 1, BACKLOG, at)
        connection.execute(f"ANALYZE {QUEUE}")
        _copy_rows(connection, 1, BACKLOG - ANALYZED_ROWS, at)
        expected = ANALYZED_ROWS
    else:
        _copy_rows(connection, 1, BACKLOG, at)
        connection.execute(f"ANALYZE {QUEUE}")
        expected = BACKLOG
    reltuples, statistics_rows = connection.execute(
        f"""
        SELECT reltuples,
               (SELECT count(*) FROM pg_statistic WHERE starelid = '{QUEUE}'::regclass)
        FROM pg_class WHERE relname = '{QUEUE}'
        """
    ).fetchone()
    assert int(reltuples) == expected, f"statistics must be {statistics}: {reltuples}"
    assert statistics_rows > 0, "the analysed rows left column statistics"
    assert connection.execute(f"SELECT count(*) FROM {QUEUE}").fetchone()[0] == BACKLOG


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
    _seed_backlog(connection, statistics=statistics)
    now = datetime.now(timezone.utc)
    sql, params = store.claim_active_processor_query(
        "pod-plan", now=now, lease_duration=REQUEST_LEASE, limit=LIMIT, **SHAPES[shape]
    )
    pinned = hot_path_planner_statements()

    default_plan = _explain(connection, sql, params, settings=(), analyze=True)
    _seed_backlog(connection, statistics=statistics)
    pinned_plan = _explain(connection, sql, params, settings=pinned, analyze=True)
    pinned_scans = _scans(pinned_plan, QUEUE)
    # Evidence, recorded rather than compared: block and row counts move with
    # the planner's exact choice and the page layout; the invariant below does
    # not.
    request.node.user_properties.extend(
        [
            ("default_shared_blocks", _shared_blocks(default_plan)),
            ("pinned_shared_blocks", _shared_blocks(pinned_plan)),
            ("default_widest_queue_read", _widest_queue_read(default_plan)),
            ("pinned_widest_queue_read", _widest_queue_read(pinned_plan)),
            ("default_execution_ms", round(default_plan.get("Execution Time", 0), 1)),
            ("pinned_execution_ms", round(pinned_plan.get("Execution Time", 0), 1)),
            ("pinned_queue_scans", json.dumps(sorted(map(str, pinned_scans)))),
        ]
    )

    # The invariant: no heap scan of the queue, every read of it index-backed,
    # and the window walked in index order so no node touches more rows than
    # the window itself.
    assert all(kind != "Seq Scan" for kind, _ in pinned_scans), pinned_scans
    assert all(name for kind, name in pinned_scans if kind != "Bitmap Heap Scan"), (
        f"every queue read is an index read: {pinned_scans}"
    )
    if shape == "include_path":
        assert "gpu_fault_processor_queue_path_claim_window" in {
            name for _, name in pinned_scans
        }, pinned_scans
    assert _widest_queue_read(pinned_plan) <= WINDOW, (
        "the window is walked in index order and stops at its LIMIT; "
        f"widest read {_widest_queue_read(pinned_plan)} rows"
    )

    # Same rows, same order: the default plan's answer (rolled back) against
    # the store's real claim, which pins its plan.
    _seed_backlog(connection, statistics=statistics)

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


def _generic_plan(connection: Any, sql: str, *, now: datetime) -> dict:
    """EXPLAIN ANALYZE of ``sql`` (``%(max_priority)s``, ``%(now)s``) as the
    plan cache's generic plan, in a rolled-back transaction."""

    from psycopg import sql as pgsql

    prepared = sql.replace("%(max_priority)s", "$1").replace("%(now)s", "$2")

    def body(conn: Any) -> dict:
        conn.execute("SET LOCAL plan_cache_mode = force_generic_plan")
        conn.execute(f"PREPARE backlog_probe(integer, timestamptz) AS {prepared}")
        try:
            row = conn.execute(
                pgsql.SQL(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
                    "EXECUTE backlog_probe({}, {})"
                ).format(pgsql.Literal(RESERVED_TIER_MAX_PRIORITY), pgsql.Literal(now))
            ).fetchone()
        finally:
            conn.execute("DEALLOCATE backlog_probe")
        plan: dict = row[0][0]
        return plan

    return _in_transaction(connection, (), body)


@pytest.mark.parametrize("statistics", ["stale", "fresh"])
def test_the_fault_backlog_probe_walks_the_claim_window_under_a_generic_plan(
    connection: Any, statistics: str, request
) -> None:
    # Five thousand routine rows and no fault row: the probe's worst case,
    # where a heap scan reads the whole queue to answer "no".
    _seed_backlog(connection, statistics=statistics)
    now = datetime.now(timezone.utc)
    plan = _generic_plan(connection, UNLEASED_FAULT_BACKLOG_SQL, now=now)
    scans = _scans(plan, QUEUE)
    # The same predicate as an EXISTS, which the generic plan priced as a heap
    # scan that stops early: recorded as evidence, not asserted.
    exists = _generic_plan(
        connection,
        "SELECT EXISTS (SELECT 1 FROM gpu_fault_processor_queue "
        "WHERE priority <= %(max_priority)s AND (status='PENDING' OR "
        "(status='LEASED' AND lease_expires_at <= %(now)s)))",
        now=now,
    )
    request.node.user_properties.extend(
        [
            ("probe_shared_blocks", _shared_blocks(plan)),
            ("probe_execution_ms", round(plan.get("Execution Time", 0), 2)),
            ("probe_queue_scans", json.dumps(sorted(map(str, scans)))),
            ("exists_shared_blocks", _shared_blocks(exists)),
            ("exists_execution_ms", round(exists.get("Execution Time", 0), 2)),
            ("exists_queue_scans", json.dumps(sorted(map(str, _scans(exists, QUEUE))))),
        ]
    )

    assert scans == {("Index Scan", "gpu_fault_processor_queue_claim_window")}, (
        f"the probe walks the claim window, not the heap: {scans}"
    )
    assert _widest_queue_read(plan) == 0, "no fault row waits, and none was read"
