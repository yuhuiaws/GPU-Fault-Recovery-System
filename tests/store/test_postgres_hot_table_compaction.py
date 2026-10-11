"""The hot-table bloat policy against the test database.

A round of synthetic churn -- rows inserted, claimed, completed and deleted,
then vacuumed -- leaves the queue's indexes and the counter tables at their
high-water mark with almost nothing live in them. The policy must measure
exactly that from the catalog, rebuild only the objects that crossed the
thresholds, bring their density back, change nothing on a second pass, and
stay out of a schema ensure's or the operator verb's way.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres import hot_table_bloat as policy
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import (
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires GPU_FAULT_TEST_POSTGRES_URL"
)
QUEUE = "gpu_fault_processor_queue"
COUNTS = "gpu_fault_processor_queue_counts"
CHURN_ROWS = 6000
HOST_TELEMETRY = "/v1/collector-events/host-telemetry"


@pytest.fixture(autouse=True)
def current_schema():
    _truncate()
    _ensure_postgres_schema()
    yield
    _truncate()


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _copy_queue_rows(connection, count: int) -> None:
    at = datetime.now(timezone.utc)
    with connection.cursor() as cursor:
        with cursor.copy(
            f"COPY {QUEUE} (request_id, status, cluster_id, correlation_key, "
            "ordering_key, priority, created_at, updated_at, payload) FROM STDIN"
        ) as copy:
            for n in range(count):
                item = processor_request(
                    HOST_TELEMETRY,
                    body=json.dumps(
                        {"node_id": f"node-{n % 512:04d}", "cpu": [n] * 64}
                    ).encode(),
                    cluster_id=f"cluster-{n % 32:02d}",
                ).model_copy(
                    update={
                        "request_id": f"churn-{n:06d}",
                        "created_at": at - timedelta(seconds=count - n),
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


def _churn_and_delete(connection) -> None:
    """One synthetic round: insert, complete (status change, no HOT), delete,
    then vacuum twice so deleted index pages are recycled like autovacuum's."""

    _copy_queue_rows(connection, CHURN_ROWS)
    connection.execute(
        f"UPDATE {QUEUE} SET status='COMPLETED', updated_at=now() WHERE status='PENDING'"
    )
    # The counter tables see one per-writer row per (cluster, shard) at a
    # fillfactor of 50; spread rows over many shards the way 128 backends do.
    connection.execute(
        f"INSERT INTO {COUNTS} (cluster_id, shard_id, incomplete_count, updated_at) "
        "SELECT 'cluster-' || lpad(c::text, 2, '0'), s, 1, now() "
        "FROM generate_series(0, 31) c, generate_series(0, 127) s "
        "ON CONFLICT DO NOTHING"
    )
    for _ in range(40):
        connection.execute(
            f"UPDATE {COUNTS} SET incomplete_count=incomplete_count + 1, updated_at=now()"
        )
    connection.execute(f"DELETE FROM {QUEUE}")
    connection.execute(f"DELETE FROM {COUNTS} WHERE shard_id > 1")
    for _ in range(2):
        connection.execute(f"VACUUM (ANALYZE) {QUEUE}")
        connection.execute(f"VACUUM (ANALYZE) {COUNTS}")


def _facts(store: PostgresStore, table: str) -> dict:
    return next(
        item for item in store.hot_table_bloat()["tables"] if item["table"] == table
    )


def test_a_fresh_schema_is_compact_and_calls_for_nothing() -> None:
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        assessment = store.hot_table_bloat()
    finally:
        store.close()
    assert [item["table"] for item in assessment["tables"]] == list(policy.HOT_TABLES)
    assert assessment["missing_tables"] == []
    assert assessment["actions"] == []
    for item in assessment["tables"]:
        assert item["heap_bloated"] is False
        assert all(index["bloated"] is False for index in item["indexes"]), (
            f"a fresh index of {item['table']} reads as bloated: {item['indexes']}"
        )
        assert item["heap_pages_per_live_row"] >= 0
        assert item["index_pages_per_live_row"] >= 0


def test_churn_bloats_indexes_and_the_policy_rebuilds_exactly_those() -> None:
    with _connect() as connection:
        _churn_and_delete(connection)
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        before = _facts(store, QUEUE)
        bloated = [index["name"] for index in before["indexes"] if index["bloated"]]
        # The partial claim indexes carried an entry per PENDING and per
        # COMPLETED version; every page is empty but still allocated.
        assert before["live_rows"] == 0
        assert before["index_pages"] >= policy.INDEX_MIN_PAGES
        assert bloated, before["indexes"]
        assert before["index_min_entries_per_page"] is not None
        assert before["index_min_entries_per_page"] < policy.INDEX_MIN_ENTRIES_PER_PAGE
        assessment = store.hot_table_bloat()
        reindexes = {
            item["target"]
            for item in assessment["actions"]
            if item["action"] == policy.ACTION_REINDEX and item["table"] == QUEUE
        }
        assert reindexes == set(bloated)
        untouched = {
            index["name"] for index in before["indexes"] if not index["bloated"]
        }
        assert untouched.isdisjoint(reindexes), (
            f"compact indexes scheduled for a rebuild: {untouched & reindexes}"
        )

        result = store.compact_hot_tables(assessment)

        assert result["status"] == "SUCCEEDED", result
        assert {item["target"] for item in result["actions"]} >= reindexes
        assert all(item["status"] == "SUCCEEDED" for item in result["actions"]), result[
            "actions"
        ]
        after = result["after"][QUEUE]
        assert after["index_pages"] < before["index_pages"]
        assert all(index["bloated"] is False for index in after["indexes"]), (
            f"an index is still bloated after the rebuild: {after['indexes']}"
        )
        assert all(index["valid"] for index in after["indexes"]), (
            f"REINDEX CONCURRENTLY left an invalid index: {after['indexes']}"
        )
        # Second pass: nothing is due any more.
        second = store.compact_hot_tables()
        assert second["status"] == "NOOP"
        assert second["actions"] == []
    finally:
        store.close()


def test_a_thinned_counter_table_is_rewritten_in_place() -> None:
    with _connect() as connection:
        _churn_and_delete(connection)
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        before = _facts(store, COUNTS)
        assert before["small"] is True
        assert before["heap_pages"] >= policy.SMALL_HEAP_MIN_PAGES, before
        assert before["heap_bloated"] is True
        assessment = store.hot_table_bloat()
        assert {
            "table": COUNTS,
            "action": policy.ACTION_VACUUM_FULL,
            "target": COUNTS,
        } in assessment["actions"]

        result = store.compact_hot_tables(assessment)

        assert result["status"] == "SUCCEEDED", result
        after = result["after"][COUNTS]
        assert after["heap_pages"] < before["heap_pages"]
        assert after["heap_bloated"] is False
        # ``n_live_tup`` was an estimate before; the rewrite's ANALYZE makes
        # it the exact count, and no row went missing.
        with _connect() as connection:
            exact = connection.execute(f"SELECT count(*) FROM {COUNTS}").fetchone()[0]
        assert after["live_rows"] == exact == 64
    finally:
        store.close()


def test_the_schema_lock_holder_makes_the_job_skip_not_wait() -> None:
    with _connect() as connection:
        _churn_and_delete(connection)
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        assessment = store.hot_table_bloat()
        assert assessment["actions"]
        with _connect() as holder:
            holder.execute(
                "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                (policy.ADVISORY_LOCK_NAME,),
            )
            result = store.compact_hot_tables(assessment)
            holder.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (policy.ADVISORY_LOCK_NAME,),
            )
        assert result["status"] == "SKIPPED"
        assert result["actions"] == []
        # Nothing was rebuilt while the lock was held.
        assert store.hot_table_bloat()["actions"] == assessment["actions"]
    finally:
        store.close()


def test_compaction_refuses_to_run_inside_a_store_transaction() -> None:
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        with store.processor_batch_transaction():
            with pytest.raises(RuntimeError, match="inside a transaction"):
                store.compact_hot_tables({"actions": []})
    finally:
        store.close()


def _timeouts(connection) -> tuple[str, str]:
    return (
        connection.execute("SHOW lock_timeout").fetchone()[0],
        connection.execute("SHOW statement_timeout").fetchone()[0],
    )


def test_the_connection_keeps_its_own_timeouts_after_the_run() -> None:
    """The pool's connections carry the store's statement timeout as a
    startup option and a caller may have set its own: the maintenance values
    must neither leak past the run nor flatten the ones found."""

    with _connect() as connection:
        _churn_and_delete(connection)
        connection.execute("SET lock_timeout = '750ms'")
        connection.execute("SET statement_timeout = '1min'")
        assert _timeouts(connection) == ("750ms", "1min")
        with connection.cursor() as cursor:
            assessment = policy.assess_hot_tables(cursor)
        assert assessment["actions"]

        result = policy.compact(connection, assessment)

        assert result["status"] == "SUCCEEDED", result
        assert _timeouts(connection) == ("750ms", "1min")
        # And the advisory lock is released again.
        assert (
            connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
                (policy.ADVISORY_LOCK_NAME,),
            ).fetchone()[0]
            is True
        )


@pytest.mark.parametrize(
    "pages,entries,expected",
    [
        (policy.INDEX_MIN_PAGES - 1, 0.0, False),  # too small to matter
        (policy.INDEX_MIN_PAGES, -1.0, False),  # never measured
        (policy.INDEX_MIN_PAGES, 0.0, True),  # empty but allocated
        (1000, 7999.0, True),
        (1000, 8000.0, False),
    ],
)
def test_the_index_rule_is_a_density_floor(pages, entries, expected) -> None:
    assert policy.index_is_bloated(pages, entries) is expected


def test_only_the_three_small_tables_are_ever_rewritten() -> None:
    assert set(policy.COMPACT_TABLES) == {
        "gpu_fault_processor_lanes",
        "gpu_fault_processor_queue_counts",
        "gpu_fault_processor_priority_count_shards",
    }
    assert "gpu_fault_gpu_metric_latest" in policy.REINDEX_TABLES
    assert policy.small_heap_is_bloated(policy.SMALL_HEAP_MIN_PAGES - 1, 0) is False
    assert policy.small_heap_is_bloated(policy.SMALL_HEAP_MIN_PAGES, 63) is True
    assert policy.small_heap_is_bloated(policy.SMALL_HEAP_MIN_PAGES, 64) is False
