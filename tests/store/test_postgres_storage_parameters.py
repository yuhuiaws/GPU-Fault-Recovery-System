"""Schema v19: the hot tables' storage parameters and HOT-eligible updates.

A day of campaign churn left ``gpu_fault_gpu_metric_latest`` with 1,216 live
rows in 23,435 heap pages and ``gpu_fault_processor_lanes`` with 40 rows in
1,188 pages because no update was HOT (``n_tup_hot_upd`` 0) and autovacuum
ran at the instance defaults. v19 gives the hot tables fillfactor and
threshold-only autovacuum and retires the two timestamp indexes whose only
reader was a retention sweep, so the metric upsert and the lane updates stay
on their heap page.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource
from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.ddl import declared_index_names
from gpu_fault.store.postgres.ddl_storage_parameters import (
    HOT_TABLE_STORAGE_PARAMETERS,
    set_storage_parameters_if_different,
)
from tests.store._postgres_processor_claim_support import (
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires GPU_FAULT_TEST_POSTGRES_URL"
)

RETIRED_INDEXES = (
    "gpu_fault_gpu_metric_latest_observed",
    "gpu_fault_processor_lanes_expiry",
)
HOT_STAT_SQL = (
    "SELECT n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables WHERE relname=%s"
)


@pytest.fixture(autouse=True)
def current_schema():
    _truncate()
    _ensure_postgres_schema()
    yield
    _truncate()


def _connect(**kwargs):
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, **kwargs)


def _reloptions(cursor, table: str) -> tuple[set[str], set[str]]:
    cursor.execute(
        "SELECT c.reloptions, t.reloptions FROM pg_class c "
        "LEFT JOIN pg_class t ON t.oid=c.reltoastrelid "
        "WHERE c.oid=to_regclass(%s)",
        (table,),
    )
    main, toast = cursor.fetchone()
    return set(main or []), set(toast or [])


def _hot_counters(connection, table: str) -> tuple[int, int]:
    deadline = time.monotonic() + 5
    while True:
        connection.execute("SELECT pg_stat_clear_snapshot()")
        updates, hot = connection.execute(HOT_STAT_SQL, (table,)).fetchone()
        if updates >= 20 or time.monotonic() >= deadline:
            return updates, hot
        time.sleep(0.05)


def test_every_hot_table_carries_its_storage_parameters() -> None:
    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        for table, options in HOT_TABLE_STORAGE_PARAMETERS.items():
            main, toast = _reloptions(cursor, table)
            expected_main = {
                f"{name}={value}"
                for name, value in options.items()
                if not name.startswith("toast.")
            }
            expected_toast = {
                f"{name.removeprefix('toast.')}={value}"
                for name, value in options.items()
                if name.startswith("toast.")
            }
            assert expected_main <= main, (table, sorted(main))
            assert expected_toast <= toast, (table, sorted(toast))
            assert "autovacuum_vacuum_scale_factor=0" in main, table
            assert "autovacuum_vacuum_cost_delay=0" in main, table


def test_fillfactor_only_where_rows_are_rewritten_in_place() -> None:
    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        for table in (
            "gpu_fault_processor_lanes",
            "gpu_fault_processor_queue_counts",
            "gpu_fault_processor_priority_count_shards",
        ):
            assert "fillfactor=50" in _reloptions(cursor, table)[0], table
        for table in ("gpu_fault_processor_queue", "gpu_fault_gpu_metric_latest"):
            assert "fillfactor=70" in _reloptions(cursor, table)[0], table
        batches, _ = _reloptions(cursor, "gpu_fault_gpu_metrics_batches")
        assert not any(item.startswith("fillfactor=") for item in batches), (
            "insert-only idempotency records gain nothing from page slack"
        )
        _, queue_toast = _reloptions(cursor, "gpu_fault_processor_queue")
        assert {
            "autovacuum_vacuum_scale_factor=0",
            "autovacuum_vacuum_threshold=50",
        } <= queue_toast


class _Recording:
    def __init__(self, cursor) -> None:
        self.cursor = cursor
        self.statements: list[str] = []

    def execute(self, query, params=None):
        self.statements.append(" ".join(str(query).split()))
        return self.cursor.execute(query, params)

    def fetchone(self):
        return self.cursor.fetchone()


def test_a_matching_catalog_issues_no_alter() -> None:
    """Item J: the ALTER takes ShareUpdateExclusiveLock even as a no-op, so a
    second ensure must read the catalog and stop."""

    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        recording = _Recording(cursor)
        for table, options in HOT_TABLE_STORAGE_PARAMETERS.items():
            set_storage_parameters_if_different(recording, table, options)
    assert recording.statements, "the catalog must be read"
    assert not [s for s in recording.statements if s.startswith("ALTER")], (
        recording.statements
    )


def test_a_changed_or_missing_option_is_reapplied_including_toast() -> None:
    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE gpu_fault_processor_queue RESET "
            "(fillfactor, toast.autovacuum_vacuum_threshold)"
        )
        cursor.execute("ALTER TABLE gpu_fault_processor_lanes SET (fillfactor=90)")
        recording = _Recording(cursor)
        for table in ("gpu_fault_processor_queue", "gpu_fault_processor_lanes"):
            set_storage_parameters_if_different(
                recording, table, HOT_TABLE_STORAGE_PARAMETERS[table]
            )
        alters = [s for s in recording.statements if s.startswith("ALTER")]
        assert alters == [
            "ALTER TABLE gpu_fault_processor_queue SET "
            "(fillfactor=70, toast.autovacuum_vacuum_threshold=50)",
            "ALTER TABLE gpu_fault_processor_lanes SET (fillfactor=50)",
        ]
        main, toast = _reloptions(cursor, "gpu_fault_processor_queue")
        assert "fillfactor=70" in main and "autovacuum_vacuum_threshold=50" in toast


@pytest.mark.parametrize(
    "options",
    [
        {"fillfactor; DROP TABLE x": "70"},
        {"fillfactor": "70 OR 1"},
        {"toast.autovacuum_enabled": "off"},
    ],
)
def test_unsafe_options_are_refused(options: dict[str, str]) -> None:
    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        with pytest.raises(ValueError, match="unsafe table option"):
            set_storage_parameters_if_different(cursor, "gpu_fault_objects", options)


def test_the_retired_timestamp_indexes_are_neither_declared_nor_present() -> None:
    names = set(declared_index_names())
    assert not names & set(RETIRED_INDEXES), sorted(names & set(RETIRED_INDEXES))
    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname = ANY(%s)",
            (list(RETIRED_INDEXES),),
        )
        assert cursor.fetchall() == []


def test_a_v18_database_loses_the_retired_indexes_on_ensure() -> None:
    """The v19 callback runs once on a database that lacks its row."""

    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            "CREATE INDEX gpu_fault_gpu_metric_latest_observed "
            "ON gpu_fault_gpu_metric_latest (observed_at)"
        )
        cursor.execute(
            "CREATE INDEX gpu_fault_processor_lanes_expiry "
            "ON gpu_fault_processor_lanes (lease_expires_at)"
        )
        cursor.execute("DELETE FROM gpu_fault_schema_migrations WHERE version>18")
        cursor.execute("UPDATE gpu_fault_schema_version SET version=18")
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    with _connect(autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname = ANY(%s)",
            (list(RETIRED_INDEXES),),
        )
        assert cursor.fetchall() == []
        cursor.execute("SELECT version FROM gpu_fault_schema_version")
        assert cursor.fetchone() == (LATEST_POSTGRES_SCHEMA_VERSION,)


def _latest(observed_at: datetime) -> GpuMetricLatest:
    return GpuMetricLatest(
        cluster_id="hp-cluster",
        node_id="worker-1",
        observed_at=observed_at,
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name="gpu_temperature_c",
            canonical_name="gpu_temperature_c",
            value=30.0,
            gpu_index="0",
            gpu_uuid="GPU-0",
        ),
    )


def test_metric_latest_upserts_are_hot() -> None:
    """Every sample rewrites its key's row; only ``observed_at`` and the
    payload change, and neither is indexed any more."""

    assert POSTGRES_URL is not None
    key = ("hp-cluster", "worker-1", "gpu_temperature_c", "GPU-0")
    base = datetime.now(UTC)
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    with _connect(autocommit=True) as connection:
        try:
            store.observe_gpu_metrics([(key, _latest(base))])
            connection.execute(
                "SELECT pg_stat_reset_single_table_counters("
                "'gpu_fault_gpu_metric_latest'::regclass)"
            )
            for index in range(1, 21):
                store.observe_gpu_metrics(
                    [(key, _latest(base + timedelta(seconds=index)))]
                )
        finally:
            # The pool's backends flush their statistics when they exit.
            store.close()
        updates, hot = _hot_counters(connection, "gpu_fault_gpu_metric_latest")
    assert updates >= 20
    assert hot == updates, "metric upserts maintained an index instead of HOT"


def test_lane_lease_updates_are_hot() -> None:
    now = datetime.now(UTC)
    with _connect(autocommit=True) as connection:
        connection.execute(
            "INSERT INTO gpu_fault_processor_lanes "
            "(ordering_key, owner_id, epoch, lease_token, lease_expires_at, updated_at) "
            "VALUES ('lane-a', 'owner-1', 1, 'token-1', %s, %s)",
            (now + timedelta(seconds=30), now),
        )
        connection.execute(
            "SELECT pg_stat_reset_single_table_counters("
            "'gpu_fault_processor_lanes'::regclass)"
        )
        for index in range(1, 21):
            connection.execute(
                "UPDATE gpu_fault_processor_lanes SET lease_expires_at=%s, "
                "updated_at=%s WHERE ordering_key='lane-a'",
                (now + timedelta(seconds=30 + index), now + timedelta(seconds=index)),
            )
        updates, hot = _hot_counters(connection, "gpu_fault_processor_lanes")
    assert updates >= 20
    assert hot == updates, "lane renewals maintained an index instead of HOT"
