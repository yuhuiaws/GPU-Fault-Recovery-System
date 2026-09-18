"""Serial SQL regressions for guarded logical copy and atomic hot-state backfill."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_fault import store_migrate
from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource
from gpu_fault.store import PostgresStore
from scripts.e2e.regional.run_cap005_postgres_suite import validate_server
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires the separately allocated serial PostgreSQL slot"
)


@pytest.fixture
def migration_store() -> Iterator[PostgresStore]:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("logical-copy PostgreSQL tests must run without xdist")
    assert POSTGRES_URL is not None, (
        "the parent must allocate a local PostgreSQL instance"
    )
    validate_server(POSTGRES_URL)
    yield from postgres_store_instance()


def sample() -> GpuMetricLatest:
    return GpuMetricLatest(
        cluster_id="migration-test",
        node_id="node-test",
        observed_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name="temperature", canonical_name="temperature", value=70.0
        ),
    )


def legacy_file(path: Path, value: GpuMetricLatest | None) -> None:
    with closing(sqlite3.connect(path)) as database, database:
        database.executescript(
            "CREATE TABLE objects(kind TEXT,key TEXT,payload TEXT);"
            "CREATE TABLE links(kind TEXT,key TEXT,value TEXT);"
        )
        if value is not None:
            database.execute(
                "INSERT INTO objects VALUES(?,?,?)",
                ("gpu_metric_latest", "migration-sample", value.model_dump_json()),
            )


def test_legacy_import_is_immediately_readable_by_dedicated_postgres(
    migration_store: PostgresStore, tmp_path: Path
) -> None:
    value = sample()
    path = tmp_path / "legacy.db"
    legacy_file(path, value)
    assert POSTGRES_URL is not None, "the fixture validated the test URL"
    result = store_migrate.migrate_sqlite_to_postgres(str(path), POSTGRES_URL)
    assert (result.objects, result.links) == (1, 0), "one logical sample was imported"
    assert migration_store.list_gpu_metrics_latest(value.cluster_id, value.node_id) == [
        value
    ], "the production dedicated reader must observe the imported legacy value"
    status = migration_store.hot_state_migration_status()["gpu_metric_latest"]
    assert status["dedicated"] == 1 and status["missing_or_mismatched"] == 0, (
        "migration success includes completed native backfill"
    )


def test_failed_backfill_rolls_back_both_legacy_and_native_rows(
    migration_store: PostgresStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = sample()
    path = tmp_path / "legacy.db"
    legacy_file(path, value)
    original = PostgresStore.backfill_hot_state_tables

    def lose_ack(store: PostgresStore):
        original(store)
        raise RuntimeError("simulated failure after native write")

    monkeypatch.setattr(PostgresStore, "backfill_hot_state_tables", lose_ack)
    assert POSTGRES_URL is not None, "the fixture validated the test URL"
    with pytest.raises(RuntimeError, match="after native write"):
        store_migrate.migrate_sqlite_to_postgres(str(path), POSTGRES_URL)
    status = migration_store.hot_state_migration_status()["gpu_metric_latest"]
    assert status["legacy"] == status["dedicated"] == 0, (
        "backfill and copy share one real SQL transaction"
    )


@pytest.mark.parametrize("side", ["source", "destination"])
def test_native_hot_state_cannot_be_omitted_by_logical_copy(
    migration_store: PostgresStore, tmp_path: Path, side: str
) -> None:
    import psycopg

    value = sample()
    assert POSTGRES_URL is not None, "the fixture validated the test URL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            """
            INSERT INTO gpu_fault_gpu_metric_latest(
                key,cluster_id,node_id,observed_at,payload
            ) VALUES(%s,%s,%s,%s,%s::jsonb)
            """,
            (
                "migration-sample",
                value.cluster_id,
                value.node_id,
                value.observed_at,
                value.model_dump_json(),
            ),
        )
    path = tmp_path / "empty-legacy.db"
    legacy_file(path, None)
    with pytest.raises(RuntimeError, match=f"{side} contains dedicated telemetry"):
        if side == "source":
            store_migrate.migrate_postgres_to_postgres(POSTGRES_URL, POSTGRES_URL)
        else:
            store_migrate.migrate_sqlite_to_postgres(str(path), POSTGRES_URL)
    assert migration_store.list_gpu_metrics_latest(value.cluster_id, value.node_id) == [
        value
    ], "refusal must preserve the original authoritative native row"
