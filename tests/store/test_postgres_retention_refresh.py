"""Two-connection retention races on the explicitly allocated PostgreSQL."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from queue import Queue

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.time import utc_text
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)
NOW = datetime(2026, 9, 14, tzinfo=UTC)
OLD = NOW - timedelta(days=40)
TABLES = {
    "gpu_finding_history": ("gpu_fault_objects", None),
    "gpu_metrics_batch": ("gpu_fault_gpu_metrics_batches", "created_at"),
    "gpu_metric_latest": ("gpu_fault_gpu_metric_latest", "observed_at"),
    "training_progress": ("gpu_fault_training_progress", "observed_at"),
    "attempt_observation": ("gpu_fault_attempt_observations", "observed_at"),
    "attempt_observation_stale": ("gpu_fault_attempt_observations", "observed_at"),
    "attempt_observation_legacy": ("gpu_fault_objects", None),
    "attempt_observation_legacy_stale": ("gpu_fault_objects", None),
}


class SweepStore(PostgresStore):
    def sweep(self, pid: Queue[int], *, stale: bool) -> dict[str, int]:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pid.put(cursor.fetchone()[0])
            return self.cleanup_hot_state(
                now=NOW,
                limit=1,
                attempt_observation_max_age=timedelta(days=30) if stale else None,
            )


@pytest.fixture
def store(mode, monkeypatch):
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", mode)
    with closing(postgres_store_instance()) as setup:
        next(setup)
    assert POSTGRES_URL is not None
    with closing(
        SweepStore(
            POSTGRES_URL,
            initialize_schema=False,
            hot_state_mode=mode,
            pool_min_size=1,
            pool_max_size=1,
        )
    ) as instance:
        try:
            yield instance
        finally:
            _truncate()


def row_values(kind: str, key: str, at: datetime, phase: str) -> dict:
    from psycopg.types.json import Jsonb

    table, timestamp = TABLES[kind]
    payload = {
        "observed_at": utc_text(at),
        "observation": {"observed_at": utc_text(at), "workload_phase": phase},
    }
    values = {"key": key, "payload": Jsonb(payload)}
    if table == "gpu_fault_objects":
        values["kind"] = "attempt_observation" if kind.startswith("attempt_") else kind
    else:
        values.update(cluster_id="retention-cluster")
        values[timestamp] = at
        if kind in {"gpu_metrics_batch", "gpu_metric_latest"}:
            values["node_id"] = "retention-node"
        else:
            values["attempt_id"] = key
        if kind == "training_progress":
            values["rank"] = 0
    return values


def insert_row(connection, kind: str, key: str, at: datetime, phase: str) -> None:
    from psycopg import sql

    values = row_values(kind, key, at, phase)
    connection.execute(
        sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(TABLES[kind][0]),
            sql.SQL(", ").join(map(sql.Identifier, values)),
            sql.SQL(", ").join(sql.Placeholder() for _ in values),
        ),
        list(values.values()),
    )


@pytest.mark.parametrize(
    ("mode", "kind", "refresh_phase"),
    [
        ("legacy", "gpu_finding_history", False),
        ("dedicated", "gpu_metrics_batch", False),
        ("dedicated", "gpu_metric_latest", False),
        ("dedicated", "training_progress", False),
        ("dedicated", "attempt_observation", False),
        ("dedicated", "attempt_observation", True),
        ("dedicated", "attempt_observation_stale", False),
        ("legacy", "attempt_observation_legacy", False),
        ("legacy", "attempt_observation_legacy", True),
        ("legacy", "attempt_observation_legacy_stale", False),
        ("dual", "attempt_observation_legacy", False),
    ],
)
def test_cleanup_skips_a_concurrent_refresh_and_spends_limit_on_expired_rows(
    store, mode: str, kind: str, refresh_phase: bool
) -> None:
    import psycopg
    from psycopg import sql

    assert POSTGRES_URL is not None
    table, timestamp = TABLES[kind]
    stale = kind.endswith("_stale")
    phase = "RUNNING" if stale else "SUCCEEDED"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as writer:
        for index, key in enumerate(("held", "expired-a", "expired-b")):
            insert_row(writer, kind, key, OLD + timedelta(hours=index), phase)
        refreshed = row_values(
            kind,
            "held",
            OLD if refresh_phase else NOW,
            "RUNNING" if refresh_phase else phase,
        )
        changed = {"payload": refreshed["payload"]}
        if timestamp is not None:
            changed[timestamp] = refreshed[timestamp]
        with ThreadPoolExecutor(max_workers=1) as executor:
            with writer.transaction():
                writer.execute(
                    sql.SQL("UPDATE {} SET {} WHERE key='held'").format(
                        sql.Identifier(table),
                        sql.SQL(", ").join(
                            sql.SQL("{}=%s").format(sql.Identifier(name))
                            for name in changed
                        ),
                    ),
                    list(changed.values()),
                )
                pids: Queue[int] = Queue()
                sweep = executor.submit(store.sweep, pids, stale=stale)
                pid = pids.get(timeout=5)
                deadline = time.monotonic() + 5
                # The old DELETE waits on this writer after its unlocked CTE
                # chose "held". Commit only once the race is established.
                while not sweep.done():
                    blockers = writer.execute(
                        "SELECT pg_backend_pid()=ANY(pg_blocking_pids(%s))", (pid,)
                    ).fetchone()[0]
                    if blockers:
                        break
                    assert time.monotonic() < deadline, (
                        "cleanup did not reach its candidates"
                    )
                    time.sleep(0.01)
            result = sweep.result(timeout=5)
        count_key = (
            "attempt_observation_legacy" if kind.endswith("legacy_stale") else kind
        )
        assert result[count_key] == 1, (
            "the batch quota must reach an unlocked expired row"
        )
        rows = writer.execute(
            sql.SQL("SELECT key, payload FROM {} ORDER BY key").format(
                sql.Identifier(table)
            )
        ).fetchall()
        assert [row[0] for row in rows] == ["expired-b", "held"], (
            "cleanup deleted a concurrently refreshed row or starved the next expired row"
        )
        assert rows[-1][1] == refreshed["payload"].obj, (
            "cleanup changed the fresh payload"
        )
        assert (
            store.cleanup_hot_state(
                now=NOW,
                limit=1,
                attempt_observation_max_age=timedelta(days=30) if stale else None,
            )[count_key]
            == 1
        )
        assert writer.execute(
            sql.SQL("SELECT key FROM {}").format(sql.Identifier(table))
        ).fetchall() == [("held",)], (
            "a later sweep must still preserve the refreshed row"
        )
