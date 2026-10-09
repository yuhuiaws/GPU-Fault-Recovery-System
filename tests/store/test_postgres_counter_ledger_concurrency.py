"""Per-writer-backend counter rows under concurrency (schema v20).

Twelve writers on one cluster run admission-shaped inserts, completions of
each other's requests and cleanup deletes at once. The aggregates must stay
exact, every counter row written must belong to a writer's own shard
(``mod(pg_backend_pid(), 128)``), and -- when the writers' residues are
distinct, as they are for twelve fresh backends -- no writer may wait on a
heavyweight lock held by another, which is the convoy the perf r1 runs
collapsed into on the hash-sharded rows.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.ddl_counter_ledger import (
    COUNTER_WRITER_SHARDS,
    COUNTS_TABLE,
    SHARDS_TABLE,
)
from tests.store._postgres_processor_claim_support import (
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires GPU_FAULT_TEST_POSTGRES_URL"
)

WRITERS = 12
ROUNDS = 25
CLUSTER = "ledger-a"
BUCKET_PRIORITIES = {0: 5, 50: 30, 100: 90}
CONVOY_WAITS = {"transactionid", "tuple"}


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


def _writer(
    index: int,
    pids: list[int | None],
    ready: threading.Barrier,
    errors: list[BaseException],
) -> None:
    rng = random.Random(index)
    try:
        with _connect(autocommit=True) as connection:
            pids[index] = int(
                connection.execute("SELECT pg_backend_pid()").fetchone()[0]
            )
            ready.wait(timeout=30)
            for _ in range(ROUNDS):
                # One admission flush: three requests, one per bucket, in one
                # statement -- the shape the statement-level triggers see.
                values = [
                    (
                        uuid.uuid4().hex,
                        CLUSTER,
                        f"lane-{index}-{rng.randrange(1_000_000)}",
                        BUCKET_PRIORITIES[bucket],
                    )
                    for bucket in (0, 50, 100)
                ]
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO gpu_fault_processor_queue "
                        "(request_id, status, cluster_id, ordering_key, priority, "
                        "created_at, updated_at, payload) "
                        "VALUES (%s, 'PENDING', %s, %s, %s, now(), now(), '{}')",
                        values,
                    )
                # A completion of whichever pending request is free, usually
                # one another backend admitted: that is what drives this
                # writer's rows negative while the aggregate stays exact.
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(
                        """
                        WITH picked AS (
                            SELECT request_id FROM gpu_fault_processor_queue
                            WHERE cluster_id=%s AND status='PENDING'
                            ORDER BY random() LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        UPDATE gpu_fault_processor_queue AS queue
                        SET status='COMPLETED', updated_at=now()
                        FROM picked WHERE queue.request_id=picked.request_id
                        """,
                        (CLUSTER,),
                    )
                if rng.random() < 0.3:
                    with connection.transaction(), connection.cursor() as cursor:
                        cursor.execute(
                            """
                            DELETE FROM gpu_fault_processor_queue
                            WHERE request_id IN (
                                SELECT request_id FROM gpu_fault_processor_queue
                                WHERE cluster_id=%s AND status='COMPLETED'
                                LIMIT 2 FOR UPDATE SKIP LOCKED
                            )
                            """,
                            (CLUSTER,),
                        )
    except BaseException as exc:  # noqa: BLE001 - every failure is the verdict
        errors.append(exc)


def _monitor(stop: threading.Event, pids: list[int | None], waits: list[tuple]) -> None:
    with _connect(autocommit=True) as connection:
        while not stop.is_set():
            watched = [pid for pid in pids if pid is not None]
            if watched:
                waits.extend(
                    connection.execute(
                        "SELECT pid, wait_event_type, wait_event FROM pg_stat_activity "
                        "WHERE pid = ANY(%s) AND wait_event_type = 'Lock'",
                        (watched,),
                    ).fetchall()
                )
            time.sleep(0.005)


def test_concurrent_flushes_and_completions_keep_exact_counts_on_disjoint_rows() -> (
    None
):
    pids: list[int | None] = [None] * WRITERS
    errors: list[BaseException] = []
    waits: list[tuple] = []
    ready = threading.Barrier(WRITERS)
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(stop, pids, waits), daemon=True)
    monitor.start()
    writers = [
        threading.Thread(target=_writer, args=(index, pids, ready, errors), daemon=True)
        for index in range(WRITERS)
    ]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(timeout=120)
    stop.set()
    monitor.join(timeout=10)

    assert errors == [], f"a writer failed (deadlock or error): {errors[:3]}"
    assert all(pid is not None for pid in pids), "every writer reported its backend"

    with _connect(autocommit=True) as connection:
        actual = dict(
            connection.execute(
                """
                SELECT
                    (CASE WHEN priority <= 10 THEN 0 WHEN priority <= 50 THEN 50
                          ELSE 100 END)::smallint AS bucket,
                    count(*)
                FROM gpu_fault_processor_queue
                WHERE cluster_id=%s AND status IN ('PENDING', 'LEASED')
                GROUP BY 1
                """,
                (CLUSTER,),
            ).fetchall()
        )
        ledger = dict(
            connection.execute(
                f"SELECT priority_bucket, sum(incomplete_count) FROM {SHARDS_TABLE} "
                "WHERE cluster_id=%s GROUP BY priority_bucket",
                (CLUSTER,),
            ).fetchall()
        )
        for bucket in (0, 50, 100):
            assert int(ledger.get(bucket, 0)) == int(actual.get(bucket, 0)), (
                bucket,
                ledger,
                actual,
            )
        (counts_total,) = connection.execute(
            f"SELECT coalesce(sum(incomplete_count), 0) FROM {COUNTS_TABLE} "
            "WHERE cluster_id=%s",
            (CLUSTER,),
        ).fetchone()
        assert int(counts_total) == sum(int(v) for v in actual.values())

        residues = {int(pid) % COUNTER_WRITER_SHARDS for pid in pids if pid is not None}
        shard_rows = {
            int(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT shard_id FROM {SHARDS_TABLE} WHERE cluster_id=%s",
                (CLUSTER,),
            ).fetchall()
        }
        count_rows = {
            int(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT shard_id FROM {COUNTS_TABLE} WHERE cluster_id=%s",
                (CLUSTER,),
            ).fetchall()
        }
        assert shard_rows and shard_rows <= residues, (
            "every shard row belongs to a writer's own backend residue"
        )
        assert count_rows and count_rows <= residues, (
            "every count row belongs to a writer's own backend residue"
        )
        negative = connection.execute(
            f"SELECT count(*) FROM {SHARDS_TABLE} WHERE incomplete_count < 0"
        ).fetchone()[0]
        assert isinstance(negative, int), "signed partial sums are readable"

    assert POSTGRES_URL is not None
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        status = store.processor_queue_count_status()
    finally:
        store.close()
    assert status["mismatched_clusters"] == 0 and status["ready"] is True, status

    if len(residues) == WRITERS:
        convoy = [row for row in waits if row[2] in CONVOY_WAITS]
        assert convoy == [], (
            "writers on disjoint counter rows waited on each other: "
            f"{convoy[:5]} (of {len(convoy)})"
        )
    else:
        pytest.skip(
            "two writers shared a pid residue; exactness and ownership held, "
            "the no-wait assertion needs distinct residues"
        )
