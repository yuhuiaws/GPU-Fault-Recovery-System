"""Native read/merge/write races for Collector status history."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres import collector_telemetry
from gpu_fault.store.shared.primitives import state_key
from gpu_fault.telemetry import CollectorKind, CollectorStatus, merge_collector_status
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)
NOW = datetime(2026, 9, 14, tzinfo=UTC)
CLUSTER = "collector-status-race"


def status(second: int, *, node: str = "node-a", **changes) -> CollectorStatus:
    value = CollectorStatus(
        cluster_id=CLUSTER,
        node_id=node,
        collector=CollectorKind.GPU_METRICS,
        observed_at=NOW + timedelta(seconds=second),
        ingested_at=NOW + timedelta(seconds=second),
    )
    return value.model_copy(update=changes)


@pytest.fixture
def peers():
    import psycopg
    from psycopg.conninfo import make_conninfo

    with closing(postgres_store_instance()) as setup:
        next(setup)
    assert POSTGRES_URL, "this fixture requires its explicitly allocated database"
    with ExitStack() as stack:
        stores = [
            stack.enter_context(
                closing(
                    PostgresStore(
                        make_conninfo(
                            POSTGRES_URL, application_name=f"collector-race-{index}"
                        ),
                        initialize_schema=False,
                        pool_min_size=1,
                        pool_max_size=1,
                    )
                )
            )
            for index in range(2)
        ]
        for instance in stores:
            assert instance.list_collector_statuses(CLUSTER) == []
        monitor = stack.enter_context(psycopg.connect(POSTGRES_URL, autocommit=True))
        pids = [
            monitor.execute(
                "SELECT pid FROM pg_stat_activity WHERE application_name=%s "
                "AND datname=current_database()",
                (f"collector-race-{index}",),
            ).fetchone()[0]
            for index in range(2)
        ]
        try:
            yield stores, monitor, pids
        finally:
            stack.close()
            _truncate()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("newer_error", [False, True])
def test_concurrent_merge_preserves_sticky_history(
    peers, monkeypatch, existing: bool, newer_error: bool
) -> None:
    stores, monitor, pids = peers
    previous = status(0, last_success_at=NOW) if existing else None
    first = status(
        2,
        last_success_at=NOW + timedelta(seconds=1),
        last_error_at=NOW + timedelta(seconds=2),
        errors=["rejected-event:first"],
    )
    newer = status(
        3,
        last_error_at=NOW + timedelta(seconds=3) if newer_error else None,
        errors=["rejected-event:second"] if newer_error else [],
    )
    if previous is not None:
        assert stores[0].save_collector_status(previous), "baseline status was rejected"
    first_read, second_read = Event(), Event()
    release_first, first_committed = Event(), Event()

    def controlled_merge(current, incoming):
        if incoming == first:
            first_read.set()
            assert release_first.wait(5), "first writer was not released"
        elif incoming == newer:
            second_read.set()
            assert first_committed.wait(5), "newer writer preceded the first commit"
        return merge_collector_status(current, incoming)

    def write_first():
        try:
            return stores[0].save_collector_status(first)
        finally:
            first_committed.set()

    monkeypatch.setattr(collector_telemetry, "merge_collector_status", controlled_merge)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_write = executor.submit(write_first)
        try:
            assert first_read.wait(5), "first writer did not read the old status"
            second_write = executor.submit(stores[1].save_collector_status, newer)
            deadline = time.monotonic() + 5
            while not second_read.is_set():
                blocked = monitor.execute(
                    "SELECT %s=ANY(pg_blocking_pids(%s))", (pids[0], pids[1])
                ).fetchone()[0]
                if blocked:
                    break
                assert not second_write.done(), "newer writer exited before merging"
                assert time.monotonic() < deadline, "second writer made no progress"
                time.sleep(0.01)
        finally:
            release_first.set()
        assert first_write.result(timeout=5) is True
        assert second_write.result(timeout=5) is True

    expected = merge_collector_status(merge_collector_status(previous, first), newer)
    assert stores[0].list_collector_statuses(CLUSTER) == [expected], (
        "the newer observation discarded committed sticky history"
    )
    assert stores[1].save_collector_status(first) is False
    recovered = status(4, last_success_at=NOW + timedelta(seconds=4))
    assert stores[1].save_collector_status(recovered) is True
    actual = stores[0].list_collector_statuses(CLUSTER)[0]
    assert actual.errors == [], "a later success must still clear rejection history"
    assert actual.last_error_at == expected.last_error_at


def test_reversed_batches_take_absent_status_locks_in_the_same_order(peers) -> None:
    stores, monitor, pids = peers
    first = [status(1, node=node) for node in ("node-a", "node-b")]
    second = [status(2, node=node) for node in ("node-b", "node-a")]
    first_key = state_key((CLUSTER, "node-a", CollectorKind.GPU_METRICS.value))
    with ThreadPoolExecutor(max_workers=2) as executor:
        with monitor.transaction():
            monitor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"collector_status/{first_key}",),
            )
            futures = [
                executor.submit(store.save_collector_statuses_batch, batch)
                for store, batch in zip(stores, (first, second), strict=True)
            ]
            deadline = time.monotonic() + 5
            while True:
                waiting = monitor.execute(
                    "SELECT count(*) FROM pg_locks "
                    "WHERE pid=ANY(%s) AND locktype='advisory' AND NOT granted",
                    (pids,),
                ).fetchone()[0]
                if waiting == 2:
                    break
                assert not any(future.done() for future in futures), (
                    "a batch bypassed the lock for a previously absent status"
                )
                assert time.monotonic() < deadline, "batch locks did not converge"
                time.sleep(0.01)
            assert (
                monitor.execute(
                    "SELECT count(*) FROM pg_locks WHERE pid=ANY(%s) "
                    "AND locktype='advisory' AND granted",
                    (pids,),
                ).fetchone()[0]
                == 0
            ), "a reversed batch locked a later key before the first"
        outcomes = [future.result(timeout=5) for future in futures]
    assert outcomes[1] == [True, True]
    assert outcomes[0] in ([True, True], [False, False])
    assert stores[0].list_collector_statuses(CLUSTER) == list(reversed(second))
