"""One coverage heartbeat row per cluster, and it never moves backwards.

The Completion Watcher posts a heartbeat after every completed full pass so an
idle cluster reads IDLE instead of UNKNOWN (completion-watcher F4). The row is
what the topology resolver reads, so it has to be an upsert keyed by cluster --
a second row would let a stale one answer -- and it has to refuse an older
``observed_at`` than the one already stored: after a watcher rollout the old
Pod's last in-flight heartbeat can arrive behind the new Pod's first one.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore
from gpu_fault.telemetry import WorkloadCoverageHeartbeat

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"


def _heartbeat(
    *,
    observed_at: datetime = NOW,
    cluster_id: str = CLUSTER,
    watcher_instance: str = "completion-watcher-0",
    watched_pods: int = 0,
    watched_attempts: int = 0,
) -> WorkloadCoverageHeartbeat:
    return WorkloadCoverageHeartbeat(
        cluster_id=cluster_id,
        observed_at=observed_at,
        watched_pods=watched_pods,
        watched_attempts=watched_attempts,
        resource_version="4711",
        watcher_instance=watcher_instance,
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def coverage_store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStore()
        return
    if request.param == "sqlite":
        store = SqliteStore(str(tmp_path / "coverage.db"))
        try:
            yield store
        finally:
            store.close()
        return
    postgres_url = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
    if not postgres_url:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    postgres = PostgresStore(postgres_url)
    try:
        yield postgres
    finally:
        postgres.close()


def test_a_coverage_heartbeat_round_trips_per_cluster(coverage_store) -> None:
    assert coverage_store.get_workload_coverage_heartbeat(CLUSTER) is None, (
        "a cluster nobody has watched must have no heartbeat row"
    )

    accepted = coverage_store.save_workload_coverage_heartbeat(
        _heartbeat(watched_pods=8, watched_attempts=2)
    )
    stored = coverage_store.get_workload_coverage_heartbeat(CLUSTER)

    assert accepted is True, "the first heartbeat of a cluster must be accepted"
    assert stored is not None, "the heartbeat was not persisted"
    assert (stored.observed_at, stored.watched_pods, stored.watched_attempts) == (
        NOW,
        8,
        2,
    ), stored
    assert stored.watcher_instance == "completion-watcher-0", stored
    assert stored.resource_version == "4711", stored


def test_a_newer_heartbeat_replaces_the_row_instead_of_adding_one(
    coverage_store,
) -> None:
    coverage_store.save_workload_coverage_heartbeat(_heartbeat())

    accepted = coverage_store.save_workload_coverage_heartbeat(
        _heartbeat(
            observed_at=NOW + timedelta(seconds=30),
            watcher_instance="completion-watcher-1",
            watched_pods=4,
        )
    )
    stored = coverage_store.get_workload_coverage_heartbeat(CLUSTER)

    assert accepted is True, "a newer heartbeat must be accepted"
    assert stored is not None, "the heartbeat row disappeared"
    assert stored.observed_at == NOW + timedelta(seconds=30), stored
    assert stored.watcher_instance == "completion-watcher-1", (
        "the row must carry the newest watcher instance, not the first one"
    )
    assert stored.watched_pods == 4, stored


def test_a_heartbeat_from_a_replaced_watcher_cannot_move_coverage_backwards(
    coverage_store,
) -> None:
    coverage_store.save_workload_coverage_heartbeat(
        _heartbeat(observed_at=NOW, watcher_instance="completion-watcher-1")
    )

    accepted = coverage_store.save_workload_coverage_heartbeat(
        _heartbeat(
            observed_at=NOW - timedelta(seconds=45),
            watcher_instance="completion-watcher-0",
        )
    )
    stored = coverage_store.get_workload_coverage_heartbeat(CLUSTER)

    assert accepted is False, (
        "the old watcher's late heartbeat must be refused, not stored"
    )
    assert stored is not None, "the heartbeat row disappeared"
    assert stored.watcher_instance == "completion-watcher-1", stored
    assert stored.observed_at == NOW, stored


def test_a_row_with_a_naive_stamp_is_replaced_rather_than_defended(
    coverage_store,
) -> None:
    """A row written before the model required a timezone must not poison a
    cluster.

    The monotonic guard compares the new stamp against the stored one, and that
    comparison raises on a naive value: the row would refuse every replacement
    for ever, and the read on the fault ingest path would raise instead of
    answering "no coverage". ``model_construct`` is how such a row is written
    without the model's validator -- which is exactly what an older release did.
    """

    poisoned = WorkloadCoverageHeartbeat.model_construct(
        cluster_id=CLUSTER,
        observed_at=NOW.replace(tzinfo=None),
        watched_pods=0,
        watched_attempts=0,
        resource_version="4711",
        watcher_instance="completion-watcher-0",
    )
    coverage_store.save_workload_coverage_heartbeat(poisoned)

    accepted = coverage_store.save_workload_coverage_heartbeat(
        _heartbeat(
            observed_at=NOW - timedelta(seconds=300),
            watcher_instance="completion-watcher-1",
        )
    )
    stored = coverage_store.get_workload_coverage_heartbeat(CLUSTER)

    assert accepted is True, (
        "a heartbeat that cannot be compared with the stored row must replace "
        "it, otherwise the cluster can never be vouched for again"
    )
    assert stored is not None, "the heartbeat row disappeared"
    assert stored.observed_at == NOW - timedelta(seconds=300), stored
    assert stored.watcher_instance == "completion-watcher-1", stored


def test_each_cluster_keeps_its_own_heartbeat(coverage_store) -> None:
    coverage_store.save_workload_coverage_heartbeat(_heartbeat())
    coverage_store.save_workload_coverage_heartbeat(
        _heartbeat(cluster_id="cluster-b", observed_at=NOW - timedelta(seconds=300))
    )

    first = coverage_store.get_workload_coverage_heartbeat(CLUSTER)
    second = coverage_store.get_workload_coverage_heartbeat("cluster-b")

    assert first is not None and first.observed_at == NOW, first
    assert second is not None and second.observed_at == NOW - timedelta(seconds=300), (
        second
    )
