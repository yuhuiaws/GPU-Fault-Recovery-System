"""One coverage heartbeat row per cluster, and it never moves backwards.

The Completion Watcher posts a heartbeat after every completed full pass so an
idle cluster reads IDLE instead of UNKNOWN (completion-watcher F4). The row is
what the topology resolver reads, so it has to be an upsert keyed by cluster --
a second row would let a stale one answer -- and it has to refuse an older
``observed_at`` than the one already stored: after a watcher rollout the old
Pod's last in-flight heartbeat can arrive behind the new Pod's first one.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore
from gpu_fault.store.shared.primitives import state_key
from gpu_fault.store.shared.telemetry_records import SharedTelemetryRecordMixin
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


def test_replacing_an_unusable_row_re_arms_the_report(coverage_store, caplog) -> None:
    """Twice bad is twice reported.

    The report is suppressed per cluster so one bad row does not log once per
    fault -- coverage is read on the ingest path. That suppression has to lift
    when the row is replaced, otherwise a watcher rolled back to a release that
    writes unusable rows again is silent for the rest of this process's life,
    while its whole cluster reads UNKNOWN and blocks every node-mutating plan.
    """

    # The suppression is keyed by cluster and this process is shared with every
    # other test, so each store backend needs its own cluster.
    cluster = f"cluster-rearm-{type(coverage_store).__name__}"

    def _naive(watcher_instance: str) -> WorkloadCoverageHeartbeat:
        return WorkloadCoverageHeartbeat.model_construct(
            cluster_id=cluster,
            observed_at=NOW.replace(tzinfo=None),
            watched_pods=0,
            watched_attempts=0,
            resource_version="4711",
            watcher_instance=watcher_instance,
        )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.telemetry"):
        coverage_store.save_workload_coverage_heartbeat(_naive("watcher-0"))
        coverage_store.save_workload_coverage_heartbeat(
            _heartbeat(cluster_id=cluster, observed_at=NOW + timedelta(seconds=30))
        )
        first_report = caplog.text.count("ignoring the stored coverage heartbeat")
        coverage_store.save_workload_coverage_heartbeat(_naive("watcher-1"))

    assert first_report == 1, (
        f"replacing the unusable row must be reported exactly once: {caplog.text}"
    )
    assert caplog.text.count("ignoring the stored coverage heartbeat") == 2, (
        f"the second unusable row is a new fact and must be reported: {caplog.text}"
    )


class _UndecodableRowStore(SharedTelemetryRecordMixin):
    """A key/value store whose stored coverage row cannot be decoded at all.

    ``_get`` ends in ``self._models[kind].model_validate_json(payload)``, so a
    row of a kind this build's ``record_models()`` does not carry raises
    ``KeyError`` and a payload that is not even a JSON object can raise
    ``TypeError`` before pydantic wraps it. Reading coverage happens on the
    fault ingest path, where any of them raising fails the ingest of every
    fault on the cluster -- the one thing this row must never do.
    """

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.rows: dict[str, WorkloadCoverageHeartbeat] = {}
        self._get_optional = self._raise
        self._put = self._store
        self._state_key = state_key
        self._state_transaction = self._no_transaction

    def _raise(self, _kind: str, _key: str) -> object:
        raise self.error

    def _store(self, _kind: str, key: str, value: object) -> None:
        self.rows[key] = cast(WorkloadCoverageHeartbeat, value)

    def _no_transaction(self, _key: str) -> AbstractContextManager[object]:
        return nullcontext()


@pytest.mark.parametrize(
    "error",
    [KeyError("workload_coverage_heartbeat"), TypeError("not a mapping"), ValueError()],
    ids=["unknown-record-kind", "not-a-mapping", "not-a-value"],
)
def test_a_row_that_cannot_be_decoded_reads_as_no_coverage(error) -> None:
    store = _UndecodableRowStore(error)

    assert store.get_workload_coverage_heartbeat(CLUSTER) is None, (
        "an undecodable row must read as absent coverage; raising here fails "
        "the ingest of every fault on the cluster"
    )
    accepted = store.save_workload_coverage_heartbeat(_heartbeat())

    assert accepted is True, (
        "the writer reads the row too, and refusing there would leave the "
        "cluster's single row poisoned for ever"
    )
    assert store.rows and next(iter(store.rows.values())).observed_at == NOW, store.rows


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
