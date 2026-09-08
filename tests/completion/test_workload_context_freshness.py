"""Workload state resolved from attempt observations must fail closed.

``WorkloadTopologyService.resolve`` used to answer IDLE whenever no attempt
observation matched the node, so a cluster with no observer feed at all (the
watcher down, the cluster silent, a stale spool) looked idle and a node-health
REBOOT plan compiled executable past the UNKNOWN gate (ARCH-B2). Monitoring
loss is UNKNOWN: IDLE needs fresh observation coverage of the cluster that
simply does not name the node (ARCH-E2E-1 关键发现 2).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.telemetry import WorkloadCoverageHeartbeat, WorkloadTopologyService
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import attempt_observation, build_store, container_observation

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"


def _observation(
    node_id: str,
    *,
    observed_at: datetime,
    attempt_id: str = "attempt-a",
    phase: WorkloadPhase = WorkloadPhase.RUNNING,
) -> AttemptObservation:
    return attempt_observation(
        "training-job",
        attempt_id,
        observed_at,
        cluster_id=CLUSTER,
        workload_phase=phase,
        workload_ids=["training/job/training-job"],
        containers=[
            container_observation(
                f"pod-{node_id}", f"pod-{node_id}", 0, node_id, gpu_uuids=["GPU-1"]
            )
        ],
    )


def _topology(store, *, freshness_seconds: float = 600) -> WorkloadTopologyService:
    return WorkloadTopologyService(store, freshness_seconds=freshness_seconds)


def test_no_observations_for_the_cluster_is_unknown_not_idle() -> None:
    store = build_store()

    context = _topology(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a cluster nobody observes must not be reported as idle"
    )
    assert context.attempt_ids == [], context


def test_a_store_without_coverage_heartbeats_resolves_unknown() -> None:
    class ObservationOnlyStore:
        """The narrowest store the resolver has ever been handed."""

        def list_attempt_observations(self, _cluster_id: str) -> list:
            return []

    context = _topology(ObservationOnlyStore()).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a store that cannot answer the coverage question must fail closed, "
        "not raise on the ingest path"
    )


def test_fresh_coverage_elsewhere_in_the_cluster_makes_the_node_idle() -> None:
    store = build_store()
    store.save_attempt_observation(
        _observation("node-2", observed_at=NOW - timedelta(seconds=30))
    )

    context = _topology(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE", (
        "the feed is alive and names another node, so this node is idle"
    )


def test_a_matching_fresh_observation_is_active() -> None:
    store = build_store()
    store.save_attempt_observation(
        _observation("node-1", observed_at=NOW - timedelta(seconds=30))
    )

    context = _topology(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "ACTIVE", context
    assert context.attempt_ids == ["attempt-a"], context


def test_coverage_older_than_the_freshness_window_is_unknown() -> None:
    store = build_store()
    store.save_attempt_observation(
        _observation("node-2", observed_at=NOW - timedelta(seconds=601))
    )

    context = _topology(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a feed that stopped ten minutes ago proves nothing about now"
    )


def test_a_terminal_attempt_still_proves_the_feed_is_alive() -> None:
    store = build_store()
    store.save_attempt_observation(
        _observation(
            "node-1",
            observed_at=NOW - timedelta(seconds=30),
            phase=WorkloadPhase.SUCCEEDED,
        )
    )

    context = _topology(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE", (
        "a finished attempt observed just now is coverage, not activity"
    )


def test_the_freshness_window_is_configurable() -> None:
    store = build_store()
    store.save_attempt_observation(
        _observation("node-2", observed_at=NOW - timedelta(seconds=90))
    )

    strict = WorkloadTopologyService(
        store, max_age_seconds=60, freshness_seconds=60
    ).resolve(CLUSTER, "node-1", NOW)
    relaxed = _topology(store, freshness_seconds=120).resolve(CLUSTER, "node-1", NOW)

    assert strict.workload_state == "UNKNOWN", strict
    assert relaxed.workload_state == "IDLE", relaxed


@pytest.mark.parametrize("seconds", [0, -1])
def test_a_non_positive_freshness_window_is_refused(seconds: int) -> None:
    with pytest.raises(ValueError, match="freshness_seconds"):
        WorkloadTopologyService(build_store(), freshness_seconds=seconds)


def test_the_freshness_window_cannot_be_shorter_than_the_match_window() -> None:
    with pytest.raises(ValueError, match="freshness_seconds"):
        WorkloadTopologyService(
            build_store(), max_age_seconds=120, freshness_seconds=60
        )


def _heartbeat(*, observed_at: datetime, cluster_id: str = CLUSTER):
    return WorkloadCoverageHeartbeat(
        cluster_id=cluster_id,
        observed_at=observed_at,
        watched_pods=0,
        watched_attempts=0,
        resource_version="4711",
        watcher_instance="completion-watcher-0",
    )


def test_fresh_coverage_heartbeat_makes_an_empty_cluster_idle() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(observed_at=NOW - timedelta(seconds=30)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE", (
        "a watcher that completed a full pass 30s ago and saw no attempt "
        "proves the cluster is idle, not unobserved"
    )
    assert context.attempt_ids == [], context


def test_stale_coverage_heartbeat_leaves_the_cluster_unknown() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(observed_at=NOW - timedelta(seconds=601)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a heartbeat older than the freshness window proves nothing about now"
    )


def test_a_coverage_heartbeat_from_another_cluster_is_not_coverage() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(
        _heartbeat(observed_at=NOW - timedelta(seconds=5), cluster_id="cluster-b")
    )

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "another cluster's watcher says nothing about this one"
    )


def test_observations_win_over_a_fresh_coverage_heartbeat() -> None:
    store = build_store()
    store.save_attempt_observation(
        _observation("node-1", observed_at=NOW - timedelta(seconds=30))
    )
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(observed_at=NOW - timedelta(seconds=1)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "ACTIVE", (
        "an attempt observed on the node outranks the coverage heartbeat"
    )
    assert context.attempt_ids == ["attempt-a"], context


def test_the_coverage_heartbeat_window_is_configurable() -> None:
    store = build_store()
    strict = WorkloadTopologyService(
        store, freshness_seconds=600, coverage_freshness_seconds=60
    )
    strict.observe_coverage(_heartbeat(observed_at=NOW - timedelta(seconds=90)))

    relaxed = WorkloadTopologyService(
        store, freshness_seconds=600, coverage_freshness_seconds=120
    )

    assert strict.resolve(CLUSTER, "node-1", NOW).workload_state == "UNKNOWN", (
        "a 90s-old heartbeat is outside a 60s coverage window"
    )
    assert relaxed.resolve(CLUSTER, "node-1", NOW).workload_state == "IDLE", (
        "a 90s-old heartbeat is inside a 120s coverage window"
    )


@pytest.mark.parametrize("seconds", [0, -1])
def test_a_non_positive_coverage_window_is_refused(seconds: int) -> None:
    with pytest.raises(ValueError, match="coverage_freshness_seconds"):
        WorkloadTopologyService(build_store(), coverage_freshness_seconds=seconds)
