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

from gpu_fault.telemetry import WorkloadTopologyService
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


# --- coverage heartbeat -------------------------------------------------------
#
# A cluster with no managed attempt produces no observation, so "no fresh
# observation" could not tell an idle cluster from a dead watcher and the
# design fell closed to UNKNOWN. The watcher now reports after every full
# scan, even an empty one; a fresh heartbeat is coverage, so an idle cluster
# resolves IDLE while a silent one still resolves UNKNOWN.


def _heartbeat(scanned_at: datetime, *, attempt_count: int = 0):
    from gpu_fault.watcher import WorkloadCoverageHeartbeat

    return WorkloadCoverageHeartbeat(
        cluster_id=CLUSTER, scanned_at=scanned_at, attempt_count=attempt_count
    )


def test_a_fresh_coverage_heartbeat_makes_an_unobserved_node_idle() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(NOW - timedelta(seconds=30)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE", (
        "the watcher scanned the cluster and found nothing: the node is idle"
    )
    assert context.attempt_ids == []


def test_a_stale_coverage_heartbeat_leaves_the_node_unknown() -> None:
    store = build_store()
    topology = _topology(store, freshness_seconds=600)
    topology.observe_coverage(_heartbeat(NOW - timedelta(seconds=601)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a heartbeat older than the freshness window is a dead watcher"
    )


def test_a_coverage_heartbeat_never_makes_a_node_active() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(NOW, attempt_count=3))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE"
    assert context.job_ids == []


def test_an_older_heartbeat_does_not_overwrite_a_newer_one() -> None:
    store = build_store()
    newer = _heartbeat(NOW)
    older = _heartbeat(NOW - timedelta(seconds=45))

    assert store.save_workload_coverage(newer) is True
    assert store.save_workload_coverage(older) is False
    assert store.get_workload_coverage(CLUSTER) == newer


def test_coverage_is_per_cluster() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(NOW))

    assert topology.resolve("cluster-b", "node-1", NOW).workload_state == "UNKNOWN"
    assert store.get_workload_coverage("cluster-b") is None
