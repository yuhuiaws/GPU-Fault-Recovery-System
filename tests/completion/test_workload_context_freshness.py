"""Workload state resolved from attempt observations must fail closed.

``WorkloadTopologyService.resolve`` used to answer IDLE whenever no attempt
observation matched the node, so a cluster with no observer feed at all (the
watcher down, the cluster silent, a stale spool) looked idle and a node-health
REBOOT plan compiled executable past the UNKNOWN gate (ARCH-B2). Monitoring
loss is UNKNOWN: IDLE needs fresh observation coverage of the cluster that
simply does not name the node (ARCH-E2E-1 关键发现 2).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

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

        def list_attempt_observation_states(
            self,
            _cluster_id: str | None = None,
            *,
            limit: int | None = None,
            newest_first: bool = False,
        ) -> list:
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


def _heartbeat(
    *,
    observed_at: datetime,
    cluster_id: str = CLUSTER,
    watched_pods: int = 0,
    watched_attempts: int = 0,
):
    return WorkloadCoverageHeartbeat(
        cluster_id=cluster_id,
        observed_at=observed_at,
        watched_pods=watched_pods,
        watched_attempts=watched_attempts,
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


@pytest.mark.parametrize(("watched_pods", "watched_attempts"), [(0, 3), (8, 0), (8, 3)])
def test_a_heartbeat_that_saw_workload_running_is_not_coverage(
    watched_pods: int, watched_attempts: int
) -> None:
    """Only "I saw nothing running" separates an idle cluster from a lossy feed.

    A pass that watched a live Pod or attempt and published no observation the
    resolver can see means the observation was lost -- a per-attempt failure
    swallowed to keep the pass alive, a dropped POST, a batch shed under load
    while this reserved-capacity path still landed. Reading that as coverage
    answers IDLE for a node that is training, and IDLE is what lets a plan skip
    CHECKPOINT and STOP_WORKLOADS before it reboots the node.
    """

    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(
        _heartbeat(
            observed_at=NOW - timedelta(seconds=5),
            watched_pods=watched_pods,
            watched_attempts=watched_attempts,
        )
    )

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a heartbeat from a pass that still saw workload running is not "
        f"evidence of an idle cluster: pods={watched_pods} "
        f"attempts={watched_attempts}"
    )


def test_a_naive_heartbeat_stamp_is_refused_at_the_boundary() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _heartbeat(observed_at=NOW.replace(tzinfo=None))


def test_a_heartbeat_stamp_from_another_offset_is_stored_as_utc() -> None:
    heartbeat = _heartbeat(observed_at=NOW.astimezone(timezone(timedelta(hours=8))))

    assert heartbeat.observed_at == NOW, heartbeat.observed_at
    assert heartbeat.observed_at.utcoffset() == timedelta(0), (
        "the stored stamp must be UTC so every reader compares like with like"
    )


def test_a_stored_heartbeat_with_a_naive_stamp_reads_as_no_coverage(caplog) -> None:
    """A row an older release wrote must not raise on the fault ingest path."""

    # Its own cluster: the report is deliberately once per cluster per process,
    # so a test that asserts it must not share a cluster with another one.
    poisoned_cluster = "cluster-with-a-naive-coverage-row"

    class PoisonedStore:
        def list_attempt_observations(self, _cluster_id: str) -> list:
            return []

        def list_attempt_observation_states(
            self,
            _cluster_id: str | None = None,
            *,
            limit: int | None = None,
            newest_first: bool = False,
        ) -> list:
            return []

        def get_workload_coverage_heartbeat(self, cluster_id: str):
            # model_construct: exactly what a payload written before the model
            # required a timezone deserialises to.
            return WorkloadCoverageHeartbeat.model_construct(
                cluster_id=cluster_id,
                observed_at=NOW.replace(tzinfo=None),
                watched_pods=0,
                watched_attempts=0,
                resource_version=None,
                watcher_instance="completion-watcher-0",
            )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.telemetry"):
        context = _topology(PoisonedStore()).resolve(poisoned_cluster, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "an unusable coverage row must read as absent coverage, not raise and "
        "not vouch for the cluster"
    )
    assert "ignoring the stored coverage heartbeat" in caplog.text, caplog.text


def test_a_row_that_goes_bad_again_is_reported_again(caplog) -> None:
    """The report is once per cluster, not once per process lifetime.

    The unusable row is a real operator signal: a watcher writing a stamp
    nothing can compare against is a watcher whose cluster reads UNKNOWN, and
    every node-mutating plan on it is BLOCKED. Suppressing the report for ever
    after the first one would hide the second occurrence -- a rolled-back
    watcher writing naive stamps again -- for as long as the control plane runs.
    """

    naive_cluster = "cluster-that-goes-bad-twice"
    good = _heartbeat(cluster_id=naive_cluster, observed_at=NOW - timedelta(seconds=5))
    bad = WorkloadCoverageHeartbeat.model_construct(
        cluster_id=naive_cluster,
        observed_at=NOW.replace(tzinfo=None),
        watched_pods=0,
        watched_attempts=0,
        resource_version=None,
        watcher_instance="completion-watcher-0",
    )

    class RollingBackStore:
        def __init__(self) -> None:
            self.heartbeat = bad

        def list_attempt_observations(self, _cluster_id: str) -> list:
            return []

        def list_attempt_observation_states(
            self,
            _cluster_id: str | None = None,
            *,
            limit: int | None = None,
            newest_first: bool = False,
        ) -> list:
            return []

        def get_workload_coverage_heartbeat(self, _cluster_id: str):
            return self.heartbeat

    store = RollingBackStore()
    topology = _topology(store)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.telemetry"):
        first = topology.resolve(naive_cluster, "node-1", NOW)
        repeat = topology.resolve(naive_cluster, "node-1", NOW)
        reports_while_bad = caplog.text.count("ignoring the stored coverage heartbeat")
        store.heartbeat = good
        recovered = topology.resolve(naive_cluster, "node-1", NOW)
        store.heartbeat = bad
        again = topology.resolve(naive_cluster, "node-1", NOW)

    assert (first.workload_state, repeat.workload_state) == ("UNKNOWN", "UNKNOWN"), (
        f"an unusable row is absent coverage: {first} {repeat}"
    )
    assert reports_while_bad == 1, (
        "coverage is read once per fault, so one bad row must not log once per "
        f"fault: {caplog.text}"
    )
    assert recovered.workload_state == "IDLE", (
        f"a readable heartbeat that saw nothing running is coverage: {recovered}"
    )
    assert again.workload_state == "UNKNOWN", again
    assert caplog.text.count("ignoring the stored coverage heartbeat") == 2, (
        f"a row that was fixed and went bad again must be reported again: {caplog.text}"
    )


def test_a_heartbeat_behind_the_stored_row_is_counted_and_reported(caplog) -> None:
    store = build_store()
    topology = _topology(store)

    assert topology.observe_coverage(_heartbeat(observed_at=NOW)) is True, (
        "the first heartbeat of a cluster is always accepted"
    )
    with caplog.at_level(logging.WARNING, logger="gpu_fault.telemetry"):
        accepted = topology.observe_coverage(
            _heartbeat(observed_at=NOW - timedelta(seconds=5))
        )

    assert accepted is False, "a heartbeat behind the stored row must be refused"
    assert topology.coverage_heartbeats_rejected_total == 1, (
        "a refused heartbeat is what two watchers publishing for one cluster "
        f"look like, so it is counted: {topology.coverage_heartbeats_rejected_total}"
    )
    assert "refused a coverage heartbeat" in caplog.text, caplog.text


# --- coverage heartbeat, the control-plane review's cases (c152aef) ------------
#
# The 2026-09-08 control-plane review added the same heartbeat independently,
# with a model that carried ``scanned_at`` / ``attempt_count`` and read *any*
# fresh heartbeat as coverage. Its cases are kept here on the data-plane
# review's model; the one whose premise differs is reconciled below and says so.


def test_a_fresh_coverage_heartbeat_makes_an_unobserved_node_idle() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(observed_at=NOW - timedelta(seconds=30)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE", (
        "the watcher scanned the cluster and found nothing: the node is idle"
    )
    assert context.attempt_ids == []


def test_a_stale_coverage_heartbeat_leaves_the_node_unknown() -> None:
    store = build_store()
    topology = _topology(store, freshness_seconds=600)
    topology.observe_coverage(_heartbeat(observed_at=NOW - timedelta(seconds=601)))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", (
        "a heartbeat older than the freshness window is a dead watcher"
    )


def test_a_coverage_heartbeat_never_makes_a_node_active() -> None:
    """A heartbeat names no node and no job, so it can never answer ACTIVE.

    The control-plane review's version of this case expected IDLE for a
    heartbeat that saw three attempts running. The data-plane review decided
    otherwise (``test_a_heartbeat_that_saw_workload_running_is_not_coverage``):
    a pass that saw workload the resolver cannot see means the observation feed
    is lossy, and IDLE would let a plan reboot a training node without a
    checkpoint. So the answer is UNKNOWN -- still never ACTIVE, and still
    without a job or attempt the heartbeat cannot know about.
    """

    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(observed_at=NOW, watched_attempts=3))

    context = topology.resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "UNKNOWN", context
    assert context.job_ids == []
    assert context.attempt_ids == []


def test_an_older_heartbeat_does_not_overwrite_a_newer_one() -> None:
    store = build_store()
    newer = _heartbeat(observed_at=NOW)
    older = _heartbeat(observed_at=NOW - timedelta(seconds=45))

    assert store.save_workload_coverage_heartbeat(newer) is True
    assert store.save_workload_coverage_heartbeat(older) is False
    assert store.get_workload_coverage_heartbeat(CLUSTER) == newer


def test_coverage_is_per_cluster() -> None:
    store = build_store()
    topology = _topology(store)
    topology.observe_coverage(_heartbeat(observed_at=NOW))

    assert topology.resolve("cluster-b", "node-1", NOW).workload_state == "UNKNOWN"
    assert store.get_workload_coverage_heartbeat("cluster-b") is None
