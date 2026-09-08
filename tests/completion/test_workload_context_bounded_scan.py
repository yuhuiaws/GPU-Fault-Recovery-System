"""``WorkloadTopologyService.resolve`` reads a bounded newest-first window.

Every fault event resolves the workload on its node, and ``resolve`` used to
do that by reading *every* retained attempt observation of the cluster and
filtering in Python (性能 2). Attempt observations are retained for a week,
so on a busy cluster with a high event rate that was O(events x attempts) and
the Postgres store materialised every row on every call. The matching window
is two minutes wide and cluster coverage is decided by the newest observation
alone, so a bounded newest-first read answers exactly the same question.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.telemetry import WorkloadTopologyService
from gpu_fault.telemetry_models import WorkloadObservationState
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import attempt_observation, build_store, container_observation

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
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


class _RecordingStore:
    """A store that records how ``resolve`` reads observations.

    ``list_attempt_observations`` is deliberately absent: the unbounded read is
    the defect, so falling back to it must fail loudly here.
    """

    def __init__(self, observations: list[AttemptObservation]) -> None:
        self.observations = observations
        self.calls: list[tuple[str | None, int | None, bool]] = []

    def list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[WorkloadObservationState]:
        self.calls.append((cluster_id, limit, newest_first))
        ordered = sorted(
            self.observations, key=lambda item: item.observed_at, reverse=newest_first
        )
        return [
            WorkloadObservationState(
                first_observed_at=item.observed_at, observation=item
            )
            for item in (ordered if limit is None else ordered[:limit])
        ]

    def get_workload_coverage(self, cluster_id: str):
        return None


def test_resolve_reads_a_bounded_newest_first_window() -> None:
    store = _RecordingStore([_observation("node-1", observed_at=NOW)])

    context = WorkloadTopologyService(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "ACTIVE", context
    assert store.calls == [
        (CLUSTER, WorkloadTopologyService.OBSERVATION_SCAN_LIMIT, True)
    ], "resolve must ask the store for the newest observations only"
    assert WorkloadTopologyService.OBSERVATION_SCAN_LIMIT >= 1


def test_the_newest_observation_alone_decides_coverage() -> None:
    # The freshest observation is older than the freshness window, so nothing
    # newer can exist: UNKNOWN without looking further.
    stale = _RecordingStore(
        [
            _observation("node-2", observed_at=NOW - timedelta(seconds=601)),
            _observation(
                "node-2", observed_at=NOW - timedelta(hours=3), attempt_id="old"
            ),
        ]
    )
    assert (
        WorkloadTopologyService(stale).resolve(CLUSTER, "node-1", NOW).workload_state
        == "UNKNOWN"
    )

    # One fresh observation naming another node is coverage: IDLE.
    fresh_elsewhere = _RecordingStore(
        [
            _observation("node-2", observed_at=NOW - timedelta(seconds=30)),
            _observation(
                "node-1", observed_at=NOW - timedelta(hours=3), attempt_id="old"
            ),
        ]
    )
    assert (
        WorkloadTopologyService(fresh_elsewhere)
        .resolve(CLUSTER, "node-1", NOW)
        .workload_state
        == "IDLE"
    )


def test_a_live_container_within_the_match_window_is_active() -> None:
    store = _RecordingStore(
        [
            _observation("node-1", observed_at=NOW - timedelta(seconds=119)),
            _observation(
                "node-2", observed_at=NOW - timedelta(seconds=5), attempt_id="other"
            ),
        ]
    )

    context = WorkloadTopologyService(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "ACTIVE", context
    assert context.attempt_ids == ["attempt-a"], context


def test_a_match_just_outside_the_window_is_coverage_not_activity() -> None:
    store = _RecordingStore(
        [
            _observation("node-1", observed_at=NOW - timedelta(seconds=121)),
            _observation(
                "node-2", observed_at=NOW - timedelta(seconds=5), attempt_id="other"
            ),
        ]
    )

    context = WorkloadTopologyService(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "IDLE", context
    assert context.attempt_ids == [], context


def test_an_observation_slightly_in_the_future_still_matches() -> None:
    # Watcher and collector clocks are not the same clock; an observation a few
    # seconds ahead of the event has always matched (negative age) and sorts
    # first in a newest-first read, so the early stop must not skip it.
    store = _RecordingStore(
        [
            _observation("node-1", observed_at=NOW + timedelta(seconds=10)),
            _observation(
                "node-2", observed_at=NOW - timedelta(seconds=5), attempt_id="other"
            ),
        ]
    )

    context = WorkloadTopologyService(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "ACTIVE", context
    assert context.attempt_ids == ["attempt-a"], context


def test_a_cluster_with_more_attempts_than_the_window_still_resolves() -> None:
    store = build_store()
    limit = WorkloadTopologyService.OBSERVATION_SCAN_LIMIT
    for index in range(limit + 50):
        store.save_attempt_observation(
            _observation(
                "node-2",
                observed_at=NOW - timedelta(hours=1, seconds=index),
                attempt_id=f"attempt-{index}",
                phase=WorkloadPhase.SUCCEEDED,
            )
        )
    store.save_attempt_observation(
        _observation("node-1", observed_at=NOW - timedelta(seconds=20))
    )

    context = WorkloadTopologyService(store).resolve(CLUSTER, "node-1", NOW)

    assert context.workload_state == "ACTIVE", context
    assert context.attempt_ids == ["attempt-a"], context


def test_the_observations_override_is_used_as_given_in_any_order() -> None:
    store = _RecordingStore([])
    fresh_match = _observation("node-1", observed_at=NOW - timedelta(seconds=30))
    stale = _observation(
        "node-1", observed_at=NOW - timedelta(hours=2), attempt_id="stale"
    )
    other = _observation(
        "node-2", observed_at=NOW - timedelta(seconds=5), attempt_id="other"
    )
    topology = WorkloadTopologyService(store)

    oldest_first = topology.resolve(
        CLUSTER, "node-1", NOW, observations=[stale, other, fresh_match]
    )
    newest_first = topology.resolve(
        CLUSTER, "node-1", NOW, observations=[other, fresh_match, stale]
    )

    assert store.calls == [], "a caller-supplied list must not trigger a store read"
    assert oldest_first == newest_first
    assert oldest_first.workload_state == "ACTIVE", oldest_first
    assert oldest_first.attempt_ids == ["attempt-a"], oldest_first


def test_the_observations_override_decides_coverage_from_any_position() -> None:
    store = _RecordingStore([])
    fresh_elsewhere = _observation("node-2", observed_at=NOW - timedelta(seconds=30))
    stale = _observation(
        "node-1", observed_at=NOW - timedelta(hours=2), attempt_id="stale"
    )

    context = WorkloadTopologyService(store).resolve(
        CLUSTER, "node-1", NOW, observations=[stale, fresh_elsewhere]
    )

    assert context.workload_state == "IDLE", (
        "coverage in an unordered caller list must be found wherever it sits"
    )
