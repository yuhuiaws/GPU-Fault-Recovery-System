"""The storeless regional executor reclaims stale warm-spare reservations.

The control-plane ``HyperPodSpareHealthController`` needs a store; the
regional executor has none, so it can only reclaim by the reservation
timestamp (ARCH-A4b). The sweep is periodic, never raises into the claim
loop and counts what it released.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor, SpareReservationSweep
from gpu_fault.hyperpod_spares import (
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    SPARE_RESERVED_AT_ANNOTATION,
    HyperPodSpareCoordinator,
)
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
)
from tests.hyperpod.test_spare_reservations import (
    NOW,
    ApiServerCore,
    FakeCore,
    FakeLifecycle,
    ReplicaCoordinator,
    _released,
    reserved_node,
    spare,
)

SWEEP_LOGGER = "gpu_fault.spare_reservation_sweep"


def _coordinator(core_nodes, nodes):
    core = FakeCore(core_nodes)
    return HyperPodSpareCoordinator(FakeLifecycle(nodes), None, core, now=lambda: NOW)


def _stale_spare_core() -> ApiServerCore:
    return ApiServerCore(
        {
            "hyperpod-i-1": reserved_node(
                "incident-old", reserved_at=NOW - timedelta(hours=2)
            )
        }
    )


def _replica(core: ApiServerCore) -> SpareReservationSweep:
    """One executor replica: its own coordinator and sweep, the shared node."""

    coordinator = ReplicaCoordinator(
        FakeLifecycle([spare("i-1")]), None, core, now=lambda: NOW
    )
    return SpareReservationSweep(
        coordinator, ttl_seconds=3600.0, interval_seconds=300.0, now=lambda: NOW
    )


def _sweep_lines(caplog, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == SWEEP_LOGGER and record.levelno == level
    ]


RECLAIM_LINE = (
    "reclaimed stale spare reservation: node=hyperpod-i-1 incident=incident-old "
    "reason=reservation by incident-old exceeded 3600s TTL"
)
ALREADY_RELEASED_LINE = (
    "stale spare reservation already released: node=hyperpod-i-1 incident=incident-old"
)


def test_sweep_reclaims_only_reservations_past_the_ttl() -> None:
    core_nodes = {
        "hyperpod-i-1": reserved_node(
            "incident-young", reserved_at=NOW - timedelta(minutes=30)
        ),
        "hyperpod-i-2": reserved_node(
            "incident-old", reserved_at=NOW - timedelta(hours=2)
        ),
        "hyperpod-i-3": reserved_node("incident-undated", reserved_at=None),
    }
    sweep = SpareReservationSweep(
        _coordinator(core_nodes, [spare("i-1"), spare("i-2"), spare("i-3")]),
        ttl_seconds=3600.0,
        interval_seconds=300.0,
        now=lambda: NOW,
    )

    released = sweep.run()

    assert released == ["hyperpod-i-2"], f"released set differs: {released}"
    assert not _released(core_nodes["hyperpod-i-1"]), "young reservation reclaimed"
    assert _released(core_nodes["hyperpod-i-2"]), "expired reservation kept"
    assert not _released(core_nodes["hyperpod-i-3"]), (
        "a reservation without a timestamp has no evidence of staleness"
    )
    assert sweep.reclaimed_total == 1, "reclaim counter did not move"


def test_sweep_is_paced_by_its_interval() -> None:
    clock = {"now": 1000.0}
    sweep = SpareReservationSweep(
        _coordinator({}, []),
        ttl_seconds=3600.0,
        interval_seconds=300.0,
        now=lambda: NOW,
        clock=lambda: clock["now"],
    )

    assert sweep.due(), "the first sweep must run immediately"
    sweep.run()
    assert not sweep.due(), "a sweep just ran; the next one is not due yet"
    clock["now"] += 299.0
    assert not sweep.due(), "the interval has not elapsed"
    clock["now"] += 1.0
    assert sweep.due(), "the interval elapsed; the sweep is due again"


def test_a_failing_sweep_does_not_break_the_claim_loop(caplog) -> None:
    class BrokenLifecycle(FakeLifecycle):
        def list_nodes(self, *, enrich=False):
            raise RuntimeError("apiserver unavailable")

    core_nodes = {
        "hyperpod-i-1": reserved_node(
            "incident-old", reserved_at=NOW - timedelta(hours=2)
        )
    }
    coordinator = HyperPodSpareCoordinator(
        BrokenLifecycle([spare("i-1")]), None, FakeCore(core_nodes), now=lambda: NOW
    )
    executor = ClusterActionExecutor(
        FakeExecutorClient([]),
        [RecordingAdapter("gpu-fault-node-agent")],
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        spare_reservation_sweep=SpareReservationSweep(
            coordinator, ttl_seconds=3600.0, interval_seconds=300.0, now=lambda: NOW
        ),
    )

    with caplog.at_level("WARNING"):
        executor.sweep_spare_reservations()

    assert any(
        "spare reservation sweep failed" in record.message for record in caplog.records
    ), "a failed sweep must be logged, not raised"
    assert (
        SPARE_RESERVATION_ANNOTATION
        in (core_nodes["hyperpod-i-1"]["metadata"]["annotations"])
    ), "a failed sweep must not release anything"
    assert executor.metrics_snapshot()["spare_reservations_reclaimed_total"] == 0, (
        "the executor counter must expose the sweep total"
    )


def test_executor_without_a_sweep_has_a_zero_counter() -> None:
    executor = ClusterActionExecutor(
        FakeExecutorClient([]),
        [RecordingAdapter("gpu-fault-node-agent")],
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
    )
    executor.sweep_spare_reservations()
    assert executor.metrics_snapshot()["spare_reservations_reclaimed_total"] == 0, (
        "no sweep configured means nothing reclaimed"
    )


@pytest.mark.parametrize("ttl", [0.0, -1.0])
def test_sweep_rejects_a_non_positive_ttl(ttl: float) -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        SpareReservationSweep(
            _coordinator({}, []), ttl_seconds=ttl, interval_seconds=300.0
        )


def test_sweep_judges_the_spare_label_on_the_kubernetes_node() -> None:
    """Live 2026-09-08 (DESTR-022): the regional executor's provider view
    carried no kubernetes_labels at all, so a sweep keyed on them skipped
    every node and a two-day-old reservation was never reclaimed. The label
    the declaration writes is on the Kubernetes Node; that is what counts."""
    from gpu_fault.hyperpod import HyperPodNode

    unenriched = HyperPodNode(
        node_logical_id="worker-i-9",
        instance_id="i-9",
        instance_group_name="workers",
        instance_type="ml.p5.48xlarge",
        status="Running",
        kubernetes_labels={"kubernetes.io/hostname": "hyperpod-i-9"},
    )
    stale = reserved_node("incident-stale", reserved_at=NOW - timedelta(days=2))
    stale["metadata"]["labels"]["gpu-fault.io/spare"] = "true"
    not_a_spare = reserved_node("incident-other", reserved_at=NOW - timedelta(days=2))
    plain = HyperPodNode(
        node_logical_id="worker-i-8",
        instance_id="i-8",
        instance_group_name="workers",
        instance_type="ml.p5.48xlarge",
        status="Running",
        kubernetes_labels={"kubernetes.io/hostname": "hyperpod-i-8"},
    )
    core_nodes = {"hyperpod-i-9": stale, "hyperpod-i-8": not_a_spare}
    sweep = SpareReservationSweep(
        _coordinator(core_nodes, [unenriched, plain]),
        ttl_seconds=86400.0,
        interval_seconds=300.0,
        now=lambda: NOW,
    )

    released = sweep.run()

    assert released == ["hyperpod-i-9"], f"released set differs: {released}"
    assert _released(stale), "the labeled Node's expired reservation was kept"
    assert not _released(not_a_spare), "a node without the spare label was swept"


# Every executor replica runs its own sweep over the same nodes (live
# DESTR-022 on plain EKS: both replicas logged the reclaim of one reservation
# 174 ms apart and each moved its counter). The release is a resourceVersion
# CAS, so it applies once; only the replica whose release applied reclaimed.


def test_a_replica_sweeping_after_its_peer_finds_nothing_to_reclaim(caplog) -> None:
    core = _stale_spare_core()
    first, second = _replica(core), _replica(core)

    with caplog.at_level(logging.INFO, logger=SWEEP_LOGGER):
        released = [first.run(), second.run()]

    assert released == [["hyperpod-i-1"], []], released
    assert (first.reclaimed_total, second.reclaimed_total) == (1, 0)
    assert len(core.patches) == 1, core.patches
    assert _sweep_lines(caplog, logging.WARNING) == [RECLAIM_LINE]
    assert _sweep_lines(caplog, logging.INFO) == [], "nothing was left to release"


@pytest.mark.parametrize(
    "peer_slips_in",
    [
        # The peer released between this replica's judgment and its release:
        # the release's own read already shows no reservation.
        "before-release",
        # The peer's patch landed between the release's read and its patch:
        # 409, and the re-read finds nothing to patch.
        "before-patch",
    ],
)
def test_a_release_the_peer_replica_already_applied_is_not_a_reclaim(
    peer_slips_in: str, caplog
) -> None:
    core = _stale_spare_core()
    late, peer = _replica(core), _replica(core)
    if peer_slips_in == "before-release":
        late.coordinator.before_release = peer.run
    else:
        core.before_patch = peer.run

    with caplog.at_level(logging.INFO, logger=SWEEP_LOGGER):
        released = late.run()

    assert released == [], f"the late replica claims a release: {released}"
    assert (late.reclaimed_total, peer.reclaimed_total) == (0, 1)
    assert len(core.patches) == 1, core.patches
    assert _released(core.nodes["hyperpod-i-1"]), "the reservation was kept"
    assert _sweep_lines(caplog, logging.WARNING) == [RECLAIM_LINE]
    assert _sweep_lines(caplog, logging.INFO) == [ALREADY_RELEASED_LINE]


def test_a_new_reservation_made_before_the_late_release_is_kept_and_not_counted(
    caplog,
) -> None:
    core = _stale_spare_core()
    late, peer = _replica(core), _replica(core)
    owner = HyperPodSpareCoordinator(FakeLifecycle([]), None, core, now=lambda: NOW)

    def peer_reclaims_then_a_new_incident_reserves() -> None:
        peer.run()
        owner.reserve(spare("i-1"), "hyperpod-i-1", "incident-new")

    late.coordinator.before_release = peer_reclaims_then_a_new_incident_reserves

    with caplog.at_level(logging.INFO, logger=SWEEP_LOGGER):
        released = late.run()

    node = core.nodes["hyperpod-i-1"]
    annotations = node["metadata"]["annotations"]
    assert released == [], f"the late replica claims a release: {released}"
    assert (late.reclaimed_total, peer.reclaimed_total) == (0, 1)
    assert annotations[SPARE_RESERVATION_ANNOTATION] == "incident-new", annotations
    assert annotations[SPARE_RESERVED_AT_ANNOTATION] == NOW.isoformat(), annotations
    assert annotations[SPARE_POOL_STATE_ANNOTATION] == "ALLOCATED", annotations
    assert node["spec"]["unschedulable"] is False, "the new reservation was undone"
    assert len(core.patches) == 2, f"only the reclaim and the reserve: {core.patches}"
    assert _sweep_lines(caplog, logging.WARNING) == [RECLAIM_LINE]
    assert _sweep_lines(caplog, logging.INFO) == [ALREADY_RELEASED_LINE]
