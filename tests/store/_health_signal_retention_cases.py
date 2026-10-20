"""One site's health-signal rows, seeded through the public store API.

Shared by the memory/SQLite retention tests and the PostgreSQL ones, so the
three backends are held to one expectation (batch 145).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.departed_nodes import (
    DepartedNodeCleanupRecord,
    DepartedNodeCleanupStatus,
)
from gpu_fault.efa_traffic_state import (
    efa_traffic_state_key,
    legacy_efa_traffic_state_key,
)
from gpu_fault.store.shared.health_signals import (
    attempt_signal_key,
    training_signal_key,
)
from gpu_fault.telemetry import WorkloadCoverageHeartbeat
from gpu_fault.watcher import WorkloadPhase
from tests._builders import attempt_observation

# Fixed and far ahead of the wall clock: the sweep takes ``now`` from its
# caller, so nothing here ages into or out of a window on its own.
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
GRACE = timedelta(days=1)
RETENTION = timedelta(days=30)
FRESH = NOW - timedelta(hours=1)  # inside the grace
OLD = NOW - timedelta(days=2)  # past the grace, inside the retention
ANCIENT = NOW - timedelta(days=31)  # past the retention

CLUSTER = "cluster-a"
GPU_GATE = f"{CLUSTER}/node-a/host_gpu_utilization_percent/GPU-1/LOW_GPU_UTILIZATION"
CPU_GATE = f"{CLUSTER}/node-a/training_rank_cpu_percent/GPU-1/LOW_CPU_UTILIZATION"
LINK = f"{CLUSTER}/node-a/network_link_down/eth0"
MOUNT = (
    f"{CLUSTER}/node-a/local_filesystem_used_percent//fsx/HIGH_LOCAL_FILESYSTEM_USAGE"
)


def window(gate: str, job_id: str, attempt_id: str) -> str:
    return attempt_signal_key(gate, job_id, attempt_id)


def claim(store, key: str, at: datetime, *, active: bool = True) -> bool:
    return store.claim_health_signal_transitions(
        [(key, active, at, 0.0)], received_at=at
    )[0]


def latch(store, key: str, at: datetime) -> None:
    store.mark_health_signal_notified(key, notified_at=at)


def sweep(store, *, limit: int = 100) -> dict[str, int]:
    return store.cleanup_health_signal_states(
        now=NOW, grace=GRACE, retention=RETENTION, limit=limit
    )


def watch(store, at: datetime = FRESH) -> None:
    """A watcher's full pass over the cluster: it vouches for the cluster."""

    store.save_workload_coverage_heartbeat(
        WorkloadCoverageHeartbeat(
            cluster_id=CLUSTER, observed_at=at, watcher_instance="watcher-a"
        )
    )


def observe(
    store, job_id: str, attempt_id: str, phase: WorkloadPhase, at: datetime = FRESH
) -> None:
    store.save_attempt_observation(
        attempt_observation(job_id, attempt_id, at, workload_phase=phase)
    )


def depart(store, node_id: str, *, confirmed_at: datetime, proven: bool = True) -> None:
    """The departed-node cleanup's ledger line for ``node_id``."""

    store.save_departed_node_cleanup(
        DepartedNodeCleanupRecord(
            cluster_id=CLUSTER,
            node_id=node_id,
            status=(
                DepartedNodeCleanupStatus.CLEANED
                if proven
                else DepartedNodeCleanupStatus.PENDING
            ),
            reference=f"departed-node-cleanup/{node_id}" if proven else None,
            first_seen_at=confirmed_at - timedelta(minutes=10),
            updated_at=confirmed_at,
            cleaned_at=confirmed_at if proven else None,
        )
    )


def efa(
    store,
    node_id: str,
    at: datetime,
    *,
    job_id: str = "job-a",
    attempt_id: str = "attempt-1",
    state_key: str | None = None,
    bps: float = 1000.0,
):
    """One EFA traffic sample; ``state_key`` names a pre-batch-145 row."""

    return store.observe_efa_traffic(
        state_key=state_key or efa_traffic_state_key(CLUSTER, node_id),
        cluster_id=CLUSTER,
        node_id=node_id,
        job_id=job_id,
        attempt_id=attempt_id,
        observed_at=at,
        bytes_per_second=bps,
        minimum_active_bps=100.0,
        spike_ratio=4.0,
        drop_ratio=0.25,
        zero_bps=1.0,
        zero_warning_seconds=60.0,
        zero_hung_seconds=600.0,
        baseline_alpha=0.2,
        startup_grace_seconds=120.0,
        spike_event_id=f"{node_id}-{at.isoformat()}-efa-traffic-spike",
    )


def health_rows(store, keys) -> set[str]:
    return {key for key in keys if store.get_health_signal_state(key) is not None}


def seed_site(store) -> tuple[dict[str, str], dict[str, str]]:
    """A mixed site. Returns ``(kept, removed)``: health-signal key -> why."""

    watch(store)
    observe(store, "job-a", "attempt-live", WorkloadPhase.RUNNING)
    observe(store, "job-a", "attempt-done", WorkloadPhase.SUCCEEDED)
    kept = {
        window(GPU_GATE, "job-a", "attempt-live"): "a live attempt, however old",
        GPU_GATE: "the unscoped gate of a live node",
        MOUNT: "a live node's row inside the retention (mount-point device)",
        window(GPU_GATE, "job-a", "attempt-recent"): "written inside the grace",
        f"{CLUSTER}/node-c/network_link_down/eth0": "an unproven departure",
        training_signal_key(CLUSTER, "attempt-live", 0, "hang"): "live training",
    }
    removed = {
        window(GPU_GATE, "job-a", "attempt-done"): "attempt_ended",
        window(CPU_GATE, "job-a", "attempt-done"): "attempt_ended (latched)",
        window(GPU_GATE, "job-a", "attempt-unseen"): "attempt_ended (unobserved)",
        training_signal_key(CLUSTER, "attempt-done", 3, "straggler"): "attempt_ended",
        f"{CLUSTER}/node-b/network_link_down/eth0": "node_departed (latched)",
        f"{CLUSTER}/node-b/host_gpu_utilization_percent/GPU-9/LOW_GPU_UTILIZATION": (
            "node_departed"
        ),
        LINK: "not_updated",
    }
    claim(store, window(GPU_GATE, "job-a", "attempt-live"), ANCIENT)
    claim(store, GPU_GATE, OLD)
    claim(store, MOUNT, OLD)
    claim(store, window(GPU_GATE, "job-a", "attempt-recent"), FRESH)
    claim(store, f"{CLUSTER}/node-c/network_link_down/eth0", OLD)
    depart(store, "node-c", confirmed_at=FRESH, proven=False)
    claim(store, training_signal_key(CLUSTER, "attempt-live", 0, "hang"), OLD)
    claim(store, window(GPU_GATE, "job-a", "attempt-done"), OLD)
    claim(store, window(CPU_GATE, "job-a", "attempt-done"), OLD)
    latch(store, window(CPU_GATE, "job-a", "attempt-done"), OLD)
    claim(store, window(GPU_GATE, "job-a", "attempt-unseen"), OLD)
    claim(store, training_signal_key(CLUSTER, "attempt-done", 3, "straggler"), OLD)
    claim(store, f"{CLUSTER}/node-b/network_link_down/eth0", OLD)
    latch(store, f"{CLUSTER}/node-b/network_link_down/eth0", OLD)
    claim(
        store,
        f"{CLUSTER}/node-b/host_gpu_utilization_percent/GPU-9/LOW_GPU_UTILIZATION",
        OLD,
    )
    depart(store, "node-b", confirmed_at=FRESH)
    claim(store, LINK, ANCIENT)
    return kept, removed


def seed_efa(store) -> tuple[set[str], set[str]]:
    """EFA rows: ``(kept, removed)`` state keys."""

    efa(store, "node-a", OLD)  # a live node's row
    efa(store, "node-b", OLD)  # node-b departed (``seed_site``)
    efa(store, "node-d", ANCIENT)  # nothing wrote it for the retention
    legacy_old = legacy_efa_traffic_state_key(
        efa_traffic_state_key(CLUSTER, "node-e"), "job-a", "attempt-old"
    )
    legacy_recent = legacy_efa_traffic_state_key(
        efa_traffic_state_key(CLUSTER, "node-e"), "job-a", "attempt-recent"
    )
    # Rows a release before batch 145 wrote: one per attempt.
    efa(store, "node-e", OLD, attempt_id="attempt-old", state_key=legacy_old)
    efa(store, "node-e", FRESH, attempt_id="attempt-recent", state_key=legacy_recent)
    kept = {efa_traffic_state_key(CLUSTER, "node-a"), legacy_recent}
    removed = {
        efa_traffic_state_key(CLUSTER, "node-b"),
        efa_traffic_state_key(CLUSTER, "node-d"),
        legacy_old,
    }
    return kept, removed


def efa_rows(store, keys) -> set[str]:
    from gpu_fault.store.shared.errors import NotFoundError

    present = set()
    for key in keys:
        try:
            store.get_efa_traffic_state(key)
        except NotFoundError:
            continue
        present.add(key)
    return present
