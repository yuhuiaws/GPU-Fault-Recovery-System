"""Full-allocation coverage from stored telemetry, without node-side attachments."""

from collections import Counter
from typing import Any

from gpu_fault.observation_health import observation_reason


def hung_task_summary(
    store: Any, finding: Any, observation: Any, nodes: list[str]
) -> dict[str, Any]:
    now = finding.observed_at
    summary = {
        "version": "hung-task-summary/v1",
        "source": "stored_control_plane_observations",
        "snapshot_at": now.isoformat(),
        "expected_node_count": len(nodes),
        "allocation_proven": observation is not None,
        "rank_inventory_basis": "watcher_containers_not_all_torch_workers",
        "rank_inventory_complete": False,
        "rdma_scope": "reporting_node_only",
        "reporting_node": finding.node_id,
        "reporting_node_rdma_bytes_per_second": finding.diagnostic_parameters.get(
            "observed_bytes_per_second"
        ),
    }
    if observation is None:
        return {**summary, "status": "ALLOCATION_UNKNOWN"}
    node_set = set(nodes)
    containers = [
        item
        for item in observation.containers
        if not item.terminated and item.node_id in node_set
    ]
    # One bulk read; NCCL diagnostics do not require application instrumentation.
    statuses = {
        (item.node_id, item.collector.value): item
        for item in store.list_collector_statuses(finding.cluster_id)
    }
    channel_counts: Counter[str] = Counter()
    rows = [
        {"node_id": container.node_id, "container_rank": container.rank}
        for container in containers[:256]
    ]
    for node in nodes:
        for channel in ("HOST_TELEMETRY", "GPU_METRICS"):
            if observation_reason(statuses.get((node, channel)), now, 420) is None:
                channel_counts[channel] += 1
    return {
        **summary,
        "status": "AVAILABLE",
        "job_id": observation.job_id,
        "attempt_id": observation.attempt_id,
        "allocation_observed_at": observation.observed_at.isoformat(),
        "allocated_container_count": len(containers),
        "fresh_node_channels": dict(channel_counts),
        "allocation_rows": rows,
        "allocation_rows_truncated": len(containers) > len(rows),
    }
