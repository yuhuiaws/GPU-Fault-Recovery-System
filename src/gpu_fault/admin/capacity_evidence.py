"""Capacity rules that encode the repository's own performance evidence.

Every constant here is a measurement from ``docs/性能压测验收方案.md``:

* section 13.4 -- one 1000-node cluster against a per-cluster queue depth of
  1024 produced 243 HTTP 429 from
  ``gpu_fault_telemetry_spool_rejected_total{scope="cluster"}``; raising the
  depth to 4096 produced 0. Four requests of depth per node is the ratio that
  held.
* section 13.3 -- 32-50 clusters x 256 nodes (about 12,800 nodes) against an
  Aurora ``0.5/96 low-start`` produced 3076 HTTP 503; pre-provisioning
  Min ACU at about 124-128 produced 0. One ACU per hundred managed nodes is
  the floor that matches the measurement.
* product requirement -- N clusters of 500+ nodes, single-AZ. A whole-cluster
  correlated fault is at least one fault-priority request per node on that
  cluster's lanes, so the fault-reserved per-cluster depth must hold the
  largest cluster, not a fixed eighth of the queue depth.

The functions are pure so that the admin config, the two manifest renderers
and the runtime guard all compute the same number from the same inputs.
"""

from __future__ import annotations

import math

DEFAULT_LARGEST_CLUSTER_NODE_COUNT = 512
DEFAULT_MANAGED_NODE_COUNT = 512
MAX_CLUSTER_QUEUE_DEPTH_PER_NODE = 4
NODES_PER_AURORA_ACU = 100
FAULT_RESERVE_DIVISOR = 8


class CapacityEvidenceError(ValueError):
    pass


def aurora_min_acu_floor(managed_node_count: int) -> float:
    """Minimum Aurora Min ACU for the fleet, rounded up to the 0.5 ACU step.

    12,800 nodes -> 128.0 (the section 13.3 measurement), 512 -> 5.5,
    2048 -> 20.5. Never below the 0.5 ACU Serverless v2 minimum.
    """

    half_steps: int = math.ceil(managed_node_count * 2 / NODES_PER_AURORA_ACU)
    return max(0.5, half_steps / 2)


def minimum_cluster_queue_depth(largest_cluster_node_count: int) -> int:
    return MAX_CLUSTER_QUEUE_DEPTH_PER_NODE * largest_cluster_node_count


def legacy_largest_cluster_node_count(max_cluster_queue_depth: int) -> int:
    """The largest cluster a pre-node-count desired config can be read as.

    Persisted configs written before node counts existed carry no topology
    and usually the old 1024 depth; reading them as the new 512-node default
    would make them invalid on load. The depth they did declare bounds the
    cluster they were sized for: 1024 -> 256 nodes, 4096 or more -> 512.
    """

    return max(
        1,
        min(
            DEFAULT_LARGEST_CLUSTER_NODE_COUNT,
            max_cluster_queue_depth // MAX_CLUSTER_QUEUE_DEPTH_PER_NODE,
        ),
    )


def fault_reserved_queue_depth(max_queue_depth: int) -> int:
    return max(1, max_queue_depth // FAULT_RESERVE_DIVISOR)


def fault_reserved_cluster_depth(
    max_cluster_queue_depth: int,
    largest_cluster_node_count: int,
) -> int:
    """Per-cluster fault reserve: an eighth of the depth or one whole-cluster wave."""

    return max(
        max_cluster_queue_depth // FAULT_RESERVE_DIVISOR,
        largest_cluster_node_count,
    )


def validate_capacity_evidence(
    *,
    largest_cluster_node_count: int,
    managed_node_count: int,
    max_cluster_queue_depth: int,
    aurora_min_acu: float,
) -> None:
    """Refuse a desired config the perf evidence says will reject traffic."""

    required_depth: int = minimum_cluster_queue_depth(largest_cluster_node_count)
    if max_cluster_queue_depth < required_depth:
        raise CapacityEvidenceError(
            f"spec.processor.maxClusterQueueDepth {max_cluster_queue_depth} must "
            f"be at least {MAX_CLUSTER_QUEUE_DEPTH_PER_NODE} x "
            f"spec.capacity.largestClusterNodeCount {largest_cluster_node_count} "
            f"= {required_depth}; a 1000-node cluster overflowed depth 1024 and "
            "needed 4096 (性能压测验收方案 §13.4)"
        )
    floor: float = aurora_min_acu_floor(managed_node_count)
    if aurora_min_acu < floor:
        raise CapacityEvidenceError(
            f"spec.aurora.minAcu {aurora_min_acu:g} is below the {floor:g} ACU "
            f"floor for spec.capacity.managedNodeCount {managed_node_count}; "
            "12,800 nodes returned 3076 HTTP 503 until Min ACU was "
            "pre-provisioned at about 128 (性能压测验收方案 §13.3)"
        )
