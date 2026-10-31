"""The node-ownership families of ``/metrics``: sparse per node, dense per cluster.

``EvidenceOperationService.ownership_metric_snapshot`` answers, for every GPU
node it evaluates -- each registered Agent's node and each node an active
attempt observation of the scanned slice names -- how many fresh active
attempts claim it and how many stale observations it keeps. This module
renders that answer.

Both per-node families used to carry every evaluated node: a zero on an idle
node, ``1`` (its single owner) on a busy one. That is two lines per registered
Agent in every publication of every control-plane process, parsed and encoded
again every 5 s and merged on every scrape: at 12,800 Agents 25,600 lines and
3.5 MB per publication, about 0.4 s of CPU per process each time, on the
interpreter the workflow-dispatch lease holder shares. Neither value is what an
alert reads: GpuFaultExclusiveNodeOwnershipInvariantViolation fires on more
than one fresh attempt, GpuFaultStaleAttemptObservation on a stale one.

``gpu_fault_ambiguous_attempt_ownership_current`` therefore carries only a node
more than one fresh attempt claims, and ``gpu_fault_stale_attempt_observations``
only a node with a stale observation; a node's series ends (Prometheus
staleness) when its condition clears. Four per-cluster gauges say what the
other lines said, for every cluster with an evaluated node, zero included:

``gpu_fault_attempt_observation_scan_nodes``
    nodes evaluated, the coverage the zero series used to show;
``gpu_fault_attempt_observation_scan_owned_nodes``
    nodes at least one fresh attempt claims, ambiguous ones included -- what
    the value-1 series carried;
``gpu_fault_ambiguous_attempt_ownership_nodes``
    nodes more than one fresh attempt claims;
``gpu_fault_stale_attempt_observation_nodes``
    nodes with at least one stale observation.

Every family keeps its HELP/TYPE lines when it has no sample. The Pod-level
merge (``process_metrics.aggregate``, ANY strategy) answers with the first
process that declares a family, so an empty family is this process's answer
and a sibling's older publication cannot stand in for it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

#: Fresh attempts on one node at which its exclusive ownership is violated:
#: GpuFaultExclusiveNodeOwnershipInvariantViolation reads ``> 1``.
AMBIGUOUS_OWNERS = 2

NodeCounts = Mapping[tuple[str, str], int]


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def ownership_metric_lines(snapshot: Mapping[str, NodeCounts]) -> list[str]:
    """Render an ownership snapshot (``{"current": ..., "stale": ...}``)."""

    current = snapshot["current"]
    stale = snapshot["stale"]
    ambiguous_rows = sorted(
        (key, count) for key, count in current.items() if count >= AMBIGUOUS_OWNERS
    )
    stale_rows = sorted((key, count) for key, count in stale.items() if count >= 1)
    evaluated = Counter(cluster for cluster, _node in current.keys() | stale.keys())
    owned = Counter(
        cluster for (cluster, _node), count in current.items() if count >= 1
    )
    ambiguous_by_cluster = Counter(cluster for (cluster, _node), _ in ambiguous_rows)
    stale_by_cluster = Counter(cluster for (cluster, _node), _ in stale_rows)
    clusters = [(cluster, _escape_label(cluster)) for cluster in sorted(evaluated)]
    lines = [
        "# HELP gpu_fault_ambiguous_attempt_ownership_current Fresh active "
        "attempts claiming one GPU node, rendered only for a node more than one "
        "claims; gpu_fault_ambiguous_attempt_ownership_nodes counts those nodes "
        "per cluster.",
        "# TYPE gpu_fault_ambiguous_attempt_ownership_current gauge",
    ]
    for (cluster_id, node_id), count in ambiguous_rows:
        lines.append(
            "gpu_fault_ambiguous_attempt_ownership_current"
            f'{{cluster_id="{_escape_label(cluster_id)}",'
            f'gpu_node="{_escape_label(node_id)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_stale_attempt_observations Latest active attempt "
            "observations older than the ownership freshness window for one GPU "
            "node, rendered only for a node that keeps one; "
            "gpu_fault_stale_attempt_observation_nodes counts those nodes per "
            "cluster.",
            "# TYPE gpu_fault_stale_attempt_observations gauge",
        ]
    )
    for (cluster_id, node_id), count in stale_rows:
        lines.append(
            "gpu_fault_stale_attempt_observations"
            f'{{cluster_id="{_escape_label(cluster_id)}",'
            f'gpu_node="{_escape_label(node_id)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_attempt_observation_scan_nodes GPU nodes the "
            "ownership and staleness families evaluated, by cluster: every "
            "registered Agent's node and every node an active observation in the "
            "scanned slice names.",
            "# TYPE gpu_fault_attempt_observation_scan_nodes gauge",
        ]
    )
    for cluster, label in clusters:
        lines.append(
            "gpu_fault_attempt_observation_scan_nodes"
            f'{{cluster_id="{label}"}} {evaluated[cluster]}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_attempt_observation_scan_owned_nodes Evaluated GPU "
            "nodes at least one fresh active attempt claims, by cluster; nodes "
            "more than one claims are included.",
            "# TYPE gpu_fault_attempt_observation_scan_owned_nodes gauge",
        ]
    )
    for cluster, label in clusters:
        lines.append(
            "gpu_fault_attempt_observation_scan_owned_nodes"
            f'{{cluster_id="{label}"}} {owned[cluster]}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_ambiguous_attempt_ownership_nodes Evaluated GPU "
            "nodes more than one fresh active attempt claims, by cluster; 0 while "
            "every node has one owner or none.",
            "# TYPE gpu_fault_ambiguous_attempt_ownership_nodes gauge",
        ]
    )
    for cluster, label in clusters:
        lines.append(
            "gpu_fault_ambiguous_attempt_ownership_nodes"
            f'{{cluster_id="{label}"}} {ambiguous_by_cluster[cluster]}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_stale_attempt_observation_nodes Evaluated GPU nodes "
            "keeping an active attempt observation older than the ownership "
            "freshness window, by cluster; 0 while none does.",
            "# TYPE gpu_fault_stale_attempt_observation_nodes gauge",
        ]
    )
    for cluster, label in clusters:
        lines.append(
            "gpu_fault_stale_attempt_observation_nodes"
            f'{{cluster_id="{label}"}} {stale_by_cluster[cluster]}'
        )
    return lines
