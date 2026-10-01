"""Agent coverage for the ``control_api`` verification check.

The fleet's ``/v1/fleet/agents`` lists every Agent record the control plane
has, whatever its lifecycle state, and the verification compares that list
with the cluster's Ready Kubernetes Nodes. Two residues of a node HyperPod
reclaimed used to turn that into a deploy that could never verify: the
record of the departed node stays ``ACTIVE`` (one Agent more than there are
nodes), and once the administrator retires it the ``REVOKED`` record would
still have been counted and failed the non-ACTIVE check.

A retired Agent is not coverage: ``REVOKED`` records are left out before the
comparison. Coverage drift stays a hard failure -- a node that is NotReady is
still a node that must be covered -- but when every extra Agent has no
Kubernetes Node at all the refusal names the remedy, so the operator reads
the next command instead of a bare set difference.
"""

from __future__ import annotations

from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError

RETIRE_DEPARTED_AGENTS_HINT = (
    "these Agents have no Kubernetes Node; if HyperPod no longer lists their "
    "instances, retire them with gpu-fault-admin workflow-reconcile --state-dir "
    "<state-dir> --retire-departed-agents --dry-run, then without --dry-run and "
    "with --reference"
)


def annotate_kubernetes_nodes(
    report: dict[str, Any], node_ids_by_cluster: dict[str, list[str]]
) -> None:
    """Record every Kubernetes Node name the report's clusters have, in place.

    ``expected_node_ids`` is the Ready subset the probe was asked about; the
    coverage refusal needs the whole list to tell an Agent whose node is gone
    from one whose node is merely NotReady.
    """

    clusters = report.get("clusters")
    if not isinstance(clusters, dict):
        return
    for cluster_id, node_ids in node_ids_by_cluster.items():
        cluster = clusters.get(cluster_id)
        if isinstance(cluster, dict):
            cluster["kubernetes_node_ids"] = sorted(node_ids)


def counted_agents(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    """The Agents that count as coverage: every lifecycle state but REVOKED."""

    return [
        item
        for item in cluster.get("agents") or []
        if isinstance(item, dict) and item.get("lifecycle_state") != "REVOKED"
    ]


def check_agent_coverage(
    cluster_id: str, cluster: dict[str, Any]
) -> list[dict[str, Any]]:
    """Refuse a cluster whose counted Agents do not match its Ready Nodes.

    Returns the counted Agents for the checks that follow (Runtime Profile,
    readiness), so a retired record is judged by none of them.
    """

    agents = counted_agents(cluster)
    if not agents:
        raise ReleaseError(f"{cluster_id} has no registered Node Agents")
    expected_nodes = {str(item) for item in cluster.get("expected_node_ids") or []}
    agent_nodes = {str(item.get("node_id")) for item in agents}
    if agent_nodes != expected_nodes:
        message = (
            f"{cluster_id} Agent coverage drift: "
            f"expected={sorted(expected_nodes)}, agents={sorted(agent_nodes)}"
        )
        extra = agent_nodes - expected_nodes
        kubernetes_nodes = cluster.get("kubernetes_node_ids")
        if (
            extra
            and isinstance(kubernetes_nodes, list)
            and not (extra & {str(item) for item in kubernetes_nodes})
        ):
            message += (
                f"; Agents without a Kubernetes Node: {sorted(extra)}; "
                + RETIRE_DEPARTED_AGENTS_HINT
            )
        raise ReleaseError(message)
    if any(item.get("lifecycle_state") != "ACTIVE" for item in agents):
        raise ReleaseError(f"{cluster_id} has a non-ACTIVE Node Agent")
    return agents
