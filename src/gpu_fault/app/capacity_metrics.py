"""The registered GPU fleet next to the declared capacity topology.

The administrator declares the site's topology -- the largest GPU cluster and
the whole managed fleet, ``spec.capacity.largestClusterNodeCount`` and
``managedNodeCount`` (``gpu_fault.admin.capacity_evidence``) -- and four
runtime limits are derived from it: the per-cluster queue floor, the
per-cluster fault reserve that must hold a whole-cluster wave of faults, the
Aurora Min ACU floor and the ingress normal-tier concurrency. Node groups and
instance groups grow without the administrator (a spot replacement, a raised
``desiredSize`` or ``TargetCount``, one more labelled group), and the installer
reconciler registers every new node's Agent, so a cluster can outgrow its
declaration while every limit stays sized for the smaller one. Two families
put the real fleet beside the declaration:

* ``gpu_fault_capacity_registered_cluster_node_count{cluster_id}`` -- one
  managed cluster's registered GPU nodes: its fleet Agents ACTIVE or DRAINING,
  one per GPU node, the count the remediation storm breaker judges with
  (``execution/storm_admission``). A departed node counts until the
  departed-node cleanup retires its Agent, never before the 10-minute
  departure guard, so the count errs high for minutes after a spot
  replacement; the alerts' ``for`` rides that out.
* ``gpu_fault_capacity_registered_managed_node_count`` -- the sum over every
  managed cluster; 0 for an empty fleet.

Synthetic capacity-drill registrations (the performance runner's clusters)
are not managed GPU clusters and are left out of both: a drill declares its
topology with a capacity preset, and ``gpu-fault-admin status`` counts the
site's own clusters only.

Cost: no Store read of its own. The closed-loop and ownership families already
read the whole agent table once per scan-cache TTL on every control-worker
process (``MetricScanCache.agents``); the tally is made from that tuple once
per TTL (``MetricScanCache.shared``), and the synthetic registrations come
from the process's in-memory regional registry snapshot. One series per
cluster, never one per node.

Which processes render them, and how the Pod merges them: the contributor is
fleet-level, so only the control-worker role renders it and ingress and
spool-worker never repeat the read. Every control-worker process reads the same
table, so the families aggregate ANY (``metric_aggregation``): the local render
wins, an empty fleet's declared-but-empty family included. A process whose read
failed renders neither family -- the contributor raises, the registry skips and
counts it (``gpu_fault_metrics_contributor_errors_total``) -- so the merge
answers with a sibling's recent good publication. A failed read is never
rendered, or kept, as a count of zero (b227's rule for the breaker's count).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from gpu_fault.app.metric_scan_cache import metric_scan_cache
from gpu_fault.app.runtime import AppRuntime
from gpu_fault.fleet import AgentLifecycleState

#: The lifecycle states of a registered GPU node's Agent: the storm breaker's
#: definition of "the cluster's GPU nodes". ``REVOKED`` is a retired Agent.
REGISTERED_AGENT_STATES = frozenset(
    {AgentLifecycleState.ACTIVE, AgentLifecycleState.DRAINING}
)
REGISTERED_CLUSTER_NODES = "gpu_fault_capacity_registered_cluster_node_count"
REGISTERED_MANAGED_NODES = "gpu_fault_capacity_registered_managed_node_count"
#: The scan-cache key of the per-cluster tally (one tally per TTL).
TALLY_KEY = "capacity-registered-gpu-nodes"


def registered_gpu_nodes(agents: Iterable[Any]) -> dict[str, int]:
    """Registered GPU nodes by cluster: the Agents ACTIVE or DRAINING."""

    return dict(
        Counter(
            str(agent.cluster_id)
            for agent in agents
            if agent.lifecycle_state in REGISTERED_AGENT_STATES
        )
    )


def synthetic_cluster_ids(context: Any) -> frozenset[str]:
    """The clusters the regional registry marks synthetic (capacity drills).

    None outside regional mode, where no registry runs. A registry whose
    snapshot has not loaded raises: rendering the drill clusters' nodes as
    managed ones would be a wrong count, not a missing one.
    """

    registry = getattr(context, "regional_registry_runtime", None)
    if registry is None:
        return frozenset()
    return frozenset(
        str(cluster_id)
        for cluster_id, registration in registry.snapshot().registrations.items()
        if registration.synthetic
    )


def managed_gpu_nodes(
    by_cluster: Mapping[str, int], synthetic: Iterable[str]
) -> dict[str, int]:
    """``by_cluster`` without the synthetic clusters."""

    excluded = frozenset(synthetic)
    return {
        cluster_id: nodes
        for cluster_id, nodes in by_cluster.items()
        if cluster_id not in excluded
    }


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def registered_node_metric_lines(runtime: AppRuntime) -> list[str]:
    """The two registered-fleet families (module docstring)."""

    cache = metric_scan_cache(runtime)
    by_cluster: dict[str, int] = cache.shared(
        TALLY_KEY, lambda: registered_gpu_nodes(cache.agents())
    )
    managed = managed_gpu_nodes(by_cluster, synthetic_cluster_ids(runtime.context))
    lines = [
        "# HELP gpu_fault_capacity_registered_cluster_node_count Registered GPU "
        "nodes of one managed GPU cluster: its fleet Agents ACTIVE or DRAINING, "
        "the count the remediation storm breaker judges with; compare with the "
        "declared largestClusterNodeCount. Synthetic capacity-drill clusters are "
        "left out.",
        "# TYPE gpu_fault_capacity_registered_cluster_node_count gauge",
    ]
    lines.extend(
        "gpu_fault_capacity_registered_cluster_node_count"
        f'{{cluster_id="{_escape_label(cluster_id)}"}} {nodes}'
        for cluster_id, nodes in sorted(managed.items())
    )
    lines.extend(
        [
            "# HELP gpu_fault_capacity_registered_managed_node_count Registered "
            "GPU nodes across every managed GPU cluster; compare with the declared "
            "managedNodeCount.",
            "# TYPE gpu_fault_capacity_registered_managed_node_count gauge",
            f"gpu_fault_capacity_registered_managed_node_count {sum(managed.values())}",
        ]
    )
    return lines
