"""The declared capacity topology against the registered GPU fleet.

``spec.capacity.largestClusterNodeCount`` and ``managedNodeCount`` size the
per-cluster fault reserve and queue floor, the Aurora Min ACU floor and the
ingress normal-tier slots (``capacity_evidence``), while the fleet registers
its nodes by itself: a spot replacement, a raised ``desiredSize`` or
``TargetCount``, one more labelled group. A cluster's registered GPU nodes are
its fleet Agents ACTIVE or DRAINING -- the remediation storm breaker's count and
the control plane's ``gpu_fault_capacity_registered_*`` gauges
(``gpu_fault.app.capacity_metrics``, which also raise the
``GpuFault*NodesExceedDeclaredTopology`` alerts); synthetic capacity-drill
registrations are not managed clusters. Two administrator paths read them:

* ``status`` (``capacity_findings``, ``capacity_topology_field``) -- from the
  engine's status report alone: the declared counts of
  ``configured_release.admin_config.capacity`` and each configured cluster's
  Agents in the ``control_api`` check. One ``capacity:`` finding, a ``WARN
  capacity:`` finding per count the fleet exceeds, and the
  ``capacity_topology`` document field. Advisory: the checks, ``healthy`` and
  the exit code stay the engine's; no read or probe of its own. While the
  ``control_api`` check carries no report (it failed), nothing is said.
* ``config`` (``refuse_lowered_node_counts``) -- a target that LOWERS a declared
  count below the registered fleet is refused before anything is written,
  unless ``--accept-below-registered-fleet``. The §5 evidence floors of the
  capacity guide are validated against the declared counts, so a count below
  the live fleet lets ``maxClusterQueueDepth`` and ``minAcu`` fall under what
  the measurements require for the fleet really running, and the per-cluster
  fault reserve, ``max(maxClusterQueueDepth / 8, largestClusterNodeCount)``,
  below one whole-cluster wave. Raising or keeping a count reads and judges
  nothing; only a lowering reads the fleet, once, through the exec channel
  every administrator command uses (``run_control_plane_script``), and a
  lowering that cannot be judged is refused as well.

The deploy preflight does not judge this: a deploy must never fail because
spot capacity grew, and the preflight has no control-plane fleet read.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.capacity_evidence import (
    MAX_CLUSTER_QUEUE_DEPTH_PER_NODE,
    aurora_min_acu_floor,
)
from gpu_fault.admin.config import AdminConfig, CapacityConfig
from gpu_fault.admin.config_parser import AdminConfigError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import run_control_plane_script

ACCEPT_FLAG = "--accept-below-registered-fleet"
STATUS_FIELD = "capacity_topology"
LARGEST = "largestClusterNodeCount"
MANAGED = "managedNodeCount"
#: The lifecycle states of a registered GPU node's Agent (the storm breaker's).
REGISTERED_STATES = frozenset({"ACTIVE", "DRAINING"})
#: Clusters named per ``status`` line; the rest are in the document field.
LINE_LIMIT = 5
_READ_ERRORS = (BootstrapError, OSError, subprocess.SubprocessError, ValueError)

# Runs in a Running api-ha Pod on the deployed image (``config`` has already
# proved it runs this release): the control plane's own count, drills left out.
FLEET_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.app.capacity_metrics import registered_gpu_nodes

json.load(sys.stdin)
store = ApplicationContext.from_environment().store
synthetic = {
    registration.cluster_id
    for registration in store.list_regional_clusters()
    if registration.synthetic
}
clusters = {
    cluster_id: nodes
    for cluster_id, nodes in registered_gpu_nodes(store.list_agents()).items()
    if cluster_id not in synthetic
}
print(json.dumps({"clusters": clusters}, sort_keys=True))
"""


@dataclass(frozen=True)
class FleetTopology:
    """The declared counts and the registered GPU nodes of each cluster."""

    largest_declared: int
    managed_declared: int
    clusters: Mapping[str, int]

    @property
    def managed(self) -> int:
        return sum(self.clusters.values())

    def ranked(self) -> list[tuple[str, int]]:
        """Largest cluster first, then by name."""

        return sorted(self.clusters.items(), key=lambda item: (-item[1], item[0]))

    def clusters_above(self) -> list[tuple[str, int]]:
        return [item for item in self.ranked() if item[1] > self.largest_declared]

    def document(self) -> dict[str, Any]:
        return {
            "declared": {
                "largest_cluster_node_count": self.largest_declared,
                "managed_node_count": self.managed_declared,
            },
            "registered": {"clusters": dict(self.ranked()), "managed": self.managed},
            "clusters_above_declared": [name for name, _ in self.clusters_above()],
            "managed_above_declared": self.managed > self.managed_declared,
        }


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _mapping(value: object, *keys: str) -> Mapping[str, Any] | None:
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value if isinstance(value, Mapping) else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _control_api_clusters(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    health = report.get("health")
    checks = health.get("checks") if isinstance(health, Mapping) else None
    for check in checks if isinstance(checks, list) else []:
        if isinstance(check, Mapping) and check.get("name") == "control_api":
            return _mapping(check, "details", "clusters")
    return None


def _registered(cluster: object) -> int:
    agents = cluster.get("agents") if isinstance(cluster, Mapping) else None
    return sum(
        1
        for agent in (agents if isinstance(agents, list) else [])
        if isinstance(agent, Mapping)
        and agent.get("lifecycle_state") in REGISTERED_STATES
    )


def fleet_topology(report: Mapping[str, Any] | None) -> FleetTopology | None:
    """The topology a ``status`` report states, or ``None`` when it states
    either half not at all (no configured release, no ``control_api`` report)."""

    if not isinstance(report, Mapping):
        return None
    capacity = _mapping(report, "configured_release", "admin_config", "capacity")
    clusters = _control_api_clusters(report)
    if capacity is None or clusters is None:
        return None
    largest = _count(capacity.get("largest_cluster_node_count"))
    managed = _count(capacity.get("managed_node_count"))
    if largest is None or managed is None:
        return None
    return FleetTopology(
        largest_declared=largest,
        managed_declared=managed,
        clusters={str(name): _registered(item) for name, item in clusters.items()},
    )


def capacity_topology_field(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """``{"capacity_topology": ...}`` for the status document, or nothing."""

    topology = fleet_topology(report)
    return {} if topology is None else {STATUS_FIELD: topology.document()}


def _summary(topology: FleetTopology) -> str:
    ranked = topology.ranked()
    named = ", ".join(f"{name} {nodes}" for name, nodes in ranked[:LINE_LIMIT])
    if len(ranked) > LINE_LIMIT:
        named += f", and {len(ranked) - LINE_LIMIT} more in {STATUS_FIELD}"
    return (
        f"capacity: registered GPU nodes {topology.managed} of the declared "
        f"{MANAGED} {topology.managed_declared}; per cluster against the declared "
        f"{LARGEST} {topology.largest_declared}: {named or 'none'}"
    )


def capacity_findings(
    report: Mapping[str, Any] | None, *, state_dir: object = None
) -> list[str]:
    """The ``status`` findings: the summary, then one ``WARN`` per excess."""

    topology = fleet_topology(report)
    if topology is None:
        return []
    command = f"gpu-fault-admin config --state-dir {state_dir or '<state-dir>'}"
    lines = [_summary(topology)]
    above = topology.clusters_above()
    lines.extend(
        f"WARN capacity: cluster {name} has {nodes} registered GPU nodes, above the "
        f"declared {LARGEST} {topology.largest_declared}; raise "
        f"spec.capacity.{LARGEST} to at least {nodes} (spec.processor."
        f"maxClusterQueueDepth at least {MAX_CLUSTER_QUEUE_DEPTH_PER_NODE * nodes}) "
        f"and run {command}"
        for name, nodes in above[:LINE_LIMIT]
    )
    if len(above) > LINE_LIMIT:
        lines.append(
            f"WARN capacity: {len(above) - LINE_LIMIT} more clusters above the "
            f"declared {LARGEST} in {STATUS_FIELD}"
        )
    if topology.managed > topology.managed_declared:
        lines.append(
            f"WARN capacity: {topology.managed} registered GPU nodes, above the "
            f"declared {MANAGED} {topology.managed_declared}; raise "
            f"spec.capacity.{MANAGED} to at least {topology.managed} with "
            f"spec.aurora.minAcu at least {aurora_min_acu_floor(topology.managed):g} "
            f"(ceil({MANAGED} / 100, 0.5 ACU)) and run {command}"
        )
    return lines


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def add_accept_flag(config: argparse.ArgumentParser) -> None:
    config.add_argument(
        ACCEPT_FLAG,
        dest="accept_below_registered_fleet",
        action="store_true",
        help=(
            "apply a lowered largestClusterNodeCount or managedNodeCount even "
            "though it is below the registered GPU nodes (a fleet being shrunk, "
            "or departed nodes not retired yet)"
        ),
    )


def lowered_node_counts(
    current: CapacityConfig, desired: CapacityConfig
) -> tuple[str, ...]:
    """The declared counts ``desired`` lowers, by their YAML spelling."""

    return tuple(
        field
        for field, before, after in (
            (
                LARGEST,
                current.largest_cluster_node_count,
                desired.largest_cluster_node_count,
            ),
            (MANAGED, current.managed_node_count, desired.managed_node_count),
        )
        if after < before
    )


def counts_below_fleet(
    fleet: Mapping[str, int], desired: CapacityConfig, lowered: Sequence[str]
) -> list[str]:
    """One finding per lowered count that falls below the registered fleet."""

    findings: list[str] = []
    ranked = sorted(fleet.items(), key=lambda item: (-item[1], item[0]))
    if LARGEST in lowered and ranked:
        name, nodes = ranked[0]
        if desired.largest_cluster_node_count < nodes:
            findings.append(
                f"spec.capacity.{LARGEST} {desired.largest_cluster_node_count} is "
                f"below the {nodes} registered GPU nodes of cluster {name}"
            )
    total = sum(fleet.values())
    if MANAGED in lowered and desired.managed_node_count < total:
        findings.append(
            f"spec.capacity.{MANAGED} {desired.managed_node_count} is below the "
            f"{total} registered GPU nodes of the managed fleet"
        )
    return findings


def read_registered_fleet(site: RenderedSite) -> dict[str, int]:
    """Registered GPU nodes per managed cluster, as the control plane counts them."""

    result = run_control_plane_script(
        site, {"mode": "registered-fleet"}, script=FLEET_SCRIPT
    )
    clusters = result.get("clusters")
    if not isinstance(clusters, dict) or not all(
        isinstance(name, str) and _count(nodes) is not None and nodes >= 0
        for name, nodes in clusters.items()
    ):
        raise BootstrapError("the registered GPU fleet read returned an invalid answer")
    return {str(name): int(nodes) for name, nodes in clusters.items()}


def _warn(finding: str) -> None:
    print(
        f"gpu-fault-admin config: {ACCEPT_FLAG}: {finding}; applying anyway",
        file=sys.stderr,
        flush=True,
    )


def refuse_lowered_node_counts(
    site: RenderedSite,
    current: AdminConfig,
    desired: AdminConfig,
    *,
    accepted: bool,
    read: Callable[[RenderedSite], Mapping[str, int]] = read_registered_fleet,
    warn: Callable[[str], None] = _warn,
) -> None:
    """Refuse a target that lowers a declared count below the registered fleet.

    Nothing is read unless a count is lowered. ``accepted`` (the flag) turns
    each refusal, a failed read included, into a warning on stderr.
    """

    lowered = lowered_node_counts(current.capacity, desired.capacity)
    if not lowered:
        return
    try:
        fleet = read(site)
    except _READ_ERRORS as exc:
        finding = (
            "cannot read the registered GPU nodes to judge the lowered "
            + ", ".join(f"spec.capacity.{field}" for field in lowered)
            + f": {exc}"
        )
        if accepted:
            warn(finding)
            return
        raise AdminConfigError(
            f"{finding}; rerun when the control plane answers, or rerun with "
            f"{ACCEPT_FLAG}"
        ) from exc
    findings = counts_below_fleet(fleet, desired.capacity, lowered)
    if not findings:
        return
    if accepted:
        warn("; ".join(findings))
        return
    raise AdminConfigError(
        "refusing to lower the declared topology below the registered GPU fleet: "
        + "; ".join(findings)
        + ". The limits derived from these counts (the per-cluster fault reserve "
        "and queue floor, the Aurora Min ACU floor, the ingress normal-tier "
        "slots) would be sized for a smaller fleet than the one running; keep "
        "each count at least the registered nodes, or rerun with "
        f"{ACCEPT_FLAG} while the fleet is being shrunk or its count errs high "
        "(an Agent the departed-node cleanup has not retired yet still counts)"
    )
