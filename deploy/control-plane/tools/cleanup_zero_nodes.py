"""The node cleanup of a GPU cluster without nodes: an empty target set, proven.

``uninstall`` and ``remove-cluster`` stop and uninstall the node runtime with a
DaemonSet pinned to the fleet's nodes (``cleanup_kubernetes.node_targets``). A
cluster whose recorded node selector selects no Kubernetes node at all -- parked
at zero by its provider, or unavailable (``gpu_fault_release.regional_parked_cluster``)
-- has no node to run that DaemonSet on, while the fleet snapshot still holds
the records of the nodes that left: the Agents the departed-node cleanup retired
(``REVOKED``) and any it has not retired yet. ``node_targets`` read that as "no
proven fleet targets" and refused, so neither verb could finish while the
cluster had no node, and the open uninstall journal then refused every deploy of
the site.

The empty target set is proven, for parked and unavailable alike, when

* the API lists no node for the cluster's recorded selector: the release
  engine's own read and proof (``gpu_fault.admin.bootstrap_zero_nodes``), so a
  node list that cannot be read proves nothing;
* every fleet record of the cluster names a node Kubernetes no longer has --
  a listed one would be a target -- and every record not yet retired (``ACTIVE``
  or ``DRAINING``) is proven departed by its provider
  (``cleanup_kubernetes.departed_fleet_node``: HyperPod no longer lists the
  instance, or EC2 reports it terminated or unknown); ``REVOKED`` records were
  retired on that same proof;
* no node of the cluster carries anything of ours outside the targets, the
  check ``node_targets`` makes for every cluster right after this one.

The proof read's state, reason and covered groups are recorded with the
decision (``zero_node_clusters.<cluster>`` in the cleanup state) and narrated;
they do not make it -- the node list and the fleet records do -- so a provider
that cannot be read is recorded as the reason of an unavailable cluster and the
cleanup still proceeds. Nothing is deleted on any node: there is none; the
node-key Secrets go with the namespaces as for every cluster.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from cleanup_state import CleanupStateError

#: The cleanup-state key of the clusters whose node cleanup had no node.
ZERO_NODE_KEY = "zero_node_clusters"
#: A retired fleet record: the departed-node cleanup ended it with its proof.
RETIRED = "REVOKED"
#: Records not yet retired, which need the provider's departure proof here.
NOT_RETIRED = frozenset({"ACTIVE", "DRAINING"})
#: The narration of the skipped node cleanup.
STEP = "node-cleanup"
RECORDED = "no node, nothing to clean"

DepartureProof = Callable[[dict[str, Any]], dict[str, Any]]


def read_proof(config: dict[str, Any], member: dict[str, Any]) -> dict[str, Any] | None:
    """The engine's proof of ``member`` when its selector selects no node, else
    ``None`` (a selected node, or a node list that cannot be read)."""

    from gpu_fault.admin.bootstrap_common import CommandRunner
    from gpu_fault.admin.bootstrap_zero_nodes import zero_node_proof_for_entry

    kubeconfig = config.get("gpu_kubeconfig")
    return zero_node_proof_for_entry(
        CommandRunner(),
        gpu_kubeconfig=Path(str(kubeconfig)) if kubeconfig else None,
        entry=member,
        region=str(config.get("aws_region") or ""),
    )


def zero_node_targets(
    config: dict[str, Any] | None,
    member: dict[str, Any],
    agents: list[dict[str, Any]],
    cluster_id: str,
    *,
    departed: list[dict[str, Any]],
    prove_departure: DepartureProof,
) -> dict[str, Any]:
    """The record of ``cluster_id``'s proven empty target set, or the refusal.

    ``member`` is the cluster's entry of the bound ``config``, ``agents`` the
    fleet snapshot, ``departed`` the departures ``node_targets`` already proved
    (its ``ACTIVE`` records); the ``DRAINING`` ones are proved here and appended
    to it. Without a bound config nothing is proven.
    """

    proof = read_proof(config, member) if config else None
    if proof is None:
        raise CleanupStateError(
            "node cleanup has no proven fleet targets: no fleet record names a "
            "node of this cluster, and its node selector still selects a node "
            "or its node list cannot be read"
        )
    from gpu_fault_release import regional_parked_cluster as parked_cluster

    proven = {str(entry.get("node_id")) for entry in departed}
    retired: list[str] = []
    for agent in agents:
        if agent.get("cluster_id") != cluster_id:
            continue
        name = str(agent.get("node_id"))
        state = agent.get("lifecycle_state")
        if state == RETIRED:
            retired.append(name)
        elif state not in NOT_RETIRED:
            raise CleanupStateError(
                f"node cleanup has no proven fleet targets: fleet record {name} "
                f"has lifecycle {state!r}, neither retired nor provable"
            )
        elif name not in proven:
            departed.append(prove_departure(agent))
            proven.add(name)
    entry = {
        **parked_cluster.summary(proof),
        "retired_agents": sorted(retired),
        "departed_agents": sorted(proven),
    }
    parked_cluster.narrate(
        cluster_id,
        state=entry["state"],
        step=STEP,
        recorded=RECORDED,
        retired=len(entry["retired_agents"]),
        departed=len(entry["departed_agents"]),
        reason=str(entry["reason"]),
    )
    return entry


def skipped_reason(document: dict[str, Any], cluster_id: str) -> str:
    """Why a cluster's node cleanup ran no DaemonSet."""

    entry = (document.get(ZERO_NODE_KEY) or {}).get(cluster_id)
    if not isinstance(entry, dict):
        return "no node runtime was installed on this cluster"
    return (
        f"GPU cluster without nodes ({entry.get('state')}): no node is selected "
        "and every fleet record is retired or proven departed; nothing to clean"
    )
