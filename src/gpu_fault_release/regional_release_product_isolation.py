"""Nodes the product itself has isolated, kept out of a release's node wave.

Live 2026-10-05 (mixed site, gpu-eks-cluster): during the EKS node wave the
product quarantined ip-10-0-41-164 (a VALIDATE_GPU failure, then the
drain-after rung's MARK_UNSCHEDULABLE + QUARANTINE). The wave's node-state gate
read the cordon and the ``gpu-fault.io/quarantined`` taint as "unsafe", raised
a plain ``ReleaseError`` -- a *global* failure -- and the whole release rolled
back for twenty minutes, HyperPod included, because one GPU node was being
repaired by the very product the release was deploying.

A node the product isolates is not an unsafe node to the release: it holds no
workload, its Agent is fenced, and the product owns its way back
(``submit-remediation --disposition restore``). The upgrade wave therefore
excludes such nodes, rolls the rest, and ends the cluster PARTIAL: the existing
cluster-local pause (``data-plane-paused`` -> ``partial-convergence``, lifecycle
PAUSED, no automatic rollback, the other clusters keep what converged) with the
excluded nodes, their incident ids and the operator commands recorded in the
release state (``product_isolated_nodes``) and printed by ``status``.
Rerunning the same ``deploy`` after the restore resumes the paused cluster and
rolls the node that was excluded.

Only product isolation is excused: a node that is Not Ready, cordoned or
tainted without the product's incident annotation or taints is still a hard
failure of the wave (``validate_target_node_state``) and still rolls back.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ClusterTarget,
)
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items
from gpu_fault_release.regional_release_narration import narrate_step

#: Release-state key: ``{cluster_id: {"release_id": ..., "nodes": [...]}}``.
PRODUCT_ISOLATED_NODES_STATE_KEY = "product_isolated_nodes"
#: Only the upgrade wave excuses product isolation. A join or bootstrap installs
#: a cluster for the first time and a rollback restores the fleet's safety net;
#: both keep refusing an isolated node, as before.
EXCLUDING_PHASES = frozenset({"upgrade"})
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
QUIESCE_TAINT = "gpu-fault.io/quiesce"
INCIDENT_ANNOTATION = "gpu-fault.io/incident-id"
FENCING_ANNOTATION = "gpu-fault.io/fencing-token"
RESUME_COMMAND = (
    'gpu-fault-admin deploy --state-dir "$STATE_DIR"'
    "  # resumes the paused cluster and rolls the restored node(s)"
)


class ProductIsolatedNodesError(ClusterLocalReleaseError):
    """The cluster rolled every node except the ones the product isolated.

    Cluster-local on purpose: the orchestrator pauses this cluster
    (``data-plane-paused``, then ``partial-convergence``) and the automatic
    rollback leaves the release alone, exactly as for an installer that failed
    on one node. ``excluded`` is what the state and ``status`` carry.
    """

    def __init__(self, cluster_id: str, excluded: Sequence[Mapping[str, Any]]) -> None:
        self.cluster_id = cluster_id
        self.excluded = tuple(dict(item) for item in excluded)
        names = ", ".join(
            f"{item.get('node_id')} ({item.get('incident_id') or 'no incident annotation'})"
            for item in self.excluded
        )
        super().__init__(
            f"{cluster_id} rolled every node except the {len(self.excluded)} the "
            f"product isolated: {names}; the release ends PARTIAL -- restore the "
            "incident(s) with submit-remediation --disposition restore, then rerun "
            "the same deploy to roll them"
        )


def product_isolation(node: Mapping[str, Any]) -> dict[str, Any] | None:
    """Why ``node`` (a ``kubectl get node`` item) is the product's to release.

    ``None`` for a node the product does not hold: a bare cordon, a Not Ready
    node or a foreign taint are not the product's isolation and stay the hard
    failures they were.
    """

    metadata = node.get("metadata") or {}
    spec = node.get("spec") or {}
    annotations = metadata.get("annotations") or {}
    taints = [item for item in spec.get("taints") or [] if isinstance(item, dict)]
    quarantine = next(
        (item for item in taints if str(item.get("key") or "") == QUARANTINE_TAINT),
        None,
    )
    quiesce = next(
        (item for item in taints if str(item.get("key") or "") == QUIESCE_TAINT), None
    )
    incident_id = str(annotations.get(INCIDENT_ANNOTATION) or "")
    cordoned = bool(spec.get("unschedulable"))
    reasons = []
    if quarantine is not None:
        reasons.append("quarantine-taint")
    if quiesce is not None:
        reasons.append("quiesce-taint")
    if cordoned and incident_id:
        reasons.append("cordoned-by-incident")
    if not reasons:
        return None
    ready = any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in (node.get("status") or {}).get("conditions") or []
        if isinstance(condition, dict)
    )
    return {
        "node_id": str(metadata.get("name") or ""),
        "reasons": reasons,
        "incident_id": incident_id or None,
        "fencing_token": str(annotations.get(FENCING_ANNOTATION) or "") or None,
        "quarantine_taint_value": (
            str(quarantine.get("value") or "") if quarantine is not None else None
        ),
        "unschedulable": cordoned,
        "ready": ready,
    }


def product_isolated_nodes(
    release: Any, target: ClusterTarget, node_names: Sequence[str]
) -> list[dict[str, Any]]:
    """The target nodes the product holds right now, from a fresh node read."""

    expected = set(node_names)
    found = []
    for item in gpu_node_items(release, target, fresh=True):
        name = str((item.get("metadata") or {}).get("name") or "")
        if name not in expected:
            continue
        isolation = product_isolation(item)
        if isolation is not None:
            found.append(isolation)
    return sorted(found, key=lambda item: str(item["node_id"]))


def record_product_isolated_nodes(
    release: Any, target: ClusterTarget, excluded: Sequence[Mapping[str, Any]]
) -> None:
    """Keep this release's exclusions in the state the next checkpoint writes."""

    state = getattr(release, "state", None)
    if not isinstance(state, dict):
        return
    existing = state.get(PRODUCT_ISOLATED_NODES_STATE_KEY)
    kept = {
        cluster_id: item
        for cluster_id, item in (existing.items() if isinstance(existing, dict) else ())
        if isinstance(item, dict) and item.get("release_id") == release.release_id
    }
    if excluded:
        kept[target.cluster_id] = {
            "release_id": release.release_id,
            "nodes": [dict(item) for item in excluded],
        }
        narrate_step(
            "fleet-node-excluded",
            cluster=target.cluster_id,
            nodes=",".join(str(item.get("node_id")) for item in excluded),
            incidents=",".join(
                str(item.get("incident_id") or "-") for item in excluded
            ),
            reason="product-isolation",
        )
    else:
        kept.pop(target.cluster_id, None)
    if kept:
        state[PRODUCT_ISOLATED_NODES_STATE_KEY] = kept
    else:
        state.pop(PRODUCT_ISOLATED_NODES_STATE_KEY, None)


def recorded_product_isolated_nodes(
    state: Mapping[str, Any] | None, cluster_id: str
) -> tuple[str, ...]:
    """The node ids a checkpoint recorded as excluded for ``cluster_id``."""

    entries = (state or {}).get(PRODUCT_ISOLATED_NODES_STATE_KEY)
    entry = entries.get(cluster_id) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return ()
    recorded_release = (state or {}).get("release_id")
    if recorded_release and entry.get("release_id") != recorded_release:
        # Another release's exclusions say nothing about this transaction.
        return ()
    nodes = entry.get("nodes")
    return tuple(
        sorted(
            str(item["node_id"])
            for item in nodes or []
            if isinstance(item, dict) and item.get("node_id")
        )
    )


def node_subset_digest(node_names: Sequence[str]) -> str:
    """What a partial wave's fleet deployment id carries: the included set.

    The fleet service binds a deployment id to its node set (``waves``), and a
    resume after the restore rolls the whole cluster again -- it must open a
    new record rather than hit "deployment_id is already bound to a different
    rollout" on the partial one.
    """

    return hashlib.sha256(
        "\n".join(sorted(str(name) for name in node_names)).encode()
    ).hexdigest()[:12]


def operator_recovery_commands(entries: Mapping[str, Any] | None) -> list[str]:
    """The exact commands that take a PARTIAL release to a complete one."""

    commands: list[str] = []
    for cluster_id, entry in sorted((entries or {}).items()):
        nodes = entry.get("nodes") if isinstance(entry, dict) else None
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            incident_id = str(node.get("incident_id") or "")
            if not incident_id:
                commands.append(
                    f"# {cluster_id}/{node.get('node_id')}: isolated without an "
                    "incident annotation; find the owning incident with "
                    "gpu-fault-admin status --full before restoring"
                )
                continue
            restore = (
                'gpu-fault-admin submit-remediation --state-dir "$STATE_DIR" '
                f"--incident-id {incident_id} --disposition restore --reference <CHG>"
            )
            commands.extend([f"{restore} --plan", restore])
    if commands:
        commands.append(RESUME_COMMAND)
    return list(dict.fromkeys(commands))
