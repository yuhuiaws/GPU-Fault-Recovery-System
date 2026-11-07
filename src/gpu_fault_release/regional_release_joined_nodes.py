"""Nodes that join a cluster while a release rolls it.

A release plans the node set it captured when it started
(``previous.agent_identities[<cluster>].node_ids``, which the release start
holds equal to the cluster's Kubernetes nodes) and fixes the fleet
deployment's node list when the cluster's roll starts. Capacity that arrives in
between -- a spot replacement, a scale-out -- used to end a healthy release:

* a node that joined before the cluster's roll and was not installed yet sat in
  the fleet deployment and in the pre-mutation gate, whose Agent check found no
  Agent for it (a paused cluster after 60 s), or whose node check found the
  steady-state installer still on it (``installer-active``: a plain
  ``ReleaseError``, answered by the automatic rollback);
* a node that joined after the roll started was in no wave, and the
  steady-state wait waited for it anyway -- for an install that only starts
  once the steady-state Reconciler is handed ``*`` and that takes minutes on a
  fresh node (key sync, installer Job, Agent registration), long past the
  900 s wait on the live site;
* the verify then judged it like any planned node (``installer is not
  Succeeded``, ``Agent coverage drift``), and the stability window failed on
  its first DaemonSet Pods.

Now a node outside the captured set is a *newcomer*. One that carries no
installation of its own when the cluster's roll starts is left to the
steady-state Reconciler (``deferred``): it is in no wave and no gate. At the
end of the roll the release still converges every planned node exactly as
before, and gives the newcomers ``NEWCOMER_GRACE_SECONDS`` from the moment the
Reconciler is handed ``*``; one that finishes inside it is ``installed``, the
rest are ``pending`` -- recorded under ``joined_nodes`` in the release state,
shown by ``status`` (``live_release.joined_nodes``) and never a reason to fail
the release: the steady-state Reconciler finishes them. A newcomer that
already carries an installation of another identity (the previous release's)
is no newcomer to wave past: it is held to the planned nodes' convergence wait,
which the Reconciler's re-install must meet, and the verify still fails on any
installed node whose artifact is not the release's (``artifact drift``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from gpu_fault.cluster_scope import node_selector, selector_matches
from gpu_fault_release import regional_parked_cluster as parked_cluster
from gpu_fault_release.regional_release_config import ClusterTarget
from gpu_fault_release.regional_release_gpu_rollout import (
    agents_converged,
    gpu_node_items,
)
from gpu_fault_release.regional_release_narration import narrate_step

#: Release-state key of the newcomers, ``{cluster_id: {release_id, phase, nodes}}``.
JOINED_NODES_STATE_KEY = "joined_nodes"
#: How long the end of a roll waits for newcomers, from the moment the
#: steady-state Reconciler is handed ``*``.
NEWCOMER_GRACE_SECONDS = 300.0
NEWCOMER_POLL_SECONDS = 15.0
INSTALLER_STATE = "gpu-fault.io/installer-state"
INSTALLER_NODE_UID = "gpu-fault.io/installer-node-uid"
_STATE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def captured_node_ids(state: object, cluster_id: str) -> frozenset[str] | None:
    """The node set the release captured for ``cluster_id``; None without one.

    A cluster the release found without nodes (``regional_parked_cluster``) was
    captured with none: every node that joins it during the release is a
    newcomer, installed by the steady-state Reconciler with the release's own
    node runtime."""

    previous = state.get("previous") if isinstance(state, Mapping) else None
    identities = (
        previous.get("agent_identities") if isinstance(previous, Mapping) else None
    )
    identity = identities.get(cluster_id) if isinstance(identities, Mapping) else None
    if parked_cluster.is_zero_node_identity(identity):
        return frozenset()
    nodes = identity.get("node_ids") if isinstance(identity, Mapping) else None
    if not isinstance(nodes, list) or not nodes:
        return None
    return frozenset(str(node) for node in nodes if str(node))


def node_name(item: Mapping[str, Any]) -> str:
    return str((item.get("metadata") or {}).get("name") or "")


def installer_state(item: Mapping[str, Any]) -> str:
    annotations = (item.get("metadata") or {}).get("annotations") or {}
    return str(annotations.get(INSTALLER_STATE) or "<none>")


def installed_here(item: Mapping[str, Any]) -> bool:
    """An installer finished on *this* Node object (its uid, not a name's)."""

    metadata = item.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    return (
        annotations.get(INSTALLER_STATE) == "Succeeded"
        and bool(metadata.get("uid"))
        and annotations.get(INSTALLER_NODE_UID) == metadata.get("uid")
    )


def _selected(release: Any, target: ClusterTarget, *, fresh: bool) -> dict[str, Any]:
    selector = node_selector(target)
    return {
        node_name(item): item
        for item in gpu_node_items(release, target, fresh=fresh)
        if node_name(item)
        and selector_matches((item.get("metadata") or {}).get("labels") or {}, selector)
    }


def record_joined_nodes(
    release: Any,
    target: ClusterTarget,
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    """Keep the newcomers in the state the next checkpoint writes."""

    state = getattr(release, "state", None)
    if not rows or not isinstance(state, dict):
        return
    with _STATE_LOCK:
        existing = state.get(JOINED_NODES_STATE_KEY)
        kept = {
            cluster_id: entry
            for cluster_id, entry in (
                existing.items() if isinstance(existing, dict) else ()
            )
            if isinstance(entry, dict) and entry.get("release_id") == release.release_id
        }
        current = kept.get(target.cluster_id) or {}
        nodes = {
            str(item.get("node_id")): dict(item)
            for item in current.get("nodes") or []
            if isinstance(item, dict)
        }
        for row in rows:
            nodes[str(row.get("node_id"))] = dict(row)
        kept[target.cluster_id] = {
            "release_id": release.release_id,
            "phase": phase,
            "nodes": [nodes[node] for node in sorted(nodes)],
        }
        state[JOINED_NODES_STATE_KEY] = kept


def joined_nodes_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """This release's newcomers, for ``status``; empty when there were none."""

    entries = state.get(JOINED_NODES_STATE_KEY)
    if not isinstance(entries, dict):
        return {}
    mine = {
        str(cluster_id): entry
        for cluster_id, entry in entries.items()
        if isinstance(entry, dict)
        and entry.get("release_id") == state.get("release_id")
    }
    return {JOINED_NODES_STATE_KEY: mine} if mine else {}


def _row(node: str, outcome: str, item: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = (item or {}).get("metadata") or {}
    return {
        "node_id": node,
        "outcome": outcome,
        "installer_state": installer_state(item) if item is not None else None,
        "joined_at": metadata.get("creationTimestamp"),
        "observed_at": _now(),
    }


def _narrate(
    target: ClusterTarget, phase: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    by_outcome: dict[str, list[str]] = {}
    for row in rows:
        node = str(row["node_id"])
        if row["outcome"] in {"pending", "deferred"}:
            node += f":{row.get('installer_state')}"
        by_outcome.setdefault(str(row["outcome"]), []).append(node)
    narrate_step(
        "joined-nodes",
        cluster=target.cluster_id,
        phase=phase,
        **{
            outcome: ",".join(sorted(nodes)) or "-"
            for outcome, nodes in sorted(by_outcome.items())
        },
    )


def defer_uninstalled_newcomers(
    release: Any,
    target: ClusterTarget,
    node_names: tuple[str, ...],
    *,
    phase: str,
) -> tuple[str, ...]:
    """``node_names`` without the newcomers no installer has finished on yet.

    Called with the node list the cluster's roll reads, inside its inventory
    scope (the Nodes are the ones that list was read from). Only the forward
    ``upgrade`` of a release that captured a node set defers anything, and
    never the whole list.
    """

    captured = captured_node_ids(getattr(release, "state", None), target.cluster_id)
    if phase != "upgrade" or captured is None:
        return node_names
    outside = [node for node in node_names if node not in captured]
    if not outside:
        return node_names
    items = {node_name(item): item for item in gpu_node_items(release, target)}
    deferred = [node for node in outside if not installed_here(items.get(node) or {})]
    if not deferred or len(deferred) == len(node_names):
        return node_names
    rows = [_row(node, "deferred", items.get(node)) for node in deferred]
    record_joined_nodes(release, target, rows, phase="node-rollout")
    _narrate(target, "node-rollout", rows)
    return tuple(node for node in node_names if node not in set(deferred))


@dataclass
class FinalizeNewcomers:
    """The newcomers the end of one cluster's roll found, and their outcome."""

    release: Any
    target: ClusterTarget
    #: Planned by the release: its captured set and the nodes the waves rolled.
    planned: frozenset[str]
    #: Newcomers seen so far.
    nodes: set[str] = field(default_factory=set)
    #: Newcomers carrying another identity's installation: held to the wait.
    drift: set[str] = field(default_factory=set)
    started: float = 0.0
    #: The forward ``upgrade`` of a release that captured a node set: the end
    #: of the roll waits for the planned nodes by name and looks for newcomers.
    applicable: bool = False

    @property
    def active(self) -> bool:
        return bool(self.nodes)

    def wait_node_names(
        self, node_names: Collection[str], departed: Collection[str]
    ) -> tuple[str, ...]:
        """The convergence wait's node set: the planned nodes and the drift."""

        return tuple(sorted((set(node_names) | self.drift) - set(departed)))

    def settle(
        self,
        *,
        artifact_sha: str,
        bundle_sha: str | None,
        template_sha: str | None,
        config_digest: str,
        runtime_profile_version: str,
        agent_identity: dict[str, Any] | None,
        dcgm_hostengine: str | None,
    ) -> list[dict[str, Any]]:
        """Give the newcomers the rest of the grace; record and narrate them.

        Looked for once more first: a node that joined while the planned nodes
        converged is a newcomer too.
        """

        if not self.applicable:
            return []
        deadline = self.started + NEWCOMER_GRACE_SECONDS
        while True:
            items = _selected(self.release, self.target, fresh=True)
            self.nodes.update(node for node in items if node not in self.planned)
            if not self.nodes:
                return []
            rows = []
            for node in sorted(self.nodes):
                item = items.get(node)
                if item is None:
                    rows.append(_row(node, "left", None))
                    continue
                aligned = agents_converged(
                    [item],
                    self.target,
                    artifact_sha,
                    bundle_sha=bundle_sha,
                    template_sha=template_sha,
                    config_digest=config_digest,
                    require_node_uid=True,
                    node_names=frozenset({node}),
                    dcgm_hostengine=dcgm_hostengine,
                ) and self.release._agent_heartbeats_converged(
                    self.target,
                    node_count=1,
                    node_names=(node,),
                    artifact_sha=artifact_sha,
                    config_digest=config_digest,
                    runtime_profile_version=runtime_profile_version,
                    bundle_sha=bundle_sha,
                    template_sha=template_sha,
                    agent_identity=agent_identity,
                )
                rows.append(_row(node, "installed" if aligned else "pending", item))
            now = time.monotonic()
            if all(row["outcome"] != "pending" for row in rows) or now >= deadline:
                break
            time.sleep(min(NEWCOMER_POLL_SECONDS, deadline - now))
        record_joined_nodes(self.release, self.target, rows, phase="finalize")
        _narrate(self.target, "finalize", rows)
        return rows


def finalize_newcomers(
    release: Any,
    target: ClusterTarget,
    *,
    phase: str,
    node_names: Collection[str],
    identity_matches: Any,
) -> FinalizeNewcomers:
    """The newcomers present when the end of the roll starts (before ``*``).

    ``identity_matches(item)`` says whether an installed Node already carries
    the candidate identity. Inert -- no read at all -- outside the forward
    ``upgrade`` of a release that captured a node set.
    """

    captured = captured_node_ids(getattr(release, "state", None), target.cluster_id)
    planned = frozenset(node_names) | (captured or frozenset())
    joined = FinalizeNewcomers(release, target, planned, started=time.monotonic())
    if phase != "upgrade" or captured is None or release.runner.dry_run:
        return joined
    joined.applicable = True
    items = _selected(release, target, fresh=True)
    joined.nodes = {node for node in items if node not in planned}
    joined.drift = {
        node
        for node in joined.nodes
        if installed_here(items[node]) and not identity_matches(items[node])
    }
    return joined


def installing_newcomers(
    release: Any, cluster_id: str, items: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """``node -> installer state`` of the given Nodes the release did not
    capture and whose installer has not succeeded; empty without a capture.

    The verify's view: the same ``installer-state`` its own check reads. A
    newcomer that did succeed is judged like every other node.
    """

    captured = captured_node_ids(getattr(release, "state", None), cluster_id)
    if captured is None:
        return {}
    return {
        node_name(item): installer_state(item)
        for item in items
        if node_name(item)
        and node_name(item) not in captured
        and installer_state(item) != "Succeeded"
    }
