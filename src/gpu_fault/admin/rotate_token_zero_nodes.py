"""The node roll of ``rotate-token`` on a GPU cluster without nodes.

A rotation re-installs the cluster's nodes (``rotate_token.roll_node_tokens``) so
that every host carries the new token. A cluster whose recorded node selector
selects no Kubernetes node -- parked at zero by its provider, or unavailable
(``gpu_fault_release.regional_parked_cluster``) -- has no host to re-install, and
the roll stopped on its empty node list ("has no GPU nodes for fleet rollout")
after the registry overlap, the connection Secret and the data plane had already
moved to the new token. The journal then stayed open, and an open rotation
refuses every deploy and join of the site (``token_rotation``).

When the release engine's proof (``regional_parked_cluster.without_nodes``: the
API lists no node for the selector, then the provider documents) says the
cluster has no node, the node roll is skipped and recorded instead: no wave
ConfigMap is read or handed over, no node is annotated and no Agent is waited
for. The journal's ``node_rollout`` keeps ``zero_nodes`` (state, provider,
reason, groups) and the nodes a forward pass had recorded or a rollback had
moved but that have left since (``departed_unrolled``); one
``zero-node-gpu-cluster ... step=rotate-token`` line says so. The rest of the
rotation runs as for any cluster: the acceptance window watches the restarted
data-plane Deployments, the only callers left on the old token.

That is safe because the connection Secret already holds the new token when the
node roll starts (``CONNECTION_SECRET_UPDATED`` comes first): a node that joins
later is installed from it by the steady-state Reconciler and authenticates with
the new token, and a node that existed before the Secret update would have been
selected now. A node list that cannot be read proves nothing, and the roll runs
-- and refuses -- as before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release import regional_parked_cluster as parked_cluster

#: The journal key of a node roll skipped on a cluster without nodes.
ZERO_NODES_KEY = "zero_nodes"
#: The narration of that skip.
STEP = "rotate-token"
RECORDED = "no node, no node token to roll"
SKIPPED = ("node-waves", "agent-wait")

NodeNames = Callable[[Any, Any], tuple[str, ...]]


def rotation_node_names(
    release: Any,
    target: Any,
    progress: dict[str, Any],
    record: Callable[[], None],
    only_nodes: frozenset[str] | None,
    node_names: NodeNames,
) -> tuple[str, ...]:
    """The nodes the roll re-installs; ``()`` for a cluster without nodes,
    whose skip is recorded in ``progress`` (and ``record``-ed) first.

    ``node_names`` is the roll's own node read (``target_node_names``), asked
    only of a cluster that has nodes; a node set that changed since the first
    wave still refuses.
    """

    proof = parked_cluster.without_nodes(release, target)
    if proof is None:
        names = node_names(release, target)
        if progress.get("node_names") is not None and set(
            progress["node_names"]
        ) != set(names):
            raise BootstrapError("rotation node inventory changed since the first wave")
        return names
    left = set(only_nodes or progress.get("node_names") or ())
    entry = {
        **parked_cluster.summary(proof),
        "skipped": list(SKIPPED),
        "departed_unrolled": sorted(left - set(progress.get("completed_nodes") or ())),
    }
    progress[ZERO_NODES_KEY] = entry
    record()
    parked_cluster.narrate(
        str(target.cluster_id),
        state=entry["state"],
        step=STEP,
        recorded=RECORDED,
        reason=str(entry["reason"]),
    )
    return ()


def zero_node_roll(progress: dict[str, Any]) -> dict[str, Any]:
    """The node roll's result when it was skipped (``rotation_node_names``)."""

    return {
        "node_count": 0,
        "waves": [],
        "reinstalled_nodes": sorted(progress.get("completed_nodes") or []),
        ZERO_NODES_KEY: dict(progress[ZERO_NODES_KEY]),
    }


def node_rollout_summary(progress: object) -> dict[str, Any]:
    """What ``rotation_summary`` says of the node roll: the re-installed nodes,
    and the zero-node record of a roll skipped on a cluster without nodes."""

    rollout = progress if isinstance(progress, dict) else {}
    return {
        "reinstalled_nodes": rollout.get("completed_nodes"),
        ZERO_NODES_KEY: rollout.get(ZERO_NODES_KEY),
    }
