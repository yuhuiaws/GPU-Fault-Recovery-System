"""Node metadata patches of a reset, retried when another writer moves the Node.

``clear_node_metadata`` (``cleanup_kubernetes.py``) lists the cluster's nodes
once, plans every node's patch from that list -- so every refusal comes before
the first mutation -- and then patches the nodes one after another. Each patch
starts with ``test /metadata/uid`` and ``test /metadata/resourceVersion``: its
operations were computed from that version of the Node, and the API server
applies them to that version only. The test stays; it is what makes the patch
safe.

The Node is not ours alone, though. The kubelet posts Node status -- and right
after the phases before this one removed our node DaemonSets and node runtime it
has fresh status to post -- and controllers write labels and conditions. Any
such write between the list and the patch moves the resourceVersion, the
``test`` fails (kubectl exit 1, "The request is invalid") and the whole
uninstall stopped (live: BOOT-029 stage 4, one HyperPod node kept its installer
annotations, its managedFields showing a ``kubelet Update status`` inside the
phase). Resuming re-ran the phase and passed, but an uninstall must not depend
on that luck.

A failed patch is therefore followed by a fresh read of the Node
(:func:`apply_node_patch`, through ``gpu_fault.admin.node_patch_retry``, the
retry the admin's ``remove-cluster`` and join-rollback annotation cleanup use
too):

* gone, or a different UID (the node was replaced): the hard error it always
  was;
* the same resourceVersion as planned: nothing moved the Node, so the patch
  failed for a reason of its own, and that failure is raised as it is (with
  kubectl's reason, :class:`cleanup_failure.CleanupCommandError`);
* a newer resourceVersion: a concurrent writer. The patch is planned again from
  the fresh Node -- every guard re-evaluated: the spare's scheduling baseline,
  quarantine, an orphan's installer-only shape, and which of our keys are still
  there (none left: done) -- and sent again, bound to the new resourceVersion.

At most ``NODE_PATCH_ATTEMPTS`` patches per node, with a short back-off inside
the command's deadline. A patch that was applied although kubectl reported a
failure (a lost acknowledgement) reads back as a newer Node with nothing left to
remove, and the node is done.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from cleanup_failure import CleanupCommandError
from cleanup_state import CleanupStateError

from gpu_fault.admin.node_patch_retry import NodePatchRetry, guarded, pointer
from gpu_fault.cluster_scope import DCGM_HOSTENGINE_LABEL, OWNERSHIP_LABEL

NODE_METADATA_RETRY = NodePatchRetry(
    subject="node metadata cleanup",
    failure=CleanupCommandError,
    refuse=CleanupStateError,
)
OWNED_PREFIX = "gpu-fault.io/"
#: The labels an administrator declares on a node group (EKS managed node
#: group ``labels``, HyperPod instance group ``KubernetesConfig.Labels``,
#: Karpenter NodePool template labels) and every node of the group carries:
#: the site's membership claim (``cluster_scope.OWNERSHIP_LABEL``, released
#: from the node groups by ``OWNERSHIP_RELEASED``) and the node image's host
#: DCGM declaration (``cluster_scope.DCGM_HOSTENGINE_LABEL``). The product
#: reads them and never writes them on a Node, and the node cleanup leaves
#: them alone, so they are no installed state: a node of a label-convention
#: cluster carries them whether or not anything was ever installed on it. As
#: labels only -- the same key as an annotation or a taint is ours.
NODE_GROUP_DECLARATION_LABELS = frozenset({OWNERSHIP_LABEL, DCGM_HOSTENGINE_LABEL})
# Node annotations the node-installer-reconciler writes while (or after) it
# tries to install a node; ``gpu-fault.io/node-key-*`` ownership annotations
# are deliberately not in this family.
INSTALLER_ANNOTATION_PREFIX = "gpu-fault.io/installer-"
SPARE_LABEL = "gpu-fault.io/spare"
QUARANTINED_LABEL = "gpu-fault.io/quarantined"
PREVIOUS_UNSCHEDULABLE = "gpu-fault.io/previous-unschedulable"


class NodeClient(Protocol):
    """The two ``CleanupClient`` calls a node patch needs."""

    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str: ...

    def get(
        self, kind: str, name: str, namespace: str = ""
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class NodePatch:
    """A JSON patch for one Node, bound to the UID and resourceVersion it read."""

    operations: list[dict[str, Any]]
    restore_scheduling: bool = False


NodePlan = Callable[[dict[str, Any]], NodePatch | None]
NodeConverged = Callable[[dict[str, Any], NodePatch], bool]


def owned_keys(inventory: dict[str, Any]) -> tuple[tuple[str, list[str]], ...]:
    return (
        ("annotations", inventory["node_annotations"]),
        ("labels", inventory["node_labels"]),
    )


def product_labels(node: dict[str, Any]) -> list[str]:
    """Our label keys on ``node``, the node-group declarations excepted."""

    labels = (node.get("metadata") or {}).get("labels") or {}
    return [
        key
        for key in labels
        if key.startswith(OWNED_PREFIX) and key not in NODE_GROUP_DECLARATION_LABELS
    ]


def product_annotations(node: dict[str, Any]) -> list[str]:
    annotations = (node.get("metadata") or {}).get("annotations") or {}
    return [key for key in annotations if key.startswith(OWNED_PREFIX)]


def product_taints(node: dict[str, Any]) -> list[str]:
    taints = (node.get("spec") or {}).get("taints") or []
    return [
        key
        for taint in taints
        if (key := str(taint.get("key", ""))).startswith(OWNED_PREFIX)
    ]


def carries_product_state(node: dict[str, Any]) -> bool:
    """Whether the product set anything on ``node``: any ``gpu-fault.io/``
    label but the node-group declarations, any such annotation or taint.

    A key nobody knows counts too, so state a later release adds blocks the
    cleanup until it is known instead of being left behind.
    """

    return bool(
        product_labels(node) or product_annotations(node) or product_taints(node)
    )


def installer_annotation_patch(node: dict[str, Any], uid: str) -> list[dict[str, Any]]:
    """Remove every installer annotation from an orphaned node, UID/RV bound."""

    operations = guarded(node, uid)
    for key in sorted(node["metadata"].get("annotations") or {}):
        if key.startswith(INSTALLER_ANNOTATION_PREFIX):
            operations.append(
                {"op": "remove", "path": f"/metadata/annotations/{pointer(key)}"}
            )
    return operations


def target_node_patch(
    node: dict[str, Any], *, uid: str, inventory: dict[str, Any]
) -> NodePatch | None:
    """The patch that strips our labels and annotations from a fleet node.

    A declared warm spare is uncordoned first when, and only when, its own
    recorded baseline says it was schedulable. ``None`` when none of our keys
    is left on the node.
    """

    metadata = node["metadata"]
    if metadata.get("uid") != uid:
        raise CleanupStateError("node metadata cleanup target changed UID")
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    restore_scheduling = False
    if labels.get(SPARE_LABEL) == "true":
        previous = annotations.get(PREVIOUS_UNSCHEDULABLE)
        if not isinstance(previous, str) or previous not in {"true", "false"}:
            raise CleanupStateError(
                "declared warm spare has no scheduling baseline on the node; "
                "release it with gpu-fault-admin (config spare --release; "
                "uninstall does it) before node cleanup"
            )
        if labels.get(QUARANTINED_LABEL) not in {None, "false"}:
            raise CleanupStateError("quarantined spare cannot be released")
        unschedulable = (node.get("spec") or {}).get("unschedulable", False)
        if type(unschedulable) is not bool:
            raise CleanupStateError("spare cleanup has unknown scheduling state")
        restore_scheduling = previous == "false" and unschedulable
    operations = guarded(node, uid)
    if restore_scheduling:
        operations.extend(
            [
                {"op": "test", "path": "/spec/unschedulable", "value": True},
                {"op": "replace", "path": "/spec/unschedulable", "value": False},
            ]
        )
    for field, keys in owned_keys(inventory):
        for key in keys:
            if key in (metadata.get(field) or {}):
                operations.append(
                    {"op": "remove", "path": f"/metadata/{field}/{pointer(key)}"}
                )
    if len(operations) == 2:
        return None
    return NodePatch(operations, restore_scheduling)


def orphan_node_patch(
    node: dict[str, Any],
    *,
    uid: str,
    still_orphaned: Callable[[dict[str, Any]], bool],
) -> NodePatch | None:
    """The patch that strips the installer annotations of an orphaned node.

    ``still_orphaned`` re-judges the node (``orphaned_installer_node``): only
    unfinished installer annotations of ours and no fleet Agent. A node that
    gained anything else of ours is the refusal the capture would raise; one
    with nothing of ours left (the node-group declarations do not count)
    needs no patch.
    """

    if node["metadata"].get("uid") != uid:
        raise CleanupStateError("orphaned installer node changed UID")
    if not carries_product_state(node):
        return None
    if not still_orphaned(node):
        raise CleanupStateError(
            "unregistered node runtime remains outside cleanup targets"
        )
    operations = installer_annotation_patch(node, uid)
    return NodePatch(operations) if len(operations) > 2 else None


def target_converged(
    node: dict[str, Any], patch: NodePatch, *, inventory: dict[str, Any]
) -> bool:
    metadata = node["metadata"]
    if any(
        key in (metadata.get(field) or {})
        for field, keys in owned_keys(inventory)
        for key in keys
    ):
        return False
    return (
        not patch.restore_scheduling
        or (node.get("spec") or {}).get("unschedulable") is False
    )


def orphan_converged(node: dict[str, Any], _patch: NodePatch) -> bool:
    return not any(
        key.startswith(OWNED_PREFIX)
        for key in node["metadata"].get("annotations") or {}
    )


def apply_node_patch(
    client: NodeClient,
    name: str,
    node: dict[str, Any],
    patch: NodePatch,
    plan: NodePlan,
    *,
    uid: str,
) -> NodePatch | None:
    """Send ``patch`` (planned from ``node``); after a failure, re-read and retry.

    Returns the patch the API server accepted, or ``None`` when a fresh read
    showed nothing left to remove (``NodePatchRetry.apply``).
    """

    def send(current: NodePatch) -> None:
        client.run(
            "patch", "node", name, "--type=json", "-p", json.dumps(current.operations)
        )

    return NODE_METADATA_RETRY.apply(
        name,
        node,
        patch,
        uid=uid,
        send=send,
        read=lambda: client.get("node", name),
        plan=plan,
    )


def clear_node(
    client: NodeClient,
    name: str,
    node: dict[str, Any],
    patch: NodePatch,
    plan: NodePlan,
    *,
    uid: str,
    converged: NodeConverged,
) -> None:
    """Patch one node and prove, on a read-back, that none of our metadata is left."""

    applied = apply_node_patch(client, name, node, patch, plan, uid=uid)
    if applied is None:
        return
    current = client.get("node", name)
    if current is None or current["metadata"].get("uid") != uid:
        raise CleanupStateError("node changed during metadata cleanup")
    if not converged(current, applied):
        raise CleanupStateError("node metadata cleanup did not converge")
