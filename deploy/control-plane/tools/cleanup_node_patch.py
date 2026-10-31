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
(:func:`apply_node_patch`):

* gone, or a different UID (the node was replaced): the hard error it always
  was;
* the same resourceVersion as planned: nothing moved the Node, so the patch
  failed for a reason of its own, and that failure is raised as it is (with
  kubectl's reason, :class:`cleanup_failure.CleanupCommandError`);
* a newer resourceVersion: a concurrent writer. The patch is planned again from
  the fresh Node -- every guard re-evaluated: the spare's scheduling baseline,
  quarantine, an orphan's installer-only shape, and which of our keys are still
  there (none left: done) -- and sent again, bound to the new resourceVersion.

At most :data:`NODE_PATCH_ATTEMPTS` patches per node, with a short back-off
(:data:`NODE_PATCH_BACKOFF_SECONDS`) inside the command's deadline. A patch that
was applied although kubectl reported a failure (a lost acknowledgement) reads
back as a newer Node with nothing left to remove, and the node is done.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from cleanup_failure import CleanupCommandError
from cleanup_state import CleanupStateError

from gpu_fault.admin.execution import remaining_timeout

NODE_PATCH_ATTEMPTS = 4
# Slept before the fresh read that precedes attempt 2, 3 and 4. A short wait
# lets a burst of status updates settle; the read comes after it, so the retried
# patch is bound to the newest resourceVersion, not one the wait made stale.
NODE_PATCH_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
OWNED_PREFIX = "gpu-fault.io/"
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


def pointer(key: str) -> str:
    """``key`` as one JSON-pointer reference token (RFC 6901)."""

    return key.replace("~", "~0").replace("/", "~1")


def guarded(node: dict[str, Any], uid: str) -> list[dict[str, Any]]:
    """The two preconditions every node patch opens with."""

    return [
        {"op": "test", "path": "/metadata/uid", "value": uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": node["metadata"]["resourceVersion"],
        },
    ]


def owned_keys(inventory: dict[str, Any]) -> tuple[tuple[str, list[str]], ...]:
    return (
        ("annotations", inventory["node_annotations"]),
        ("labels", inventory["node_labels"]),
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
    with nothing of ours left needs no patch.
    """

    metadata = node["metadata"]
    if metadata.get("uid") != uid:
        raise CleanupStateError("orphaned installer node changed UID")
    keys = [
        *(metadata.get("labels") or {}),
        *(metadata.get("annotations") or {}),
        *(
            str(taint.get("key", ""))
            for taint in (node.get("spec") or {}).get("taints") or []
        ),
    ]
    if not any(key.startswith(OWNED_PREFIX) for key in keys):
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


def back_off(attempt: int) -> None:
    """Wait before the fresh read that precedes attempt ``attempt + 1``."""

    delay = NODE_PATCH_BACKOFF_SECONDS[
        min(attempt, len(NODE_PATCH_BACKOFF_SECONDS)) - 1
    ]
    time.sleep(min(delay, remaining_timeout(delay)))


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
    showed nothing left to remove. Raises the patch's own failure when the
    Node's resourceVersion did not move, and a refusal naming the node when it
    moved before every one of :data:`NODE_PATCH_ATTEMPTS` attempts.
    """

    attempt = 1
    while True:
        try:
            client.run(
                "patch", "node", name, "--type=json", "-p", json.dumps(patch.operations)
            )
        except CleanupCommandError as failure:
            last = attempt == NODE_PATCH_ATTEMPTS
            if not last:
                back_off(attempt)
            fresh = client.get("node", name)
            if fresh is None:
                raise CleanupStateError(
                    f"node metadata cleanup target disappeared: {name}"
                ) from failure
            if fresh["metadata"].get("uid") != uid:
                raise CleanupStateError(
                    f"node metadata cleanup target changed UID: {name} was replaced"
                ) from failure
            planned = node["metadata"]["resourceVersion"]
            if fresh["metadata"].get("resourceVersion") == planned:
                failure.add_note(
                    f"node {name}: its resourceVersion did not move, so no "
                    "concurrent update explains the failure; not retried"
                )
                raise
            if last:
                raise CleanupStateError(
                    f"node metadata cleanup of {name} did not settle: the Node's "
                    f"resourceVersion moved before each of {NODE_PATCH_ATTEMPTS} "
                    "patch attempts (concurrent kubelet or controller updates); "
                    f"last failure: {failure}"
                ) from failure
            node = fresh
            replanned = plan(node)
            if replanned is None:
                return None
            patch = replanned
            attempt += 1
        else:
            return patch


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
