"""A guarded JSON patch to one Node, retried when another writer moved the Node.

Every node cleanup patch opens with ``test /metadata/uid`` and
``test /metadata/resourceVersion``: its operations were computed from that
version of the Node, and the API server applies them to that version only. The
test stays; it is what makes the patch safe.

The Node is not the cleanup's alone, though. The kubelet posts Node status --
and right after a cleanup removed our node DaemonSets and node runtime it has
fresh status to post -- and controllers write labels and conditions. Any such
write between the cleanup's read and its patch moves the resourceVersion, the
``test`` fails (kubectl exit 1, "The request is invalid") and the cleanup used to
stop there (live: BOOT-029 stage 4, one HyperPod node of a reset; the same
list-then-patch shape runs in ``remove-cluster`` right after its node DaemonSets
are deleted). :meth:`NodePatchRetry.apply` therefore answers a failed patch with
a fresh read of the Node:

* gone, or a different UID (the node was replaced): the hard error it always
  was;
* the same resourceVersion as planned: nothing moved the Node, so the patch
  failed for a reason of its own, and that failure is raised as it is (with
  kubectl's classified reason, ``gpu_fault.admin.kubectl_failure``);
* a newer resourceVersion: a concurrent writer. The caller's planner builds the
  patch again from the fresh Node -- every guard re-evaluated, nothing left to
  remove meaning done -- and it is sent again, bound to the new resourceVersion.

At most :data:`NODE_PATCH_ATTEMPTS` patches per node, with a short back-off
(:data:`NODE_PATCH_BACKOFF_SECONDS`) inside the caller's deadline. A patch that
was applied although kubectl reported a failure (a lost acknowledgement) reads
back as a newer Node with nothing left to remove, and the node is done.

Used by the reset's node metadata cleanup
(``deploy/control-plane/tools/cleanup_node_patch.py``) and by the installer
annotation cleanup of ``remove-cluster`` and the join rollback
(``cluster_removal_kubernetes``). Each brings its own wording and error types.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from gpu_fault.admin.execution import remaining_timeout

NODE_PATCH_ATTEMPTS = 4
# Slept before the fresh read that precedes attempt 2, 3 and 4. A short wait
# lets a burst of status updates settle; the read comes after it, so the retried
# patch is bound to the newest resourceVersion, not one the wait made stale.
NODE_PATCH_BACKOFF_SECONDS = (0.5, 1.0, 2.0)

PatchT = TypeVar("PatchT")


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


def back_off(attempt: int) -> None:
    """Wait before the fresh read that precedes attempt ``attempt + 1``."""

    delay = NODE_PATCH_BACKOFF_SECONDS[
        min(attempt, len(NODE_PATCH_BACKOFF_SECONDS)) - 1
    ]
    time.sleep(min(delay, remaining_timeout(delay)))


@dataclass(frozen=True)
class NodePatchRetry:
    """How one cleanup retries its node patches.

    ``subject`` opens every refusal ("node metadata cleanup"); ``failure`` is the
    exception the caller's ``send`` raises when the kubectl call itself failed
    (anything else propagates untouched); ``refuse`` builds the caller's own
    error for a refusal.
    """

    subject: str
    failure: type[Exception]
    refuse: Callable[[str], Exception]

    def apply(
        self,
        name: str,
        node: dict[str, Any],
        patch: PatchT,
        *,
        uid: str,
        send: Callable[[PatchT], object],
        read: Callable[[], dict[str, Any] | None],
        plan: Callable[[dict[str, Any]], PatchT | None],
    ) -> PatchT | None:
        """Send ``patch`` (planned from ``node``); after a failure, re-read and retry.

        Returns the patch the API server accepted, or ``None`` when a fresh read
        showed nothing left to remove. Raises the patch's own failure when the
        Node's resourceVersion did not move, and a refusal naming the node when
        it moved before every one of :data:`NODE_PATCH_ATTEMPTS` attempts.
        """

        attempt = 1
        while True:
            try:
                send(patch)
            except self.failure as failure:
                last = attempt == NODE_PATCH_ATTEMPTS
                if not last:
                    back_off(attempt)
                fresh = read()
                if fresh is None:
                    raise self.refuse(
                        f"{self.subject} target disappeared: {name}"
                    ) from failure
                if fresh["metadata"].get("uid") != uid:
                    raise self.refuse(
                        f"{self.subject} target changed UID: {name} was replaced"
                    ) from failure
                planned = node["metadata"]["resourceVersion"]
                if fresh["metadata"].get("resourceVersion") == planned:
                    failure.add_note(
                        f"node {name}: its resourceVersion did not move, so no "
                        "concurrent update explains the failure; not retried"
                    )
                    raise
                if last:
                    raise self.refuse(
                        f"{self.subject} of {name} did not settle: the Node's "
                        f"resourceVersion moved before each of {NODE_PATCH_ATTEMPTS}"
                        " patch attempts (concurrent kubelet or controller "
                        f"updates); last failure: {failure}"
                    ) from failure
                node = fresh
                replanned = plan(node)
                if replanned is None:
                    return None
                patch = replanned
                attempt += 1
            else:
                return patch
