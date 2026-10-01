"""Conditional Kubernetes mutations used only by HA acceptance fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

from scripts.e2e.regional.regional_commands import RegionalFixtureError

NODE_OWNER = "gpu-fault.io/ha002-owner"
NODE_OWNER_PATH = "/metadata/annotations/gpu-fault.io~1ha002-owner"


def require_uid(uid: Any) -> str:
    if not isinstance(uid, str) or not uid:
        raise RegionalFixtureError("HA mutation requires an observed resource UID")
    return uid


def delete_pod(
    command: Callable[[list[str], str], Any],
    namespace: str,
    pod: Mapping[str, Any],
    *,
    force: bool = False,
) -> None:
    uid = require_uid(pod.get("uid"))
    name = pod.get("name")
    if not isinstance(name, str) or not name or not namespace:
        raise RegionalFixtureError("HA Pod deletion identity is incomplete")
    options: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": uid},
        "propagationPolicy": "Foreground",
    }
    if force:
        options["gracePeriodSeconds"] = 0
    command(
        [
            "delete",
            "--raw",
            f"/api/v1/namespaces/{quote(namespace, safe='')}/pods/{quote(name, safe='')}",
            "-f",
            "-",
        ],
        json.dumps(options),
    )


def node_cordon_patch(
    snapshot: Mapping[str, Any], *, uid: str, owner: str
) -> list[dict[str, Any]]:
    if (
        snapshot.get("uid") != require_uid(uid)
        or snapshot.get("unschedulable") is not False
        or snapshot.get("ha_owner") is not None
        or not snapshot.get("resource_version")
        or not owner
    ):
        raise RegionalFixtureError("HA CPU Node changed before cordon")
    patch: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": snapshot["resource_version"],
        },
    ]
    if not snapshot["annotations_present"]:
        patch.append({"op": "add", "path": "/metadata/annotations", "value": {}})
    patch.extend(
        [
            {"op": "add", "path": NODE_OWNER_PATH, "value": owner},
            {"op": "add", "path": "/spec/unschedulable", "value": True},
        ]
    )
    return patch


CORDON_ATTEMPTS = 3


def cordon_node_with_retry(
    *,
    read_node: Callable[[], Mapping[str, Any]],
    apply_patch: Callable[[list[dict[str, Any]]], None],
    uid: str,
    owner: str,
    retryable: Callable[[BaseException], bool],
    attempts: int = CORDON_ATTEMPTS,
    log: Callable[[str], None] | None = None,
) -> tuple[Mapping[str, Any], int]:
    """Cordon through ``node_cordon_patch`` with a bounded optimistic-lock retry.

    The patch tests the Node's ``resourceVersion``, so a kubelet status update
    (an image pull landing on the node, a condition heartbeat) between the read
    and the patch makes the API server reject the request (live 2026-10-01,
    HA-002). Only that race is retried: the Node is re-read, the cordon
    invariants are re-checked by ``node_cordon_patch`` (a UID change, an existing
    owner or an already-cordoned node still fail closed), and the patch is
    rebuilt from the fresh ``resourceVersion``. A rejection while the
    ``resourceVersion`` did not move is not the race and is re-raised at once.
    Returns the snapshot the successful patch was built from and the attempt
    number (1-based) for the evidence.
    """
    if attempts < 1:
        raise RegionalFixtureError("cordon retry budget must allow one attempt")
    for attempt in range(1, attempts + 1):
        snapshot = read_node()
        patch = node_cordon_patch(snapshot, uid=uid, owner=owner)
        try:
            apply_patch(patch)
        except Exception as exc:
            if attempt >= attempts or not retryable(exc):
                raise
            current = read_node()
            if current.get("resource_version") == snapshot.get("resource_version"):
                raise
            if log is not None:
                log(
                    "cordon patch lost the resourceVersion race "
                    f"(attempt {attempt}/{attempts}); re-reading the node"
                )
            continue
        return snapshot, attempt
    raise RegionalFixtureError("cordon retry loop exhausted without a verdict")


def node_restore_patch(
    baseline: Mapping[str, Any], *, owner: str
) -> list[dict[str, Any]]:
    if baseline.get("unschedulable") is not False or not owner:
        raise RegionalFixtureError("HA CPU Node baseline is not restorable")
    patch: list[dict[str, Any]] = [
        {
            "op": "test",
            "path": "/metadata/uid",
            "value": require_uid(baseline.get("uid")),
        },
        {"op": "test", "path": NODE_OWNER_PATH, "value": owner},
        {"op": "test", "path": "/spec/unschedulable", "value": True},
    ]
    if baseline.get("taints_present"):
        patch.append(
            {"op": "test", "path": "/spec/taints", "value": baseline["taints"]}
        )
    patch.extend(
        [
            {"op": "add", "path": "/spec/unschedulable", "value": False},
            {"op": "remove", "path": NODE_OWNER_PATH},
        ]
    )
    return patch


def scale_patch(uid: str, before: int, after: int) -> list[dict[str, Any]]:
    require_uid(uid)
    if any(type(value) is not int or value < 1 for value in (before, after)):
        raise RegionalFixtureError("HA scale requires positive replica counts")
    return [
        {"op": "test", "path": "/metadata/uid", "value": uid},
        {"op": "test", "path": "/spec/replicas", "value": before},
        {"op": "replace", "path": "/spec/replicas", "value": after},
    ]
