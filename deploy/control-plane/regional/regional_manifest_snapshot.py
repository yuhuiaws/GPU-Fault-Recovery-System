"""Capture and restore the live objects a release manifest applies.

Some components are applied straight from the candidate checkout rather than
from a versioned artifact: the ADOT collector and the control-plane NLB Service
are both ``kubectl apply -f <file in the tree>``. Rollback cannot rebuild their
previous form from the previous release's pins, because the previous form only
ever existed as a file in the previous checkout.

Reading the live objects before the upgrade mutates them removes that
asymmetry. What is captured is the previous release's own applied state, so
restoring it restores the previous manifest *and* whatever the previous
manifest substituted into it -- images, ARNs, subnet lists -- without needing
the previous checkout. The manifest in the tree is used only to decide *which*
objects an upgrade may mutate, which is exactly the set a rollback has to put
back.

Restore is the inverse of apply over that same object set: an object the
manifest declares that was not live yet is one the candidate adds, so rollback
deletes it. Objects outside the set are touched by neither direction.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]
from regional_release_config import ReleaseError

# Fields the API server owns. Applying them back either fails outright
# (``resourceVersion`` conflicts) or reintroduces state that belongs to the
# object's current life, not to the configuration being restored.
SERVER_METADATA = (
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
)
SERVER_ANNOTATIONS = (
    "deployment.kubernetes.io/revision",
    "kubectl.kubernetes.io/last-applied-configuration",
)


def resource_argument(api_version: str, kind: str) -> str:
    """Return the ``kind.group`` form ``kubectl`` accepts for an object.

    Qualifying the kind with its API group keeps the capture unambiguous when a
    CRD in the cluster shares a kind with a built-in resource.
    """

    group = api_version.rsplit("/", 1)[0] if "/" in api_version else ""
    return f"{kind.lower()}.{group}" if group else kind.lower()


def declared_manifest_objects(
    manifest_text: str,
    *,
    label: str,
) -> tuple[dict[str, str], ...]:
    """Return the objects a manifest applies, in manifest order.

    ``label`` names the manifest in errors, e.g. ``"ADOT manifest"``.
    """

    declared: list[dict[str, str]] = []
    for document in yaml.safe_load_all(manifest_text):
        if not isinstance(document, dict):
            continue
        api_version = str(document.get("apiVersion") or "")
        kind = str(document.get("kind") or "")
        name = str((document.get("metadata") or {}).get("name") or "")
        if not api_version or not kind or not name:
            raise ReleaseError(f"{label} has an unidentifiable object")
        declared.append(
            {
                "resource": resource_argument(api_version, kind),
                "kind": kind,
                "name": name,
            }
        )
    if not declared:
        raise ReleaseError(f"{label} declares no objects")
    return tuple(declared)


def restorable_object(document: object, name: str, *, label: str) -> dict[str, Any]:
    """Strip the server-owned fields that must not be applied back.

    ``label`` names the object in errors, e.g. ``"live ADOT object"``.
    """

    if not isinstance(document, dict):
        raise ReleaseError(f"{label} {name} is not a Kubernetes object")
    value = copy.deepcopy(document)
    value.pop("status", None)
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ReleaseError(f"{label} {name} has no metadata")
    for field in SERVER_METADATA:
        metadata.pop(field, None)
    annotations = metadata.get("annotations")
    if isinstance(annotations, dict):
        for field in SERVER_ANNOTATIONS:
            annotations.pop(field, None)
        if not annotations:
            metadata.pop("annotations", None)
    return value


def capture_declared_objects(
    release: Any,
    declared: tuple[dict[str, str], ...],
    *,
    label: str,
) -> dict[str, Any]:
    """Read the live form of every declared object in the release namespace."""

    namespace = release.config.namespace
    objects: list[dict[str, Any]] = []
    absent: list[dict[str, str]] = []
    for item in declared:
        text = release.runner.run(
            release._cpu(
                "-n",
                namespace,
                "get",
                item["resource"],
                item["name"],
                "-o",
                "json",
                "--ignore-not-found",
            ),
            capture=True,
        ).strip()
        if not text:
            absent.append({"resource": item["resource"], "name": item["name"]})
            continue
        objects.append(restorable_object(json.loads(text), item["name"], label=label))
    return {"namespace": namespace, "objects": objects, "absent": absent}


def snapshot_parts(snapshot: object, *, label: str) -> tuple[str, list[Any], list[Any]]:
    """Validate a captured snapshot before anything is mutated.

    ``label`` names the snapshot in errors, e.g. ``"previous ADOT collector
    snapshot"``. An empty object list is rejected by the callers that know the
    objects must have been live; a caller whose objects may legitimately all be
    new passes them through and relies on ``absent``.
    """

    if not isinstance(snapshot, dict):
        raise ReleaseError(f"{label} is invalid")
    namespace = str(snapshot.get("namespace") or "")
    objects = snapshot.get("objects")
    absent = snapshot.get("absent") or []
    if not namespace or not isinstance(objects, list) or not isinstance(absent, list):
        raise ReleaseError(f"{label} is invalid")
    for item in absent:
        if (
            not isinstance(item, dict)
            or not item.get("name")
            or not item.get("resource")
        ):
            raise ReleaseError(f"{label} is invalid")
    return namespace, objects, absent


def apply_snapshot_objects(release: Any, objects: list[Any]) -> None:
    """Apply the captured objects as one list, so a partial apply is one call."""

    if not objects:
        return
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "previous-objects.json"
        path.write_text(
            json.dumps({"apiVersion": "v1", "kind": "List", "items": objects}),
            encoding="utf-8",
        )
        release.runner.run(release._cpu("apply", "-f", str(path)))


def delete_absent_objects(
    release: Any,
    namespace: str,
    absent: list[Any],
) -> None:
    """Remove the objects the candidate added, in reverse declaration order.

    Reverse order matters for the same reason manifest order matters going
    forward: a RoleBinding declared after the Role it binds has to go first.
    """

    for item in reversed(absent):
        release.runner.run(
            release._cpu(
                "-n",
                namespace,
                "delete",
                str(item["resource"]),
                str(item["name"]),
                "--ignore-not-found",
            )
        )


def restart_snapshot_deployments(
    release: Any,
    namespace: str,
    objects: list[Any],
    *,
    timeout: str,
) -> None:
    """Replace the Pods of every restored Deployment and wait for them.

    Restoring a mounted ConfigMap only takes effect once the Pod is replaced,
    and a rollback has to find out whether the restored workload comes back up
    rather than assume it.
    """

    for item in objects:
        if not isinstance(item, dict) or str(item.get("kind")) != "Deployment":
            continue
        name = str((item.get("metadata") or {}).get("name") or "")
        if not name:
            raise ReleaseError("restored Deployment has no name")
        release.runner.run(
            release._cpu("-n", namespace, "rollout", "restart", f"deployment/{name}")
        )
        release.runner.run(
            release._cpu(
                "-n",
                namespace,
                "rollout",
                "status",
                f"deployment/{name}",
                f"--timeout={timeout}",
            )
        )
