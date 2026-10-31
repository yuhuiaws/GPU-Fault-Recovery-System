"""UID-bound Kubernetes cleanup for a proven cluster removal attempt."""

from __future__ import annotations

import functools
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, cast
from urllib.parse import quote

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_limits import DEPLOY_CONCURRENCY
from gpu_fault.cluster_scope import node_selector as scope_selector
from gpu_fault.cluster_scope import selector_matches
from gpu_fault.admin.execution import deadline_scope
from gpu_fault.admin.kubectl_failure import KubectlFailure
from gpu_fault.admin.node_patch_retry import NodePatchRetry, guarded, pointer
from gpu_fault.admin.process_supervisor import ensure_supervision_safe

Command = Callable[..., subprocess.CompletedProcess[str]]


class NodePatchCommandFailed(BootstrapError):
    """A guarded installer annotation patch failed at kubectl.

    ``failure`` names the call, kubectl's exit status (none when the call ran
    out of time), a reason class and kubectl's redacted last stderr line
    (``gpu_fault.admin.kubectl_failure``). The annotation cleanup's retry
    (``NODE_ANNOTATION_RETRY``) answers this type with a fresh read of the Node.
    """

    def __init__(self, failure: KubectlFailure) -> None:
        self.failure = failure
        super().__init__(f"node annotation cleanup failed its guarded patch {failure}")


NODE_ANNOTATION_RETRY = NodePatchRetry(
    subject="node annotation cleanup",
    failure=NodePatchCommandFailed,
    refuse=BootstrapError,
)


def namespace_document(
    runner: Command, kubectl: Sequence[str], namespace: str
) -> dict[str, Any] | None:
    result = runner(
        [
            *kubectl,
            "get",
            "namespace",
            namespace,
            "--ignore-not-found",
            "-o",
            "json",
            "--request-timeout=15s",
        ],
        timeout_seconds=20,
    )
    if result.returncode:
        raise BootstrapError(
            f"cannot read namespace identity (exit {result.returncode})"
        )
    if not result.stdout.strip():
        return None
    try:
        document = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise BootstrapError("namespace query returned invalid JSON") from None
    metadata = document.get("metadata") if isinstance(document, dict) else None
    if (
        not isinstance(metadata, dict)
        or document.get("kind") != "Namespace"
        or metadata.get("name") != namespace
        or not isinstance(metadata.get("uid"), str)
        or not metadata["uid"]
    ):
        raise BootstrapError("namespace query returned a conflicting identity")
    return cast(dict[str, Any], document)


def require_namespace_uid(expected: object) -> str:
    if not isinstance(expected, str) or not expected:
        raise BootstrapError("namespace cleanup has no proven UID")
    return expected


def request_namespace_deletion(
    runner: Command,
    kubectl: Sequence[str],
    namespace: str,
    expected_uid: object,
) -> None:
    uid = require_namespace_uid(expected_uid)
    document = namespace_document(runner, kubectl, namespace)
    if document is None:
        return
    if document["metadata"]["uid"] != uid:
        raise BootstrapError("target namespace was recreated before deletion")
    result = runner(
        [
            *kubectl,
            "delete",
            "--raw",
            "/api/v1/namespaces/" + quote(namespace, safe=""),
            "-f",
            "-",
        ],
        input_text=json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {"uid": uid},
                "propagationPolicy": "Foreground",
            }
        ),
        timeout_seconds=30,
    )
    if result.returncode:
        # A lost delete acknowledgement is harmless only after a fresh,
        # successful read proves absence. An error response alone proves nothing.
        if namespace_document(runner, kubectl, namespace) is None:
            return
        raise BootstrapError(
            f"failed to delete target GPU namespace (exit {result.returncode})"
        )


def wait_namespace_absent(
    runner: Command,
    kubectl: Sequence[str],
    namespace: str,
    expected_uid: object,
    *,
    timeout_seconds: float = 600,
    interval_seconds: float = 5,
) -> None:
    uid = require_namespace_uid(expected_uid)
    try:
        with deadline_scope("target GPU namespace deletion", timeout_seconds) as budget:
            while True:
                ensure_supervision_safe()
                document = namespace_document(runner, kubectl, namespace)
                budget.remaining()
                if document is None:
                    return
                if document["metadata"]["uid"] != uid:
                    raise BootstrapError(
                        "target namespace was recreated during deletion"
                    )
                time.sleep(min(interval_seconds, budget.remaining()))
    except (TimeoutError, subprocess.TimeoutExpired):
        raise BootstrapError(
            "target GPU namespace deletion exceeded its deadline"
        ) from None


def _target_node(item: Any, selector: str) -> bool:
    """A well-formed Node document of this cluster's node selector."""

    metadata = item["metadata"]
    name = metadata["name"]
    return bool(
        item.get("kind") == "Node"
        and isinstance(name, str)
        and name
        and isinstance(metadata.get("resourceVersion"), str)
        and metadata["resourceVersion"]
        and selector_matches(metadata.get("labels") or {}, selector)
        and (
            metadata.get("annotations") is None
            or isinstance(metadata["annotations"], dict)
        )
    )


def _annotation_patch(
    item: dict[str, Any], *, uid: str, annotations: Sequence[str]
) -> list[dict[str, Any]] | None:
    """The UID/resourceVersion-bound removal of the annotations ``item`` still has."""

    present = item["metadata"].get("annotations") or {}
    removals = [
        {"op": "remove", "path": "/metadata/annotations/" + pointer(annotation)}
        for annotation in annotations
        if annotation in present
    ]
    return [*guarded(item, uid), *removals] if removals else None


def _read_target_node(
    runner: Command, kubectl: Sequence[str], name: str, selector: str
) -> dict[str, Any] | None:
    """A fresh read of one target Node after its guarded patch failed."""

    arguments = ["get", "node", name, "--ignore-not-found", "-o", "json"]
    result = runner([*kubectl, *arguments], timeout_seconds=30)
    if result.returncode:
        raise BootstrapError(
            f"cannot re-read node {name} for annotation cleanup "
            f"{KubectlFailure.of(arguments, result.returncode, result.stderr)}"
        )
    if not result.stdout.strip():
        return None
    try:
        item = json.loads(result.stdout)
        if not _target_node(item, selector) or item["metadata"]["name"] != name:
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BootstrapError(
            f"node identity drifted during annotation cleanup: {name}"
        ) from None
    return cast(dict[str, Any], item)


def clear_installer_annotations(
    runner: Command,
    kubectl: Sequence[str],
    *,
    hyperpod_name: str = "",
    node_uids: Mapping[str, str],
    annotations: Sequence[str],
    node_selector: str = "",
) -> None:
    """Strip installer annotations from the named nodes after proving identity.

    ``node_selector`` is the cluster's GPU node selector
    (``gpu_fault.cluster_scope.node_selector``); ``hyperpod_name`` alone still
    works for HyperPod callers and derives the same label text.

    One list proves every node's identity before any patch; each patch is then
    bound to the UID and resourceVersion that list returned. ``remove-cluster``
    runs this right after its node DaemonSets are deleted, when the kubelet
    posts Node status: a patch refused because the Node moved is re-read and
    rebuilt from the fresh Node (``NODE_ANNOTATION_RETRY``, the retry of the
    reset's node metadata cleanup): a new UID or a vanished Node is a hard
    error, an unchanged resourceVersion is the patch's own failure, raised with
    kubectl's classified reason (:class:`NodePatchCommandFailed`).
    """

    if not node_uids:
        return
    selector = node_selector or scope_selector(
        None, provider="hyperpod", hyperpod_cluster_name=hyperpod_name
    )
    if any(
        not isinstance(name, str) or not name or not isinstance(uid, str) or not uid
        for name, uid in node_uids.items()
    ):
        raise BootstrapError("node annotation cleanup has incomplete UID bindings")
    result = runner(
        [
            *kubectl,
            "get",
            "nodes",
            "-l",
            selector,
            "-o",
            "json",
        ],
        timeout_seconds=120,
    )
    if result.returncode:
        raise BootstrapError("cannot verify nodes before annotation cleanup")
    try:
        document = json.loads(result.stdout)
        items = document["items"]
        if not isinstance(items, list):
            raise ValueError
        live: dict[str, dict[str, Any]] = {}
        for item in items:
            metadata = item["metadata"]
            name = metadata["name"]
            if (
                not _target_node(item, selector)
                or name not in node_uids
                or name in live
                or metadata.get("uid") != node_uids[name]
            ):
                raise ValueError
            live[name] = item
        if set(live) != set(node_uids):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BootstrapError(
            "node identity drifted before annotation cleanup"
        ) from None
    patches: list[tuple[str, list[dict[str, Any]]]] = []
    for name, item in live.items():
        patch = _annotation_patch(item, uid=node_uids[name], annotations=annotations)
        if patch is not None:
            patches.append((name, patch))

    def send(name: str, patch: list[dict[str, Any]]) -> None:
        arguments = ["patch", "node", name, "--type=json", "-p", json.dumps(patch)]
        try:
            completed = runner([*kubectl, *arguments], timeout_seconds=30)
        except subprocess.TimeoutExpired:
            raise NodePatchCommandFailed(KubectlFailure.of(arguments, None)) from None
        if completed.returncode:
            raise NodePatchCommandFailed(
                KubectlFailure.of(arguments, completed.returncode, completed.stderr)
            )

    def apply_patch(name: str, patch: list[dict[str, Any]]) -> None:
        uid = node_uids[name]
        NODE_ANNOTATION_RETRY.apply(
            name,
            live[name],
            patch,
            uid=uid,
            send=functools.partial(send, name),
            read=functools.partial(_read_target_node, runner, kubectl, name, selector),
            plan=functools.partial(_annotation_patch, uid=uid, annotations=annotations),
        )

    if not patches:
        return
    errors: list[BaseException] = []
    with ThreadPoolExecutor(
        max_workers=min(DEPLOY_CONCURRENCY.metadata_cleanup, len(patches))
    ) as executor:
        futures = [
            executor.submit(copy_context().run, apply_patch, name, patch)
            for name, patch in patches
        ]
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
                for pending in futures:
                    pending.cancel()
    if errors:
        raise next(
            (error for error in errors if not isinstance(error, Exception)), errors[0]
        )
