"""UID-bound Kubernetes cleanup for a proven cluster removal attempt."""

from __future__ import annotations

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
from gpu_fault.admin.execution import deadline_scope
from gpu_fault.admin.process_supervisor import ensure_supervision_safe

Command = Callable[..., subprocess.CompletedProcess[str]]


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


def clear_installer_annotations(
    runner: Command,
    kubectl: Sequence[str],
    *,
    hyperpod_name: str,
    node_uids: Mapping[str, str],
    annotations: Sequence[str],
) -> None:
    if not node_uids:
        return
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
            f"sagemaker.amazonaws.com/cluster-name={hyperpod_name}",
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
                item.get("kind") != "Node"
                or not isinstance(name, str)
                or name not in node_uids
                or name in live
                or metadata.get("uid") != node_uids[name]
                or not isinstance(metadata.get("resourceVersion"), str)
                or not metadata["resourceVersion"]
                or (metadata.get("labels") or {}).get(
                    "sagemaker.amazonaws.com/cluster-name"
                )
                != hyperpod_name
                or metadata.get("annotations") is not None
                and not isinstance(metadata["annotations"], dict)
            ):
                raise ValueError
            live[name] = metadata
        if set(live) != set(node_uids):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BootstrapError(
            "node identity drifted before annotation cleanup"
        ) from None
    patches: list[tuple[str, list[dict[str, str]]]] = []
    for name, metadata in live.items():
        present = metadata.get("annotations") or {}
        removals = [
            {
                "op": "remove",
                "path": "/metadata/annotations/"
                + annotation.replace("~", "~0").replace("/", "~1"),
            }
            for annotation in annotations
            if annotation in present
        ]
        if removals:
            patches.append(
                (
                    name,
                    [
                        {
                            "op": "test",
                            "path": "/metadata/uid",
                            "value": node_uids[name],
                        },
                        {
                            "op": "test",
                            "path": "/metadata/resourceVersion",
                            "value": metadata["resourceVersion"],
                        },
                        *removals,
                    ],
                )
            )

    def apply_patch(name: str, patch: list[dict[str, str]]) -> None:
        completed = runner(
            [*kubectl, "patch", "node", name, "--type=json", "-p", json.dumps(patch)],
            timeout_seconds=30,
        )
        if completed.returncode:
            raise BootstrapError("node annotation cleanup failed its guarded patch")

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
