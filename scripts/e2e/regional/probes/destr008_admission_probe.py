"""Read-only server-side proof that the run's spare activation fence is active."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

PROBE_ANNOTATION = "gpu-fault.io/acceptance-activation-probe"
DENIAL_PREFIX = "GPU_FAULT_ACCEPTANCE_ACTIVATION_DENIED:"
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,252}")


class AdmissionProbeError(RuntimeError):
    pass


class NodeApi(Protocol):
    def read_node(self, name: str, **kwargs: Any) -> Any: ...

    def patch_node(
        self, name: str, body: list[dict[str, Any]], **kwargs: Any
    ) -> Any: ...


def source_identity() -> str:
    pinned = globals().get("_SOURCE_SHA256")
    if pinned is not None:
        if not isinstance(pinned, str) or not re.fullmatch(r"[0-9a-f]{64}", pinned):
            raise AdmissionProbeError("pinned admission probe identity is invalid")
        return pinned
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_identity(*values: str) -> None:
    if any(
        not isinstance(value, str) or not IDENTIFIER.fullmatch(value)
        for value in values
    ):
        raise AdmissionProbeError("activation fence identity is invalid")


def node_version(node: Any, *, name: str, uid: str) -> str:
    metadata = getattr(node, "metadata", None)
    spec = getattr(node, "spec", None)
    if (
        metadata is None
        or getattr(metadata, "name", None) != name
        or getattr(metadata, "uid", None) != uid
        or not getattr(metadata, "resource_version", None)
        or spec is None
        or getattr(spec, "unschedulable", None) is not True
    ):
        raise AdmissionProbeError("the protected spare is not the bound cordoned node")
    return str(metadata.resource_version)


def is_fence_denial(
    error: ApiException, *, node: str, policy: str, binding: str, marker: str
) -> bool:
    if (
        error.status not in {403, 422}
        or not isinstance(error.body, str)
        or len(error.body) > 65536
    ):
        return False
    try:
        status = json.loads(error.body)
    except ValueError:
        return False
    if not isinstance(status, dict):
        return False
    details = status.get("details")
    message = status.get("message")
    return (
        status.get("kind") == "Status"
        and status.get("status") == "Failure"
        and type(status.get("code")) is int
        and status["code"] == error.status
        and status.get("reason") in {"Forbidden", "Invalid"}
        and isinstance(details, dict)
        and details.get("name") == node
        and details.get("kind") in {"Node", "nodes"}
        and isinstance(message, str)
        and policy in message
        and binding in message
        and DENIAL_PREFIX + marker in message
    )


def probe_fence(
    api: NodeApi, *, node: str, uid: str, policy: str, binding: str, marker: str
) -> dict[str, Any]:
    validate_identity(node, uid, policy, binding, marker)
    before = api.read_node(node, _request_timeout=(5, 10))
    version = node_version(before, name=node, uid=uid)
    annotations = dict(before.metadata.annotations or {})
    annotations[PROBE_ANNOTATION] = marker
    tests: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": uid},
        {"op": "test", "path": "/metadata/resourceVersion", "value": version},
    ]
    safe = api.patch_node(
        node,
        [*tests, {"op": "add", "path": "/metadata/annotations", "value": annotations}],
        dry_run="All",
        _request_timeout=(5, 10),
    )
    node_version(safe, name=node, uid=uid)
    if (safe.metadata.annotations or {}).get(PROBE_ANNOTATION) != marker:
        raise AdmissionProbeError("safe dry-run update was not acknowledged")
    try:
        api.patch_node(
            node,
            [*tests, {"op": "add", "path": "/spec/unschedulable", "value": False}],
            dry_run="All",
            _request_timeout=(5, 10),
        )
    except ApiException as exc:
        if not is_fence_denial(
            exc, node=node, policy=policy, binding=binding, marker=marker
        ):
            raise AdmissionProbeError(
                "activation refusal was not issued by the bound fence"
            ) from None
    else:
        raise AdmissionProbeError("the API accepted a forbidden activation dry run")
    after = api.read_node(node, _request_timeout=(5, 10))
    node_version(after, name=node, uid=uid)
    if (after.metadata.annotations or {}).get(PROBE_ANNOTATION) != (
        before.metadata.annotations or {}
    ).get(PROBE_ANNOTATION):
        raise AdmissionProbeError("dry-run probe changed persistent node metadata")
    return {
        "state": "DENYING_ACTIVATION",
        "node": node,
        "node_uid": uid,
        "policy": policy,
        "binding": binding,
        "marker": marker,
        "safe_dry_run_acknowledged": True,
        "activation_dry_run_denied": True,
        "probe_not_persisted": True,
        "source_sha256": source_identity(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        "kubeconfig",
        "context",
        "node",
        "uid",
        "policy",
        "binding",
        "marker",
    ):
        parser.add_argument(f"--{argument}", required=True)
    options = parser.parse_args(argv)
    try:
        configuration = client.Configuration()
        config.load_kube_config(
            config_file=options.kubeconfig,
            context=options.context,
            client_configuration=configuration,
        )
        if (
            configuration.verify_ssl is not True
            or urlsplit(configuration.host).scheme != "https"
        ):
            raise AdmissionProbeError("the API requires verified HTTPS")
        with client.ApiClient(configuration) as api_client:
            result = probe_fence(
                client.CoreV1Api(api_client),
                node=options.node,
                uid=options.uid,
                policy=options.policy,
                binding=options.binding,
                marker=options.marker,
            )
    except Exception as exc:
        print(json.dumps({"state": "FAILED", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
