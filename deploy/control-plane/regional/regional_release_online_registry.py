from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError
from regional_release_registry import registry


CLIENT = r"""
import json
import os
import sys
import urllib.request

method, path = sys.argv[1:]
body = sys.stdin.buffer.read()
request = urllib.request.Request(
    "http://127.0.0.1:8080" + path,
    data=body or None,
    method=method,
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""


def _request(
    release: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("no running CPU ingress Pod for registry update")
    output = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-c",
            CLIENT,
            method,
            path,
        ),
        input_text=json.dumps(payload or {}, separators=(",", ":")),
        capture=True,
        sensitive=True,
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ReleaseError("regional registry API returned a non-object")
    return value


def _registrations(
    release: Any,
    lifecycle_overrides: dict[str, str],
) -> list[dict[str, Any]]:
    result = []
    for source in registry(release):
        item = dict(source)
        token = str(item.pop("token", ""))
        if token:
            item["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
        if "token_sha256" not in item:
            raise ReleaseError("regional registry entry has no token digest source")
        item["lifecycle_state"] = lifecycle_overrides.get(
            str(item["cluster_id"]),
            "ACTIVE",
        )
        result.append(item)
    return sorted(result, key=lambda item: str(item["cluster_id"]))


def publish_current_registry(
    release: Any,
    *,
    reason: str,
    lifecycle_overrides: dict[str, str] | None = None,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    current = _request(release, "GET", "/v1/regional/registry/status")
    published = _request(
        release,
        "POST",
        "/v1/regional/registry/revisions",
        {
            "expected_generation": int(current["generation"]),
            "registrations": _registrations(
                release,
                lifecycle_overrides or {},
            ),
            "reason": reason,
        },
    )
    generation = int(published["generation"])
    digest = str(published["content_sha256"])
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = _request(release, "GET", "/v1/regional/registry/status")
        if (
            int(status["generation"]) == generation
            and str(status["content_sha256"]) == digest
            and status.get("converged") is True
        ):
            return status
        time.sleep(1)
    raise ReleaseError(f"regional registry generation {generation} did not converge")


def prepare_join_registry(release: Any, cluster_id: str) -> None:
    publish_current_registry(
        release,
        reason=f"join {cluster_id} pending",
        lifecycle_overrides={cluster_id: "PENDING"},
    )


def activate_join_registry(release: Any, cluster_id: str) -> None:
    publish_current_registry(
        release,
        reason=f"join {cluster_id} active",
    )


def drain_registry_cluster(release: Any, cluster_id: str) -> None:
    release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    publish_current_registry(
        release,
        reason=f"remove {cluster_id} draining",
        lifecycle_overrides={cluster_id: "DRAINING"},
    )


def revoke_registry_cluster(release: Any, cluster_id: str) -> None:
    publish_current_registry(
        release,
        reason=f"remove {cluster_id} revoked",
        lifecycle_overrides={cluster_id: "REVOKED"},
    )


def purge_registry_cluster(release: Any, cluster_id: str) -> None:
    publish_current_registry(
        release,
        reason=f"remove {cluster_id} purged",
    )
