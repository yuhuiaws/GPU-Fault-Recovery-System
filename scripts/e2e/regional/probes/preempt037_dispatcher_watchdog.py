"""Bounded CPU-only restoration for the PREEMPT-037 dispatcher window."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from kubernetes import client, config
from urllib3.exceptions import HTTPError

VARIABLE = "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER"
DEPLOYMENT = "gpu-fault-control-worker"
CONTAINER = "control-worker"


def deployment_patch(
    deployment: Any, *, uid: str, baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    if not uid or deployment.metadata.uid != uid:
        raise RuntimeError("dispatcher Deployment identity changed")
    if not deployment.metadata.resource_version:
        raise RuntimeError("dispatcher Deployment version is unknown")
    if (
        type(baseline.get("present")) is not bool
        or (baseline["present"] and baseline.get("value") != "true")
        or (not baseline["present"] and baseline.get("value") is not None)
    ):
        raise RuntimeError("dispatcher baseline is invalid")
    containers = deployment.spec.template.spec.containers
    matches = [
        (index, item) for index, item in enumerate(containers) if item.name == CONTAINER
    ]
    if len(matches) != 1:
        raise RuntimeError("dispatcher container identity changed")
    container_index, container = matches[0]
    if not baseline.get("image") or container.image != baseline["image"]:
        raise RuntimeError("dispatcher image changed during the window")
    variables = [
        (index, item)
        for index, item in enumerate(container.env or [])
        if item.name == VARIABLE
    ]
    if len(variables) > 1 or any(item.value_from is not None for _, item in variables):
        raise RuntimeError("dispatcher setting is ambiguous")
    current = {
        "present": bool(variables),
        "value": variables[0][1].value if variables else None,
    }
    original = {key: baseline[key] for key in ("present", "value")}
    if current == original:
        return []
    if current != {"present": True, "value": "false"}:
        raise RuntimeError("dispatcher setting is no longer owned by this window")
    env_index = variables[0][0]
    path = f"/spec/template/spec/containers/{container_index}/env/{env_index}"
    change = (
        {"op": "replace", "path": path + "/value", "value": baseline["value"]}
        if baseline["present"]
        else {"op": "remove", "path": path}
    )
    return [
        {"op": "test", "path": "/metadata/uid", "value": uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": deployment.metadata.resource_version,
        },
        {"op": "test", "path": path + "/name", "value": VARIABLE},
        {"op": "test", "path": path + "/value", "value": "false"},
        change,
    ]


def restore(api: Any, namespace: str, uid: str, baseline: dict[str, Any]) -> None:
    current = api.read_namespaced_deployment(
        DEPLOYMENT, namespace, _request_timeout=(5, 10)
    )
    patch = deployment_patch(current, uid=uid, baseline=baseline)
    if patch:
        api.patch_namespaced_deployment(
            DEPLOYMENT, namespace, patch, _request_timeout=(5, 10)
        )
    restored = api.read_namespaced_deployment(
        DEPLOYMENT, namespace, _request_timeout=(5, 10)
    )
    if deployment_patch(restored, uid=uid, baseline=baseline):
        raise RuntimeError("dispatcher restoration was not confirmed")


def main() -> None:
    restore_at = float(os.environ["PREEMPT037_RESTORE_AT"])
    if not math.isfinite(restore_at) or restore_at <= 0:
        raise RuntimeError("watchdog deadline is invalid")
    namespace = os.environ["PREEMPT037_NAMESPACE"]
    uid = os.environ["PREEMPT037_DEPLOYMENT_UID"]
    baseline = json.loads(os.environ["PREEMPT037_BASELINE"])
    config.load_incluster_config()
    api = client.AppsV1Api()
    current = api.read_namespaced_deployment(
        DEPLOYMENT, namespace, _request_timeout=(5, 10)
    )
    if deployment_patch(current, uid=uid, baseline=baseline):
        raise RuntimeError("dispatcher window was opened before the watchdog was armed")
    print(
        json.dumps(
            {
                "state": "ARMED",
                "restore_at": restore_at,
                "deployment_uid": uid,
                "image": baseline["image"],
            }
        ),
        flush=True,
    )
    while time.time() < restore_at:
        time.sleep(max(0, min(5, restore_at - time.time())))
    deadline = time.monotonic() + 120
    while True:
        try:
            restore(api, namespace, uid, baseline)
            print(json.dumps({"state": "RESTORED"}), flush=True)
            return
        except client.exceptions.ApiException as exc:
            if exc.status not in {409, 429, 500, 502, 503, 504}:
                raise RuntimeError("watchdog could not confirm restoration") from None
        except (HTTPError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("watchdog could not confirm restoration")
        time.sleep(2)


if __name__ == "__main__":
    main()
