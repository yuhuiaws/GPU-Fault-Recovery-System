"""GPU-local, UID-fenced restoration of one acceptance-owned plugin affinity."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

WINDOW_KEY = "gpu-fault.io/collector-plugin-window"
WINDOW_PATH = "/metadata/annotations/gpu-fault.io~1collector-plugin-window"
AFFINITY_PATH = "/spec/template/spec/affinity"


def restoration_patch(
    document: dict[str, Any], record: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = document.get("metadata") or {}
    spec = document.get("spec") or {}
    if metadata.get("uid") != record["daemonset_uid"] or not metadata.get(
        "resourceVersion"
    ):
        raise RuntimeError("plugin watchdog DaemonSet identity changed")
    if (spec.get("updateStrategy") or {}).get("type") != "OnDelete":
        raise RuntimeError("plugin watchdog requires the approved OnDelete policy")
    pod_spec = spec["template"]["spec"]
    baseline = record["baseline"]
    restored = ("affinity" in pod_spec) == baseline["present"] and pod_spec.get(
        "affinity"
    ) == baseline["value"]
    owner = (metadata.get("annotations") or {}).get(WINDOW_KEY)
    if owner == "CLOSED:" + record["window_id"] and restored:
        return []
    if owner != record["window_id"]:
        raise RuntimeError("plugin watchdog no longer owns the affinity window")
    if not restored and pod_spec.get("affinity") != record["excluded_affinity"]:
        raise RuntimeError("plugin watchdog refuses foreign affinity")
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": record["daemonset_uid"]},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": metadata["resourceVersion"],
        },
        {"op": "test", "path": WINDOW_PATH, "value": record["window_id"]},
        {"op": "test", "path": "/spec/updateStrategy/type", "value": "OnDelete"},
    ]
    if not restored:
        patch.append(
            {"op": "replace", "path": AFFINITY_PATH, "value": baseline["value"]}
            if baseline["present"]
            else {"op": "remove", "path": AFFINITY_PATH}
        )
    patch.append(
        {"op": "replace", "path": WINDOW_PATH, "value": "CLOSED:" + record["window_id"]}
    )
    return patch


def restore(api: Any, record: dict[str, Any]) -> None:
    def read() -> dict[str, Any]:
        value = api.read_namespaced_daemon_set(
            record["daemonset"], record["namespace"], _request_timeout=(5, 10)
        )
        return dict(api.api_client.sanitize_for_serialization(value))

    patch = restoration_patch(read(), record)
    if patch:
        api.patch_namespaced_daemon_set(
            record["daemonset"], record["namespace"], patch, _request_timeout=(5, 10)
        )
    if restoration_patch(read(), record):
        raise RuntimeError("plugin watchdog restoration readback failed")


def main() -> None:
    from kubernetes import client, config

    record = json.loads(os.environ["COLLECTOR_PLUGIN_WINDOW"])
    restore_at = record["restore_at"]
    if isinstance(restore_at, bool) or not math.isfinite(restore_at) or restore_at <= 0:
        raise RuntimeError("plugin watchdog deadline is invalid")
    config.load_incluster_config()
    api = client.AppsV1Api()
    current = api.api_client.sanitize_for_serialization(
        api.read_namespaced_daemon_set(
            record["daemonset"], record["namespace"], _request_timeout=(5, 10)
        )
    )
    # Prove restoration is authorized before acknowledging the exclusion.
    restoration_patch(current, record)
    print(
        json.dumps(
            {
                "state": "ARMED",
                "window_id": record["window_id"],
                "daemonset_uid": record["daemonset_uid"],
                "restore_at": restore_at,
            }
        ),
        flush=True,
    )
    monotonic_end = time.monotonic() + max(0, restore_at - time.time())
    while time.time() < restore_at and time.monotonic() < monotonic_end:
        time.sleep(
            max(0, min(2, restore_at - time.time(), monotonic_end - time.monotonic()))
        )
    retry_end = time.monotonic() + 180
    while True:
        try:
            restore(api, record)
            print(
                json.dumps({"state": "RESTORED", "window_id": record["window_id"]}),
                flush=True,
            )
            return
        except client.exceptions.ApiException as exc:
            if (
                exc.status not in {409, 429, 500, 502, 503, 504}
                or time.monotonic() >= retry_end
            ):
                raise RuntimeError("plugin watchdog restoration unconfirmed") from None
            time.sleep(2)


if __name__ == "__main__":
    main()
