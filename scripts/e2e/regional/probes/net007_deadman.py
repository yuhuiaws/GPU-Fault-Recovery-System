"""Run-owned, bounded removal of the NET-007 admission webhook."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from kubernetes import client, config
from urllib3.exceptions import HTTPError

RUN_LABEL = "gpu-fault.io/acceptance-run"


def remove_webhook(api: Any, *, name: str, run_id: str) -> bool:
    try:
        current = api.read_validating_webhook_configuration(
            name, _request_timeout=(5, 10)
        )
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return True
        raise
    metadata = current.metadata
    if (
        not run_id
        or not metadata.uid
        or not metadata.resource_version
        or (metadata.labels or {}).get(RUN_LABEL) != run_id
    ):
        raise RuntimeError("webhook ownership changed; refusing deadman deletion")
    api.delete_validating_webhook_configuration(
        name,
        body=client.V1DeleteOptions(
            preconditions=client.V1Preconditions(
                uid=metadata.uid, resource_version=metadata.resource_version
            )
        ),
        _request_timeout=(5, 10),
    )
    try:
        remaining = api.read_validating_webhook_configuration(
            name, _request_timeout=(5, 10)
        )
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return True
        raise
    if remaining.metadata.uid != metadata.uid:
        raise RuntimeError("webhook was replaced during deadman deletion")
    return False


def watch(api: Any, *, name: str, run_id: str, restore_at: float) -> None:
    if (
        not name
        or not run_id
        or not math.isfinite(restore_at)
        or restore_at <= time.time()
    ):
        raise RuntimeError("deadman identity or deadline is invalid")
    try:
        api.read_validating_webhook_configuration(name, _request_timeout=(5, 10))
    except client.exceptions.ApiException as exc:
        if exc.status != 404:
            raise
    else:
        raise RuntimeError("webhook already exists before deadman admission")
    print(
        json.dumps({"state": "ARMED", "run_id": run_id, "restore_at": restore_at}),
        flush=True,
    )
    while time.time() < restore_at:
        time.sleep(min(5, max(0, restore_at - time.time())))
    deadline = time.monotonic() + 120
    while True:
        try:
            if remove_webhook(api, name=name, run_id=run_id):
                print(json.dumps({"state": "REMOVED", "run_id": run_id}), flush=True)
                return
        except client.exceptions.ApiException as exc:
            if exc.status not in {404, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError("deadman removal was refused") from None
        except (HTTPError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("deadman could not confirm webhook removal")
        time.sleep(2)


def main() -> None:
    config.load_incluster_config()
    watch(
        client.AdmissionregistrationV1Api(),
        name=os.environ["NET007_WEBHOOK_NAME"],
        run_id=os.environ["NET007_RUN_ID"],
        restore_at=float(os.environ["NET007_RESTORE_AT"]),
    )


if __name__ == "__main__":
    main()
