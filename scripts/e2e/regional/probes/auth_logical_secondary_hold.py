#!/usr/bin/env python3
"""Hold the AUTH-007/008 synthetic secondary's executor Pod open.

The second logical cluster's Pod exists so the case can ``kubectl exec`` its
claim probes from inside a container that carries B's connection environment
(``GPU_FAULT_CLUSTER_ID``, B's token, the control-plane URL and CA, the
executor pins). Nothing here talks to the control plane: the deployed claim
loop is not started, so B's only authenticated traffic is what the case sends,
and a disabled or removed registration is observed exactly where the case
looks for it. The Pod reports readiness the way every seeded-command probe
does -- ``/state/ready.json`` -- and leaves on SIGTERM or when the Pod's
``activeDeadlineSeconds`` ends it.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from pathlib import Path

STATE_DIRECTORY = Path(os.getenv("GPU_FAULT_SYNTHETIC_SECONDARY_STATE", "/state"))
READY_FILE = "ready.json"
POLL_SECONDS = 1.0


def write_ready(directory: Path) -> dict[str, object]:
    """Publish the identity this container holds, atomically."""

    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "cluster_id": os.environ["GPU_FAULT_CLUSTER_ID"],
        "hostname": socket.gethostname(),
        "started_at_epoch": time.time(),
        "claims_issued": 0,
    }
    temporary = directory / (READY_FILE + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, directory / READY_FILE)
    return payload


def hold_until_stopped(
    stopped: list[bool], *, poll_seconds: float = POLL_SECONDS
) -> int:
    """Sleep until ``stopped`` holds a truthy flag; the signal handlers set it."""

    while not any(stopped):
        time.sleep(poll_seconds)
    return 0


def main() -> int:
    stopped: list[bool] = []

    def stop(signum: int, _frame: object) -> None:
        stopped.append(True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write_ready(STATE_DIRECTORY)
    return hold_until_stopped(stopped)


if __name__ == "__main__":
    raise SystemExit(main())
