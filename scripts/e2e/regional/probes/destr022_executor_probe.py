#!/usr/bin/env python3
"""Read one cluster executor Pod's claim-state breadcrumb and sweep env.

GF-REGIONAL-DESTR-022 runs this inside each ``gpu-fault-cluster-executor``
Pod (``kubectl exec -i <pod> -- python3 -``, stdin-fed). The executor Pod has
no ``/metrics`` listener; its counters -- including
``spare_reservations_reclaimed_total`` -- ride in the claim-state file the
claim loop rewrites on every successful claim round-trip. Read-only: it opens
the file and the environment and prints one JSON document. Stdlib only, since
it runs with whatever the executor image ships.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any

CLAIM_STATE_PATH_ENV = "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH"
DEFAULT_CLAIM_STATE_PATH = "/tmp/executor-claim-state.json"
ENVIRONMENT_KEYS = (
    "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER",
    "GPU_FAULT_HYPERPOD_SPARE_LABEL",
    "GPU_FAULT_ALLOW_HYPERPOD_REPLACE",
    CLAIM_STATE_PATH_ENV,
)


def read_claim_state(path: str) -> tuple[dict[str, Any] | None, str | None]:
    """The parsed breadcrumb, or ``None`` plus why it could not be read."""

    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None, "claim state file does not exist"
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "claim state is not a JSON object"
    return value, None


def report(environ: dict[str, str], hostname: str) -> dict[str, Any]:
    path = environ.get(CLAIM_STATE_PATH_ENV) or DEFAULT_CLAIM_STATE_PATH
    claim_state, error = read_claim_state(path)
    return {
        "hostname": hostname,
        "claim_state_path": path,
        "claim_state": claim_state,
        "claim_state_error": error,
        "env": {key: environ.get(key) for key in ENVIRONMENT_KEYS},
    }


def main() -> int:
    print(
        json.dumps(
            report(dict(os.environ), socket.gethostname()),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
