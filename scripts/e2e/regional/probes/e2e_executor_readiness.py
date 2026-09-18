"""Read the running executor's claim proof without claiming any commands."""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# An acceptance ceiling; authenticated readiness also enforces the live limit.
MAX_CLAIM_AGE_SECONDS = 300


def claim_snapshot(
    cluster_id: str, pod_name: str, container_started_at: str
) -> dict[str, Any]:
    if os.environ.get("GPU_FAULT_CLUSTER_ID") != cluster_id:
        raise ValueError("executor cluster identity differs")
    if socket.gethostname() != pod_name:
        raise ValueError("executor Pod identity differs")
    executor_id = os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_ID", f"{cluster_id}/{pod_name}")
    path = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        "/tmp/executor-claim-state.json",
    )
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("executor_id") != executor_id:
        raise ValueError("executor claim identity is missing or differs")
    owners = state.get("execution_owners")
    if (
        not isinstance(owners, list)
        or not owners
        or any(not isinstance(owner, str) or not owner.strip() for owner in owners)
        or len(set(owners)) != len(owners)
    ):
        raise ValueError("executor claim owners are missing or malformed")
    raw_claim = state.get("last_successful_claim_at")
    if not isinstance(raw_claim, str):
        raise ValueError("executor has no successful claim timestamp")
    claimed_at = datetime.fromisoformat(raw_claim)
    started_at = datetime.fromisoformat(container_started_at)
    observed_at = datetime.now(timezone.utc)
    if claimed_at.tzinfo is None or started_at.tzinfo is None:
        raise ValueError("executor claim or container timestamp has no timezone")
    age = (observed_at - claimed_at).total_seconds()
    if claimed_at < started_at or not 0 <= age <= MAX_CLAIM_AGE_SECONDS:
        raise ValueError("executor claim is stale, future or predates its container")
    return {
        "executor_id": executor_id,
        "execution_owners": sorted(owners),
        "last_successful_claim_at": claimed_at.isoformat(),
        "observed_at": observed_at.isoformat(),
        "claim_age_seconds": age,
    }


def readiness_snapshot(
    cluster_id: str, pod_name: str, container_started_at: str
) -> dict[str, Any]:
    from gpu_fault.cluster_executor.bootstrap import readiness_probe

    before = claim_snapshot(cluster_id, pod_name, container_started_at)
    if readiness_probe() != 0:
        raise ValueError("authenticated executor readiness failed")
    after = claim_snapshot(cluster_id, pod_name, container_started_at)
    if (
        before["executor_id"] != after["executor_id"]
        or before["execution_owners"] != after["execution_owners"]
        or datetime.fromisoformat(after["last_successful_claim_at"])
        < datetime.fromisoformat(before["last_successful_claim_at"])
    ):
        raise ValueError("executor claim identity changed or timestamp regressed")
    return {
        "cluster_id": cluster_id,
        "pod": pod_name,
        "authenticated_ready": True,
        "claim_before": before,
        "claim_after": after,
    }


def main() -> None:
    try:
        result = readiness_snapshot(*sys.argv[1:])
    except Exception as exc:
        # Do not relay runtime exception payloads or the unfiltered breadcrumb.
        raise SystemExit(
            f"executor readiness evidence failed: {type(exc).__name__}"
        ) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
