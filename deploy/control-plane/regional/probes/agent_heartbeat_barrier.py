"""Wait until every expected agent has heartbeat *again* against a new pin.

Request on stdin::

    {"expected_by_cluster": {"<cluster>": ["<node>", ...]},
     "required_identity": {...} | null,
     "timeout_seconds": 600, "poll_seconds": 5,
     "minimum_lease_remaining_seconds": 30}

Response: one JSON object on stdout with ``status`` PASSED or FAILED.

The barrier is "advanced past a baseline", not "recent": a node that stopped
heartbeating still has a ``last_seen_at`` that looks fresh for a while, so
comparing against a timestamp taken before the pin moved is the only reading
that proves the agent is alive *now* and running the identity we just shipped.

``blockers`` is capped at 100 entries while ``blocker_count`` stays exact: a
fleet-wide failure would otherwise put thousands of records through a kubectl
exec pipe, and the first hundred are enough to tell which failure it is.
"""

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

from gpu_fault.app import ApplicationContext


def main() -> None:
    expected = json.load(sys.stdin)
    expected_by_cluster = {
        str(cluster_id): {str(node_id) for node_id in node_ids}
        for cluster_id, node_ids in expected["expected_by_cluster"].items()
    }
    context = ApplicationContext.from_environment()
    required_identity = expected.get("required_identity") or None

    def read_agents() -> dict[tuple[str, str], Any]:
        records: dict[tuple[str, str], Any] = {}
        for cluster_id, node_ids in expected_by_cluster.items():
            records.update(
                {
                    (cluster_id, item.node_id): item
                    for item in context.store.list_agents(cluster_id)
                    if item.node_id in node_ids
                }
            )
        return records

    def pin_aligned(item: Any) -> bool:
        if required_identity is None:
            return True
        return bool(
            item.artifact_sha256 == required_identity["artifact_sha256"]
            and (item.compatibility_digest or item.artifact_sha256)
            == required_identity["compatibility_digest"]
            and str(item.agent_protocol_version)
            == str(required_identity["agent_protocol_version"])
            and item.config_digest == required_identity["config_digest"]
        )

    baseline = {key: item.last_seen_at for key, item in read_agents().items()}
    deadline = time.monotonic() + float(expected["timeout_seconds"])
    minimum_lease = float(expected["minimum_lease_remaining_seconds"])
    while True:
        now = datetime.now(timezone.utc)
        records = read_agents()
        blockers: list[dict[str, Any]] = []
        refreshed = 0
        for cluster_id, node_ids in expected_by_cluster.items():
            for node_id in sorted(node_ids):
                key = (cluster_id, node_id)
                item = records.get(key)
                if item is None:
                    if len(blockers) < 100:
                        blockers.append(
                            {
                                "cluster_id": cluster_id,
                                "node_id": node_id,
                                "reason": "missing",
                            }
                        )
                    continue
                lifecycle = getattr(item.lifecycle_state, "value", item.lifecycle_state)
                previous_seen = baseline.get(key)
                heartbeat_advanced = (
                    previous_seen is None or item.last_seen_at > previous_seen
                )
                lease_remaining = (
                    None
                    if item.lease_expires_at is None
                    else (item.lease_expires_at - now).total_seconds()
                )
                if lifecycle != "ACTIVE":
                    reason = "lifecycle-" + str(lifecycle)
                elif not heartbeat_advanced:
                    reason = "heartbeat-not-refreshed"
                elif lease_remaining is None:
                    reason = "lease-missing"
                elif lease_remaining < minimum_lease:
                    reason = "lease-margin"
                elif not pin_aligned(item):
                    reason = "pin-not-aligned"
                else:
                    refreshed += 1
                    continue
                if len(blockers) < 100:
                    blockers.append(
                        {
                            "cluster_id": cluster_id,
                            "node_id": node_id,
                            "reason": reason,
                            "last_seen_at": item.last_seen_at.isoformat(),
                            "lease_expires_at": (
                                item.lease_expires_at.isoformat()
                                if item.lease_expires_at is not None
                                else None
                            ),
                            "lease_remaining_seconds": lease_remaining,
                        }
                    )
        expected_count = sum(len(node_ids) for node_ids in expected_by_cluster.values())
        if refreshed == expected_count:
            print(
                json.dumps(
                    {
                        "status": "PASSED",
                        "expected_count": expected_count,
                        "refreshed_count": refreshed,
                    },
                    sort_keys=True,
                )
            )
            break
        if time.monotonic() >= deadline:
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "expected_count": expected_count,
                        "refreshed_count": refreshed,
                        "blocker_count": expected_count - refreshed,
                        "blockers": blockers,
                    },
                    sort_keys=True,
                )
            )
            break
        time.sleep(float(expected["poll_seconds"]))


main()
