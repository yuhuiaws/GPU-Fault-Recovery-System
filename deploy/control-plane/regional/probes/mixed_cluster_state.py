"""Report whether a cluster's agents are all on identities this release owns.

Request on stdin::

    {"cluster_id": "...", "node_ids": [...], "deployment_id": "...",
     "allowed_identities": [{"protocol": ..., "artifact": ..., ...}, ...]}

Response: one JSON object on stdout with the agent blockers and the fleet
deployment record (null when absent).

This is what a resume checks: an interrupted release can leave a cluster
half-upgraded, and resuming is only safe if every agent sits on an identity the
transaction itself put there. Anything else -- a node on an identity from
*outside* the transaction -- means the fleet moved under the release, and the
resume has to stop rather than pave over it.
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError


def main() -> None:
    expected = json.load(sys.stdin)
    context = ApplicationContext.from_environment()
    now = datetime.now(timezone.utc)
    allowed = {
        (
            item["protocol"],
            item["artifact"],
            item["compatibility"],
            item["bundle"],
            item["template"],
            item["profile"],
            item["config"],
        )
        for item in expected["allowed_identities"]
    }
    expected_nodes = set(expected["node_ids"])
    records = {
        item.node_id: item
        for item in context.store.list_agents(expected["cluster_id"])
        if item.node_id in expected_nodes
    }
    blockers: list[dict[str, Any]] = []
    blocker_count = 0
    for node_id in sorted(expected_nodes):
        item = records.get(node_id)
        if item is None:
            blocker_count += 1
            if len(blockers) < 100:
                blockers.append({"node_id": node_id, "reason": "missing"})
            continue
        identity = (
            item.agent_protocol_version,
            item.artifact_sha256,
            item.compatibility_digest or item.artifact_sha256,
            item.installer_bundle_sha256,
            item.installer_template_sha256,
            item.runtime_profile_version,
            item.config_digest,
        )
        lifecycle = getattr(item.lifecycle_state, "value", item.lifecycle_state)
        if lifecycle != "ACTIVE":
            reason = "lifecycle-" + str(lifecycle)
        elif item.lease_expires_at is None or item.lease_expires_at <= now:
            reason = "lease-expired"
        elif identity not in allowed:
            reason = "identity-outside-transaction"
        else:
            continue
        blocker_count += 1
        if len(blockers) < 100:
            blockers.append(
                {
                    "node_id": node_id,
                    "reason": reason,
                    "artifact": item.artifact_sha256,
                    "protocol": item.agent_protocol_version,
                }
            )
    try:
        deployment = context.store.get_fleet_deployment(expected["deployment_id"])
    except NotFoundError:
        deployment_value = None
    else:
        deployment_value = deployment.model_dump(mode="json")
    print(
        json.dumps(
            {
                "agent_blocker_count": blocker_count,
                "agent_blockers": blockers,
                "deployment": deployment_value,
            },
            sort_keys=True,
        )
    )


main()
