"""Report the lifecycle and lease facts of named Agent records, in any state.

Request on stdin::

    {"cluster_id": "...", "node_ids": ["<node>", ...]}

Response: one JSON object on stdout::

    {"read_at": "<iso>",
     "agents": {"<node>": {"lifecycle_state": "ACTIVE" | "DRAINING" | "REVOKED",
                           "lease_expires_at": "<iso>" | null,
                           "last_seen_at": "<iso>" | null,
                           "lease_live": true | false,
                           "node_instance_id": "<id>" | null}}}

A node with no record is absent from ``agents``. ``lease_live`` is judged on
the Pod's clock, the clock the lease was issued against.

``active_agent_identities`` lists live Agents only. The engine asks this probe
about a node that is *missing from Kubernetes*, to tell a retired or silent
record (the node left; its lease lapsed or an administrator revoked it) from
one still heartbeating (the node did not leave, whatever the node list says).
``node_instance_id`` is the EC2 instance id the installer recorded on an
eks-managed node, the only provider handle such a node has once it is gone.
``getattr`` on it is deliberate: the Pod runs whatever ``gpu_fault`` is
deployed, which may predate the field.
"""

import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext


def main() -> None:
    request = json.load(sys.stdin)
    wanted = {str(node_id) for node_id in request["node_ids"]}
    now = datetime.now(timezone.utc)
    agents = {}
    for item in ApplicationContext.from_environment().store.list_agents(
        request["cluster_id"]
    ):
        if item.node_id not in wanted:
            continue
        lease_expires_at = item.lease_expires_at
        last_seen_at = getattr(item, "last_seen_at", None)
        agents[item.node_id] = {
            "lifecycle_state": getattr(
                item.lifecycle_state, "value", item.lifecycle_state
            ),
            "lease_expires_at": (
                lease_expires_at.isoformat() if lease_expires_at is not None else None
            ),
            "last_seen_at": (
                last_seen_at.isoformat() if last_seen_at is not None else None
            ),
            "lease_live": lease_expires_at is not None and lease_expires_at > now,
            "node_instance_id": getattr(item, "node_instance_id", None),
        }
    print(json.dumps({"read_at": now.isoformat(), "agents": agents}, sort_keys=True))


main()
