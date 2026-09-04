"""Count how many of a cluster's live agents match a full desired identity.

Request on stdin::

    {"cluster_id": "...", "node_names": [...], "artifact": ..., "config": ...,
     "profile": ..., "bundle": ..., "template": ..., "protocol": ...,
     "version": ..., "compatibility": ..., "policy": ..., "key_version": ...}

Response: one JSON object on stdout, ``{"active": N, "aligned": M}``.

The engine, not this probe, decides convergence: it requires both counts to
equal the node count, so "aligned but one node missing" cannot pass.

Every identity field except ``artifact``, ``config`` and ``profile`` is
optional, and a null there means "do not compare". That is what lets the same
probe serve a release that pins the full identity and a rollback to a release
that never recorded some of these fields -- comparing against a null would fail
forever instead.

``node_names`` empty means the whole cluster.
"""

import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext


def main() -> None:
    expected = json.load(sys.stdin)
    expected_nodes = set(expected["node_names"])
    now = datetime.now(timezone.utc)
    agents = [
        item
        for item in ApplicationContext.from_environment().store.list_agents(
            expected["cluster_id"]
        )
        if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
        and item.lease_expires_at is not None
        and item.lease_expires_at > now
        and (not expected_nodes or item.node_id in expected_nodes)
    ]
    aligned = [
        item
        for item in agents
        if item.artifact_sha256 == expected["artifact"]
        and (
            expected["protocol"] is None
            or item.agent_protocol_version == expected["protocol"]
        )
        and (expected["version"] is None or item.agent_version == expected["version"])
        and (
            expected["compatibility"] is None
            or (item.compatibility_digest or item.artifact_sha256)
            == expected["compatibility"]
        )
        and (expected["policy"] is None or item.policy_version == expected["policy"])
        and item.config_digest == expected["config"]
        and item.runtime_profile_version == expected["profile"]
        and (
            expected["key_version"] is None
            or item.node_action_key_version == expected["key_version"]
        )
        and (
            expected["bundle"] is None
            or item.installer_bundle_sha256 == expected["bundle"]
        )
        and (
            expected["template"] is None
            or item.installer_template_sha256 == expected["template"]
        )
    ]
    print(json.dumps({"active": len(agents), "aligned": len(aligned)}))


main()
