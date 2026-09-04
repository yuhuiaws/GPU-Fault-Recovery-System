"""Dump the full identity of every live agent, for the rollback snapshot.

Request: nothing on stdin.
Response: a JSON array of one object per live agent, on stdout.

This is the "previous" side of a release transaction: the engine records it
before mutating anything and requires each cluster's live agents to agree on a
single identity, so a rollback has one identity per cluster to restore to.

``getattr(..., None)`` on the installer fields is deliberate: this probe is
executed by whatever ``gpu_fault`` is *already* deployed, which may predate those
fields. A plain attribute read would make capture fail against an older Pod,
which is exactly the case a rollback has to work in.
"""

import json
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext


def main() -> None:
    now = datetime.now(timezone.utc)
    records = [
        item
        for item in ApplicationContext.from_environment().store.list_agents()
        if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
        and item.lease_expires_at is not None
        and item.lease_expires_at > now
    ]
    print(
        json.dumps(
            [
                {
                    "cluster_id": item.cluster_id,
                    "node_id": item.node_id,
                    "agent_protocol_version": item.agent_protocol_version,
                    "agent_version": item.agent_version,
                    "artifact_sha256": item.artifact_sha256,
                    "compatibility_digest": (
                        item.compatibility_digest or item.artifact_sha256
                    ),
                    "installer_bundle_sha256": getattr(
                        item, "installer_bundle_sha256", None
                    ),
                    "installer_template_sha256": getattr(
                        item, "installer_template_sha256", None
                    ),
                    "policy_version": item.policy_version,
                    "runtime_profile_version": item.runtime_profile_version,
                    "config_digest": item.config_digest,
                    "node_action_key_version": item.node_action_key_version,
                }
                for item in records
            ],
            sort_keys=True,
        )
    )


main()
