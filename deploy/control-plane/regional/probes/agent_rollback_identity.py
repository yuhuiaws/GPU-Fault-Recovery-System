"""Verify every live agent is back on its cluster's pre-release identity.

Request on stdin: ``{"clusters": {"<cluster>": {"protocol": ..., ...}}}``.
Response: ``{"active": N, "mismatches": [[node_id, field], ...]}`` on stdout,
then **exit 1** if there are no live agents or any mismatch.

The non-zero exit is the point: this is a rollback gate, not a report, so a
partially-restored fleet has to fail the command rather than land in a dict the
caller might not check. ``not agents`` fails too -- a rollback that verified
nothing is not a rollback that verified success.

``bundle`` and ``template`` are compared only when the snapshot recorded them:
a rollback to a release that predates the installer identity fields has nothing
to compare against, and demanding equality with null would make that rollback
impossible.
"""

import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext


def main() -> None:
    expected = json.load(sys.stdin)
    now = datetime.now(timezone.utc)
    agents = [
        item
        for item in ApplicationContext.from_environment().store.list_agents()
        if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
        and item.lease_expires_at is not None
        and item.lease_expires_at > now
    ]
    mismatches: list[list[str]] = []
    for item in agents:
        cluster = expected["clusters"].get(item.cluster_id) or {}
        if item.agent_protocol_version != cluster.get("protocol"):
            mismatches.append([item.node_id, "protocol"])
        if item.agent_version != cluster.get("version"):
            mismatches.append([item.node_id, "version"])
        if item.artifact_sha256 != cluster.get("artifact"):
            mismatches.append([item.node_id, "artifact"])
        if (item.compatibility_digest or item.artifact_sha256) != cluster.get(
            "compatibility"
        ):
            mismatches.append([item.node_id, "compatibility"])
        if item.policy_version != cluster.get("policy"):
            mismatches.append([item.node_id, "policy"])
        if item.config_digest != cluster.get("config"):
            mismatches.append([item.node_id, "config"])
        if item.node_action_key_version != cluster.get("key_version"):
            mismatches.append([item.node_id, "key_version"])
        if item.runtime_profile_version != cluster.get("profile"):
            mismatches.append([item.node_id, "profile"])
        if (
            cluster.get("bundle") is not None
            and item.installer_bundle_sha256 != cluster["bundle"]
        ):
            mismatches.append([item.node_id, "bundle"])
        if (
            cluster.get("template") is not None
            and item.installer_template_sha256 != cluster["template"]
        ):
            mismatches.append([item.node_id, "template"])
    print(json.dumps({"active": len(agents), "mismatches": mismatches}))
    if not agents or mismatches:
        raise SystemExit(1)


main()
