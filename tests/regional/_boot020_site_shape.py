"""What BOOT-020's per-cluster verdicts read, for the runner's test doubles.

``boot020_site_topology`` judges each stage cluster by cluster: the site shape
node safety reports, the Agent identities every full snapshot captures, the
Fleet waves of the Agent and FULL rollbacks, the engine's zero-node records
around each deploy and the FULL candidate's Runtime Profile versions. These
helpers give a fake backend those observations in the live layout.
"""

from __future__ import annotations

from typing import Any

#: The FULL candidate's suffixed profile of a cluster with nodes, and its pin.
SITE_PROFILE = "profile-v1"
FULL_PROFILE = "profile-v1-boot020"
SITE_DIGEST = "c" * 64
FULL_DIGEST = "d" * 64
#: Agent identity fields per release the fakes roll: the site's own release,
#: candidate C (node runtime) and candidate D (node runtime and profile).
RELEASES = {
    "site": ("node-v1", "bundle-v1", SITE_PROFILE, SITE_DIGEST),
    "agent": ("node-v2", "bundle-v2", SITE_PROFILE, SITE_DIGEST),
    "full": ("node-v3", "bundle-v3", FULL_PROFILE, FULL_DIGEST),
}
PARKED_PROOF = {
    "state": "parked",
    "provider": "eks-managed",
    "reason": (
        "every managed node group the node selector covers is ACTIVE at "
        "desiredSize 0 and no Karpenter NodePool can launch a node it selects"
    ),
    "groups": [
        {
            "name": "gpu-group",
            "status": "ACTIVE",
            "desired": 0,
            "min": 0,
            "max": 2,
            "instance_types": ["p5en.48xlarge"],
            "labels": {},
        }
    ],
}


def agent_identity(nodes: list[str], release: str = "site") -> dict[str, Any]:
    """One cluster's Agent identity, the way the capture records it."""

    artifact, bundle, profile, digest = RELEASES[release]
    return {
        "agent_protocol_version": 1,
        "agent_version": "unit",
        "artifact_sha256": artifact,
        "compatibility_digest": artifact,
        "installer_bundle_sha256": bundle,
        "installer_template_sha256": "t" * 64,
        "policy_version": 1,
        "runtime_profile_version": profile,
        "config_digest": digest,
        "node_action_key_version": 1,
        "node_ids": sorted(nodes),
    }


def zero_node_identity(proof: dict[str, Any] | None = None) -> dict[str, Any]:
    """The explicit entry of a cluster without nodes (``zero_node_identity``)."""

    return {"node_ids": [], "zero_nodes": dict(proof or PARKED_PROOF)}


def node_shape(nodes: list[str]) -> dict[str, Any]:
    return {"nodes": sorted(nodes)}


def zero_node_shape(proof: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"zero_nodes": dict(proof or PARKED_PROOF)}


def waves(nodes: list[str], start: float = 1_000.0) -> dict[str, Any]:
    """One completed single-node wave per node, its five events in order."""

    result: dict[str, Any] = {}
    for index, node in enumerate(sorted(nodes)):
        base = start + 60.0 * index
        result[node] = {
            "nodes": [node],
            "status": "COMPLETED",
            "started_at_epoch": base,
            "safety_started_at_epoch": base,
            "safety_completed_at_epoch": base + 9.0,
            "reconciler_applied_at_epoch": base + 20.0,
            "agents_converged_at_epoch": base + 50.0,
            "completed_at_epoch": base + 55.0,
            "duration_seconds": 55.0,
        }
    return result


def cluster_timing(
    components: list[str], node_waves: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A cluster's completed rollback timing entry, with its waves if any."""

    entry: dict[str, Any] = {
        "status": "COMPLETED",
        "started_at_epoch": 900.0,
        "completed_at_epoch": 1_400.0,
        "details": {"components": sorted(components)},
    }
    if node_waves:
        entry["waves"] = node_waves
    return entry


def profile_versions(cluster_ids: list[str]) -> dict[str, Any]:
    """``profile_versions("full")``: every cluster's FULL profile is suffixed."""

    return {
        "clusters": {
            cluster_id: {
                "version": FULL_PROFILE,
                "agent_config_digest": FULL_DIGEST,
                "own_profile": False,
            }
            for cluster_id in cluster_ids
        },
        "site_clusters": {
            cluster_id: {
                "version": SITE_PROFILE,
                "agent_config_digest": SITE_DIGEST,
                "own_profile": False,
            }
            for cluster_id in cluster_ids
        },
        "registered_versions": [FULL_PROFILE],
        "permanent": True,
    }
