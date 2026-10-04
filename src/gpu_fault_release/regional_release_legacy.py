from __future__ import annotations

from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError

AGENT_IDENTITY_FIELDS = (
    "agent_protocol_version",
    "agent_version",
    "artifact_sha256",
    "compatibility_digest",
    "installer_bundle_sha256",
    "installer_template_sha256",
    "policy_version",
    "runtime_profile_version",
    "config_digest",
    "node_action_key_version",
)
LEGACY_OPTIONAL_IDENTITY_FIELDS = (
    "installer_bundle_sha256",
    "installer_template_sha256",
)
LEGACY_NODE_ANNOTATION_FIELDS = (
    "gpu-fault.io/installer-artifact-sha256",
    "gpu-fault.io/installer-config-digest",
    "gpu-fault.io/installer-node-uid",
    "gpu-fault.io/installer-state",
)
FLEET_REQUEST_IDENTITY_FIELDS = {
    "desired_agent_protocol_version": "agent_protocol_version",
    "desired_agent_version": "agent_version",
    "desired_policy_version": "policy_version",
}
ROLLBACK_CONTROLLER_CONFIG_FIELDS = {
    "GPU_FAULT_REQUIRED_AGENT_VERSION": "agent_version",
    "GPU_FAULT_REQUIRED_POLICY_VERSION": "policy_version",
    "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "runtime_profile_version",
}


def apply_fleet_request_identity(
    request: dict[str, Any],
    identity: dict[str, Any] | None,
) -> None:
    if not identity:
        return
    for request_field, identity_field in FLEET_REQUEST_IDENTITY_FIELDS.items():
        value = identity.get(identity_field)
        if value is not None:
            request[request_field] = value


def rollback_controller_config(
    identities: dict[str, dict[str, Any]],
    *,
    site_runtime_profile_version: str | None = None,
    cluster_runtime_profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """The site pins the rolled-back control plane is patched with.

    Agent version and policy version must be one value fleet-wide. The Runtime
    Profile version is the site's: a cluster running its own profile (named in
    ``cluster_runtime_profiles``, the previous release's per-cluster overrides)
    reports that profile's version instead, and keeps it through the restored
    registry registration, not through this env. Without the overrides map the
    historical single-value rule applies.
    """

    if not identities:
        raise ReleaseError("previous Agent identity snapshot is empty")
    result: dict[str, str] = {}
    for config_field, identity_field in ROLLBACK_CONTROLLER_CONFIG_FIELDS.items():
        if (
            identity_field == "runtime_profile_version"
            and cluster_runtime_profiles is not None
        ):
            result[config_field] = _site_runtime_profile_version(
                identities, site_runtime_profile_version, cluster_runtime_profiles
            )
            continue
        values = {
            str(identity.get(identity_field) or "") for identity in identities.values()
        }
        if len(values) != 1 or not next(iter(values)):
            raise ReleaseError(
                f"previous Agent {identity_field} differs across clusters"
            )
        result[config_field] = values.pop()
    return result


def _site_runtime_profile_version(
    identities: dict[str, dict[str, Any]],
    site_version: str | None,
    overrides: dict[str, dict[str, Any]],
) -> str:
    site = str(site_version or "")
    if not site:
        shared = {
            str(identity.get("runtime_profile_version") or "")
            for cluster_id, identity in identities.items()
            if cluster_id not in overrides
        }
        if len(shared) != 1 or not next(iter(shared)):
            raise ReleaseError(
                "previous Agent runtime_profile_version differs across the clusters "
                "on the site profile: " + ", ".join(sorted(shared))
            )
        site = shared.pop()
    for cluster_id, identity in sorted(identities.items()):
        reported = str(identity.get("runtime_profile_version") or "")
        expected = str((overrides.get(cluster_id) or {}).get("version") or site)
        if reported != expected:
            raise ReleaseError(
                f"{cluster_id} previous Agent runtime_profile_version {reported!r} "
                f"is neither the site's {site!r} nor the cluster's own "
                f"{expected!r}"
            )
    return site


def validate_rollback_agent_identity(
    cluster_id: str,
    identity: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact: str,
    compatibility: str,
    config_digest: str,
    runtime_profile_version: str,
) -> bool:
    expected = {
        "agent_protocol_version": int(
            metadata.get("required-agent-protocol-version") or 0
        ),
        "artifact_sha256": artifact,
        "compatibility_digest": compatibility,
        "config_digest": config_digest,
        "runtime_profile_version": runtime_profile_version,
        "node_action_key_version": int(
            metadata.get("required-node-action-key-version") or 0
        ),
    }
    mismatches = [
        field for field, value in expected.items() if identity.get(field) != value
    ]
    if mismatches:
        raise ReleaseError(
            f"{cluster_id} previous Agent identity disagrees with pins: "
            + ", ".join(mismatches)
        )
    optional = tuple(identity.get(field) for field in LEGACY_OPTIONAL_IDENTITY_FIELDS)
    if (optional[0] is None) != (optional[1] is None):
        raise ReleaseError(
            f"{cluster_id} previous Agent optional identity is incomplete"
        )
    if not identity.get("agent_version") or not identity.get("policy_version"):
        raise ReleaseError(f"{cluster_id} previous Agent version or policy is missing")
    if not identity.get("node_ids"):
        raise ReleaseError(f"{cluster_id} previous Agent node set is empty")
    return all(value is None for value in optional)
