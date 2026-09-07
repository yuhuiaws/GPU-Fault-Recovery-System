from __future__ import annotations

import os
from typing import Any

from gpu_fault_release.regional_notifications import notification_digest
from gpu_fault_release.regional_release_config import ReleaseConfig, ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent
from gpu_fault_release.regional_release_progress import RollbackCompensationPlan
from gpu_fault_release.regional_release_rendering import (
    admin_config_renderer_environment,
)


def rollback_target_arguments(
    release: Any,
    *,
    previous: dict[str, Any],
    metadata: dict[str, Any],
    artifact: str,
    config_digest: str,
    profile: str,
    executor_artifact: str,
    executor_compatibility: str,
    runtime_image: str,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "config_digest": config_digest,
        "runtime_profile_version": profile,
        "executor_artifact": executor_artifact,
        "executor_compatibility": executor_compatibility,
        "node_compatibility": (
            metadata.get("required-agent-compatibility-digest") or artifact
        ),
        "runtime_image": runtime_image,
        "node_installer_image": (
            previous.get("node_installer_image") or release.node_installer_image
        ),
    }


def rollback_identity_context(
    release: Any,
    previous: dict[str, Any],
    compensation: RollbackCompensationPlan,
) -> tuple[dict[str, Any], str, str, str, str, str, str, str]:
    metadata = previous.get("metadata") or {}
    agent_identities = previous.get("agent_identities") or {}
    if compensation.needs_controller:
        missing = sorted(
            target.cluster_id
            for target in release.config.clusters
            if ReleaseComponent.AGENT in compensation.for_cluster(target.cluster_id)
            and target.cluster_id not in agent_identities
        )
        if missing:
            raise ReleaseError(
                "previous Agent identities are missing for: " + ", ".join(missing)
            )
    cpu_wheel = str(previous.get("cpu_wheel") or "")
    artifact = str(metadata.get("required-agent-artifact-sha256") or "")
    config_digest = str(metadata.get("required-agent-config-digest") or "")
    node_runtime_required = compensation.restores_cpu or any(
        components.intersection({ReleaseComponent.RECONCILER, ReleaseComponent.AGENT})
        for components in compensation.cluster_components.values()
    )
    if (
        compensation.restores_cpu
        and not cpu_wheel
        or node_runtime_required
        and not all((artifact, config_digest))
    ):
        raise ReleaseError("previous release pins are incomplete")
    profile = str(previous.get("runtime_profile_version") or "hyperpod-v1")
    executor_artifact = str(
        metadata.get("required-regional-executor-artifact-sha256") or ""
    )
    executor_compatibility = str(
        metadata.get("required-regional-executor-compatibility-digest")
        or executor_artifact
    )
    runtime_image = str(previous.get("runtime_image") or release.runtime_image)
    return (
        metadata,
        cpu_wheel,
        artifact,
        config_digest,
        profile,
        executor_artifact,
        executor_compatibility,
        runtime_image,
    )


def build_rollback_environment(
    *,
    rollback_config: ReleaseConfig,
    metadata: dict[str, str],
    cpu_wheel: str,
    cpu_sha: str,
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
    runtime_image: str,
    preserve_role_config_maps: bool = False,
) -> dict[str, str]:
    legacy_component_pins = not any(
        metadata.get(name)
        for name in (
            "required-agent-compatibility-digest",
            "required-regional-executor-artifact-sha256",
            "required-regional-executor-compatibility-digest",
        )
    )
    return {
        **os.environ,
        **admin_config_renderer_environment(rollback_config.admin_config),
        "KUBECONFIG": rollback_config.cpu_kubeconfig,
        "GPU_FAULT_AWS_REGION": rollback_config.aws_region,
        "GPU_FAULT_NAMESPACE": rollback_config.namespace,
        "GPU_FAULT_WHEEL_CONFIGMAP": cpu_wheel,
        "GPU_FAULT_WHEEL_SHA256": cpu_sha,
        "GPU_FAULT_RUNTIME_IMAGE": runtime_image,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": artifact,
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": (
            metadata.get("required-agent-compatibility-digest") or artifact
        ),
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config_digest,
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": (runtime_profile_version),
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": metadata.get(
            "required-agent-protocol-version", "3"
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": metadata.get(
            "required-regional-executor-protocol-version", "2"
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": metadata.get(
            "required-regional-executor-artifact-sha256", ""
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
            metadata.get("required-regional-executor-compatibility-digest")
            or metadata.get(
                "required-regional-executor-artifact-sha256",
                "",
            )
        ),
        "GPU_FAULT_ALLOW_EMAIL": str(rollback_config.notifications.allow_email).lower(),
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": str(
            rollback_config.notifications.acknowledge_external_alert_channel
        ).lower(),
        "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": notification_digest(
            rollback_config.notifications
        ),
        "GPU_FAULT_ADMIN_CONFIG_SHA256": rollback_config.admin_config.sha256(),
        "GPU_FAULT_ADMIN_CONFIG_INGRESS_SHA256": (
            rollback_config.admin_config.role_sha256()["ingress"]
        ),
        "GPU_FAULT_ADMIN_CONFIG_WORKER_SHA256": (
            rollback_config.admin_config.role_sha256()["worker"]
        ),
        "GPU_FAULT_ADMIN_CONFIG_SPOOL_SHA256": (
            rollback_config.admin_config.role_sha256()["spool"]
        ),
        "GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS": "spool,worker,ingress",
        "GPU_FAULT_LEGACY_COMPONENT_PINS": str(legacy_component_pins).lower(),
        "GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS": str(preserve_role_config_maps).lower(),
        "GPU_FAULT_FORCE_ROLE_RESTART": "true",
        "GPU_FAULT_FINALIZE_AGENT_PIN": "true",
        "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": "true",
    }
