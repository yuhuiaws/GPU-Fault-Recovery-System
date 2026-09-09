from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_notifications import notification_digest
from gpu_fault_release.regional_release_config import ReleaseConfig, ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent
from gpu_fault_release.regional_release_progress import RollbackCompensationPlan
from gpu_fault_release.regional_release_rendering import (
    admin_config_renderer_environment,
)
from gpu_fault_release.regional_release_state import (
    SENSITIVE_CONFIG_KEY,
    require_digest_pinned_image,
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
        "runtime_image": require_digest_pinned_image(
            "rollback target runtime", runtime_image
        ),
        "node_installer_image": require_digest_pinned_image(
            "rollback target Node Installer",
            previous.get("node_installer_image") or release.node_installer_image,
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
    runtime_image = require_digest_pinned_image(
        "rollback runtime", previous.get("runtime_image") or release.runtime_image
    )
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


# The renderer reads the previous container environment from this file on
# rollback; see CONTAINER_ENV_FILE_VARIABLE in render_control_plane_role_split.
CONTAINER_ENV_FILE_VARIABLE = "GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE"
CONTAINER_ENV_FILE_NAME = "previous-container-env.json"


def _invalid_container_env(detail: str) -> ReleaseError:
    return ReleaseError(
        f"previous CPU role container environment snapshot is invalid: {detail}"
    )


def previous_container_env_snapshot(
    snapshot: object,
) -> dict[str, dict[str, dict[str, list[Any]]]] | None:
    """The validated ``cpu_role_container_env`` snapshot, or None to fall back.

    None means the transaction was opened before the capture existed; the
    caller renders the previous Deployments from the current template as it
    always did and records that it did. Anything present must be exactly the
    shape `cpu_role_container_env` writes -- every CPU role Deployment, each
    container carrying its ``env`` and ``envFrom`` lists, no literal under a
    sensitive-looking name -- because the renderer applies it verbatim and a
    half-covered environment is what this snapshot exists to prevent.
    """

    if snapshot is None or snapshot == {}:
        return None
    if not isinstance(snapshot, dict):
        raise _invalid_container_env("expected a Deployment mapping")
    expected = set(inventory.CPU_RUNTIME_DEPLOYMENTS)
    names = {name for name in snapshot if isinstance(name, str)}
    if len(names) != len(snapshot) or names != expected:
        raise _invalid_container_env(
            "Deployments do not match the CPU role inventory: "
            + ", ".join(sorted(str(name) for name in snapshot))
        )
    validated: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        containers = snapshot[deployment]
        if not isinstance(containers, dict) or not containers:
            raise _invalid_container_env(f"{deployment} lists no containers")
        validated[deployment] = {}
        for container, spec in containers.items():
            if (
                not isinstance(container, str)
                or not isinstance(spec, dict)
                or set(spec) != {"env", "envFrom"}
                or not isinstance(spec["env"], list)
                or not isinstance(spec["envFrom"], list)
            ):
                raise _invalid_container_env(
                    f"{deployment}/{container} must carry exactly env and envFrom lists"
                )
            for item in spec["env"]:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise _invalid_container_env(
                        f"{deployment}/{container} has an env entry without a name"
                    )
                if ("value" in item) == ("valueFrom" in item):
                    raise _invalid_container_env(
                        f"{deployment}/{container} env {item['name']} must define "
                        "exactly one of value or valueFrom"
                    )
                if "value" in item and SENSITIVE_CONFIG_KEY.search(item["name"]):
                    raise _invalid_container_env(
                        f"{deployment}/{container} sensitive env {item['name']} "
                        "carries a literal value"
                    )
            if not all(isinstance(source, dict) for source in spec["envFrom"]):
                raise _invalid_container_env(
                    f"{deployment}/{container} has a malformed envFrom entry"
                )
            validated[deployment][container] = {
                "env": copy.deepcopy(spec["env"]),
                "envFrom": copy.deepcopy(spec["envFrom"]),
            }
    return validated


def write_rollback_container_env(
    snapshot: dict[str, dict[str, dict[str, list[Any]]]],
    directory: Path,
) -> str:
    """Write the validated snapshot where the renderer will read it from."""

    path = directory / CONTAINER_ENV_FILE_NAME
    path.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")
    return str(path)


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
    previous_container_env_file: str | None = None,
) -> dict[str, str]:
    legacy_component_pins = not any(
        metadata.get(name)
        for name in (
            "required-agent-compatibility-digest",
            "required-regional-executor-artifact-sha256",
            "required-regional-executor-compatibility-digest",
        )
    )
    environment = {
        **os.environ,
        **admin_config_renderer_environment(rollback_config.admin_config),
        # A rolled-back role without its alert channel would fail the
        # startup guard, so the channel travels with the rollback too.
        **rollback_config.notification_environment(),
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
    # Set only from the validated snapshot; never inherited from the caller's
    # shell, so a stray variable cannot turn a forward render into a rollback.
    environment.pop(CONTAINER_ENV_FILE_VARIABLE, None)
    if previous_container_env_file:
        environment[CONTAINER_ENV_FILE_VARIABLE] = previous_container_env_file
    return environment
