from __future__ import annotations

from typing import Any

import regional_deployment_inventory as inventory
from gpu_fault.regional_compatibility import (
    RegionalExecutorCompatibilityPolicy,
)
from regional_release_config import ClusterTarget, ReleaseError
from regional_release_diff import (
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    build_execution_plan,
)
from regional_release_rendering import render_gpu_rollout_manifests


FAST_ROLLOUT_TIMEOUT = "5m"


def agents_converged(
    items: list[dict[str, Any]],
    target: ClusterTarget,
    artifact_sha: str,
    *,
    bundle_sha: str | None = None,
    template_sha: str | None = None,
    config_digest: str | None = None,
    require_node_uid: bool = False,
    node_names: frozenset[str] | None = None,
) -> bool:
    nodes = [
        item
        for item in items
        if (
            item.get("metadata", {})
            .get("labels", {})
            .get("sagemaker.amazonaws.com/cluster-name")
            == target.hyperpod_cluster_name
            and (
                node_names is None
                or str(item.get("metadata", {}).get("name") or "") in node_names
            )
        )
    ]
    aligned = [
        item
        for item in nodes
        if (
            item.get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.io/installer-state")
            == "Succeeded"
            and item.get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.io/installer-artifact-sha256")
            == artifact_sha
            and (
                config_digest is None
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-config-digest")
                == config_digest
            )
            and (
                not require_node_uid
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-node-uid")
                == item.get("metadata", {}).get("uid")
            )
            and (
                bundle_sha is None
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-bundle-sha256")
                == bundle_sha
            )
            and (
                template_sha is None
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-template-sha256")
                == template_sha
            )
        )
    ]
    return bool(nodes) and len(aligned) == len(nodes)


def executor_pin_rejection(
    metadata: dict[str, str],
    *,
    protocol_version: int,
    artifact_sha: str,
    compatibility_digest: str,
) -> str | None:
    policy = RegionalExecutorCompatibilityPolicy.from_mapping(
        {
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": (
                metadata.get(
                    "required-regional-executor-protocol-version",
                    str(protocol_version),
                )
            ),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": (
                metadata.get(
                    "compatible-regional-executor-protocol-versions",
                    "",
                )
            ),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": (
                metadata.get(
                    "required-regional-executor-artifact-sha256",
                    "",
                )
            ),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S": (
                metadata.get(
                    "compatible-regional-executor-artifact-sha256s",
                    "",
                )
            ),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
                metadata.get(
                    "required-regional-executor-compatibility-digest",
                    "",
                )
            ),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS": (
                metadata.get(
                    "compatible-regional-executor-compatibility-digests",
                    "",
                )
            ),
        }
    )
    return policy.rejection_reason(
        protocol_version,
        artifact_sha,
        compatibility_digest,
    )


def require_executor_pin(
    release: Any,
    *,
    artifact_sha: str,
    compatibility_digest: str,
) -> None:
    metadata = release._config_map_data("gpu-fault-release-metadata")
    try:
        reason = executor_pin_rejection(
            metadata,
            protocol_version=release.config.executor_protocol_version,
            artifact_sha=artifact_sha,
            compatibility_digest=compatibility_digest,
        )
    except ValueError as exc:
        raise ReleaseError(f"invalid regional executor pin metadata: {exc}") from exc
    if reason:
        raise ReleaseError(f"executor pin preflight rejected rollout: {reason}")


def apply_gpu_deployments(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    deployment_names: frozenset[str] | None = None,
    runtime_image: str | None = None,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
) -> None:
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or release.config.component_digests.get("executor")
        or artifact_sha
    )
    require_executor_pin(
        release,
        artifact_sha=artifact_sha,
        compatibility_digest=compatibility_digest,
    )
    for deployment, text in render_gpu_rollout_manifests(
        release,
        target,
        wheel_cm,
        deployment_names=deployment_names,
        runtime_image=runtime_image,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        executor_artifact_sha=artifact_sha,
        executor_compatibility_digest=compatibility_digest,
    ):
        release.runner.run(
            release._gpu(target, "apply", "-f", "-"),
            input_text=text,
        )
        release.runner.run(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "rollout",
                "status",
                f"deployment/{deployment}",
                f"--timeout={FAST_ROLLOUT_TIMEOUT}",
            )
        )


def upgrade_gpu_target(
    release: Any,
    target: ClusterTarget,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan | None = None,
) -> None:
    active_plan = plan or build_execution_plan(diff)
    if active_plan.has(ReleaseComponent.ENDPOINT):
        release._ensure_connection_secret(target)
        release._verify_gpu_control_plane_endpoint(target)
    if active_plan.has(ReleaseComponent.DCGM):
        release._apply_gpu_dcgm_exporter(target)
    deployment_names = {
        deployment
        for component, deployment in (
            (
                ReleaseComponent.EXECUTOR,
                inventory.GPU_EXECUTOR_DEPLOYMENT,
            ),
            (
                ReleaseComponent.WATCHER,
                inventory.GPU_WATCHER_DEPLOYMENT,
            ),
            (
                ReleaseComponent.COLLECTOR,
                inventory.GPU_COLLECTOR_DEPLOYMENT,
            ),
        )
        if active_plan.has(component)
    }
    if deployment_names:
        release._apply_gpu_deployments(
            target,
            release.executor_wheel_cm,
            deployment_names=frozenset(deployment_names),
        )
    if active_plan.has(ReleaseComponent.AGENT):
        release._roll_node_runtime(
            target,
            phase="upgrade",
            wheel_cm=release.executor_wheel_cm,
            bundle_cm=release.bundle_cm,
            artifact_sha=release.node_wheel_sha,
            config_digest=release.config.agent_config_digest,
        )
    elif active_plan.has(ReleaseComponent.RECONCILER):
        release._deploy_reconciler(
            target,
            wheel_cm=release.executor_wheel_cm,
            bundle_cm=release.bundle_cm,
            artifact_sha=release.node_wheel_sha,
            config_digest=release.config.agent_config_digest,
        )


def join_target(release: Any, cluster_id: str) -> ClusterTarget:
    target = release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    return target
