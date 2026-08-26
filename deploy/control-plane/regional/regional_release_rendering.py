from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
from regional_notifications import notification_digest
from regional_release_config import ClusterTarget, ReleaseError

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"
DEFAULT_DCGM_EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04"


def build_cpu_apply_environment(
    release: Any,
    *,
    finalize: bool,
    runtime_profile_version: str | None = None,
) -> dict[str, str]:
    config = release.config
    return {
        **os.environ,
        "KUBECONFIG": config.cpu_kubeconfig,
        "GPU_FAULT_AWS_REGION": config.aws_region,
        "GPU_FAULT_NAMESPACE": config.namespace,
        "GPU_FAULT_WHEEL_CONFIGMAP": release.wheel_cm,
        "GPU_FAULT_WHEEL_SHA256": release.wheel_sha,
        "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": release.node_wheel_sha,
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": (
            config.component_digests.get("node_runtime") or release.node_wheel_sha
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": (
            release.executor_wheel_sha
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
            config.component_digests.get("executor") or release.executor_wheel_sha
        ),
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config.agent_config_digest,
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": (
            runtime_profile_version or config.runtime_profile_version
        ),
        "GPU_FAULT_ALLOW_EMAIL": str(config.notifications.allow_email).lower(),
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": str(
            config.notifications.acknowledge_external_alert_channel
        ).lower(),
        "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": notification_digest(
            config.notifications
        ),
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": str(config.agent_protocol_version),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": str(
            config.executor_protocol_version
        ),
        "GPU_FAULT_FINALIZE_AGENT_PIN": str(finalize).lower(),
        "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": str(finalize).lower(),
    }


def render_gpu_rollout_manifests(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
) -> list[tuple[str, str]]:
    config = release.config
    profile_version = runtime_profile_version or config.runtime_profile_version
    replacements = {
        "gpu-fault-executor-wheel-0100": wheel_cm,
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl": (
            executor_wheel_filename or release.config.executor_wheel.name
        ),
        "namespace: gpu-fault-system": f"namespace: {config.namespace}",
        DEFAULT_RUNTIME_IMAGE: release.runtime_image,
        "REPLACE_WITH_AWS_REGION": target.region,
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION": profile_version,
        "REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN": target.executor_irsa_role_arn,
        "REPLACE_WITH_EXECUTOR_ARTIFACT_SHA256": (release.executor_wheel_sha),
        "REPLACE_WITH_EXECUTOR_COMPATIBILITY_DIGEST": (
            config.component_digests.get("executor") or release.executor_wheel_sha
        ),
    }
    rendered = []
    for filename, deployment in inventory.GPU_ROLLOUT_DEPLOYMENTS:
        text = (ROOT / "deploy/dataplane" / filename).read_text(encoding="utf-8")
        for source, destination in replacements.items():
            text = text.replace(source, destination)
        if "REPLACE_WITH" in text:
            raise ReleaseError(f"{filename} still contains a placeholder")
        rendered.append((deployment, release._stamp_gpu_deployments(text)))
    return rendered


def build_reconciler_environment(
    release: Any,
    target: ClusterTarget,
    *,
    wheel_cm: str,
    bundle_cm: str,
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
) -> dict[str, str]:
    config = release.config
    environment = {
        **os.environ,
        "GPU_FAULT_KUBECTL_CONTEXT": target.context,
        "GPU_FAULT_NAMESPACE": config.namespace,
        "GPU_FAULT_CLUSTER_ID": target.cluster_id,
        "GPU_FAULT_HYPERPOD_CLUSTER": target.hyperpod_cluster_name,
        "GPU_FAULT_INSTALLER_CONFIG_MAP": bundle_cm,
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": config_digest,
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": artifact_sha,
        "GPU_FAULT_NODE_COMPATIBILITY_DIGEST": (
            config.component_digests.get("node_runtime") or release.node_wheel_sha
        ),
        "GPU_FAULT_WHEEL_CONFIG_MAP": wheel_cm,
        "GPU_FAULT_EXECUTOR_WHEEL_FILENAME": (
            executor_wheel_filename or config.executor_wheel.name
        ),
        "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        "GPU_FAULT_RUNTIME_PROFILE": (
            runtime_profile_version or config.runtime_profile_version
        ),
    }
    if target.fleet_master_file:
        environment["GPU_FAULT_FLEET_MASTER_FILE"] = target.fleet_master_file
        environment["GPU_FAULT_CONTROL_PLANE_KUBECONFIG"] = config.cpu_kubeconfig
    return environment
