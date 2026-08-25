from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ClusterTarget, ReleaseError

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"


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
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": release.wheel_sha,
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config.agent_config_digest,
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": (
            runtime_profile_version or config.runtime_profile_version
        ),
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "3",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "2",
        "GPU_FAULT_FINALIZE_AGENT_PIN": str(finalize).lower(),
        "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": str(finalize).lower(),
    }


def render_gpu_rollout_manifests(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    runtime_profile_version: str | None = None,
) -> list[tuple[str, str]]:
    config = release.config
    profile_version = runtime_profile_version or config.runtime_profile_version
    replacements = {
        "gpu-fault-control-plane-wheel-0100": wheel_cm,
        "namespace: gpu-fault-system": f"namespace: {config.namespace}",
        DEFAULT_RUNTIME_IMAGE: release.runtime_image,
        "REPLACE_WITH_AWS_REGION": target.region,
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION": profile_version,
        "REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN": target.executor_irsa_role_arn,
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
        "GPU_FAULT_WHEEL_CONFIG_MAP": wheel_cm,
        "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        "GPU_FAULT_RUNTIME_PROFILE": (
            runtime_profile_version or config.runtime_profile_version
        ),
    }
    if target.fleet_master_file:
        environment["GPU_FAULT_FLEET_MASTER_FILE"] = target.fleet_master_file
        environment["GPU_FAULT_CONTROL_PLANE_KUBECONFIG"] = config.cpu_kubeconfig
    return environment
