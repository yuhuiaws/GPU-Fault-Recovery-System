from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
import yaml  # type: ignore[import-untyped,unused-ignore]
from gpu_fault_release.regional_notifications import notification_digest
from gpu_fault_release.regional_release_config import (
    ClusterTarget,
    ReleaseError,
    render_nlb_manifest,
)

from gpu_fault.admin.config import AdminConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"
DEFAULT_DCGM_EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04"


def _number_text(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def admin_config_renderer_environment(
    admin_config: AdminConfig,
) -> dict[str, str]:
    capacity = admin_config.capacity
    remediation = capacity.remediation
    spool = capacity.telemetry_spool
    processor = admin_config.processor
    workflow = admin_config.workflow
    notification = admin_config.notification_delivery
    evidence = admin_config.evidence
    return {
        "GPU_FAULT_CONTROL_WORKER_REPLICAS": str(capacity.control_worker_replicas),
        # The declared topology travels with the release so the live state can
        # be read back on rollback and the runtime can check its own reserve.
        "GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT": str(
            capacity.largest_cluster_node_count
        ),
        "GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT": str(capacity.managed_node_count),
        "GPU_FAULT_TELEMETRY_SPOOL": str(spool.enabled).lower(),
        "GPU_FAULT_TELEMETRY_SPOOL_REPLICAS": str(spool.replicas),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": str(remediation.max_active_region),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": str(
            remediation.max_active_per_cluster
        ),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": str(
            remediation.max_active_per_node
        ),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": str(
            remediation.max_active_per_failure_domain
        ),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": str(
            remediation.max_active_per_resource_class
        ),
        "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH": str(processor.max_queue_depth),
        "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": str(
            processor.max_cluster_queue_depth
        ),
        "GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH": str(
            admin_config.fault_reserved_queue_depth()
        ),
        "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH": str(
            admin_config.fault_reserved_cluster_depth()
        ),
        "GPU_FAULT_PROCESSOR_GLOBAL_ADMISSION_GUARD": str(
            min(256, processor.max_queue_depth)
        ),
        "GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS": str(processor.retry_after_seconds),
        "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS": str(
            processor.retry_backoff_seconds
        ),
        "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS": str(
            processor.retry_backoff_max_seconds
        ),
        "GPU_FAULT_PROCESSOR_RETRYABLE_RESPONSE_MAX_AGE_SECONDS": str(
            max(300, processor.retry_backoff_max_seconds)
        ),
        "GPU_FAULT_PROCESSOR_COMPLETED_RETENTION_SECONDS": str(
            processor.completed_retention_seconds
        ),
        "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS": _number_text(
            workflow.poll_interval_seconds
        ),
        "GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS": str(workflow.dispatcher_workers),
        "GPU_FAULT_NOTIFICATION_BATCH_SIZE": str(notification.batch_size),
        "GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS": str(notification.max_attempts),
        "GPU_FAULT_EVIDENCE_RETENTION_HOURS": str(evidence.retention_hours),
        "GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE": str(evidence.max_records_per_node),
    }


def _documents(text: str) -> list[dict[str, Any]]:
    return [
        document for document in yaml.safe_load_all(text) if isinstance(document, dict)
    ]


# In execution order; deploy/hyperpod/deploy.sh and the regional schema tool
# run the same three (F-J3 three-step index method).
SCHEMA_JOB_MANIFESTS = (
    "deploy/migrations/postgres-index-build-job.yaml",
    "deploy/migrations/postgres-schema-ensure-job.yaml",
    "deploy/migrations/postgres-schema-preflight-job.yaml",
)


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def render_release_payload(release: Any) -> dict[str, Any]:
    """The manifests and digests a release delivers, as one JSON-able payload.

    When the release carries an ``approved_manifest_digest`` -- the digest the
    plan was reviewed and approved against -- the freshly rendered payload is
    verified against it. Rendering reads the working tree, so re-reading it at
    apply opens a TOCTOU window: the tree can change after the plan is approved
    and before it is applied. Failing closed on a mismatch means the applied
    artifact can only ever be the one that was planned.
    """

    payload = _render_release_payload(release)
    approved = str(getattr(release, "approved_manifest_digest", "") or "")
    if approved:
        actual = _payload_digest(payload)
        if actual != approved:
            raise ReleaseError(
                "release manifests changed between plan and apply: the working "
                f"tree renders {actual} but the approved plan pinned {approved}"
            )
    return payload


def rendered_release_manifest_sha256(release: Any) -> str:
    return _payload_digest(render_release_payload(release))


def _render_release_payload(release: Any) -> dict[str, Any]:
    config = release.config
    generated = ROOT / "deploy/control-plane/regional/generated"
    manifest_names = [
        item.strip()
        for item in (generated / "manifest-list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if item.strip() and not item.startswith("#")
    ]
    cpu_documents: dict[str, list[dict[str, Any]]] = {}
    for filename in manifest_names:
        text = (generated / filename).read_text(encoding="utf-8")
        replacements = {
            "gpu-fault-control-plane-wheel-0100": release.wheel_cm,
            "namespace: gpu-fault-system": f"namespace: {config.namespace}",
            "REPLACE_WITH_AWS_REGION": config.aws_region,
            "REPLACE_WITH_RUNTIME_PROFILE_VERSION": (config.runtime_profile_version),
            DEFAULT_RUNTIME_IMAGE: release.runtime_image,
            "GPU_FAULT_ALLOW_EMAIL: 'true'": (
                "GPU_FAULT_ALLOW_EMAIL: "
                f"'{str(config.notifications.allow_email).lower()}'"
            ),
            "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL: 'false'": (
                "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL: "
                f"'{str(config.notifications.acknowledge_external_alert_channel).lower()}'"
            ),
        }
        for source, destination in replacements.items():
            text = text.replace(source, destination)
        text = text.replace(
            "gpu-fault.io/artifact-sha256: "
            "209840015cc3057e191931f113d35dff733cf1b7483d68dd6b6b26a7de9a112b",
            f"gpu-fault.io/artifact-sha256: {release.wheel_sha}",
        )
        cpu_documents[filename] = _documents(text)

    gpu_documents = {
        target.cluster_id: {
            deployment: _documents(text)
            for deployment, text in render_gpu_rollout_manifests(
                release,
                target,
                release.executor_wheel_cm,
            )
        }
        for target in config.clusters
    }
    # The schema stage is three Jobs (index-build, ensure, preflight; F-J3),
    # run in that order by deploy/control-plane/tools/ensure-postgres-schema.sh;
    # the release digest covers all three manifests.
    schema_documents: list[dict[str, Any]] = []
    for manifest in SCHEMA_JOB_MANIFESTS:
        schema_text = (ROOT / manifest).read_text(encoding="utf-8")
        for source, destination in {
            "namespace: gpu-fault-system": f"namespace: {config.namespace}",
            "REPLACE_WITH_WHEEL_CONFIGMAP": release.wheel_cm,
            DEFAULT_RUNTIME_IMAGE: release.runtime_image,
        }.items():
            schema_text = schema_text.replace(source, destination)
        schema_documents.extend(_documents(schema_text))
    refresh_text = (
        ROOT / "deploy/control-plane/regional/aurora-credential-refresh.yaml"
    ).read_text(encoding="utf-8")
    refresh_text = refresh_text.replace(
        "namespace: gpu-fault-system",
        f"namespace: {config.namespace}",
    ).replace(
        DEFAULT_RUNTIME_IMAGE,
        release.runtime_image,
    )

    nlb_documents = (
        _documents(
            render_nlb_manifest(
                config,
                (
                    ROOT
                    / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
                ).read_text(encoding="utf-8"),
            )
        )
        if config.nlb
        else []
    )
    dcgm_text = (ROOT / "deploy/dataplane/hyperpod-dcgm-exporter.yaml").read_text(
        encoding="utf-8"
    )
    dcgm_text = dcgm_text.replace(
        "namespace: gpu-fault-system",
        f"namespace: {config.namespace}",
    ).replace(
        DEFAULT_DCGM_EXPORTER_IMAGE,
        release.dcgm_exporter_image,
    )
    observability_text = (
        ROOT / "deploy/observability/adot-control-plane.yaml"
    ).read_text(encoding="utf-8")
    observability_text = observability_text.replace(
        "namespace: gpu-fault-system",
        f"namespace: {config.namespace}",
    ).replace(
        "REPLACE_WITH_ADOT_IMAGE",
        release.adot_image,
    )
    payload = {
        "cpu": cpu_documents,
        "admin_config": config.admin_config.as_dict(),
        "gpu": gpu_documents,
        "schema": schema_documents,
        "aurora_refresh": _documents(refresh_text),
        "nlb": nlb_documents,
        "dcgm": _documents(dcgm_text),
        "observability": _documents(observability_text),
        "reconciler": {
            target.cluster_id: {
                "wheel_config_map": release.executor_wheel_cm,
                "bundle_config_map": release.bundle_cm,
                "artifact_sha256": release.node_wheel_sha,
                "bundle_sha256": release.bundle_sha,
                "template_source_sha256": release.node_template_sha,
                "config_digest": config.agent_config_digest,
                "runtime_profile_version": config.runtime_profile_version,
                "runtime_image": release.runtime_image,
                "node_installer_image": release.node_installer_image,
            }
            for target in config.clusters
        },
    }
    return payload


def stamp_gpu_deployments(
    release: Any,
    text: str,
    *,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
    runtime_image: str | None = None,
) -> str:
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or release.config.component_digests.get("executor")
        or artifact_sha
    )
    documents = list(yaml.safe_load_all(text))
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != "Deployment":
            continue
        annotations = (
            document.setdefault("spec", {})
            .setdefault("template", {})
            .setdefault("metadata", {})
            .setdefault("annotations", {})
        )
        annotations.update(
            {
                "gpu-fault.io/artifact-sha256": artifact_sha,
                "gpu-fault.io/release-rollout": release.release_id,
                "gpu-fault.io/release-sha256": artifact_sha,
                "gpu-fault.io/release-wheel-sha256": artifact_sha,
                "gpu-fault.io/executor-wheel-sha256": artifact_sha,
                "gpu-fault.io/executor-compatibility-digest": compatibility_digest,
                "gpu-fault.io/runtime-image": (runtime_image or release.runtime_image),
            }
        )
    return yaml.safe_dump_all(
        documents,
        sort_keys=False,
        width=72,
    )


def build_cpu_apply_environment(
    release: Any,
    *,
    finalize: bool,
    runtime_profile_version: str | None = None,
) -> dict[str, str]:
    config = release.config
    return {
        **os.environ,
        **admin_config_renderer_environment(config.admin_config),
        # Present only when site.yaml spec.retention turns archive-first
        # deletion on; the renderer forwards these to the control-worker.
        **config.retention.environment(),
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
        "GPU_FAULT_ADMIN_CONFIG_SHA256": release.admin_config_digest,
        "GPU_FAULT_ADMIN_CONFIG_INGRESS_SHA256": (
            release.admin_config_role_digests["ingress"]
        ),
        "GPU_FAULT_ADMIN_CONFIG_WORKER_SHA256": (
            release.admin_config_role_digests["worker"]
        ),
        "GPU_FAULT_ADMIN_CONFIG_SPOOL_SHA256": (
            release.admin_config_role_digests["spool"]
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
    deployment_names: frozenset[str] | None = None,
    runtime_image: str | None = None,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
) -> list[tuple[str, str]]:
    config = release.config
    profile_version = runtime_profile_version or config.runtime_profile_version
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or config.component_digests.get("executor")
        or artifact_sha
    )
    replacements = {
        "gpu-fault-executor-wheel-0100": wheel_cm,
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl": (
            executor_wheel_filename or release.config.executor_wheel.name
        ),
        "namespace: gpu-fault-system": f"namespace: {config.namespace}",
        DEFAULT_RUNTIME_IMAGE: runtime_image or release.runtime_image,
        "REPLACE_WITH_AWS_REGION": target.region,
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION": profile_version,
        "REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN": target.executor_irsa_role_arn,
        "REPLACE_WITH_EXECUTOR_ARTIFACT_SHA256": artifact_sha,
        "REPLACE_WITH_EXECUTOR_COMPATIBILITY_DIGEST": compatibility_digest,
    }
    rendered = []
    for filename, deployment in inventory.GPU_ROLLOUT_DEPLOYMENTS:
        if deployment_names is not None and deployment not in deployment_names:
            continue
        text = (ROOT / "deploy/dataplane" / filename).read_text(encoding="utf-8")
        for source, destination in replacements.items():
            text = text.replace(source, destination)
        if "REPLACE_WITH" in text:
            raise ReleaseError(f"{filename} still contains a placeholder")
        rendered.append(
            (
                deployment,
                stamp_gpu_deployments(
                    release,
                    text,
                    executor_artifact_sha=artifact_sha,
                    executor_compatibility_digest=compatibility_digest,
                    runtime_image=runtime_image,
                ),
            )
        )
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
    node_compatibility_digest: str | None = None,
    bundle_sha256: str | None = None,
    template_sha256: str | None = None,
    template_config_map: str | None = None,
    allowed_node_names: tuple[str, ...] | None = None,
    max_unavailable: int | None = None,
    sync_registry: bool = True,
    runtime_image: str | None = None,
    node_installer_image: str | None = None,
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
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": (bundle_sha256 or release.bundle_sha),
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": (
            template_sha256 or release.node_template_sha
        ),
        "GPU_FAULT_NODE_COMPATIBILITY_DIGEST": (
            node_compatibility_digest
            or config.component_digests.get("node_runtime")
            or artifact_sha
        ),
        "GPU_FAULT_WHEEL_CONFIG_MAP": wheel_cm,
        "GPU_FAULT_EXECUTOR_WHEEL_FILENAME": (
            executor_wheel_filename or config.executor_wheel.name
        ),
        "GPU_FAULT_RUNTIME_IMAGE": runtime_image or release.runtime_image,
        "GPU_FAULT_NODE_INSTALLER_IMAGE": (
            node_installer_image or release.node_installer_image
        ),
        "GPU_FAULT_INSTALLER_ALLOWED_NODES": (
            "*" if allowed_node_names is None else ",".join(sorted(allowed_node_names))
        ),
        "GPU_FAULT_INSTALLER_MAX_UNAVAILABLE": str(
            max_unavailable or config.upgrade_max_unavailable
        ),
        "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": str(sync_registry).lower(),
        "GPU_FAULT_RUNTIME_PROFILE": (
            runtime_profile_version or config.runtime_profile_version
        ),
    }
    if target.fleet_master_file:
        environment["GPU_FAULT_FLEET_MASTER_FILE"] = target.fleet_master_file
        environment["GPU_FAULT_CONTROL_PLANE_KUBECONFIG"] = config.cpu_kubeconfig
    if template_config_map:
        environment["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] = template_config_map
    return environment
