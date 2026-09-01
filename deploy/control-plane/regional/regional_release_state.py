from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
import yaml

from gpu_fault.admin_config import (
    AdminConfig,
    AdminConfigError,
    default_admin_config,
)
from regional_release_config import ClusterTarget, ReleaseError
from regional_release_legacy import AGENT_IDENTITY_FIELDS
from regional_release_runtime_identity import CONTROL_PLANE_PYTHON

STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
SENSITIVE_CONFIG_KEY = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY)")
DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def get_json(release: Any, args: list[str]) -> dict[str, Any]:
    raw = release.runner.run(args + ["-o", "json"], capture=True)
    return json.loads(raw) if raw else {}


def config_map_data(release: Any, name: str) -> dict[str, str]:
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            name,
        )
    )
    return dict(value.get("data") or {})


def deployment_wheel(
    release: Any,
    args: list[str],
    deployment: str,
) -> str | None:
    value = release._get_json(
        args
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    for volume in (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
    ):
        if volume.get("name") == "artifact":
            return (volume.get("configMap") or {}).get("name")
    return None


def deployment_image(
    release: Any,
    args: list[str],
    deployment: str,
    *,
    container_name: str | None = None,
) -> str | None:
    value = release._get_json(
        args
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    containers = (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    )
    for container in containers:
        if container_name is None or container.get("name") == container_name:
            image = str(container.get("image") or "").strip()
            return image or None
    return None


def deployment_env_value(
    release: Any,
    args: list[str],
    deployment: str,
    name: str,
) -> str | None:
    value = release._get_json(
        args
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    containers = (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    )
    for container in containers:
        for environment in container.get("env", []):
            if environment.get("name") == name:
                result = str(environment.get("value") or "").strip()
                return result or None
    return None


def template_container_image(
    text: str,
    *,
    container_name: str,
) -> str | None:
    for document in yaml.safe_load_all(text):
        containers = (
            (document or {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for container in containers:
            if container.get("name") == container_name:
                image = str(container.get("image") or "").strip()
                return image or None
    return None


def require_consistent_images(
    description: str,
    images: dict[str, str | None],
) -> str:
    missing = sorted(name for name, image in images.items() if not image)
    if missing:
        raise ReleaseError(
            f"cannot capture previous {description} image from: " + ", ".join(missing)
        )
    distinct = {str(image) for image in images.values()}
    if len(distinct) != 1:
        raise ReleaseError(
            f"previous {description} images are inconsistent across: "
            + ", ".join(sorted(images))
        )
    return distinct.pop()


def config_map_binary_key(
    release: Any,
    kubectl: list[str],
    name: str | None,
) -> str | None:
    if not name:
        return None
    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            name,
        ]
    )
    keys = sorted((value.get("binaryData") or {}).keys())
    return keys[0] if len(keys) == 1 else None


def cpu_role_config_maps(release: Any) -> dict[str, dict[str, str]]:
    names: set[str] = set()
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        value = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                deployment,
            )
        )
        pod_spec = value.get("spec", {}).get("template", {}).get("spec", {})
        for container in [
            *pod_spec.get("initContainers", []),
            *pod_spec.get("containers", []),
        ]:
            for source in container.get("envFrom", []):
                name = (source.get("configMapRef") or {}).get("name")
                if (
                    isinstance(name, str)
                    and name.startswith("gpu-fault-")
                    and "-config-" in name
                ):
                    names.add(name)

    snapshots: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        data = release._config_map_data(name)
        sensitive = sorted(key for key in data if SENSITIVE_CONFIG_KEY.search(key))
        if sensitive:
            raise ReleaseError(
                f"role ConfigMap {name} contains sensitive-looking keys: "
                + ", ".join(sensitive)
            )
        snapshots[name] = data
    return snapshots


def captured_admin_config(
    release: Any,
    snapshots: dict[str, dict[str, str]],
) -> AdminConfig:
    try:
        defaults = default_admin_config()
        capacity_defaults = defaults.capacity
        worker = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-fault-control-worker",
            )
        )
        spool = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-fault-telemetry-spool-worker",
            )
        )
        ingress_telemetry = snapshots.get(
            "gpu-fault-api-ha-config-telemetry",
            {},
        )
        worker_core = snapshots.get("gpu-fault-control-worker-config-core", {})
        ingress_processor = snapshots.get(
            "gpu-fault-api-ha-config-processor",
            {},
        )
        ingress_recovery = snapshots.get(
            "gpu-fault-api-ha-config-recovery",
            {},
        )
        ingress_notification = snapshots.get(
            "gpu-fault-api-ha-config-notification",
            {},
        )
        spool_enabled = ingress_telemetry.get(
            "GPU_FAULT_TELEMETRY_SPOOL",
            str(capacity_defaults.telemetry_spool.enabled).lower(),
        )
        if spool_enabled not in {"true", "false"}:
            raise AdminConfigError("live ingress telemetry spool value is invalid")
        worker_replicas = (worker.get("spec") or {}).get("replicas")
        spool_replicas = (spool.get("spec") or {}).get("replicas")
        remediation = capacity_defaults.remediation
        processor = defaults.processor
        workflow = defaults.workflow
        notification = defaults.notification_delivery
        evidence = defaults.evidence
        return AdminConfig.from_mapping(
            {
                "schema_version": 1,
                "capacity": {
                    "control_worker_replicas": int(
                        capacity_defaults.control_worker_replicas
                        if worker_replicas is None
                        else worker_replicas
                    ),
                    "telemetry_spool": {
                        "enabled": spool_enabled == "true",
                        "replicas": int(
                            capacity_defaults.telemetry_spool.replicas
                            if spool_replicas is None
                            else spool_replicas
                        ),
                    },
                    "remediation": {
                        "max_active_region": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION",
                                remediation.max_active_region,
                            )
                        ),
                        "max_active_per_cluster": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER",
                                remediation.max_active_per_cluster,
                            )
                        ),
                        "max_active_per_node": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE",
                                remediation.max_active_per_node,
                            )
                        ),
                        "max_active_per_failure_domain": int(
                            worker_core.get(
                                ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN"),
                                remediation.max_active_per_failure_domain,
                            )
                        ),
                        "max_active_per_resource_class": int(
                            worker_core.get(
                                ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS"),
                                remediation.max_active_per_resource_class,
                            )
                        ),
                    },
                },
                "processor": {
                    "max_queue_depth": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH",
                            processor.max_queue_depth,
                        )
                    ),
                    "max_cluster_queue_depth": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH",
                            processor.max_cluster_queue_depth,
                        )
                    ),
                    "retry_after_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS",
                            processor.retry_after_seconds,
                        )
                    ),
                    "retry_backoff_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS",
                            processor.retry_backoff_seconds,
                        )
                    ),
                    "retry_backoff_max_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS",
                            processor.retry_backoff_max_seconds,
                        )
                    ),
                    "completed_retention_seconds": int(
                        ingress_processor.get(
                            ("GPU_FAULT_PROCESSOR_COMPLETED_RETENTION_SECONDS"),
                            processor.completed_retention_seconds,
                        )
                    ),
                },
                "workflow": {
                    "poll_interval_seconds": float(
                        ingress_recovery.get(
                            "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS",
                            workflow.poll_interval_seconds,
                        )
                    ),
                    "dispatcher_workers": int(
                        ingress_recovery.get(
                            "GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS",
                            workflow.dispatcher_workers,
                        )
                    ),
                },
                "notification_delivery": {
                    "batch_size": int(
                        ingress_notification.get(
                            "GPU_FAULT_NOTIFICATION_BATCH_SIZE",
                            notification.batch_size,
                        )
                    ),
                    "max_attempts": int(
                        ingress_notification.get(
                            "GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS",
                            notification.max_attempts,
                        )
                    ),
                },
                "evidence": {
                    "retention_hours": int(
                        ingress_processor.get(
                            "GPU_FAULT_EVIDENCE_RETENTION_HOURS",
                            evidence.retention_hours,
                        )
                    ),
                    "max_records_per_node": int(
                        ingress_processor.get(
                            "GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE",
                            evidence.max_records_per_node,
                        )
                    ),
                },
            }
        )
    except (AdminConfigError, KeyError, TypeError, ValueError) as exc:
        raise ReleaseError("cannot capture a valid live administrator config") from exc


def capture_agent_identities(release: Any) -> dict[str, dict[str, Any]]:
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("cannot capture previous Agent identity without CPU ingress")
    script = """
import json
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext

now = datetime.now(timezone.utc)
records = [
    item
    for item in ApplicationContext.from_environment().store.list_agents()
    if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
    and item.lease_expires_at is not None
    and item.lease_expires_at > now
]
print(json.dumps([
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
], sort_keys=True))
"""
    raw = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            "-i",
            pod,
            "--",
            CONTROL_PLANE_PYTHON,
            "-c",
            script,
        ),
        capture=True,
    )
    records = json.loads(raw)
    result: dict[str, dict[str, Any]] = {}
    for target in release.config.clusters:
        selected = [
            item for item in records if item.get("cluster_id") == target.cluster_id
        ]
        if not selected:
            raise ReleaseError(
                f"{target.cluster_id} has no active Agent identity to capture"
            )
        identities = {
            tuple(item.get(field) for field in AGENT_IDENTITY_FIELDS)
            for item in selected
        }
        if len(identities) != 1:
            raise ReleaseError(f"{target.cluster_id} active Agent identities differ")
        identity = dict(zip(AGENT_IDENTITY_FIELDS, identities.pop(), strict=True))
        identity["node_ids"] = sorted(str(item["node_id"]) for item in selected)
        result[target.cluster_id] = identity
    return result


def capture_previous(release: Any) -> dict[str, Any]:
    live_state = dict(release.state) if release.state else release._load_state()
    remote = release._remote_command_stats()
    metadata = release._config_map_data("gpu-fault-release-metadata")
    clusters = {}
    runtime_images = {
        f"cpu/{deployment}": deployment_image(
            release,
            release._cpu(),
            deployment,
        )
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
    }
    node_installer_images: dict[str, str | None] = {}
    for target in release.config.clusters:
        template = release._deployment_template_name(target)
        executor_wheel = release._deployment_wheel(
            release._gpu(target),
            inventory.GPU_EXECUTOR_DEPLOYMENT,
        )
        reconciler_wheel = release._deployment_wheel(
            release._gpu(target),
            inventory.GPU_RECONCILER_DEPLOYMENT,
        )
        bundle_name = release._template_bundle(target, template) if template else None
        template_sha256 = deployment_env_value(
            release,
            release._gpu(target),
            inventory.GPU_RECONCILER_DEPLOYMENT,
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
        )
        template_text = None
        if template:
            template_value = release._get_json(
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "get",
                    "configmap",
                    template,
                )
            )
            template_text = (template_value.get("data") or {}).get("job.yaml")
            if template_text and not template_sha256:
                template_sha256 = hashlib.sha256(template_text.encode()).hexdigest()
        node_installer_images[target.cluster_id] = (
            template_container_image(
                template_text,
                container_name="installer",
            )
            if template_text
            else None
        )
        for deployment in (
            *inventory.DEPLOYMENTS,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        ):
            runtime_images[f"{target.cluster_id}/{deployment}"] = deployment_image(
                release,
                release._gpu(target),
                deployment,
            )
        bundle_key = release._config_map_binary_key(
            release._gpu(target),
            bundle_name,
        )
        bundle_sha256 = (
            release._config_map_sha(
                release._gpu(target),
                bundle_name,
                bundle_key or release.config.bundle.name,
            )
            if bundle_name
            else None
        )
        clusters[target.cluster_id] = {
            "wheel": executor_wheel,
            "wheel_key": release._config_map_binary_key(
                release._gpu(target),
                executor_wheel,
            ),
            "reconciler_wheel": reconciler_wheel,
            "reconciler_wheel_key": release._config_map_binary_key(
                release._gpu(target),
                reconciler_wheel,
            ),
            "template": template,
            "template_sha256": template_sha256,
            "bundle": bundle_name,
            "bundle_key": bundle_key,
            "bundle_sha256": bundle_sha256,
            "dcgm_image": (
                release._get_json(
                    release._gpu(
                        target,
                        "-n",
                        release.config.namespace,
                        "get",
                        "daemonset",
                        "gpu-fault-dcgm-exporter",
                    )
                )
                .get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("containers", [{}])[0]
                .get("image")
            ),
        }
    live_runtime_image = require_consistent_images("runtime", runtime_images)
    runtime_image = live_runtime_image
    adopted_live_runtime_image = str(
        live_state.get("adopted_live_runtime_image") or ""
    ).strip()
    rollback_runtime_image = str(live_state.get("runtime_image") or "").strip()
    if adopted_live_runtime_image:
        if live_runtime_image != adopted_live_runtime_image:
            raise ReleaseError(
                "live runtime image drifted after legacy release-state adoption"
            )
        if not DIGEST_IMAGE.fullmatch(rollback_runtime_image):
            raise ReleaseError(
                "legacy release-state adoption has no immutable rollback runtime image"
            )
        runtime_image = rollback_runtime_image
    node_installer_image = require_consistent_images(
        "Node Installer",
        node_installer_images,
    )
    adot_image = deployment_image(
        release,
        release._cpu(),
        "gpu-fault-adot",
        container_name="collector",
    )
    if not adot_image:
        raise ReleaseError(
            "cannot capture previous ADOT image from deployment/gpu-fault-adot"
        )
    capture_identities = getattr(release, "_capture_agent_identities", None)
    agent_identities = (
        capture_identities()
        if capture_identities is not None
        else capture_agent_identities(release)
    )
    for target in release.config.clusters:
        expected_nodes = set(release._target_node_names(target))
        captured_nodes = set(agent_identities[target.cluster_id]["node_ids"])
        if expected_nodes != captured_nodes:
            raise ReleaseError(
                f"{target.cluster_id} active Agent set does not match HyperPod nodes"
            )
    role_config_maps = cpu_role_config_maps(release)
    admin_config = captured_admin_config(release, role_config_maps)
    return {
        "metadata": metadata,
        "agent_identities": agent_identities,
        "runtime_profile_version": release._config_map_data(
            "gpu-fault-api-ha-config-core"
        ).get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"),
        "cpu_wheel": release._deployment_wheel(
            release._cpu(),
            inventory.CPU_INGRESS_DEPLOYMENT,
        ),
        "cpu_role_config_maps": role_config_maps,
        "admin_config": admin_config.as_dict(),
        "executor_internal_error_total": int(
            remote.get("executor_internal_error_total", 0) or 0
        ),
        "release_delivery_sha256": live_state.get("release_delivery_sha256"),
        "rendered_manifest_sha256": live_state.get("rendered_manifest_sha256"),
        "node_template_sha256": live_state.get("node_template_sha256"),
        "live_runtime_image": live_runtime_image,
        "runtime_image": runtime_image,
        "node_installer_image": node_installer_image,
        "adot_image": adot_image,
        "clusters": clusters,
    }


def deployment_template_name(
    release: Any,
    target: ClusterTarget,
) -> str | None:
    value = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.GPU_RECONCILER_DEPLOYMENT,
        )
    )
    for volume in (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
    ):
        if volume.get("name") == "installer-template":
            return (volume.get("configMap") or {}).get("name")
    return None


def template_bundle(
    release: Any,
    target: ClusterTarget,
    template_name: str,
) -> str | None:
    value = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            template_name,
        )
    )
    text = (value.get("data") or {}).get("job.yaml")
    if not text:
        return None
    for document in yaml.safe_load_all(text):
        for volume in (
            (document or {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("volumes", [])
        ):
            if volume.get("name") == "installer":
                return (volume.get("configMap") or {}).get("name")
    return None


def save_state(release: Any, phase: str, **updates: Any) -> None:
    if release.state.get("release_id") not in {None, release.release_id}:
        release.state.pop("adopted_live_runtime_image", None)
    release.state.update(
        {
            "phase": phase,
            "release_id": release.release_id,
            "wheel_sha256": release.wheel_sha,
            "executor_wheel_sha256": release.executor_wheel_sha,
            "node_wheel_sha256": release.node_wheel_sha,
            "bundle_sha256": release.bundle_sha,
            "wheel_config_map": release.wheel_cm,
            "executor_wheel_config_map": release.executor_wheel_cm,
            "bundle_config_map": release.bundle_cm,
            "database_schema_version": release.config.database_schema_version,
            "agent_protocol_version": release.config.agent_protocol_version,
            "executor_protocol_version": release.config.executor_protocol_version,
            "component_digests": release.config.component_digests,
            "runtime_profile_version": release.config.runtime_profile_version,
            "runtime_profile_sha256": release.runtime_profile_sha,
            "runtime_profile_source_sha256": release.runtime_profile_sha,
            "runtime_profile_template_sha256": release.runtime_profile_template_sha,
            "runtime_profile_policy_sha256": release.runtime_profile_policy_sha,
            "runtime_profile_registration_cluster_id": (
                release.config.runtime_profile_registration_cluster_id
            ),
            "agent_config_digest": release.config.agent_config_digest,
            "endpoint_digest": release.endpoint_digest,
            "dcgm_digest": release.dcgm_digest,
            "notification_digest": release.notification_digest,
            "cluster_ids": sorted(item.cluster_id for item in release.config.clusters),
            "cluster_registry_digest": release.cluster_registry_digest,
            "admin_config_sha256": release.admin_config_digest,
            "admin_config_role_sha256": release.admin_config_role_digests,
            "admin_config": release.config.admin_config.as_dict(),
            "release_manifest_schema_version": (
                release.config.release_manifest_schema_version
            ),
            "release_delivery_sha256": (release.config.release_delivery_sha256),
            "cpu_manifest_sha256": (
                release.config.delivery_component_digests.get("cpu")
            ),
            "executor_manifest_sha256": (
                release.config.delivery_component_digests.get("executor")
            ),
            "watcher_manifest_sha256": (
                release.config.delivery_component_digests.get("watcher")
            ),
            "collector_manifest_sha256": (
                release.config.delivery_component_digests.get("collector")
            ),
            "dcgm_manifest_sha256": (
                release.config.delivery_component_digests.get("dcgm")
            ),
            "node_manifest_sha256": (
                release.config.delivery_component_digests.get("node")
            ),
            "observability_manifest_sha256": (
                release.config.delivery_component_digests.get("observability")
            ),
            "schema_manifest_sha256": (
                release.config.delivery_component_digests.get("schema")
            ),
            "endpoint_manifest_sha256": (
                release.config.delivery_component_digests.get("endpoint")
            ),
            "rendered_manifest_sha256": release.rendered_manifest_digest,
            "node_template_sha256": release.node_template_sha,
            "runtime_image": release.runtime_image,
            "node_installer_image": release.node_installer_image,
            "dcgm_image": release.dcgm_exporter_image,
            "adot_image": release.adot_image,
            "updated_at_epoch": int(time.time()),
            **updates,
        }
    )
    if release.runner.dry_run:
        return
    with tempfile.TemporaryDirectory() as directory:
        state_file = Path(directory) / "state.json"
        state_file.write_text(
            json.dumps(release.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rendered = release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "create",
                "configmap",
                STATE_CONFIG_MAP,
                f"--from-file=state.json={state_file}",
                "--dry-run=client",
                "-o",
                "yaml",
            ),
            capture=True,
        )
        release.runner.run(
            release._cpu("apply", "-f", "-"),
            input_text=rendered,
        )


def load_state(release: Any) -> dict[str, Any]:
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            STATE_CONFIG_MAP,
        )
    )
    raw = (value.get("data") or {}).get("state.json")
    if not raw:
        raise ReleaseError("regional release state is missing")
    release.state = json.loads(raw)
    return release.state
