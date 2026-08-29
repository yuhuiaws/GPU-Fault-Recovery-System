from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ClusterTarget, ReleaseError
from regional_release_gpu_rollout import agents_converged
from regional_release_legacy import apply_fleet_request_identity
from regional_release_rendering import build_reconciler_environment
from regional_release_runtime_identity import CONTROL_PLANE_PYTHON


ROOT = Path(__file__).resolve().parents[3]


def backup_secret(
    release: Any,
    kubectl: list[str],
    *,
    source: str,
    backup: str,
    required: bool,
) -> str | None:
    exists = (
        subprocess.run(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "get",
                "secret",
                source,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not exists:
        if required:
            raise ReleaseError(f"required Secret is missing: {source}")
        return None
    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "secret",
            source,
        ]
    )
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": backup,
            "namespace": release.config.namespace,
            "labels": {
                "gpu-fault.io/release-secret-backup": "true",
            },
            "annotations": {
                "gpu-fault.io/source-secret": source,
                "gpu-fault.io/release-id": release.release_id,
            },
        },
        "type": value.get("type") or "Opaque",
        "data": dict(value.get("data") or {}),
    }
    release.runner.run(
        kubectl + ["apply", "-f", "-"],
        input_text=json.dumps(document),
        sensitive=True,
    )
    return backup


def restore_secret(
    release: Any,
    kubectl: list[str],
    *,
    source: str,
    backup: str,
) -> None:
    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "secret",
            backup,
        ]
    )
    if not value.get("data"):
        raise ReleaseError(f"release Secret backup is missing: {backup}")
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": source,
            "namespace": release.config.namespace,
        },
        "type": value.get("type") or "Opaque",
        "data": dict(value.get("data") or {}),
    }
    release.runner.run(
        kubectl + ["apply", "-f", "-"],
        input_text=json.dumps(document),
        sensitive=True,
    )


def backup_release_secrets(release: Any) -> dict[str, Any]:
    created: list[tuple[list[str], str]] = []
    try:
        cpu_backup = release._backup_secret(
            release._cpu(),
            source="gpu-fault-email",
            backup=f"gpu-fault-email-rollback-{release.release_id}",
            required=False,
        )
        if cpu_backup:
            created.append((release._cpu(), cpu_backup))
        clusters: dict[str, dict[str, str]] = {}
        for target in release.config.clusters:
            backup = release._backup_secret(
                release._gpu(target),
                source="gpu-fault-regional-connection",
                backup=("gpu-fault-regional-connection-rollback-" + release.release_id),
                required=True,
            )
            if backup is None:
                raise ReleaseError(
                    f"{target.cluster_id} connection Secret backup failed"
                )
            created.append((release._gpu(target), backup))
            clusters[target.cluster_id] = {
                "source": "gpu-fault-regional-connection",
                "backup": backup,
            }
        return {
            "cpu": (
                {
                    "source": "gpu-fault-email",
                    "backup": cpu_backup,
                }
                if cpu_backup
                else None
            ),
            "clusters": clusters,
        }
    except Exception:
        for kubectl, name in created:
            release.runner.run(
                kubectl
                + [
                    "-n",
                    release.config.namespace,
                    "delete",
                    "secret",
                    name,
                    "--ignore-not-found",
                ],
                sensitive=True,
            )
        raise


def delete_release_secret_backups(
    release: Any,
    previous: dict[str, Any],
) -> None:
    backups = previous.get("secret_backups") or {}
    entries: list[tuple[list[str], str]] = []
    cpu = backups.get("cpu") or {}
    if cpu.get("backup"):
        entries.append((release._cpu(), str(cpu["backup"])))
    by_cluster = backups.get("clusters") or {}
    for target in release.config.clusters:
        item = by_cluster.get(target.cluster_id) or {}
        if item.get("backup"):
            entries.append((release._gpu(target), str(item["backup"])))
    for kubectl, name in entries:
        release.runner.run(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "delete",
                "secret",
                name,
                "--ignore-not-found",
            ],
            sensitive=True,
        )


def target_node_names(
    release: Any,
    target: ClusterTarget,
) -> tuple[str, ...]:
    value = release._get_json(release._gpu(target, "get", "nodes"))
    names = tuple(
        sorted(
            str(item.get("metadata", {}).get("name") or "")
            for item in value.get("items", [])
            if (
                item.get("metadata", {})
                .get("labels", {})
                .get("sagemaker.amazonaws.com/cluster-name")
                == target.hyperpod_cluster_name
                and item.get("metadata", {}).get("name")
            )
        )
    )
    if not names:
        raise ReleaseError(
            f"{target.cluster_id} has no HyperPod nodes for fleet rollout"
        )
    return names


def fleet_command(
    release: Any,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
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
        raise ReleaseError("no running CPU ingress Pod for fleet rollout")
    script = """
import json
import sys

from gpu_fault import __version__
from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import FleetDeploymentRequest
from gpu_fault.policy import load_xid_policy

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
registry = context.fleet_registry
if registry is None:
    raise RuntimeError("fleet registry is disabled")
operation = payload["operation"]
if operation == "create":
    request = dict(payload["request"])
    request.setdefault("desired_agent_version", __version__)
    request.setdefault("desired_policy_version", load_xid_policy().mapping_version)
    result = registry.create_deployment(
        FleetDeploymentRequest.model_validate(request)
    )
elif operation == "get":
    result = context.store.get_fleet_deployment(payload["deployment_id"])
elif operation == "next-wave":
    result = registry.start_next_wave(payload["deployment_id"])
else:
    raise ValueError(f"unsupported fleet operation: {operation}")
print(result.model_dump_json())
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
        input_text=json.dumps({"operation": operation, **payload}),
        capture=True,
    )
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ReleaseError("fleet rollout command returned a non-object")
    return result


def fleet_deployment_id(
    release: Any,
    target: ClusterTarget,
    *,
    phase: str,
    artifact_sha: str,
    bundle_sha: str | None,
    template_sha: str | None,
    config_digest: str,
    runtime_profile_version: str,
) -> str:
    identity = hashlib.sha256(
        json.dumps(
            {
                "phase": phase,
                "release_id": release.release_id,
                "cluster_id": target.cluster_id,
                "artifact_sha256": artifact_sha,
                "bundle_sha256": bundle_sha,
                "template_sha256": template_sha,
                "config_digest": config_digest,
                "runtime_profile_version": runtime_profile_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]
    return f"release-{phase}-{release.release_id}-{identity}"


def nodes_have_legacy_installer_identity(
    release: Any,
    target: ClusterTarget,
    *,
    node_names: tuple[str, ...],
    artifact_sha: str,
    config_digest: str,
) -> bool:
    expected = set(node_names)
    value = release._get_json(release._gpu(target, "get", "nodes"))
    selected = [
        item
        for item in value.get("items", [])
        if str(item.get("metadata", {}).get("name") or "") in expected
    ]
    if len(selected) != len(expected):
        return False
    for node in selected:
        metadata = node.get("metadata", {})
        annotations = metadata.get("annotations") or {}
        if (
            annotations.get("gpu-fault.io/installer-artifact-sha256") != artifact_sha
            or annotations.get("gpu-fault.io/installer-config-digest") != config_digest
            or annotations.get("gpu-fault.io/installer-node-uid") != metadata.get("uid")
            or annotations.get("gpu-fault.io/installer-state") != "Succeeded"
        ):
            return False
    return True


def finish_legacy_node_runtime_rollback(
    release: Any,
    target: ClusterTarget,
    *,
    enabled: bool,
    node_names: tuple[str, ...],
    paused_identity: tuple[str, str],
    wheel_cm: str,
    bundle_cm: str,
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str,
    executor_wheel_filename: str | None,
    node_compatibility_digest: str,
    template_config_map: str | None,
    runtime_image: str | None,
    steady_runtime_image: str | None,
    steady_template_config_map: str | None,
    node_installer_image: str | None,
) -> tuple[str, str] | None:
    if not enabled or not nodes_have_legacy_installer_identity(
        release,
        target,
        node_names=node_names,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
    ):
        return None
    desired_bundle, desired_template = paused_identity
    final_identity = release._deploy_reconciler(
        target,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=node_compatibility_digest,
        bundle_sha256=desired_bundle,
        template_sha256=desired_template,
        template_config_map=steady_template_config_map or template_config_map,
        allowed_node_names=None,
        runtime_image=steady_runtime_image or runtime_image,
        node_installer_image=node_installer_image,
    )
    if final_identity != paused_identity:
        raise ReleaseError(f"{target.cluster_id} legacy installer identity changed")
    return final_identity


def node_rollout_max_unavailable(node_count: int) -> int:
    try:
        configured = int(os.getenv("GPU_FAULT_INSTALLER_MAX_UNAVAILABLE", "1"))
    except ValueError as exc:
        raise ReleaseError(
            "GPU_FAULT_INSTALLER_MAX_UNAVAILABLE must be an integer"
        ) from exc
    return min(node_count, max(1, configured))


def roll_node_runtime(
    release: Any,
    target: ClusterTarget,
    *,
    phase: str,
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
    runtime_image: str | None = None,
    steady_runtime_image: str | None = None,
    steady_template_config_map: str | None = None,
    node_installer_image: str | None = None,
    allow_legacy_identity: bool = False,
    agent_identity: dict[str, Any] | None = None,
) -> tuple[str, str]:
    expected_profile = runtime_profile_version or release.config.runtime_profile_version
    expected_compatibility = (
        node_compatibility_digest
        or release.config.component_digests.get("node_runtime")
        or artifact_sha
    )
    if release.runner.dry_run:
        identity = release._deploy_reconciler(
            target,
            wheel_cm=wheel_cm,
            bundle_cm=bundle_cm,
            artifact_sha=artifact_sha,
            config_digest=config_digest,
            runtime_profile_version=expected_profile,
            executor_wheel_filename=executor_wheel_filename,
            node_compatibility_digest=expected_compatibility,
            bundle_sha256=bundle_sha256,
            template_sha256=template_sha256,
            template_config_map=template_config_map,
            runtime_image=runtime_image,
            node_installer_image=node_installer_image,
        )
        release._wait_agents(
            target,
            artifact_sha,
            bundle_sha=identity[0],
            template_sha=identity[1],
            config_digest=config_digest,
            runtime_profile_version=expected_profile,
            legacy_identity=allow_legacy_identity,
            agent_identity=agent_identity,
        )
        return identity
    node_names = release._target_node_names(target)
    paused_identity = release._deploy_reconciler(
        target,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=expected_profile,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=expected_compatibility,
        bundle_sha256=bundle_sha256,
        template_sha256=template_sha256,
        template_config_map=template_config_map,
        allowed_node_names=(),
        runtime_image=runtime_image,
        node_installer_image=node_installer_image,
    )
    desired_bundle, desired_template = paused_identity
    legacy_identity = finish_legacy_node_runtime_rollback(
        release,
        target,
        enabled=allow_legacy_identity,
        node_names=node_names,
        paused_identity=paused_identity,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=expected_profile,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=expected_compatibility,
        template_config_map=template_config_map,
        runtime_image=runtime_image,
        steady_runtime_image=steady_runtime_image,
        steady_template_config_map=steady_template_config_map,
        node_installer_image=node_installer_image,
    )
    if legacy_identity is not None:
        return legacy_identity
    fleet_bundle = None if allow_legacy_identity else desired_bundle
    fleet_template = None if allow_legacy_identity else desired_template
    max_unavailable = node_rollout_max_unavailable(len(node_names))
    deployment_id = release._fleet_deployment_id(
        target,
        phase=phase,
        artifact_sha=artifact_sha,
        bundle_sha=fleet_bundle,
        template_sha=fleet_template,
        config_digest=config_digest,
        runtime_profile_version=expected_profile,
    )
    request = {
        "deployment_id": deployment_id,
        "cluster_id": target.cluster_id,
        "node_ids": list(node_names),
        "desired_artifact_sha256": artifact_sha,
        "desired_compatibility_digest": expected_compatibility,
        "desired_bundle_sha256": fleet_bundle,
        "desired_template_sha256": fleet_template,
        "desired_runtime_profile_version": expected_profile,
        "desired_config_digest": config_digest,
        "max_unavailable": max_unavailable,
    }
    apply_fleet_request_identity(request, agent_identity)
    deployment = release._fleet_command("create", {"request": request})
    while deployment.get("status") != "SUCCEEDED":
        if deployment.get("status") == "FAILED":
            raise ReleaseError(f"{target.cluster_id} fleet deployment failed")
        lease = release._fleet_command(
            "next-wave",
            {"deployment_id": deployment_id},
        )
        wave = tuple(str(item) for item in lease.get("node_ids", []))
        if not wave:
            raise ReleaseError(
                f"{target.cluster_id} fleet deployment returned an empty wave"
            )
        identity = release._deploy_reconciler(
            target,
            wheel_cm=wheel_cm,
            bundle_cm=bundle_cm,
            artifact_sha=artifact_sha,
            config_digest=config_digest,
            runtime_profile_version=expected_profile,
            executor_wheel_filename=executor_wheel_filename,
            node_compatibility_digest=expected_compatibility,
            bundle_sha256=desired_bundle,
            template_sha256=desired_template,
            template_config_map=template_config_map,
            allowed_node_names=wave,
            runtime_image=runtime_image,
            node_installer_image=node_installer_image,
        )
        if identity != paused_identity:
            raise ReleaseError(
                f"{target.cluster_id} installer identity changed between waves"
            )
        release._wait_agents(
            target,
            artifact_sha,
            bundle_sha=desired_bundle,
            template_sha=desired_template,
            config_digest=config_digest,
            runtime_profile_version=expected_profile,
            node_names=wave,
            legacy_identity=allow_legacy_identity,
            agent_identity=agent_identity,
        )
        deployment = release._fleet_command(
            "get",
            {"deployment_id": deployment_id},
        )

    final_identity = release._deploy_reconciler(
        target,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=expected_profile,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=expected_compatibility,
        bundle_sha256=desired_bundle,
        template_sha256=desired_template,
        template_config_map=steady_template_config_map or template_config_map,
        allowed_node_names=None,
        runtime_image=steady_runtime_image or runtime_image,
        node_installer_image=node_installer_image,
    )
    if final_identity != paused_identity:
        raise ReleaseError(
            f"{target.cluster_id} steady-state installer identity changed"
        )
    release._wait_agents(
        target,
        artifact_sha,
        bundle_sha=desired_bundle,
        template_sha=desired_template,
        config_digest=config_digest,
        runtime_profile_version=expected_profile,
        legacy_identity=allow_legacy_identity,
        agent_identity=agent_identity,
    )
    return final_identity


def deploy_reconciler(
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
    runtime_image: str | None = None,
    node_installer_image: str | None = None,
) -> tuple[str, str]:
    release._cancel_active_installer_jobs(target)
    release._retry_failed_installer_jobs(target)
    environment = build_reconciler_environment(
        release,
        target,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=node_compatibility_digest,
        bundle_sha256=bundle_sha256,
        template_sha256=template_sha256,
        template_config_map=template_config_map,
        allowed_node_names=allowed_node_names,
        runtime_image=runtime_image,
        node_installer_image=node_installer_image,
    )
    release.runner.run(
        [str(ROOT / "deploy/node/deploy-node-installer-reconciler.sh")],
        env=environment,
        sensitive=bool(target.fleet_master_file),
    )
    if release.runner.dry_run:
        return (
            bundle_sha256 or release.bundle_sha,
            template_sha256 or release.node_template_sha,
        )
    deployment = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.GPU_RECONCILER_DEPLOYMENT,
        )
    )
    containers = (
        deployment.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    container = next(
        (item for item in containers if item.get("name") == "reconciler"),
        None,
    )
    if container is None:
        raise ReleaseError("Reconciler Deployment has no reconciler container")
    current = {item.get("name"): item.get("value") for item in container.get("env", [])}
    bundle = str(current.get("GPU_FAULT_INSTALLER_BUNDLE_SHA256") or "")
    template = str(current.get("GPU_FAULT_INSTALLER_TEMPLATE_SHA256") or "")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (bundle, template)):
        raise ReleaseError("Reconciler installer identity is invalid")
    return bundle, template


def wait_agents(
    release: Any,
    target: ClusterTarget,
    artifact_sha: str,
    *,
    bundle_sha: str | None = None,
    template_sha: str | None = None,
    config_digest: str | None = None,
    runtime_profile_version: str | None = None,
    node_names: tuple[str, ...] = (),
    timeout_seconds: int = 900,
    legacy_identity: bool = False,
    agent_identity: dict[str, Any] | None = None,
) -> None:
    expected_bundle = None
    expected_template = None
    if not legacy_identity:
        expected_bundle = (
            bundle_sha
            if bundle_sha is not None
            else (
                release.bundle_sha
                if release.config.release_manifest_schema_version >= 3
                else None
            )
        )
        expected_template = (
            template_sha
            if template_sha is not None
            else (
                release.node_template_sha
                if release.config.release_manifest_schema_version >= 3
                else None
            )
        )
    expected_config = config_digest or release.config.agent_config_digest
    expected_profile = runtime_profile_version or release.config.runtime_profile_version
    expected_nodes = frozenset(node_names) if node_names else None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = release._get_json(release._gpu(target, "get", "nodes"))
        nodes = value.get("items", [])
        selected_nodes = [
            item
            for item in nodes
            if (
                item.get("metadata", {})
                .get("labels", {})
                .get("sagemaker.amazonaws.com/cluster-name")
                == target.hyperpod_cluster_name
                and (
                    expected_nodes is None
                    or str(item.get("metadata", {}).get("name") or "") in expected_nodes
                )
            )
        ]
        if agents_converged(
            nodes,
            target,
            artifact_sha,
            bundle_sha=expected_bundle,
            template_sha=expected_template,
            config_digest=expected_config,
            require_node_uid=True,
            node_names=expected_nodes,
        ) and release._agent_heartbeats_converged(
            target,
            node_count=len(selected_nodes),
            node_names=tuple(
                sorted(
                    str(item.get("metadata", {}).get("name") or "")
                    for item in selected_nodes
                )
            ),
            artifact_sha=artifact_sha,
            config_digest=expected_config,
            runtime_profile_version=expected_profile,
            bundle_sha=expected_bundle,
            template_sha=expected_template,
            agent_identity=agent_identity,
        ):
            return
        if release.runner.dry_run:
            return
        time.sleep(5)
    raise ReleaseError(f"{target.cluster_id} agents did not converge")


def agent_heartbeats_converged(
    release: Any,
    target: ClusterTarget,
    *,
    node_count: int,
    node_names: tuple[str, ...],
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str,
    bundle_sha: str | None,
    template_sha: str | None,
    agent_identity: dict[str, Any] | None = None,
) -> bool:
    if release.runner.dry_run:
        return True
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
    script = """
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext

expected = json.load(sys.stdin)
expected_nodes = set(expected["node_names"])
now = datetime.now(timezone.utc)
agents = [
    item
    for item in ApplicationContext.from_environment().store.list_agents(
        expected["cluster_id"]
    )
    if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
    and item.lease_expires_at is not None
    and item.lease_expires_at > now
    and (not expected_nodes or item.node_id in expected_nodes)
]
aligned = [
    item
    for item in agents
    if item.artifact_sha256 == expected["artifact"]
    and (
        expected["protocol"] is None
        or item.agent_protocol_version == expected["protocol"]
    )
    and (
        expected["version"] is None
        or item.agent_version == expected["version"]
    )
    and (
        expected["compatibility"] is None
        or (item.compatibility_digest or item.artifact_sha256)
        == expected["compatibility"]
    )
    and (
        expected["policy"] is None
        or item.policy_version == expected["policy"]
    )
    and item.config_digest == expected["config"]
    and item.runtime_profile_version == expected["profile"]
    and (
        expected["key_version"] is None
        or item.node_action_key_version == expected["key_version"]
    )
    and (
        expected["bundle"] is None
        or item.installer_bundle_sha256 == expected["bundle"]
    )
    and (
        expected["template"] is None
        or item.installer_template_sha256 == expected["template"]
    )
]
print(json.dumps({"active": len(agents), "aligned": len(aligned)}))
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
        input_text=json.dumps(
            {
                "cluster_id": target.cluster_id,
                "node_names": list(node_names),
                "artifact": artifact_sha,
                "config": config_digest,
                "profile": runtime_profile_version,
                "bundle": bundle_sha,
                "template": template_sha,
                "protocol": (agent_identity or {}).get("agent_protocol_version"),
                "version": (agent_identity or {}).get("agent_version"),
                "compatibility": (agent_identity or {}).get("compatibility_digest"),
                "policy": (agent_identity or {}).get("policy_version"),
                "key_version": (agent_identity or {}).get("node_action_key_version"),
            }
        ),
        capture=True,
    )
    result = json.loads(raw)
    return (
        node_count > 0
        and int(result.get("active", 0)) == node_count
        and int(result.get("aligned", 0)) == node_count
    )
