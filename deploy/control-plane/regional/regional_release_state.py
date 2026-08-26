from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
import yaml
from regional_release_config import ClusterTarget, ReleaseError

STATE_CONFIG_MAP = "gpu-fault-regional-release-state"


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


def capture_previous(release: Any) -> dict[str, Any]:
    metadata = release._config_map_data("gpu-fault-release-metadata")
    clusters = {}
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
            "bundle": (
                release._template_bundle(target, template) if template else None
            ),
        }
    return {
        "metadata": metadata,
        "runtime_profile_version": release._config_map_data(
            "gpu-fault-api-ha-config-core"
        ).get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"),
        "cpu_wheel": release._deployment_wheel(
            release._cpu(),
            inventory.CPU_INGRESS_DEPLOYMENT,
        ),
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
