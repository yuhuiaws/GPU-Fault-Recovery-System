from __future__ import annotations

import json
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError
from regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from regional_release_gpu_rollout import agents_converged, gpu_node_items
from regional_release_probes import probe_source
from regional_release_runtime_identity import CONTROL_PLANE_PYTHON
from regional_release_validation import validate_gpu_rollback_target

GPU_COMPONENTS = frozenset(
    {
        ReleaseComponent.ENDPOINT,
        ReleaseComponent.DCGM,
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.WATCHER,
        ReleaseComponent.COLLECTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    }
)
DEPLOYMENT_COMPONENTS = (
    (ReleaseComponent.EXECUTOR, inventory.GPU_EXECUTOR_DEPLOYMENT),
    (ReleaseComponent.WATCHER, inventory.GPU_WATCHER_DEPLOYMENT),
    (ReleaseComponent.COLLECTOR, inventory.GPU_COLLECTOR_DEPLOYMENT),
)


def _csv_values(value: object) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _require_pin_window(
    metadata: dict[str, Any],
    *,
    required_key: str,
    compatible_key: str,
    previous: str,
    candidate: str,
    finalized: bool,
) -> None:
    required = str(metadata.get(required_key) or "")
    compatible = _csv_values(metadata.get(compatible_key))
    if finalized:
        if required != candidate or compatible:
            raise ReleaseError(f"resume finalized pin drifted: {required_key}")
        return
    if candidate and required == candidate and not compatible:
        # The finalize apply promotes the window and the ``cpu-finalized``
        # checkpoint is only written afterwards, so anything that interrupts
        # finalize in between -- a failed fleet barrier, a killed process --
        # leaves the window fully promoted with the phase unrecorded. That is
        # mid-finalize, not drift: it is byte-for-byte the state the finalized
        # branch above accepts, and only this release's own finalize apply can
        # produce it, because ``candidate`` is this release's digest. Rejecting
        # it wedges the transaction, since resume is then refused while
        # rollback has already been ruled out by the promotion itself.
        return
    if previous and required != previous:
        raise ReleaseError(f"resume staged required pin drifted: {required_key}")
    if candidate and candidate not in ({required} | compatible):
        raise ReleaseError(
            f"resume staged compatibility window lost candidate: {required_key}"
        )


def _validate_cpu_pin_window(
    release: Any,
    loaded: dict[str, Any],
    previous: dict[str, Any],
) -> None:
    metadata = release._config_map_data("gpu-fault-release-metadata")
    old = previous.get("metadata") or {}
    finalized = "cpu-finalized" in set(loaded.get("completed_phases") or [])
    previous_agent_artifact = str(old.get("required-agent-artifact-sha256") or "")
    previous_agent_compatibility = str(
        old.get("required-agent-compatibility-digest") or previous_agent_artifact
    )
    previous_agent_protocol = str(
        old.get("required-agent-protocol-version")
        or release.config.agent_protocol_version
    )
    previous_agent_config = str(old.get("required-agent-config-digest") or "")
    candidate_agent_compatibility = str(
        release.config.component_digests.get("node_runtime") or release.node_wheel_sha
    )
    previous_executor_artifact = str(
        old.get("required-regional-executor-artifact-sha256") or ""
    )
    previous_executor_compatibility = str(
        old.get("required-regional-executor-compatibility-digest")
        or previous_executor_artifact
    )
    previous_executor_protocol = str(
        old.get("required-regional-executor-protocol-version")
        or release.config.executor_protocol_version
    )
    candidate_executor_compatibility = str(
        release.config.component_digests.get("executor") or release.executor_wheel_sha
    )
    for required_key, compatible_key, previous_value, candidate_value in (
        (
            "required-agent-artifact-sha256",
            "compatible-agent-artifact-sha256s",
            previous_agent_artifact,
            release.node_wheel_sha,
        ),
        (
            "required-agent-compatibility-digest",
            "compatible-agent-compatibility-digests",
            previous_agent_compatibility,
            candidate_agent_compatibility,
        ),
        (
            "required-agent-protocol-version",
            "compatible-agent-protocol-versions",
            previous_agent_protocol,
            str(release.config.agent_protocol_version),
        ),
        (
            "required-agent-config-digest",
            "compatible-agent-config-digests",
            previous_agent_config,
            release.config.agent_config_digest,
        ),
        (
            "required-regional-executor-artifact-sha256",
            "compatible-regional-executor-artifact-sha256s",
            previous_executor_artifact,
            release.executor_wheel_sha,
        ),
        (
            "required-regional-executor-compatibility-digest",
            "compatible-regional-executor-compatibility-digests",
            previous_executor_compatibility,
            candidate_executor_compatibility,
        ),
        (
            "required-regional-executor-protocol-version",
            "compatible-regional-executor-protocol-versions",
            previous_executor_protocol,
            str(release.config.executor_protocol_version),
        ),
    ):
        _require_pin_window(
            metadata,
            required_key=required_key,
            compatible_key=compatible_key,
            previous=previous_value,
            candidate=candidate_value,
            finalized=finalized,
        )


def _container_image(deployment: dict[str, Any], container_name: str) -> str:
    containers = (
        deployment.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    selected = next(
        (item for item in containers if item.get("name") == container_name),
        containers[0] if containers else {},
    )
    return str(selected.get("image") or "")


def _require_ready_deployment(
    release: Any,
    target: Any,
    deployment_name: str,
    *,
    expected_wheel: str,
) -> dict[str, Any]:
    if (
        release._deployment_wheel(
            release._gpu(target),
            deployment_name,
        )
        != expected_wheel
    ):
        raise ReleaseError(
            f"{target.cluster_id} resume {deployment_name} wheel drifted"
        )
    deployment = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment_name,
        )
    )
    desired = int((deployment.get("spec") or {}).get("replicas") or 0)
    ready = int((deployment.get("status") or {}).get("readyReplicas") or 0)
    if desired < 1 or ready != desired:
        raise ReleaseError(
            f"{target.cluster_id} resume {deployment_name} is not fully Ready"
        )
    if _container_image(deployment, deployment_name) != release.runtime_image:
        raise ReleaseError(
            f"{target.cluster_id} resume {deployment_name} image drifted"
        )
    return deployment


def _validate_candidate_components(
    release: Any,
    target: Any,
    components: frozenset[ReleaseComponent],
    node_names: tuple[str, ...],
) -> None:
    for component, deployment_name in DEPLOYMENT_COMPONENTS:
        if component in components:
            _require_ready_deployment(
                release,
                target,
                deployment_name,
                expected_wheel=release.executor_wheel_cm,
            )
    if components.intersection({ReleaseComponent.RECONCILER, ReleaseComponent.AGENT}):
        deployment = _require_ready_deployment(
            release,
            target,
            inventory.GPU_RECONCILER_DEPLOYMENT,
            expected_wheel=release.executor_wheel_cm,
        )
        containers = (
            deployment.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        reconciler = next(
            (item for item in containers if item.get("name") == "reconciler"),
            {},
        )
        environment = {
            str(item.get("name") or ""): str(item.get("value") or "")
            for item in reconciler.get("env", [])
        }
        if (
            environment.get("GPU_FAULT_INSTALLER_BUNDLE_SHA256") != release.bundle_sha
            or environment.get("GPU_FAULT_INSTALLER_TEMPLATE_SHA256")
            != release.node_template_sha
        ):
            raise ReleaseError(
                f"{target.cluster_id} resume Reconciler identity drifted"
            )
    if ReleaseComponent.DCGM in components:
        dcgm = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "daemonset",
                "gpu-fault-dcgm-exporter",
            )
        )
        if _container_image(dcgm, "dcgm-exporter") != release.dcgm_exporter_image:
            raise ReleaseError(f"{target.cluster_id} resume DCGM image drifted")
    if ReleaseComponent.ENDPOINT in components:
        release._verify_gpu_control_plane_endpoint(target)
    if ReleaseComponent.AGENT not in components:
        return
    if not agents_converged(
        gpu_node_items(release, target, fresh=True),
        target,
        release.node_wheel_sha,
        bundle_sha=release.bundle_sha,
        template_sha=release.node_template_sha,
        config_digest=release.config.agent_config_digest,
        require_node_uid=True,
        node_names=frozenset(node_names),
    ):
        raise ReleaseError(
            f"{target.cluster_id} resume candidate node annotations drifted"
        )
    if not release._agent_heartbeats_converged(
        target,
        node_count=len(node_names),
        node_names=node_names,
        artifact_sha=release.node_wheel_sha,
        config_digest=release.config.agent_config_digest,
        runtime_profile_version=release.config.runtime_profile_version,
        bundle_sha=release.bundle_sha,
        template_sha=release.node_template_sha,
    ):
        raise ReleaseError(
            f"{target.cluster_id} resume candidate Agent heartbeats drifted"
        )


def _progress_components(
    loaded: dict[str, Any],
    cluster_id: str,
    status: str,
) -> frozenset[ReleaseComponent]:
    raw = ((loaded.get("component_progress") or {}).get("clusters") or {}).get(
        cluster_id
    ) or {}
    selected = set()
    for name, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("status") != status:
            continue
        try:
            selected.add(ReleaseComponent(str(name)))
        except ValueError:
            continue
    return frozenset(selected).intersection(GPU_COMPONENTS)


def _allowed_agent_identity(
    *,
    protocol: object,
    artifact: object,
    compatibility: object,
    bundle: object,
    template: object,
    profile: object,
    config: object,
) -> dict[str, Any]:
    return {
        "protocol": int(protocol),
        "artifact": str(artifact or ""),
        "compatibility": str(compatibility or artifact or ""),
        "bundle": str(bundle) if bundle is not None else None,
        "template": str(template) if template is not None else None,
        "profile": str(profile or ""),
        "config": str(config or ""),
    }


def _validate_incomplete_agent_state(
    release: Any,
    target: Any,
    previous: dict[str, Any],
    node_names: tuple[str, ...],
    *,
    category: str,
) -> None:
    old = (previous.get("agent_identities") or {}).get(target.cluster_id) or {}
    deployment_id = release._fleet_deployment_id(
        target,
        phase="upgrade",
        artifact_sha=release.node_wheel_sha,
        bundle_sha=release.bundle_sha,
        template_sha=release.node_template_sha,
        config_digest=release.config.agent_config_digest,
        runtime_profile_version=release.config.runtime_profile_version,
    )
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
        raise ReleaseError("resume validation has no running CPU ingress Pod")
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
            probe_source("mixed_cluster_state"),
        ),
        input_text=json.dumps(
            {
                "cluster_id": target.cluster_id,
                "node_ids": list(node_names),
                "deployment_id": deployment_id,
                "allowed_identities": [
                    _allowed_agent_identity(
                        protocol=old.get("agent_protocol_version"),
                        artifact=old.get("artifact_sha256"),
                        compatibility=old.get("compatibility_digest"),
                        bundle=old.get("installer_bundle_sha256"),
                        template=old.get("installer_template_sha256"),
                        profile=old.get("runtime_profile_version"),
                        config=old.get("config_digest"),
                    ),
                    _allowed_agent_identity(
                        protocol=release.config.agent_protocol_version,
                        artifact=release.node_wheel_sha,
                        compatibility=(
                            release.config.component_digests.get("node_runtime")
                            or release.node_wheel_sha
                        ),
                        bundle=release.bundle_sha,
                        template=release.node_template_sha,
                        profile=release.config.runtime_profile_version,
                        config=release.config.agent_config_digest,
                    ),
                ],
            }
        ),
        capture=True,
    )
    result = json.loads(raw)
    if int(result.get("agent_blocker_count") or 0):
        raise ReleaseError(
            f"{target.cluster_id} resume Agent identity is outside the "
            "transaction compatibility window: "
            + json.dumps(result.get("agent_blockers") or [], sort_keys=True)
        )
    deployment = result.get("deployment")
    if not isinstance(deployment, dict):
        raise ReleaseError(f"{target.cluster_id} resume FleetDeployment is missing")
    if (
        deployment.get("cluster_id") != target.cluster_id
        or {str(item.get("node_id") or "") for item in deployment.get("nodes") or []}
        != set(node_names)
        or deployment.get("desired_artifact_sha256") != release.node_wheel_sha
        or deployment.get("desired_config_digest") != release.config.agent_config_digest
        or deployment.get("desired_bundle_sha256") != release.bundle_sha
        or deployment.get("desired_template_sha256") != release.node_template_sha
    ):
        raise ReleaseError(
            f"{target.cluster_id} resume FleetDeployment contract drifted"
        )
    statuses = {str(item.get("status") or "") for item in deployment.get("nodes") or []}
    if not statuses <= {"PENDING", "INSTALLING", "READY", "FAILED"}:
        raise ReleaseError(
            f"{target.cluster_id} resume FleetDeployment has invalid node states"
        )
    if category == "paused" and statuses.intersection({"INSTALLING", "FAILED"}):
        raise ReleaseError(
            f"{target.cluster_id} paused checkpoint has a non-terminal wave"
        )


def validate_resume_checkpoint(
    release: Any,
    *,
    loaded: dict[str, Any],
    previous: dict[str, Any],
    plan: ReleaseExecutionPlan,
) -> None:
    expected_ids = {target.cluster_id for target in release.config.clusters}
    if set(loaded.get("cluster_ids") or []) != expected_ids:
        raise ReleaseError("resume cluster membership drifted")
    if loaded.get("cluster_registry_digest") != release.cluster_registry_digest:
        raise ReleaseError("resume cluster identity digest drifted")
    attempt_states = {
        str(cluster_id): str((entry or {}).get("state") or "")
        for cluster_id, entry in (loaded.get("cluster_attempts") or {}).items()
        if isinstance(entry, dict)
    }
    completed = {
        cluster_id
        for cluster_id, state in attempt_states.items()
        if state == "CONVERGED"
    } or set(loaded.get("completed_cluster_ids") or [])
    failed = {
        cluster_id for cluster_id, state in attempt_states.items() if state == "FAILED"
    } or set(loaded.get("failed_cluster_ids") or [])
    paused = {
        cluster_id for cluster_id, state in attempt_states.items() if state == "PAUSED"
    } or set(loaded.get("paused_cluster_ids") or [])
    not_started = {
        cluster_id for cluster_id, state in attempt_states.items() if state == "PENDING"
    } or set(loaded.get("not_started_cluster_ids") or [])
    categories = (completed, failed, paused, not_started)
    if any(
        left.intersection(right)
        for index, left in enumerate(categories)
        for right in categories[index + 1 :]
    ):
        raise ReleaseError("resume cluster checkpoint categories overlap")
    if not set().union(*categories) <= expected_ids:
        raise ReleaseError("resume cluster checkpoint names an unknown cluster")
    _validate_cpu_pin_window(release, loaded, previous)
    planned_gpu = frozenset(plan.nodes).intersection(GPU_COMPONENTS)
    if not planned_gpu:
        return
    previous_runtime = str(previous.get("runtime_image") or release.runtime_image)
    for target in release.config.clusters:
        identity = (previous.get("agent_identities") or {}).get(target.cluster_id)
        if not isinstance(identity, dict):
            raise ReleaseError(
                f"{target.cluster_id} resume previous Agent identity is missing"
            )
        node_names = tuple(
            sorted(str(value) for value in identity.get("node_ids") or [])
        )
        if not node_names:
            raise ReleaseError(f"{target.cluster_id} resume previous node set is empty")
        if set(release._target_node_names(target)) != set(node_names):
            raise ReleaseError(f"{target.cluster_id} resume node set drifted")
        if target.cluster_id in completed:
            _validate_candidate_components(
                release,
                target,
                planned_gpu,
                node_names,
            )
            continue
        completed_components = _progress_components(
            loaded,
            target.cluster_id,
            "COMPLETED",
        )
        active_components = _progress_components(
            loaded,
            target.cluster_id,
            "STARTED",
        ) | _progress_components(
            loaded,
            target.cluster_id,
            "FAILED",
        )
        if target.cluster_id in not_started or (
            not completed_components and not active_components
        ):
            validate_gpu_rollback_target(
                release,
                previous,
                previous_runtime,
                target,
                components=planned_gpu,
                run_verifier=False,
            )
            continue
        if completed_components:
            _validate_candidate_components(
                release,
                target,
                completed_components,
                node_names,
            )
        previous_components = planned_gpu - completed_components - active_components
        if previous_components:
            validate_gpu_rollback_target(
                release,
                previous,
                previous_runtime,
                target,
                components=previous_components,
                run_verifier=False,
            )
        if ReleaseComponent.AGENT in active_components:
            category = (
                "paused"
                if target.cluster_id in paused
                else "failed"
                if target.cluster_id in failed
                else "incomplete"
            )
            _validate_incomplete_agent_state(
                release,
                target,
                previous,
                node_names,
                category=category,
            )
