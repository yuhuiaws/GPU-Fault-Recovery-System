"""Compensate only the selected components of one GPU cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpu_fault.node_installer_rendering import (
    load_installer_template,
    manifest_object,
    validate_node_dependencies,
)
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent
from gpu_fault_release.regional_release_images import (
    NodeDependencyTarget,
    previous_node_dependency_environment,
    previous_node_template_environment,
)
from gpu_fault_release.regional_release_legacy import validate_rollback_agent_identity

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease


def validate_rollback_node_template(
    release: RegionalRelease,
    target: ClusterTarget,
    *,
    previous: dict[str, Any],
) -> None:
    """Prove the old Job before rollback mutates any component of this cluster."""
    old = (previous.get("clusters") or {}).get(target.cluster_id) or {}
    dependencies = previous_node_dependency_environment(
        previous,
        cluster_id=target.cluster_id,
        bundle_cm=old.get("bundle") or "",
        bundle_sha256=old.get("bundle_sha256"),
    )
    template_identity = previous_node_template_environment(
        previous,
        cluster_id=target.cluster_id,
        template_config_map=old.get("template") or "",
        template_sha256=old.get("template_sha256"),
    )
    if release.runner.dry_run:
        return
    template_value = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            template_identity["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"],
        )
    )
    try:
        text = manifest_object(
            template_value.get("data"), "rollback template ConfigMap data"
        ).get("job.yaml")
        if not isinstance(text, str):
            raise ValueError("rollback template ConfigMap has no job.yaml")
        job = load_installer_template(
            text,
            expected_sha256=template_identity[
                "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"
            ],
            origin="previous installer Job",
        )
        validate_node_dependencies(
            job,
            image=dependencies["GPU_FAULT_NODE_DEPENDENCY_IMAGE"],
            inventory_sha256=dependencies["GPU_FAULT_NODE_WHEELHOUSE_SHA256"],
        )
    except (RuntimeError, ValueError) as exc:
        raise ReleaseError(
            f"{target.cluster_id} rollback template identity is invalid: {exc}"
        ) from exc


def rollback_target(
    self: RegionalRelease,
    target: ClusterTarget,
    *,
    previous: dict[str, Any],
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
    executor_artifact: str,
    executor_compatibility: str,
    node_compatibility: str,
    runtime_image: str,
    node_installer_image: str,
    components: frozenset[ReleaseComponent],
) -> None:
    old = (previous.get("clusters") or {}).get(target.cluster_id) or {}
    if components & {ReleaseComponent.AGENT, ReleaseComponent.RECONCILER}:
        validate_rollback_node_template(self, target, previous=previous)
    if ReleaseComponent.ENDPOINT in components:
        secret = ((previous.get("secret_backups") or {}).get("clusters") or {}).get(
            target.cluster_id
        ) or {}
        if not secret.get("backup") or not secret.get("source"):
            raise ReleaseError(
                f"{target.cluster_id} rollback connection Secret backup is missing"
            )
        self._restore_secret(
            self._gpu(target),
            source=str(secret["source"]),
            backup=str(secret["backup"]),
        )
    wheel = old.get("wheel")
    if ReleaseComponent.ENDPOINT in components:
        self._verify_gpu_control_plane_endpoint(target)
    if ReleaseComponent.DCGM in components:
        if not old.get("dcgm_image"):
            raise ReleaseError(
                f"{target.cluster_id} previous DCGM image is unavailable"
            )
        self._apply_gpu_dcgm_exporter(target, image=old["dcgm_image"])
    deployment_names = {
        deployment
        for component, deployment in (
            (ReleaseComponent.EXECUTOR, inventory.GPU_EXECUTOR_DEPLOYMENT),
            (ReleaseComponent.WATCHER, inventory.GPU_WATCHER_DEPLOYMENT),
            (ReleaseComponent.COLLECTOR, inventory.GPU_COLLECTOR_DEPLOYMENT),
        )
        if component in components
    }
    if deployment_names:
        if not wheel:
            raise ReleaseError(
                f"{target.cluster_id} previous Executor wheel is unavailable"
            )
        rollback_artifact = executor_artifact or self._config_map_sha(
            self._gpu(target),
            wheel,
            old.get("wheel_key") or self.config.executor_wheel.name,
        )
        self._apply_gpu_deployments(
            target,
            wheel,
            deployment_names=frozenset(deployment_names),
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("wheel_key"),
            executor_artifact_sha=rollback_artifact,
            executor_compatibility_digest=executor_compatibility or rollback_artifact,
            runtime_image=runtime_image,
        )
    if ReleaseComponent.AGENT in components:
        missing = [name for name in ("reconciler_wheel", "bundle") if not old.get(name)]
        if missing:
            raise ReleaseError(
                f"{target.cluster_id} previous Agent rollback resources are missing: "
                + ", ".join(missing)
            )
        agent_identity = (previous.get("agent_identities") or {}).get(
            target.cluster_id
        ) or {}
        legacy_agent_identity = validate_rollback_agent_identity(
            target.cluster_id,
            agent_identity,
            metadata=previous.get("metadata") or {},
            artifact=artifact,
            compatibility=node_compatibility,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
        )
        self._roll_node_runtime(
            target,
            phase="rollback",
            wheel_cm=old["reconciler_wheel"],
            bundle_cm=old["bundle"],
            artifact_sha=artifact,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("reconciler_wheel_key"),
            node_compatibility_digest=node_compatibility,
            bundle_sha256=old.get("bundle_sha256"),
            template_sha256=old.get("template_sha256"),
            template_config_map=None,
            runtime_image=self.executor_image,
            steady_runtime_image=runtime_image,
            steady_template_config_map=old.get("template"),
            node_installer_image=node_installer_image,
            allow_legacy_identity=legacy_agent_identity,
            agent_identity=agent_identity,
        )
    elif ReleaseComponent.RECONCILER in components:
        missing = [name for name in ("reconciler_wheel", "bundle") if not old.get(name)]
        if missing:
            raise ReleaseError(
                f"{target.cluster_id} previous Reconciler resources are missing: "
                + ", ".join(missing)
            )
        self._deploy_reconciler(
            target,
            wheel_cm=old["reconciler_wheel"],
            bundle_cm=old["bundle"],
            artifact_sha=artifact,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("reconciler_wheel_key"),
            node_compatibility_digest=node_compatibility,
            bundle_sha256=old.get("bundle_sha256"),
            template_sha256=old.get("template_sha256"),
            template_config_map=old.get("template"),
            allowed_node_names=None,
            runtime_image=runtime_image,
            node_installer_image=node_installer_image,
            node_dependency_target=NodeDependencyTarget.PREVIOUS,
        )
