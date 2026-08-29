from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import regional_deployment_inventory as inventory
from regional_notifications import notification_digest
from regional_release_config import ClusterTarget, ReleaseConfig, ReleaseError
from regional_release_diff import (
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    build_execution_plan,
)
from regional_release_legacy import (
    rollback_controller_config,
    validate_rollback_agent_identity,
)
from regional_runtime_profile import ensure_runtime_profile


ROOT = Path(__file__).resolve().parents[3]
NON_TRANSACTIONAL_CHANGES = frozenset(
    {
        "endpoint",
        "endpoint_manifests",
        "clusters",
        "observability_manifests",
        "adot_image",
    }
)


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
) -> dict[str, str]:
    legacy_component_pins = not any(
        metadata.get(name)
        for name in (
            "required-agent-compatibility-digest",
            "required-regional-executor-artifact-sha256",
            "required-regional-executor-compatibility-digest",
        )
    )
    return {
        **os.environ,
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
        "GPU_FAULT_LEGACY_COMPONENT_PINS": str(legacy_component_pins).lower(),
        "GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS": str(preserve_role_config_maps).lower(),
        "GPU_FAULT_FORCE_ROLE_RESTART": "true",
        "GPU_FAULT_FINALIZE_AGENT_PIN": "true",
        "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": "true",
    }


def _default_release_diff() -> ReleaseDiff:
    return ReleaseDiff(
        kind=ReleaseChangeKind.FULL,
        changed=frozenset(
            {
                "control_plane_wheel",
                "executor_wheel",
                "node_runtime_wheel",
                "node_bundle",
                "database_schema",
                "agent_protocol",
                "executor_protocol",
                "agent_config",
                "runtime_profile",
                "endpoint",
                "dcgm",
            }
        ),
    )


def _validate_upgrade_transaction(
    self: Any,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
) -> None:
    if (
        "database_schema" in diff.changed
        and self.config.auto_rollback
        and not self.config.schema_rollback_compatible
    ):
        raise ReleaseError(
            "automatic rollback across this PostgreSQL schema change "
            "is not declared backward-compatible"
        )
    non_transactional = diff.changed.intersection(NON_TRANSACTIONAL_CHANGES)
    if self.config.auto_rollback and non_transactional:
        raise ReleaseError(
            "automatic rollback is not yet transactional for: "
            + ", ".join(sorted(non_transactional))
        )


def _upgrade_context(
    self: Any,
    *,
    resume: bool,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
) -> tuple[dict[str, Any], set[str], set[str], bool]:
    loaded = self._load_state() if resume else {}
    if resume:
        if loaded.get("release_id") != self.release_id:
            raise ReleaseError("resume release_id does not match the candidate")
        if loaded.get("release_diff") != diff.as_dict():
            raise ReleaseError("resume release diff does not match the candidate")
        if loaded.get("execution_plan") != plan.as_dict():
            raise ReleaseError("resume execution plan does not match the candidate")
        self.state = dict(loaded)
        previous = loaded.get("previous")
        completed_phases = set(loaded.get("completed_phases") or [])
        completed_clusters = set(loaded.get("completed_cluster_ids") or [])
        registry_staged = bool(loaded.get("registry_staged", False))
    else:
        previous = self._capture_previous()
        previous["secret_backups"] = self._backup_release_secrets()
        completed_phases = set()
        completed_clusters = set()
        registry_staged = False
    if not isinstance(previous, dict) or not previous:
        raise ReleaseError("previous release state is unavailable")
    return (
        previous,
        completed_phases,
        completed_clusters,
        registry_staged,
    )


def _upgrade_gpu_clusters(
    self: Any,
    *,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> None:
    if not plan.has(
        ReleaseComponent.ENDPOINT,
        ReleaseComponent.DCGM,
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.WATCHER,
        ReleaseComponent.COLLECTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    ):
        return
    workers = min(4, max(1, len(self.config.clusters)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                self._upgrade_gpu_target,
                target,
                diff,
                plan,
            ): target.cluster_id
            for target in self.config.clusters
            if target.cluster_id not in completed_clusters
        }
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise ReleaseError(f"{cluster_id} rollout failed: {exc}") from exc
            completed_clusters.add(cluster_id)
            self._save_state(
                "data-plane-progress",
                previous=previous,
                release_diff=diff.as_dict(),
                execution_plan=plan.as_dict(),
                completed_phases=sorted(completed_phases),
                completed_cluster_ids=sorted(completed_clusters),
                registry_staged=registry_staged,
            )


def run_upgrade_phases(
    self: Any,
    *,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> bool:
    def checkpoint(phase: str, **updates: Any) -> None:
        completed_phases.add(phase)
        self._save_state(
            phase,
            previous=previous,
            release_diff=diff.as_dict(),
            execution_plan=plan.as_dict(),
            completed_phases=sorted(completed_phases),
            completed_cluster_ids=sorted(completed_clusters),
            registry_staged=registry_staged,
            **updates,
        )

    if "uploaded" not in completed_phases:
        self._upload_release(diff)
        checkpoint("uploaded")
    if plan.has(ReleaseComponent.SCHEMA) and ("schema-ready" not in completed_phases):
        self._ensure_schema()
    if "schema-ready" not in completed_phases:
        checkpoint("schema-ready")
    if plan.has(ReleaseComponent.REGISTRY) and (
        "registry-staged" not in completed_phases
    ):
        registry_staged = self._stage_registry()
        checkpoint("registry-staged")
    if plan.has(ReleaseComponent.CPU_STAGE) and ("cpu-staged" not in completed_phases):
        self._apply_cpu(finalize=False, force_restart=registry_staged)
        checkpoint("cpu-staged")
    if plan.has(ReleaseComponent.RUNTIME_PROFILE) and (
        "profile-ready" not in completed_phases
    ):
        ensure_runtime_profile(self)
        checkpoint("profile-ready")
    if plan.has(ReleaseComponent.ENDPOINT) and (
        "endpoint-ready" not in completed_phases
    ):
        self._apply_nlb()
        checkpoint("endpoint-ready")
    if plan.has(ReleaseComponent.OBSERVABILITY) and (
        "observability-ready" not in completed_phases
    ):
        self._apply_observability()
        checkpoint("observability-ready")
    _upgrade_gpu_clusters(
        self,
        diff=diff,
        plan=plan,
        previous=previous,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
        registry_staged=registry_staged,
    )
    if "data-converged" not in completed_phases:
        checkpoint("data-converged")
    if plan.has(ReleaseComponent.CPU_FINALIZE) and (
        "cpu-finalized" not in completed_phases
    ):
        if plan.has(ReleaseComponent.RUNTIME_PROFILE):
            self._ensure_profile_transition_safe(
                previous.get("runtime_profile_version")
            )
        self._apply_cpu(finalize=True)
        checkpoint("cpu-finalized")
    if "verified" not in completed_phases:
        self._validate_release_quick(plan)
        checkpoint("verified")
    if registry_staged:
        self._commit_registry_update()
    checkpoint("complete")
    return registry_staged


def _record_upgrade_failure(
    self: Any,
    *,
    error: Exception,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> None:
    self._save_state(
        "failed",
        previous=previous,
        release_diff=diff.as_dict(),
        execution_plan=plan.as_dict(),
        completed_phases=sorted(completed_phases),
        completed_cluster_ids=sorted(completed_clusters),
        registry_staged=registry_staged,
        original_failure=f"{type(error).__name__}: {error}",
    )


def upgrade_release(
    self: Any,
    *,
    resume: bool = False,
    diff: ReleaseDiff | None = None,
) -> None:
    self._ensure_contexts()
    self._require_cpu_secrets()
    if not self._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    active_diff = diff or _default_release_diff()
    plan = build_execution_plan(active_diff)
    _validate_upgrade_transaction(self, active_diff, plan)
    (
        previous,
        completed_phases,
        completed_clusters,
        registry_staged,
    ) = _upgrade_context(
        self,
        resume=resume,
        diff=active_diff,
        plan=plan,
    )
    if not resume:
        self._save_state(
            "preflight",
            previous=previous,
            release_diff=active_diff.as_dict(),
            execution_plan=plan.as_dict(),
            completed_phases=[],
            completed_cluster_ids=[],
            registry_staged=False,
        )
    try:
        run_upgrade_phases(
            self,
            diff=active_diff,
            plan=plan,
            previous=previous,
            completed_phases=completed_phases,
            completed_clusters=completed_clusters,
            registry_staged=registry_staged,
        )
    except Exception as upgrade_error:
        _record_upgrade_failure(
            self,
            error=upgrade_error,
            diff=active_diff,
            plan=plan,
            previous=previous,
            completed_phases=completed_phases,
            completed_clusters=completed_clusters,
            registry_staged=registry_staged,
        )
        if self.config.auto_rollback:
            try:
                self.rollback(state=previous)
            except Exception as rollback_error:
                self._save_state(
                    "rollback-failed",
                    previous=previous,
                    release_diff=active_diff.as_dict(),
                    execution_plan=plan.as_dict(),
                    original_failure=(
                        f"{type(upgrade_error).__name__}: {upgrade_error}"
                    ),
                    rollback_failure=(
                        f"{type(rollback_error).__name__}: {rollback_error}"
                    ),
                )
                raise ReleaseError(
                    "release upgrade failed and automatic rollback also failed: "
                    f"upgrade={type(upgrade_error).__name__}: {upgrade_error}; "
                    f"rollback={type(rollback_error).__name__}: {rollback_error}"
                ) from upgrade_error
        raise


def _rollback_context(
    self: Any,
    state: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = dict(self.state)
    if state is None:
        loaded = self._load_state()
    self.state = dict(loaded)
    previous = state or loaded.get("previous")
    if not previous:
        if loaded.get("phase") in {
            "bootstrap-started",
            "bootstrap-failed",
        }:
            self._cleanup_bootstrap()
            return loaded, {}
        raise ReleaseError("previous release state is unavailable")
    changed = set(((loaded.get("release_diff") or {}).get("changed") or []))
    unsupported = changed.intersection(NON_TRANSACTIONAL_CHANGES)
    if unsupported:
        raise ReleaseError(
            "rollback is not transactional for: " + ", ".join(sorted(unsupported))
        )
    if "database_schema" in changed and not self.config.schema_rollback_compatible:
        raise ReleaseError(
            "rollback across this PostgreSQL schema change is not "
            "declared backward-compatible"
        )
    return loaded, previous


def _restore_rollback_cpu(
    self: Any,
    *,
    previous: dict[str, Any],
    metadata: dict[str, Any],
    cpu_wheel: str,
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
    runtime_image: str,
) -> None:
    cpu_secret = (previous.get("secret_backups") or {}).get("cpu") or {}
    if cpu_secret.get("backup"):
        self._restore_secret(
            self._cpu(),
            source=str(cpu_secret["source"]),
            backup=str(cpu_secret["backup"]),
        )
    self._restore_registry_backup()
    preserve_role_config_maps = self._restore_cpu_role_config_maps(
        previous.get("cpu_role_config_maps")
    )
    cpu_sha = self._config_map_sha(
        self._cpu(),
        cpu_wheel,
        self.config.wheel.name,
    )
    environment = build_rollback_environment(
        rollback_config=self.config.for_rollback(config_digest),
        metadata=metadata,
        cpu_wheel=cpu_wheel,
        cpu_sha=cpu_sha,
        artifact=artifact,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        runtime_image=runtime_image,
        preserve_role_config_maps=preserve_role_config_maps,
    )
    self.runner.run(
        [
            "bash",
            str(ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"),
        ],
        env=environment,
    )
    refresh_exists = (
        subprocess.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "cronjob",
                "gpu-fault-aurora-credential-refresh",
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if refresh_exists:
        self.runner.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "set",
                "image",
                "cronjob/gpu-fault-aurora-credential-refresh",
                f"refresh={runtime_image}",
            )
        )


def _stage_rollback_controller(
    self: Any,
    *,
    previous: dict[str, Any],
    metadata: dict[str, Any],
    cpu_wheel: str,
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
) -> None:
    cpu_secret = (previous.get("secret_backups") or {}).get("cpu") or {}
    if cpu_secret.get("backup"):
        self._restore_secret(
            self._cpu(),
            source=str(cpu_secret["source"]),
            backup=str(cpu_secret["backup"]),
        )
    self._restore_registry_backup()
    controller_config = rollback_controller_config(
        previous.get("agent_identities") or {}
    )
    patch = json.dumps({"data": controller_config}, sort_keys=True)
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        self.runner.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "patch",
                "configmap",
                f"{deployment}-config-core",
                "--type=merge",
                "-p",
                patch,
            )
        )
    cpu_sha = self._config_map_sha(
        self._cpu(),
        cpu_wheel,
        self.config.wheel.name,
    )
    environment = build_rollback_environment(
        rollback_config=self.config.for_rollback(config_digest),
        metadata=metadata,
        cpu_wheel=cpu_wheel,
        cpu_sha=cpu_sha,
        artifact=artifact,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        runtime_image=self.runtime_image,
        preserve_role_config_maps=True,
    )
    self.runner.run(
        [
            "bash",
            str(ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"),
        ],
        env=environment,
    )


def rollback_target(
    self: Any,
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
) -> None:
    old = (previous.get("clusters") or {}).get(target.cluster_id, {})
    secret = ((previous.get("secret_backups") or {}).get("clusters") or {}).get(
        target.cluster_id
    ) or {}
    if secret.get("backup"):
        self._restore_secret(
            self._gpu(target),
            source=str(secret["source"]),
            backup=str(secret["backup"]),
        )
    wheel = old.get("wheel")
    if wheel:
        rollback_artifact = executor_artifact or self._config_map_sha(
            self._gpu(target),
            wheel,
            old.get("wheel_key") or self.config.executor_wheel.name,
        )
        self._verify_gpu_control_plane_endpoint(target)
        if old.get("dcgm_image"):
            self._apply_gpu_dcgm_exporter(
                target,
                image=old["dcgm_image"],
            )
        else:
            self._apply_gpu_dcgm_exporter(target)
        self._apply_gpu_deployments(
            target,
            wheel,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("wheel_key"),
            executor_artifact_sha=rollback_artifact,
            executor_compatibility_digest=(executor_compatibility or rollback_artifact),
            runtime_image=runtime_image,
        )
    if all(old.get(name) for name in ("reconciler_wheel", "bundle")):
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
            runtime_image=self.runtime_image,
            steady_runtime_image=runtime_image,
            steady_template_config_map=old.get("template"),
            node_installer_image=node_installer_image,
            allow_legacy_identity=legacy_agent_identity,
            agent_identity=agent_identity,
        )


def _rollback_gpu_clusters(
    self: Any,
    *,
    previous: dict[str, Any],
    loaded: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    target_arguments: dict[str, Any],
) -> None:
    workers = min(4, max(1, len(self.config.clusters)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                rollback_target,
                self,
                target,
                previous=previous,
                **target_arguments,
            ): target.cluster_id
            for target in self.config.clusters
            if target.cluster_id not in completed_clusters
        }
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise ReleaseError(f"{cluster_id} rollback failed: {exc}") from exc
            completed_clusters.add(cluster_id)
            self._save_state(
                "rollback-data-progress",
                previous=previous,
                rollback_completed_phases=sorted(completed_phases),
                rollback_completed_cluster_ids=sorted(completed_clusters),
                original_failure=loaded.get("original_failure"),
            )


def rollback_release(
    self: Any,
    *,
    state: dict[str, Any] | None = None,
) -> None:
    loaded, previous = _rollback_context(self, state)
    if not previous:
        return
    metadata = previous.get("metadata") or {}
    agent_identities = previous.get("agent_identities") or {}
    missing_agent_identities = sorted(
        target.cluster_id
        for target in self.config.clusters
        if target.cluster_id not in agent_identities
    )
    if missing_agent_identities:
        raise ReleaseError(
            "previous Agent identities are missing for: "
            + ", ".join(missing_agent_identities)
        )
    cpu_wheel = previous.get("cpu_wheel")
    artifact = metadata.get("required-agent-artifact-sha256")
    config_digest = metadata.get("required-agent-config-digest")
    if not all((cpu_wheel, artifact, config_digest)):
        raise ReleaseError("previous release pins are incomplete")
    profile = previous.get("runtime_profile_version") or "hyperpod-v1"
    executor_artifact = metadata.get(
        "required-regional-executor-artifact-sha256",
        "",
    )
    executor_compatibility = (
        metadata.get("required-regional-executor-compatibility-digest")
        or executor_artifact
    )
    runtime_image = previous.get("runtime_image") or self.runtime_image
    completed_phases = set(loaded.get("rollback_completed_phases") or [])
    completed_clusters = set(loaded.get("rollback_completed_cluster_ids") or [])
    if (
        loaded.get("phase") == "rollback-data-restored"
        and "rollback-verified" not in completed_phases
    ):
        completed_phases.discard("rollback-data-restored")
        completed_clusters.clear()

    def checkpoint(phase: str) -> None:
        completed_phases.add(phase)
        self._save_state(
            phase,
            previous=previous,
            rollback_completed_phases=sorted(completed_phases),
            rollback_completed_cluster_ids=sorted(completed_clusters),
            original_failure=loaded.get("original_failure"),
        )

    if "rollback-controller-staged" not in completed_phases:
        _stage_rollback_controller(
            self,
            previous=previous,
            metadata=metadata,
            cpu_wheel=cpu_wheel,
            artifact=artifact,
            config_digest=config_digest,
            runtime_profile_version=profile,
        )
        checkpoint("rollback-controller-staged")
    if "rollback-data-restored" not in completed_phases:
        _rollback_gpu_clusters(
            self,
            previous=previous,
            loaded=loaded,
            completed_phases=completed_phases,
            completed_clusters=completed_clusters,
            target_arguments={
                "artifact": artifact,
                "config_digest": config_digest,
                "runtime_profile_version": profile,
                "executor_artifact": executor_artifact,
                "executor_compatibility": executor_compatibility,
                "node_compatibility": (
                    metadata.get("required-agent-compatibility-digest") or artifact
                ),
                "runtime_image": runtime_image,
                "node_installer_image": (
                    previous.get("node_installer_image") or self.node_installer_image
                ),
            },
        )
        checkpoint("rollback-data-restored")
    if "rollback-cpu-restored" not in completed_phases:
        _restore_rollback_cpu(
            self,
            previous=previous,
            metadata=metadata,
            cpu_wheel=cpu_wheel,
            artifact=artifact,
            config_digest=config_digest,
            runtime_profile_version=profile,
            runtime_image=runtime_image,
        )
        checkpoint("rollback-cpu-restored")
    self._validate_rollback(previous)
    checkpoint("rollback-verified")
    self._delete_release_secret_backups(previous)
    self._save_state(
        "rolled-back",
        previous=previous,
        rollback_completed_phases=sorted(completed_phases),
        rollback_completed_cluster_ids=sorted(completed_clusters),
        original_failure=loaded.get("original_failure"),
        rollback_result={"status": "PASSED"},
    )


def commit_release(self: Any) -> None:
    loaded = self._load_state()
    if loaded.get("phase") != "complete":
        raise ReleaseError("only a complete release can be committed")
    previous = loaded.get("previous")
    if isinstance(previous, dict):
        self._delete_release_secret_backups(previous)
    self._save_state(
        "complete",
        transaction_committed=True,
        previous=previous,
        release_diff=loaded.get("release_diff"),
        execution_plan=loaded.get("execution_plan"),
        completed_phases=loaded.get("completed_phases", []),
        completed_cluster_ids=loaded.get("completed_cluster_ids", []),
    )
