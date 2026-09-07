from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
import yaml  # type: ignore[import-untyped,unused-ignore]
from gpu_fault_release.regional_release_config import (
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    build_execution_plan,
)
from gpu_fault_release.regional_release_rendering import render_gpu_rollout_manifests
from gpu_fault_release.regional_release_rollout_wait import wait_deployment_rollout

from gpu_fault.regional_compatibility import (
    RegionalExecutorCompatibilityPolicy,
)

ProgressSelection = ReleaseComponent | tuple[ReleaseComponent, ...]
ProgressCallback = Callable[[ProgressSelection, str, dict[str, Any] | None], None]
COMPLETION_WATCHER_STATE_CONFIG_MAP = "gpu-fault-completion-watcher-outbox"
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"
NODE_INVENTORY_ATTRIBUTE = "_gpu_node_inventory"


def gpu_node_command(release: Any, target: ClusterTarget) -> list[str]:
    """Build the one node-inventory read every release path shares.

    The HyperPod cluster label is pushed to the API server so a GPU EKS cluster
    that also hosts non-HyperPod nodes never ships them over the wire, and so
    every caller produces an identical `_get_json` cache key: inside
    `read_snapshot` the whole release reads each cluster's nodes once.
    """

    return release._gpu(
        target,
        "get",
        "nodes",
        "-l",
        f"{HYPERPOD_CLUSTER_LABEL}={target.hyperpod_cluster_name}",
    )


@contextmanager
def node_inventory_scope(release: Any):
    """Pin each cluster's node inventory for one derivation step.

    Values derived from the same node list (the fleet node set and its
    failure-domain map) must describe one observation, and re-listing every
    node once per derivation is the most expensive read in the rollout. Safety
    gates stay outside this scope and read with ``fresh=True``.
    """

    previous = getattr(release, NODE_INVENTORY_ATTRIBUTE, None)
    setattr(release, NODE_INVENTORY_ATTRIBUTE, {})
    try:
        yield
    finally:
        setattr(release, NODE_INVENTORY_ATTRIBUTE, previous)


def gpu_node_items(
    release: Any,
    target: ClusterTarget,
    *,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    # `fresh` opts out of the pinned inventory: convergence polling and safety
    # gates must observe live node state, never a value another step captured
    # seconds earlier. They also must not run inside `read_snapshot`, whose
    # cache would otherwise re-serve the first read until the deadline expires.
    cache = None if fresh else getattr(release, NODE_INVENTORY_ATTRIBUTE, None)
    if cache is not None and target.cluster_id in cache:
        return cache[target.cluster_id]
    value = release._get_json(gpu_node_command(release, target))
    items = list(value.get("items", []))
    if cache is not None:
        cache[target.cluster_id] = items
    return items


def preserve_completion_watcher_state(
    release: Any,
    target: ClusterTarget,
    text: str,
) -> str:
    if release.runner.dry_run:
        return text
    returncode, _stdout, stderr = release.runner.probe_output(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            COMPLETION_WATCHER_STATE_CONFIG_MAP,
        )
    )
    if returncode:
        if "NotFound" in stderr or "not found" in stderr:
            return text
        raise ReleaseError(
            f"{target.cluster_id} cannot inspect Completion Watcher state: "
            + stderr.strip()
        )
    documents = [
        document
        for document in yaml.safe_load_all(text)
        if not (
            isinstance(document, dict)
            and document.get("kind") == "ConfigMap"
            and (document.get("metadata") or {}).get("name")
            == COMPLETION_WATCHER_STATE_CONFIG_MAP
        )
    ]
    return yaml.safe_dump_all(documents, sort_keys=False)


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


def _gpu_deployment_manifests(
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
    require_live_pin: bool,
) -> dict[str, str]:
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or release.config.component_digests.get("executor")
        or artifact_sha
    )
    if require_live_pin:
        require_executor_pin(
            release,
            artifact_sha=artifact_sha,
            compatibility_digest=compatibility_digest,
        )
    manifests = dict(
        render_gpu_rollout_manifests(
            release,
            target,
            wheel_cm,
            deployment_names=deployment_names,
            runtime_image=runtime_image,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=executor_wheel_filename,
            executor_artifact_sha=artifact_sha,
            executor_compatibility_digest=compatibility_digest,
        )
    )
    if inventory.GPU_WATCHER_DEPLOYMENT in manifests:
        manifests[inventory.GPU_WATCHER_DEPLOYMENT] = preserve_completion_watcher_state(
            release,
            target,
            manifests[inventory.GPU_WATCHER_DEPLOYMENT],
        )
    return manifests


def preflight_gpu_deployments(
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
    manifests = _gpu_deployment_manifests(
        release,
        target,
        wheel_cm,
        deployment_names=deployment_names,
        runtime_image=runtime_image,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        executor_artifact_sha=executor_artifact_sha,
        executor_compatibility_digest=executor_compatibility_digest,
        require_live_pin=False,
    )
    for manifest in manifests.values():
        release.runner.run(
            release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
            input_text=manifest,
        )


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
    manifests = _gpu_deployment_manifests(
        release,
        target,
        wheel_cm,
        deployment_names=deployment_names,
        runtime_image=runtime_image,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        executor_artifact_sha=executor_artifact_sha,
        executor_compatibility_digest=executor_compatibility_digest,
        require_live_pin=True,
    )
    waves = []
    if inventory.GPU_EXECUTOR_DEPLOYMENT in manifests:
        waves.append((inventory.GPU_EXECUTOR_DEPLOYMENT,))
    secondary = tuple(
        deployment
        for deployment in (
            inventory.GPU_WATCHER_DEPLOYMENT,
            inventory.GPU_COLLECTOR_DEPLOYMENT,
        )
        if deployment in manifests
    )
    if secondary:
        waves.append(secondary)
    known = {deployment for wave in waves for deployment in wave}
    waves.extend((deployment,) for deployment in sorted(set(manifests) - known))
    for manifest in manifests.values():
        release.runner.run(
            release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
            input_text=manifest,
        )
    for wave in waves:
        for deployment in wave:
            manifest = manifests[deployment]
            release.runner.run(
                release._gpu(target, "apply", "-f", "-"),
                input_text=manifest,
            )
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {
                executor.submit(
                    wait_deployment_rollout,
                    release,
                    target,
                    deployment,
                ): deployment
                for deployment in wave
            }
            for future in as_completed(futures):
                deployment = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise ReleaseError(
                        f"{target.cluster_id} Deployment {deployment} rollout "
                        f"failed: {exc}"
                    ) from exc


def upgrade_gpu_target(
    release: Any,
    target: ClusterTarget,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan | None = None,
    *,
    progress: ProgressCallback | None = None,
    candidate_preflighted: bool = False,
) -> None:
    active_plan = plan or build_execution_plan(diff)

    def run_component(
        components: tuple[ReleaseComponent, ...],
        action: Callable[[], None],
    ) -> None:
        if progress is not None:
            progress(components, "STARTED", None)
        try:
            action()
        except Exception:
            if progress is not None:
                progress(components, "FAILED", None)
            raise
        if progress is not None:
            progress(components, "COMPLETED", None)

    if active_plan.has(ReleaseComponent.ENDPOINT):
        run_component(
            (ReleaseComponent.ENDPOINT,),
            lambda: (
                release._ensure_connection_secret(target),
                release._verify_gpu_control_plane_endpoint(target),
            ),
        )
    if active_plan.has(ReleaseComponent.DCGM):
        run_component(
            (ReleaseComponent.DCGM,),
            lambda: release._apply_gpu_dcgm_exporter(
                target,
            ),
        )
    deployment_components = tuple(
        (component, deployment)
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
    )
    if deployment_components:
        run_component(
            tuple(component for component, _deployment in deployment_components),
            lambda: release._apply_gpu_deployments(
                target,
                release.executor_wheel_cm,
                deployment_names=frozenset(
                    deployment for _component, deployment in deployment_components
                ),
            ),
        )
    if active_plan.has(ReleaseComponent.AGENT):
        components = (
            (ReleaseComponent.RECONCILER, ReleaseComponent.AGENT)
            if active_plan.has(ReleaseComponent.RECONCILER)
            else (ReleaseComponent.AGENT,)
        )
        mutation_started = False

        def record_mutation_started() -> None:
            nonlocal mutation_started
            if progress is not None:
                progress(components, "STARTED", None)
            mutation_started = True

        try:
            release._roll_node_runtime(
                target,
                phase="upgrade",
                wheel_cm=release.executor_wheel_cm,
                bundle_cm=release.bundle_cm,
                artifact_sha=release.node_wheel_sha,
                config_digest=release.config.agent_config_digest,
                mutation_started=record_mutation_started,
                candidate_preflight_completed=candidate_preflighted,
            )
        except Exception:
            if progress is not None and mutation_started:
                progress(components, "FAILED", None)
            raise
        if progress is not None:
            progress(components, "COMPLETED", None)
    elif active_plan.has(ReleaseComponent.RECONCILER):
        run_component(
            (ReleaseComponent.RECONCILER,),
            lambda: release._deploy_reconciler(
                target,
                wheel_cm=release.executor_wheel_cm,
                bundle_cm=release.bundle_cm,
                artifact_sha=release.node_wheel_sha,
                config_digest=release.config.agent_config_digest,
            ),
        )


def join_target(release: Any, cluster_id: str) -> ClusterTarget:
    target = release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    return target
