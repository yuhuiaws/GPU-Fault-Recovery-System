from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_fleet_rollout import (
    FleetWaveContext,
    NodeRolloutPolicy,
    ensure_rollout_wave_safe,
    finish_legacy_node_runtime_rollback,
    live_agent_node_names,
    node_rollout_policy,
    rollback_node_rollout_policy,
    run_fleet_waves,
    target_node_failure_domains,
)
from gpu_fault_release.regional_release_gpu_rollout import node_inventory_scope
from gpu_fault_release.regional_release_legacy import apply_fleet_request_identity
from gpu_fault_release.regional_release_node_preflight import (
    NodeMutationPreflight,
    ensure_node_candidate_preflight,
    ensure_pre_node_mutation_barrier,
)
from gpu_fault_release.regional_release_rollout_cleanup import (
    terminalize_stranded_cluster_rollouts,
)


def _ensure_runtime_safe(*args: Any, **kwargs: Any) -> None:
    """The barrier's safety gate: raise or return, never hand back evidence.

    ``ensure_rollout_wave_safe`` returns the snapshot it decided on so the wave
    loop can re-decide the post-lease margin from it without a second exec. The
    pre-mutation barrier has no wave lease to take, so it wants the gate and not
    the evidence.
    """

    ensure_rollout_wave_safe(*args, **kwargs)


def _create_fleet_rollout(
    release: Any,
    target: ClusterTarget,
    candidate: NodeMutationPreflight,
    policy: NodeRolloutPolicy,
    failure_domains: dict[str, str],
    paused_identity: tuple[str, str],
    *,
    allow_legacy_identity: bool,
    agent_identity: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    release._fleet_command("normalize-records", {})
    desired_bundle, desired_template = paused_identity
    fleet_bundle = None if allow_legacy_identity else desired_bundle
    fleet_template = None if allow_legacy_identity else desired_template
    deployment_id = release._fleet_deployment_id(
        target,
        phase=candidate.phase,
        artifact_sha=candidate.artifact_sha,
        bundle_sha=fleet_bundle,
        template_sha=fleet_template,
        config_digest=candidate.config_digest,
        runtime_profile_version=candidate.runtime_profile_version,
    )
    request = {
        "deployment_id": deployment_id,
        "cluster_id": target.cluster_id,
        "node_ids": list(candidate.node_names),
        "desired_artifact_sha256": candidate.artifact_sha,
        "desired_compatibility_digest": candidate.node_compatibility_digest,
        "desired_bundle_sha256": fleet_bundle,
        "desired_template_sha256": fleet_template,
        "desired_runtime_profile_version": candidate.runtime_profile_version,
        "desired_config_digest": candidate.config_digest,
        "max_unavailable": policy.max_unavailable,
        "first_wave_max_unavailable": policy.first_wave_max_unavailable,
        "max_unavailable_per_failure_domain": (
            policy.max_unavailable_per_failure_domain
        ),
        "node_failure_domains": failure_domains,
    }
    apply_fleet_request_identity(request, agent_identity)
    # This release holds the lock: any other non-terminal rollout record for
    # the cluster was abandoned by a dead or superseded transaction, and while
    # it stands the destructive-workflow fence holds the whole cluster.
    terminalize_stranded_cluster_rollouts(
        release,
        cluster_id=target.cluster_id,
        keep_deployment_id=deployment_id,
        reason=(
            f"superseded by release {release.release_id} fleet deployment "
            f"{deployment_id} before reaching a terminal state"
        ),
    )
    deployment = release._fleet_command("create", {"request": request})
    if deployment.get("status") in {"FAILED", "CANCELLED"} and (
        deployment.get("status") == "CANCELLED"
        or bool(release.state.get("partial_convergence"))
    ):
        deployment = release._fleet_command(
            "retry-failed",
            {"deployment_id": deployment_id},
        )
    return deployment_id, deployment


def _finalize_node_runtime(
    release: Any,
    target: ClusterTarget,
    candidate: NodeMutationPreflight,
    paused_identity: tuple[str, str],
    *,
    steady_runtime_image: str | None,
    steady_template_config_map: str | None,
    allow_legacy_identity: bool,
    agent_identity: dict[str, Any] | None,
) -> tuple[str, str]:
    desired_bundle, desired_template = paused_identity
    final_identity = release._deploy_reconciler(
        target,
        wheel_cm=candidate.wheel_cm,
        bundle_cm=candidate.bundle_cm,
        artifact_sha=candidate.artifact_sha,
        config_digest=candidate.config_digest,
        runtime_profile_version=candidate.runtime_profile_version,
        executor_wheel_filename=candidate.executor_wheel_filename,
        node_compatibility_digest=candidate.node_compatibility_digest,
        bundle_sha256=desired_bundle,
        template_sha256=desired_template,
        template_config_map=(
            steady_template_config_map or candidate.template_config_map
        ),
        allowed_node_names=None,
        max_unavailable=candidate.max_unavailable,
        runtime_image=steady_runtime_image or candidate.runtime_image,
        node_installer_image=candidate.node_installer_image,
    )
    if final_identity != paused_identity:
        raise ReleaseError(
            f"{target.cluster_id} steady-state installer identity changed"
        )
    release._wait_agents(
        target,
        candidate.artifact_sha,
        bundle_sha=desired_bundle,
        template_sha=desired_template,
        config_digest=candidate.config_digest,
        runtime_profile_version=candidate.runtime_profile_version,
        legacy_identity=allow_legacy_identity,
        agent_identity=agent_identity,
    )
    return final_identity


def _dry_run_node_runtime(
    release: Any,
    target: ClusterTarget,
    candidate: NodeMutationPreflight,
    *,
    allow_legacy_identity: bool,
    agent_identity: dict[str, Any] | None,
) -> tuple[str, str]:
    identity = release._deploy_reconciler(
        target,
        wheel_cm=candidate.wheel_cm,
        bundle_cm=candidate.bundle_cm,
        artifact_sha=candidate.artifact_sha,
        config_digest=candidate.config_digest,
        runtime_profile_version=candidate.runtime_profile_version,
        executor_wheel_filename=candidate.executor_wheel_filename,
        node_compatibility_digest=candidate.node_compatibility_digest,
        bundle_sha256=candidate.bundle_sha256,
        template_sha256=candidate.template_sha256,
        template_config_map=candidate.template_config_map,
        runtime_image=candidate.runtime_image,
        node_installer_image=candidate.node_installer_image,
    )
    release._wait_agents(
        target,
        candidate.artifact_sha,
        bundle_sha=identity[0],
        template_sha=identity[1],
        config_digest=candidate.config_digest,
        runtime_profile_version=candidate.runtime_profile_version,
        legacy_identity=allow_legacy_identity,
        agent_identity=agent_identity,
    )
    return identity


def _node_runtime_candidate(
    release: Any,
    target: ClusterTarget,
    *,
    phase: str,
    wheel_cm: str,
    bundle_cm: str,
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str | None,
    executor_wheel_filename: str | None,
    node_compatibility_digest: str | None,
    bundle_sha256: str | None,
    template_sha256: str | None,
    template_config_map: str | None,
    runtime_image: str | None,
    node_installer_image: str | None,
) -> tuple[NodeMutationPreflight, NodeRolloutPolicy, dict[str, str]]:
    expected_profile = runtime_profile_version or release.config.runtime_profile_version
    expected_compatibility = (
        node_compatibility_digest
        or release.config.component_digests.get("node_runtime")
        or artifact_sha
    )
    # The node set and its failure-domain map are two views of one node list,
    # so they share a single read instead of listing every node twice.
    with node_inventory_scope(release):
        node_names = (
            () if release.runner.dry_run else release._target_node_names(target)
        )
        failure_domains = (
            {}
            if release.runner.dry_run
            else target_node_failure_domains(release, target, node_names)
        )
    policy = (
        NodeRolloutPolicy(
            # A dry run lists no nodes, so there is no cap to resolve "auto"
            # against: it reports the one-node floor rather than the sentinel.
            max_unavailable=max(1, release.config.upgrade_max_unavailable),
            first_wave_max_unavailable=1,
            max_unavailable_per_failure_domain=1,
        )
        if release.runner.dry_run
        else rollback_node_rollout_policy(release, failure_domains)
        if phase == "rollback"
        else node_rollout_policy(release, failure_domains, phase=phase)
    )
    return (
        NodeMutationPreflight(
            phase=phase,
            node_names=node_names,
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
            max_unavailable=policy.max_unavailable,
            runtime_image=runtime_image,
            node_installer_image=node_installer_image,
        ),
        policy,
        failure_domains,
    )


def preflight_node_runtime(
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
    node_installer_image: str | None = None,
) -> None:
    candidate, _policy, _failure_domains = _node_runtime_candidate(
        release,
        target,
        phase=phase,
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
        runtime_image=runtime_image,
        node_installer_image=node_installer_image,
    )
    ensure_node_candidate_preflight(release, target, candidate)


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
    mutation_started: Callable[[], None] | None = None,
    candidate_preflight_completed: bool = False,
) -> tuple[str, str]:
    candidate, policy, failure_domains = _node_runtime_candidate(
        release,
        target,
        phase=phase,
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
        runtime_image=runtime_image,
        node_installer_image=node_installer_image,
    )
    expected_profile = candidate.runtime_profile_version
    expected_compatibility = candidate.node_compatibility_digest
    node_names = candidate.node_names
    if release.runner.dry_run:
        return _dry_run_node_runtime(
            release,
            target,
            candidate,
            allow_legacy_identity=allow_legacy_identity,
            agent_identity=agent_identity,
        )
    if not candidate_preflight_completed:
        ensure_node_candidate_preflight(release, target, candidate)
    ensure_pre_node_mutation_barrier(
        release,
        target,
        candidate,
        ensure_runtime_safe=_ensure_runtime_safe,
    )
    if mutation_started is not None:
        mutation_started()
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
        max_unavailable=policy.max_unavailable,
        sync_registry=False,
        runtime_image=runtime_image,
        node_installer_image=node_installer_image,
    )
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
        max_unavailable=policy.max_unavailable,
    )
    if legacy_identity is not None:
        return legacy_identity
    deployment_id, deployment = _create_fleet_rollout(
        release,
        target,
        candidate,
        policy,
        failure_domains,
        paused_identity,
        allow_legacy_identity=allow_legacy_identity,
        agent_identity=agent_identity,
    )
    desired_bundle, desired_template = paused_identity
    safety_node_names = node_names
    if phase == "join":
        # The wave safety gate protects the capacity outside the wave, and a
        # joining cluster's nodes hold no live agent yet -- or only records a
        # removed cluster left behind (live 2026-09-12: leases expired for
        # hours, every retry blocked on "lease-margin"). Neither is capacity a
        # wave can take away; only nodes with a live agent are.
        safety_node_names = live_agent_node_names(release, target, node_names)
    run_fleet_waves(
        release,
        target,
        FleetWaveContext(
            phase=phase,
            deployment_id=deployment_id,
            node_names=safety_node_names,
            paused_identity=paused_identity,
            wheel_cm=wheel_cm,
            bundle_cm=bundle_cm,
            artifact_sha=artifact_sha,
            config_digest=config_digest,
            expected_profile=expected_profile,
            executor_wheel_filename=executor_wheel_filename,
            expected_compatibility=expected_compatibility,
            desired_bundle=desired_bundle,
            desired_template=desired_template,
            template_config_map=template_config_map,
            max_unavailable=policy.max_unavailable,
            runtime_image=runtime_image,
            node_installer_image=node_installer_image,
            allow_legacy_identity=allow_legacy_identity,
            agent_identity=agent_identity,
        ),
        deployment,
    )
    return _finalize_node_runtime(
        release,
        target,
        candidate,
        paused_identity,
        steady_runtime_image=steady_runtime_image,
        steady_template_config_map=steady_template_config_map,
        allow_legacy_identity=allow_legacy_identity,
        agent_identity=agent_identity,
    )
