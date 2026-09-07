from __future__ import annotations

from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory


def build_release_plan(mode: str) -> list[str]:
    steps = ["validate CPU/GPU contexts and release artifacts"]
    if mode in {"bootstrap", "deploy"}:
        steps.extend(
            [
                "apply CPU prerequisites and require durable secrets",
                "create/update regional registry and GPU connection secrets",
                "initialize PostgreSQL schema",
            ]
        )
    if mode in {
        "bootstrap",
        "deploy",
        "upgrade",
        "resume",
    }:
        steps.extend(
            [
                "capture previous release state",
                "upload changed component wheels and the node bundle",
                "ensure PostgreSQL schema migration history",
                "stage stable required pins plus candidate compatibility",
                "roll CPU ingress and workers",
                "ensure the declared regional Runtime Profile exists without drift",
                "roll each GPU executor/watcher/collector/reconciler",
                "wait for every current node Agent to converge",
                "finalize strict pins and verify the fleet",
            ]
        )
    elif mode == "rollback":
        steps.extend(
            [
                "restore previous required pins",
                "restore the CPU control-plane and GPU Executor wheels",
                "restore previous node installer bundle",
                "verify previous fleet readiness",
            ]
        )
    elif mode == "join-cluster":
        steps.extend(
            [
                "append the target cluster to the regional registry",
                "roll CPU roles and ensure the shared Runtime Profile",
                "create the target GPU connection Secret",
                "deploy the target GPU data plane and wait for Agents",
            ]
        )
    elif mode == "remove-cluster":
        steps.extend(
            [
                "require an idle remote-command queue",
                "scale the target GPU data plane to zero",
                "remove the target cluster from the registry and roll CPU roles",
            ]
        )
    return steps


def build_release_status(release: Any) -> dict[str, Any]:
    config = release.config
    result: dict[str, Any] = {
        "site_name": config.site_name,
        "configured_runtime_image": release.runtime_image,
        "configured_release": {
            "release_id": release.release_id,
            "control_plane_wheel_sha256": release.wheel_sha,
            "executor_wheel_sha256": release.executor_wheel_sha,
            "node_wheel_sha256": release.node_wheel_sha,
            "node_bundle_sha256": release.bundle_sha,
            "database_schema_version": config.database_schema_version,
            "agent_protocol_version": config.agent_protocol_version,
            "executor_protocol_version": config.executor_protocol_version,
            "component_digests": config.component_digests,
            "release_manifest_schema_version": (config.release_manifest_schema_version),
            "release_delivery_sha256": config.release_delivery_sha256,
            "delivery_component_digests": config.delivery_component_digests,
            "rendered_manifest_sha256": release.rendered_manifest_digest,
            "admin_config_sha256": release.admin_config_digest,
            "admin_config_role_sha256": release.admin_config_role_digests,
            "admin_config": config.admin_config.as_dict(),
            "node_template_sha256": release.node_template_sha,
            "images": {
                "runtime": release.runtime_image,
                "node_installer": release.node_installer_image,
                "dcgm_exporter": release.dcgm_exporter_image,
                "adot": release.adot_image,
            },
            "endpoint_digest": release.endpoint_digest,
            "dcgm_digest": release.dcgm_digest,
        },
        "configured_runtime_profile": {
            "source": str(config.runtime_profile_source),
            "template_source": str(config.runtime_profile_template_source),
            "version": config.runtime_profile_version,
            "registration_cluster_id": (config.runtime_profile_registration_cluster_id),
            "source_sha256": release.runtime_profile_sha,
            "template_sha256": release.runtime_profile_template_sha,
            "policy_sha256": release.runtime_profile_policy_sha,
        },
        "configured_health": {
            "aurora_cluster_id": config.health.aurora_cluster_id,
            "amp_workspace_id": config.health.amp_workspace_id,
            "amp_rule_namespace": config.health.amp_rule_namespace,
            "sns_topic_arn": config.health.sns_topic_arn,
            "certificate_min_validity_days": (
                config.health.certificate_min_validity_days
            ),
            "remote_command_max_unclaimed_seconds": (
                config.health.remote_command_max_unclaimed_seconds
            ),
        },
        "release_metadata": release._config_map_data("gpu-fault-release-metadata"),
        "cpu_wheel": release._deployment_wheel(
            release._cpu(),
            inventory.CPU_INGRESS_DEPLOYMENT,
        ),
        "clusters": {},
    }
    for target in config.clusters:
        result["clusters"][target.cluster_id] = {
            "context": target.context,
            "executor_wheel": release._deployment_wheel(
                release._gpu(target),
                inventory.GPU_EXECUTOR_DEPLOYMENT,
            ),
            "reconciler_wheel": release._deployment_wheel(
                release._gpu(target),
                inventory.GPU_RECONCILER_DEPLOYMENT,
            ),
            "template": release._deployment_template_name(target),
        }
    return result
