from __future__ import annotations

from typing import Any

import regional_deployment_inventory as inventory


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
                "upload content-addressed wheel and bundle",
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
                "restore CPU and GPU Deployment wheels",
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
        "configured_runtime_image": release.runtime_image,
        "configured_runtime_profile": {
            "source": str(config.runtime_profile_source),
            "version": config.runtime_profile_version,
            "registration_cluster_id": (config.runtime_profile_registration_cluster_id),
            "sha256": release.runtime_profile_sha,
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
