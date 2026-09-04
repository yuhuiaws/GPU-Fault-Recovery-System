from __future__ import annotations

from typing import Any

import regional_deployment_inventory as inventory
from regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)


def preflight_upgrade_mutations(
    release: Any,
    plan: ReleaseExecutionPlan,
) -> None:
    deployment_names = frozenset(
        deployment
        for component, deployment in (
            (ReleaseComponent.EXECUTOR, inventory.GPU_EXECUTOR_DEPLOYMENT),
            (ReleaseComponent.WATCHER, inventory.GPU_WATCHER_DEPLOYMENT),
            (ReleaseComponent.COLLECTOR, inventory.GPU_COLLECTOR_DEPLOYMENT),
        )
        if plan.has(component)
    )
    for target in release.config.clusters:
        if plan.has(ReleaseComponent.DCGM):
            release._preflight_gpu_dcgm_exporter(target)
        if deployment_names:
            release._preflight_gpu_deployments(
                target,
                release.executor_wheel_cm,
                deployment_names=deployment_names,
            )
        if plan.has(ReleaseComponent.RECONCILER, ReleaseComponent.AGENT):
            release._preflight_node_runtime(
                target,
                phase="upgrade",
                wheel_cm=release.executor_wheel_cm,
                bundle_cm=release.bundle_cm,
                artifact_sha=release.node_wheel_sha,
                config_digest=release.config.agent_config_digest,
            )
