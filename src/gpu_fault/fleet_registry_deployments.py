from __future__ import annotations

from typing import Any, cast

from gpu_fault.fleet_deployment import (
    DeploymentNodeStatus,
    DeploymentNodeUpdate,
    DeploymentStatus,
    DeploymentWaveLease,
    FleetDeployment,
)


def _agent_is_active(agent: Any, now: Any) -> bool:
    return (
        getattr(agent.lifecycle_state, "value", agent.lifecycle_state) == "ACTIVE"
        and agent.lease_expires_at is not None
        and agent.lease_expires_at > now
    )


def update_deployment_node(
    registry: Any,
    deployment_id: str,
    node_id: str,
    update: DeploymentNodeUpdate,
    *,
    from_heartbeat: bool = False,
) -> FleetDeployment:
    for _ in range(5):
        deployment = cast(
            FleetDeployment,
            registry.store.get_fleet_deployment(deployment_id),
        )
        current = next(
            (item for item in deployment.nodes if item.node_id == node_id),
            None,
        )
        if current is None:
            raise ValueError("node is not part of deployment")
        if current.status is update.status and current.reason == update.reason:
            return deployment
        if update.status is DeploymentNodeStatus.READY and not from_heartbeat:
            raise ValueError("READY can only be set by a matching agent heartbeat")
        if update.status is DeploymentNodeStatus.FAILED and current.status not in {
            DeploymentNodeStatus.INSTALLING,
            DeploymentNodeStatus.FAILED,
        }:
            raise ValueError("only an INSTALLING node can be marked FAILED")
        if (
            not from_heartbeat
            and update.status is not current.status
            and (current.status, update.status)
            not in {
                (DeploymentNodeStatus.PENDING, DeploymentNodeStatus.INSTALLING),
                (DeploymentNodeStatus.INSTALLING, DeploymentNodeStatus.FAILED),
            }
        ):
            raise ValueError(
                f"invalid deployment node transition "
                f"{current.status.value}->{update.status.value}"
            )
        if (
            update.status is DeploymentNodeStatus.INSTALLING
            and current.status is not DeploymentNodeStatus.INSTALLING
        ):
            active_wave = registry._active_wave(deployment)
            if active_wave is None or node_id not in active_wave:
                raise ValueError("node is not in the active deployment wave")
            installing = sum(
                item.status is DeploymentNodeStatus.INSTALLING
                for item in deployment.nodes
            )
            if installing >= deployment.max_unavailable:
                raise ValueError("deployment max_unavailable would be exceeded")
        now = registry.now()
        nodes = [
            item.model_copy(
                update={
                    "status": update.status,
                    "reason": update.reason,
                    "updated_at": now,
                }
            )
            if item.node_id == node_id
            else item
            for item in deployment.nodes
        ]
        replacement = cast(
            FleetDeployment,
            deployment.model_copy(
                update={
                    "nodes": nodes,
                    "status": registry._deployment_status(nodes),
                    "updated_at": now,
                }
            ),
        )
        if registry.store.replace_fleet_deployment_if_matches(
            replacement,
            deployment,
        ):
            return replacement
    raise ValueError("fleet deployment conflicted with concurrent updates")


def cancel_deployment(
    registry: Any,
    deployment_id: str,
    *,
    reason: str,
) -> FleetDeployment:
    for _ in range(5):
        deployment = cast(
            FleetDeployment,
            registry.store.get_fleet_deployment(deployment_id),
        )
        now = registry.now()
        replacement = cast(
            FleetDeployment,
            deployment.model_copy(
                update={
                    "status": DeploymentStatus.FAILED,
                    "nodes": [
                        item.model_copy(
                            update={
                                "status": DeploymentNodeStatus.FAILED,
                                "reason": reason,
                                "updated_at": now,
                            }
                        )
                        if item.status
                        in {
                            DeploymentNodeStatus.PENDING,
                            DeploymentNodeStatus.INSTALLING,
                            DeploymentNodeStatus.FAILED,
                        }
                        else item
                        for item in deployment.nodes
                    ],
                    "updated_at": now,
                }
            ),
        )
        if registry.store.replace_fleet_deployment_if_matches(
            replacement,
            deployment,
        ):
            return replacement
    raise ValueError("fleet deployment cancellation conflicted with concurrent updates")


def retry_failed_deployment(registry: Any, deployment_id: str) -> FleetDeployment:
    for _ in range(5):
        deployment = cast(
            FleetDeployment,
            registry.store.get_fleet_deployment(deployment_id),
        )
        if not any(
            item.status is DeploymentNodeStatus.FAILED for item in deployment.nodes
        ):
            return deployment
        now = registry.now()
        expected = (
            deployment.desired_agent_protocol_version,
            deployment.desired_agent_version,
            deployment.desired_artifact_sha256,
            deployment.desired_compatibility_digest
            or deployment.desired_artifact_sha256,
            deployment.desired_bundle_sha256,
            deployment.desired_template_sha256,
            deployment.desired_policy_version,
            deployment.desired_runtime_profile_version,
            deployment.desired_config_digest,
        )
        agents = {
            item.node_id: item
            for item in registry.store.list_agents(deployment.cluster_id)
        }
        nodes = []
        for item in deployment.nodes:
            if item.status is not DeploymentNodeStatus.FAILED:
                nodes.append(item)
                continue
            agent = agents.get(item.node_id)
            recovered = (
                agent is not None
                and _agent_is_active(agent, now)
                and agent.identity == expected
            )
            nodes.append(
                item.model_copy(
                    update={
                        "status": (
                            DeploymentNodeStatus.READY
                            if recovered
                            else DeploymentNodeStatus.PENDING
                        ),
                        "reason": None,
                        "updated_at": now,
                    }
                )
            )
        replacement = cast(
            FleetDeployment,
            deployment.model_copy(
                update={
                    "nodes": nodes,
                    "status": registry._deployment_status(nodes),
                    "updated_at": now,
                }
            ),
        )
        if registry.store.replace_fleet_deployment_if_matches(
            replacement,
            deployment,
        ):
            return replacement
    raise ValueError("fleet deployment retry conflicted with concurrent updates")


def start_next_wave(registry: Any, deployment_id: str) -> DeploymentWaveLease:
    for _ in range(5):
        deployment = cast(
            FleetDeployment,
            registry.store.get_fleet_deployment(deployment_id),
        )
        if deployment.status is DeploymentStatus.SUCCEEDED:
            raise ValueError("deployment is already complete")
        failed = [
            item.node_id
            for item in deployment.nodes
            if item.status is DeploymentNodeStatus.FAILED
        ]
        if failed:
            summary = ",".join(failed[:20])
            suffix = "" if len(failed) <= 20 else f" (+{len(failed) - 20} more)"
            raise ValueError("deployment has failed nodes: " + summary + suffix)
        by_node = {item.node_id: item.status for item in deployment.nodes}
        for wave_index, wave in enumerate(deployment.waves):
            if all(by_node[node_id] is DeploymentNodeStatus.READY for node_id in wave):
                continue
            installing = [
                node_id
                for node_id in wave
                if by_node[node_id] is DeploymentNodeStatus.INSTALLING
            ]
            if installing:
                return DeploymentWaveLease(
                    deployment_id=deployment_id,
                    wave_index=wave_index,
                    node_ids=installing,
                    issued_at=registry.now(),
                )
            pending = [
                node_id
                for node_id in wave
                if by_node[node_id] is DeploymentNodeStatus.PENDING
            ]
            if not pending:
                raise ValueError("active deployment wave has no pending nodes")
            installing_count = sum(
                status is DeploymentNodeStatus.INSTALLING for status in by_node.values()
            )
            if installing_count + len(pending) > deployment.max_unavailable:
                raise ValueError("deployment max_unavailable would be exceeded")
            now = registry.now()
            selected = set(pending)
            nodes = [
                item.model_copy(
                    update={
                        "status": DeploymentNodeStatus.INSTALLING,
                        "reason": None,
                        "updated_at": now,
                    }
                )
                if item.node_id in selected
                else item
                for item in deployment.nodes
            ]
            replacement = deployment.model_copy(
                update={
                    "nodes": nodes,
                    "status": registry._deployment_status(nodes),
                    "updated_at": now,
                }
            )
            if not registry.store.replace_fleet_deployment_if_matches(
                replacement,
                deployment,
            ):
                break
            return DeploymentWaveLease(
                deployment_id=deployment_id,
                wave_index=wave_index,
                node_ids=pending,
                issued_at=now,
            )
        else:
            raise ValueError("deployment has no runnable wave")
    raise ValueError("fleet deployment conflicted with concurrent wave starts")


def reconcile_deployments(registry: Any, record: Any) -> None:
    for deployment in registry.store.list_active_fleet_deployments(record.cluster_id):
        reconcile_deployment(registry, deployment, record)


def reconcile_deployment(
    registry: Any,
    deployment: FleetDeployment,
    record: Any,
) -> None:
    if getattr(record.lifecycle_state, "value", record.lifecycle_state) != "ACTIVE":
        return
    target = next(
        (item for item in deployment.nodes if item.node_id == record.node_id),
        None,
    )
    if target is None:
        return
    expected = (
        deployment.desired_agent_protocol_version,
        deployment.desired_agent_version,
        deployment.desired_artifact_sha256,
        deployment.desired_compatibility_digest or deployment.desired_artifact_sha256,
        deployment.desired_bundle_sha256,
        deployment.desired_template_sha256,
        deployment.desired_policy_version,
        deployment.desired_runtime_profile_version,
        deployment.desired_config_digest,
    )
    if record.identity != expected:
        return
    for _ in range(16):
        current = cast(
            FleetDeployment,
            registry.store.get_fleet_deployment(deployment.deployment_id),
        )
        now = registry.now()
        agents = {
            item.node_id: item
            for item in registry.store.list_agents(current.cluster_id)
        }
        changed = False
        nodes = []
        for item in current.nodes:
            agent = agents.get(item.node_id)
            ready = (
                agent is not None
                and _agent_is_active(agent, now)
                and agent.identity == expected
            )
            if ready and item.status is not DeploymentNodeStatus.READY:
                changed = True
                nodes.append(
                    item.model_copy(
                        update={
                            "status": DeploymentNodeStatus.READY,
                            "reason": None,
                            "updated_at": now,
                        }
                    )
                )
            else:
                nodes.append(item)
        if not changed:
            return
        replacement = current.model_copy(
            update={
                "nodes": nodes,
                "status": registry._deployment_status(nodes),
                "updated_at": now,
            }
        )
        if registry.store.replace_fleet_deployment_if_matches(replacement, current):
            return
    raise ValueError("fleet deployment conflicted with concurrent heartbeats")
