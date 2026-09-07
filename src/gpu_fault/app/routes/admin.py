from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from gpu_fault import __version__, module_digest
from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.operation_registry import OPERATION_REGISTRY


@dataclass(frozen=True)
class AdminRouterDependencies:
    context: Any
    processor: Any | None
    processor_mode: str
    service_role: str
    environment: Mapping[str, str]
    regional_registry_runtime: Any | None


def get_admin_dependencies() -> AdminRouterDependencies:
    raise RuntimeError("admin router dependencies are not configured")


router = APIRouter(tags=["admin"])

AGENT_PAGE_LIMIT = 500


@router.get("/healthz")
@authorization_bucket("public")
async def healthz(
    dependencies: AdminRouterDependencies = Depends(get_admin_dependencies),
):
    ctx = dependencies.context
    processor = dependencies.processor
    leadership = processor.leadership if processor is not None else None
    spool_consumer_alive = dependencies.service_role != "spool-worker" or (
        processor is not None and processor.spool_consumer_running
    )
    processor_healthy = (
        processor is None or processor.is_healthy()
    ) and spool_consumer_alive
    registry_ready = (
        dependencies.regional_registry_runtime is None
        or dependencies.regional_registry_runtime.is_ready()
    )
    payload = {
        "status": "ok" if processor_healthy and registry_ready else "unhealthy",
        "executor": ctx.executor_mode,
        "dispatcher": ("enabled" if ctx.dispatcher.config.enabled else "disabled"),
        "agent_registry": ("enabled" if ctx.fleet_registry is not None else "disabled"),
        "processor_mode": dependencies.processor_mode,
        "processor_role": (
            "inactive"
            if dependencies.service_role == "ingress"
            else "spool-consumer"
            if dependencies.service_role == "spool-worker"
            else "active-consumer"
            if processor is not None and processor.active_consumers
            else "leader"
            if processor is not None and processor.is_leader()
            else "standby"
            if processor is not None
            else "active"
        ),
        "processor_epoch": (str(leadership.epoch) if leadership is not None else ""),
        "processor_unhealthy_reason": (
            "telemetry spool consumer thread exited"
            if not spool_consumer_alive
            else processor.unhealthy_reason
            if processor is not None
            else None
        ),
        "service_role": (
            dependencies.environment.get("GPU_FAULT_SERVICE_ROLE") or "combined"
        ),
        "regional_registry": (
            dependencies.regional_registry_runtime.status()
            if dependencies.regional_registry_runtime is not None
            else None
        ),
    }
    if not processor_healthy or not registry_ready:
        # The registry status carries datetimes; an unencoded payload made the
        # 503 branch itself raise, which the probe read as a crash, not a 503.
        return JSONResponse(status_code=503, content=jsonable_encoder(payload))
    return payload


@router.get("/livez")
@authorization_bucket("public")
async def livez(
    dependencies: AdminRouterDependencies = Depends(get_admin_dependencies),
) -> Any:
    """Process-local liveness for the kubelet.

    Deliberately blind to Aurora: it asserts that the event loop answers and
    that the background threads this process cannot run without are alive.
    Registry freshness, store round trips and leadership belong to readiness
    (``/healthz``); a writer failover must make Pods NotReady, never restart
    them, because the restarted process would need Aurora again to start.
    """

    processor = dependencies.processor
    dead_threads: list[str] = []
    if (
        dependencies.service_role == "spool-worker"
        and processor is not None
        and not processor.spool_consumer_running
    ):
        dead_threads.append("telemetry-spool-consumer")
    payload = {
        "status": "alive" if not dead_threads else "dead",
        "service_role": (
            dependencies.environment.get("GPU_FAULT_SERVICE_ROLE") or "combined"
        ),
        "dead_threads": dead_threads,
    }
    if dead_threads:
        return JSONResponse(status_code=503, content=payload)
    return payload


@router.get("/v1/version")
@authorization_bucket("execution-token")
async def version(
    dependencies: AdminRouterDependencies = Depends(get_admin_dependencies),
) -> dict[str, Any]:
    """Report the code and fleet pins loaded by this process."""

    return {
        "version": __version__,
        "module_digest": module_digest(),
        "service_role": (
            dependencies.environment.get("GPU_FAULT_SERVICE_ROLE") or "combined"
        ),
        "deployment_mode": (
            "regional" if dependencies.context.regional_mode else "single-cluster"
        ),
        "required_agent_artifact_sha256": (
            dependencies.environment.get("GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256")
            or None
        ),
        "compatible_agent_artifact_sha256s": sorted(
            value.strip()
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S",
                "",
            ).split(",")
            if value.strip()
        ),
        "required_agent_compatibility_digest": (
            dependencies.environment.get(
                "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST"
            )
            or None
        ),
        "compatible_agent_compatibility_digests": sorted(
            value.strip()
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS",
                "",
            ).split(",")
            if value.strip()
        ),
        "required_agent_protocol_version": int(
            dependencies.environment.get(
                "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION",
                "3",
            )
        ),
        "compatible_agent_protocol_versions": sorted(
            int(value.strip())
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS",
                "",
            ).split(",")
            if value.strip()
        ),
        "required_agent_config_digest": (
            dependencies.environment.get("GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST")
            or None
        ),
        "compatible_agent_config_digests": sorted(
            value.strip()
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS",
                "",
            ).split(",")
            if value.strip()
        ),
        "required_runtime_profile_version": (
            dependencies.environment.get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION")
            or None
        ),
        "required_node_action_key_version": int(
            dependencies.environment.get(
                "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION",
                "2",
            )
        ),
        "required_regional_executor_protocol_version": int(
            dependencies.environment.get(
                "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION",
                "2",
            )
        ),
        "compatible_regional_executor_protocol_versions": sorted(
            int(value.strip())
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS",
                "",
            ).split(",")
            if value.strip()
        ),
        "required_regional_executor_artifact_sha256": (
            dependencies.environment.get(
                "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256"
            )
            or None
        ),
        "compatible_regional_executor_artifact_sha256s": sorted(
            value.strip()
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S",
                "",
            ).split(",")
            if value.strip()
        ),
        "required_regional_executor_compatibility_digest": (
            dependencies.environment.get(
                "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST"
            )
            or None
        ),
        "compatible_regional_executor_compatibility_digests": sorted(
            value.strip()
            for value in dependencies.environment.get(
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS",
                "",
            ).split(",")
            if value.strip()
        ),
    }


@router.get("/v1/capabilities/operations")
@authorization_bucket("execution-token")
async def operation_capabilities(
    cluster_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    dependencies: AdminRouterDependencies = Depends(get_admin_dependencies),
) -> dict[str, Any]:
    """Report operation semantics and the per-agent allowlists.

    The agent section is paged. ``cluster_id`` is pushed down to the
    store; ``limit``/``offset`` bound the response so a fleet-sized
    registry cannot turn a capability lookup into a multi-megabyte body.
    """

    configured = {
        item.value
        for item in dependencies.context.production_executor_config.allowed_operations
    }
    page_size = max(1, min(limit, AGENT_PAGE_LIMIT))
    start = max(0, offset)
    all_agents = dependencies.context.store.list_agents(cluster_id)
    agents = all_agents[start : start + page_size]
    return {
        "configured_operations": sorted(configured),
        "operations": [
            {
                "operation": operation.value,
                "enabled": operation.value in configured,
                "scope": semantics.scope.value,
                "destructive": semantics.destructive,
                "rank": semantics.recovery_rank,
                "capability": semantics.capability.value,
                "adapters": sorted(item.value for item in semantics.adapters),
                "requires_barrier": semantics.multi_node_barrier,
            }
            for operation, semantics in sorted(
                OPERATION_REGISTRY.items(),
                key=lambda item: item[0].value,
            )
        ],
        "agent_count": len(all_agents),
        "agent_offset": start,
        "agent_limit": page_size,
        "agents": [
            {
                "cluster_id": agent.cluster_id,
                "node_id": agent.node_id,
                "allowed_operations": sorted(
                    item.value for item in agent.allowed_operations
                ),
            }
            for agent in agents
        ],
    }
