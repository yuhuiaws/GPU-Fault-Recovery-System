from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


COLLECTOR_EVENT_PREFIX = "/v1/collector-events/"


class ChannelPriorityMode(StrEnum):
    FAULT = "FAULT"
    EVIDENCE = "EVIDENCE"
    ROUTINE = "ROUTINE"
    EDGE_FILTERED = "EDGE_FILTERED"


class ChannelPool(StrEnum):
    FAULT = "fault"
    OBSERVATION = "observation"
    GPU = "gpu"
    HOST = "host"


class ChannelLane(StrEnum):
    NODE = "NODE"
    ATTEMPT = "ATTEMPT"
    GPU_INVENTORY = "GPU_INVENTORY"
    EDGE_SUMMARY = "EDGE_SUMMARY"


@dataclass(frozen=True)
class ProcessorChannel:
    path: str
    priority_mode: ChannelPriorityMode
    pool: ChannelPool
    lane: ChannelLane
    spoolable: bool = False
    edge_filtered: bool = False
    latest_wins: bool = False
    batchable: bool = False
    receipt: bool = False
    correlated_fault: bool = False
    # Payloads on this channel can end up in ``incident.node_ids``; the
    # processor must give them node correlation keys so the aggregation gate
    # sees them in flight (F-B8).
    incident_scoped: bool = False
    snapshot_bypass: bool = False
    spool_weight: int = 0
    summary_lane_suffix: str | None = None
    routine_reasons: frozenset[str] = field(default_factory=frozenset)
    routine_reason_prefixes: tuple[str, ...] = ()
    empty_reasons_are_routine: bool = True

    def is_routine_payload(self, payload: dict[str, Any]) -> bool:
        if self.priority_mode is ChannelPriorityMode.ROUTINE:
            return True
        if self.priority_mode is not ChannelPriorityMode.EDGE_FILTERED:
            return False
        if payload.get("collection_errors"):
            return False
        reasons = payload.get("edge_filter_reasons") or []
        if not isinstance(reasons, list):
            return False
        if not reasons and not self.empty_reasons_are_routine:
            return False
        return all(
            isinstance(reason, str)
            and (
                reason in self.routine_reasons
                or reason.startswith(self.routine_reason_prefixes)
            )
            for reason in reasons
        )

    def priority(self, payload: dict[str, Any] | None) -> int:
        if self.priority_mode is ChannelPriorityMode.FAULT:
            return 0
        if self.priority_mode is ChannelPriorityMode.ROUTINE:
            return 100
        if self.priority_mode is ChannelPriorityMode.EDGE_FILTERED:
            return (
                100 if payload is not None and self.is_routine_payload(payload) else 50
            )
        return 50


ROUTINE_EDGE_REASONS = frozenset(
    {
        "health-summary",
        "initial-baseline",
        "baseline",
        "filter-disabled",
    }
)
ROUTINE_EDGE_REASON_PREFIXES = ("baseline:",)

NVIDIA_KERNEL_PATH = "/v1/collector-events/nvidia-kernel"
FABRIC_MANAGER_PATH = "/v1/collector-events/fabric-manager"
GPU_INVENTORY_PATH = "/v1/collector-events/gpu-inventory"
GPU_METRICS_PATH = "/v1/collector-events/gpu-metrics"
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
NODE_LOG_PATH = "/v1/collector-events/node-logs"
COLLECTOR_HEALTH_PATH = "/v1/collector-events/collector-health"
WORKLOAD_OBSERVATIONS_PATH = "/v1/workload-observations"
TRAINING_PROGRESS_PATH = "/v1/training-progress"


CHANNEL_REGISTRY: dict[str, ProcessorChannel] = {
    NVIDIA_KERNEL_PATH: ProcessorChannel(
        path=NVIDIA_KERNEL_PATH,
        incident_scoped=True,
        priority_mode=ChannelPriorityMode.FAULT,
        pool=ChannelPool.FAULT,
        lane=ChannelLane.NODE,
        receipt=True,
        correlated_fault=True,
    ),
    FABRIC_MANAGER_PATH: ProcessorChannel(
        path=FABRIC_MANAGER_PATH,
        incident_scoped=True,
        priority_mode=ChannelPriorityMode.FAULT,
        pool=ChannelPool.FAULT,
        lane=ChannelLane.NODE,
        receipt=True,
        correlated_fault=True,
    ),
    GPU_INVENTORY_PATH: ProcessorChannel(
        path=GPU_INVENTORY_PATH,
        incident_scoped=True,
        priority_mode=ChannelPriorityMode.ROUTINE,
        pool=ChannelPool.GPU,
        lane=ChannelLane.GPU_INVENTORY,
        spoolable=True,
        latest_wins=True,
        batchable=True,
        receipt=True,
        snapshot_bypass=True,
        spool_weight=1,
    ),
    GPU_METRICS_PATH: ProcessorChannel(
        path=GPU_METRICS_PATH,
        incident_scoped=True,
        priority_mode=ChannelPriorityMode.EDGE_FILTERED,
        pool=ChannelPool.GPU,
        lane=ChannelLane.EDGE_SUMMARY,
        spoolable=True,
        edge_filtered=True,
        latest_wins=True,
        batchable=True,
        receipt=True,
        spool_weight=2,
        summary_lane_suffix="gpu-metrics-summary",
        routine_reasons=ROUTINE_EDGE_REASONS,
        routine_reason_prefixes=ROUTINE_EDGE_REASON_PREFIXES,
    ),
    HOST_TELEMETRY_PATH: ProcessorChannel(
        path=HOST_TELEMETRY_PATH,
        incident_scoped=True,
        priority_mode=ChannelPriorityMode.EDGE_FILTERED,
        pool=ChannelPool.HOST,
        lane=ChannelLane.EDGE_SUMMARY,
        spoolable=True,
        edge_filtered=True,
        latest_wins=True,
        batchable=True,
        receipt=True,
        spool_weight=1,
        summary_lane_suffix="host-summary",
        routine_reasons=ROUTINE_EDGE_REASONS,
        routine_reason_prefixes=ROUTINE_EDGE_REASON_PREFIXES,
    ),
    NODE_LOG_PATH: ProcessorChannel(
        path=NODE_LOG_PATH,
        incident_scoped=True,
        priority_mode=ChannelPriorityMode.EDGE_FILTERED,
        pool=ChannelPool.HOST,
        lane=ChannelLane.EDGE_SUMMARY,
        spoolable=True,
        edge_filtered=True,
        latest_wins=True,
        batchable=True,
        receipt=True,
        spool_weight=1,
        summary_lane_suffix="node-log-summary",
        routine_reasons=ROUTINE_EDGE_REASONS,
        routine_reason_prefixes=ROUTINE_EDGE_REASON_PREFIXES,
        empty_reasons_are_routine=False,
    ),
    COLLECTOR_HEALTH_PATH: ProcessorChannel(
        path=COLLECTOR_HEALTH_PATH,
        priority_mode=ChannelPriorityMode.ROUTINE,
        pool=ChannelPool.HOST,
        lane=ChannelLane.EDGE_SUMMARY,
        latest_wins=True,
        receipt=True,
        summary_lane_suffix="collector-health",
    ),
    WORKLOAD_OBSERVATIONS_PATH: ProcessorChannel(
        path=WORKLOAD_OBSERVATIONS_PATH,
        priority_mode=ChannelPriorityMode.EVIDENCE,
        pool=ChannelPool.OBSERVATION,
        lane=ChannelLane.ATTEMPT,
        receipt=True,
    ),
    TRAINING_PROGRESS_PATH: ProcessorChannel(
        path=TRAINING_PROGRESS_PATH,
        priority_mode=ChannelPriorityMode.ROUTINE,
        pool=ChannelPool.OBSERVATION,
        lane=ChannelLane.ATTEMPT,
        receipt=True,
    ),
}

COLLECTOR_CHANNEL_PATHS = frozenset(
    path for path in CHANNEL_REGISTRY if path.startswith(COLLECTOR_EVENT_PREFIX)
)
BATCHABLE_CHANNEL_PATHS = frozenset(
    path for path, channel in CHANNEL_REGISTRY.items() if channel.batchable
)
SPOOLABLE_CHANNEL_PATHS = frozenset(
    path for path, channel in CHANNEL_REGISTRY.items() if channel.spoolable
)
FAULT_CHANNEL_PATHS = frozenset(
    path
    for path, channel in CHANNEL_REGISTRY.items()
    if channel.priority_mode is ChannelPriorityMode.FAULT
)
TELEMETRY_SPOOL_PATH_SCHEDULE = tuple(
    path
    for path, channel in CHANNEL_REGISTRY.items()
    for _ in range(channel.spool_weight)
)
# A decided fault reported by a node or a provider: the raw material of a
# storm. Tier 10 -- reserved depth, fault pool, never spooled -- but ordered
# after the control-plane actions below so a storm cannot starve its own cure.
DEVICE_EVENT_PATH_PREFIXES = (
    "/v1/gpu-events/",
    "/v1/provider-events/",
)
# The control plane acting on a decided fault, and the attempt lifecycle events
# that finalize a workflow. Tier 0: claimed before any device event.
CONTROL_PLANE_ACTION_PATH_PREFIXES = (
    "/v1/incidents/",
    "/v1/workflows/",
    "/v1/attempts/",
    "/v1/recovery-plans/",
)
FAULT_PATH_PREFIXES = DEVICE_EVENT_PATH_PREFIXES + CONTROL_PLANE_ACTION_PATH_PREFIXES


def validate_channel_registry() -> None:
    for path, channel in CHANNEL_REGISTRY.items():
        if path != channel.path or not path.startswith("/v1/"):
            raise RuntimeError(f"invalid processor channel path: {path}")
        if channel.edge_filtered != (
            channel.priority_mode is ChannelPriorityMode.EDGE_FILTERED
        ):
            raise RuntimeError(f"{path}: edge-filtered mode and flag disagree")
        if channel.edge_filtered and not channel.routine_reasons:
            raise RuntimeError(
                f"{path}: edge-filtered channel has no routine vocabulary"
            )
        if channel.spoolable and not channel.batchable:
            raise RuntimeError(f"{path}: spoolable channel must be batchable")
        if (
            channel.summary_lane_suffix is not None
            and channel.lane is not ChannelLane.EDGE_SUMMARY
        ):
            raise RuntimeError(f"{path}: summary suffix requires EDGE_SUMMARY lane")
        if channel.spool_weight < 0:
            raise RuntimeError(f"{path}: spool weight cannot be negative")
        if channel.correlated_fault and not channel.incident_scoped:
            raise RuntimeError(f"{path}: a correlated fault channel is incident-scoped")
        if (
            channel.pool in {ChannelPool.FAULT, ChannelPool.GPU, ChannelPool.HOST}
            and path != COLLECTOR_HEALTH_PATH
            and not channel.incident_scoped
        ):
            # Every channel whose findings can name a node in an incident must
            # produce node correlation keys, or the aggregation gate cannot see
            # its in-flight requests (F-B8).
            raise RuntimeError(
                f"{path}: channel writes incident nodes without node keys"
            )


def validate_collector_routes(paths: set[str]) -> None:
    registered = set(COLLECTOR_CHANNEL_PATHS)
    if paths != registered:
        missing = sorted(registered - paths)
        unregistered = sorted(paths - registered)
        raise RuntimeError(
            "collector channel registry mismatch: "
            f"missing_routes={missing} unregistered_routes={unregistered}"
        )


def channel_for_path(path: str) -> ProcessorChannel | None:
    return CHANNEL_REGISTRY.get(path)


def is_fault_path(path: str) -> bool:
    channel = channel_for_path(path)
    return bool(
        path.startswith(FAULT_PATH_PREFIXES)
        or (channel is not None and channel.priority_mode is ChannelPriorityMode.FAULT)
    )


def is_control_plane_action_path(path: str) -> bool:
    return path.startswith(CONTROL_PLANE_ACTION_PATH_PREFIXES)


def paths_for_pool(pool: ChannelPool) -> frozenset[str]:
    return frozenset(
        path for path, channel in CHANNEL_REGISTRY.items() if channel.pool is pool
    )


validate_channel_registry()
