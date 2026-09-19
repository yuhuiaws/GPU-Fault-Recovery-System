"""The one table of what the control plane tracks per collector kind.

``COLLECTOR_KINDS`` has one row per ``CollectorKind``: the producer name a
collector status carries, the systemd unit the node agent reports it under and
the silence threshold after which the kind counts as missing. ``telemetry`` and
``collector_requirements`` re-export the derived views the rest of the code
already reads.

This is a leaf module on purpose. A component wheel is the import-walk closure
of its entry points, and the walk follows ``"module:attr"`` factory strings.
While these rows lived in ``collector_registry`` next to the
``COLLECTOR_REGISTRY`` CLI table, every collector implementation and
``collectors_cli`` rode into the control-plane wheel -- 26 modules only the data
plane runs -- and a collector-only change re-rolled the control plane (found
2026-09-19).
Nothing here may import ``gpu_fault.collectors``, ``gpu_fault.collectors_cli``
or ``gpu_fault.plugins``; the CLI table that names the collector factories stays
in ``collector_registry``, which imports this module, never the reverse.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class CollectorKind(StrEnum):
    GPU_INVENTORY = "GPU_INVENTORY"
    GPU_METRICS = "GPU_METRICS"
    HOST_TELEMETRY = "HOST_TELEMETRY"
    NODE_LOGS = "NODE_LOGS"
    NVIDIA_KERNEL = "NVIDIA_KERNEL"
    FABRIC_MANAGER_LOG = "FABRIC_MANAGER_LOG"
    HMA_NODE = "HMA_NODE"
    HMA_CLOUDWATCH = "HMA_CLOUDWATCH"


@dataclass(frozen=True)
class SilentThreshold:
    env: str
    default: float
    read: Callable[[], float]


def silent_threshold(name: str, default: str) -> SilentThreshold:
    """Declare the silence threshold of a kind as the environment read it is."""

    def read() -> float:
        return float(os.getenv(name, default))

    return SilentThreshold(env=name, default=float(default), read=read)


@dataclass(frozen=True)
class CollectorKindSpec:
    kind: CollectorKind
    producer: str
    # ``None`` for kinds produced off the node (cluster singletons, Lambda).
    systemd_unit: str | None
    # ``None`` for kinds the control plane does not expect on a schedule.
    silent_threshold: SilentThreshold | None
    retired: bool = False


COLLECTOR_KINDS: dict[CollectorKind, CollectorKindSpec] = {
    CollectorKind.GPU_INVENTORY: CollectorKindSpec(
        kind=CollectorKind.GPU_INVENTORY,
        producer="DCGM_METRICS_COLLECTOR",
        systemd_unit="gpu-fault-metrics-collector",
        silent_threshold=silent_threshold(
            "GPU_FAULT_GPU_INVENTORY_SILENT_AFTER_SECONDS", "180"
        ),
    ),
    CollectorKind.GPU_METRICS: CollectorKindSpec(
        kind=CollectorKind.GPU_METRICS,
        producer="DCGM_METRICS_COLLECTOR",
        systemd_unit="gpu-fault-metrics-collector",
        silent_threshold=silent_threshold(
            "GPU_FAULT_GPU_METRICS_SILENT_AFTER_SECONDS", "420"
        ),
    ),
    CollectorKind.HOST_TELEMETRY: CollectorKindSpec(
        kind=CollectorKind.HOST_TELEMETRY,
        producer="HOST_TELEMETRY_COLLECTOR",
        systemd_unit="gpu-fault-host-collector",
        silent_threshold=silent_threshold(
            "GPU_FAULT_HOST_TELEMETRY_SILENT_AFTER_SECONDS", "420"
        ),
    ),
    CollectorKind.NODE_LOGS: CollectorKindSpec(
        kind=CollectorKind.NODE_LOGS,
        producer="NODE_LOG_COLLECTOR",
        systemd_unit="gpu-fault-log-collector",
        silent_threshold=silent_threshold(
            "GPU_FAULT_NODE_LOG_SILENT_AFTER_SECONDS", "900"
        ),
    ),
    CollectorKind.NVIDIA_KERNEL: CollectorKindSpec(
        kind=CollectorKind.NVIDIA_KERNEL,
        producer="KERNEL_LOG_COLLECTOR",
        systemd_unit="gpu-fault-kernel-collector",
        silent_threshold=silent_threshold(
            "GPU_FAULT_KERNEL_SILENT_AFTER_SECONDS", "900"
        ),
    ),
    CollectorKind.FABRIC_MANAGER_LOG: CollectorKindSpec(
        kind=CollectorKind.FABRIC_MANAGER_LOG,
        producer="FABRIC_MANAGER_LOG_COLLECTOR",
        systemd_unit="gpu-fault-fabric-manager-collector",
        silent_threshold=silent_threshold(
            "GPU_FAULT_FABRIC_MANAGER_SILENT_AFTER_SECONDS", "900"
        ),
    ),
    CollectorKind.HMA_NODE: CollectorKindSpec(
        kind=CollectorKind.HMA_NODE,
        producer="KUBERNETES_HMA_NODE_COLLECTOR",
        systemd_unit=None,
        silent_threshold=None,
        retired=True,
    ),
    CollectorKind.HMA_CLOUDWATCH: CollectorKindSpec(
        kind=CollectorKind.HMA_CLOUDWATCH,
        producer="CLOUDWATCH_HMA_COLLECTOR",
        systemd_unit=None,
        silent_threshold=None,
        retired=True,
    ),
}

# Derived views. ``telemetry`` and ``collector_requirements`` re-export them
# under the names their callers have always imported.
COLLECTOR_PRODUCER_BY_CHANNEL: dict[CollectorKind, str] = {
    kind: spec.producer for kind, spec in COLLECTOR_KINDS.items()
}

COLLECTOR_SYSTEMD_UNITS: dict[CollectorKind, str] = {
    kind: spec.systemd_unit
    for kind, spec in COLLECTOR_KINDS.items()
    if spec.systemd_unit is not None
}


def collector_silent_thresholds() -> dict[CollectorKind, float]:
    return {
        kind: spec.silent_threshold.read()
        for kind, spec in COLLECTOR_KINDS.items()
        if spec.silent_threshold is not None
    }
