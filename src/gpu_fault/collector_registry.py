"""The one table a collector is added to.

``COLLECTOR_KINDS`` has one row per ``CollectorKind`` (what the control plane
tracks: producer name, systemd unit, silence threshold) and
``COLLECTOR_REGISTRY`` has one row per ``gpu-fault-collector`` subcommand (what
the CLI builds: factory, channels, context needs). ``telemetry`` and
``collector_requirements`` re-export the derived views the rest of the code
already reads, so a new collector edits these two tables, adds a channel to
``channel_registry`` and a systemd unit to ``deploy/``, and nothing else.

``validate_collector_registry()`` runs at import time like the operation,
channel and node-action registries: a row that names a channel the processor
does not serve, an export the ``gpu_fault.collectors`` package does not
publish, or a kind no subcommand produces fails the first import.

Third-party collectors join through the ``gpu_fault.collectors`` entry-point
group; ``collector_registry_with_plugins()`` merges them under the same
validator when the CLI starts, never at import time.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, cast

from gpu_fault import collectors as collectors_package
from gpu_fault.channel_registry import (
    CHANNEL_REGISTRY,
    COLLECTOR_HEALTH_PATH,
    DEVICE_EVENT_PATH_PREFIXES,
    FABRIC_MANAGER_PATH,
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    NVIDIA_KERNEL_PATH,
    TRAINING_PROGRESS_PATH,
)
from gpu_fault.plugins import PluginGroup, discover_plugins

if TYPE_CHECKING:
    from gpu_fault.collectors.models import CollectorContext
    from gpu_fault.collectors.sinks import EventSink


class CollectorKind(StrEnum):
    GPU_INVENTORY = "GPU_INVENTORY"
    GPU_METRICS = "GPU_METRICS"
    HOST_TELEMETRY = "HOST_TELEMETRY"
    NODE_LOGS = "NODE_LOGS"
    NVIDIA_KERNEL = "NVIDIA_KERNEL"
    FABRIC_MANAGER_LOG = "FABRIC_MANAGER_LOG"
    HMA_NODE = "HMA_NODE"
    HMA_CLOUDWATCH = "HMA_CLOUDWATCH"


class RunnableCollector(Protocol):
    def run(self) -> None: ...


# A factory is named as ``"module:callable"`` so the registry never imports a
# collector module: the CLI resolves the one it is about to run.
ContextualCollectorFactory = Callable[
    ["EventSink", "CollectorContext", argparse.Namespace], RunnableCollector
]
ContextFreeCollectorFactory = Callable[
    ["EventSink", argparse.Namespace], RunnableCollector
]
CollectorRuntime = Literal["node", "cluster", "workload"]


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


@dataclass(frozen=True)
class CollectorDescriptor:
    cli_command: str
    # The kinds this subcommand produces. Two subcommands may produce the same
    # kind (``dcgm`` and its ``nvidia-smi`` fallback); a subcommand with no
    # ``CollectorKind`` (``training-progress``) declares none.
    kinds: tuple[CollectorKind, ...]
    channel_paths: tuple[str, ...]
    # The class name ``gpu_fault.collectors`` publishes lazily, or ``None`` for
    # a plugin that ships outside the package.
    export_name: str | None
    factory: str
    runs_in: CollectorRuntime
    needs_context: bool = True
    needs_product_discovery: bool = False

    def build(
        self,
        sink: EventSink,
        context: CollectorContext | None,
        arguments: argparse.Namespace,
    ) -> RunnableCollector:
        factory = _load_factory(self.factory)
        if not self.needs_context:
            return cast(ContextFreeCollectorFactory, factory)(sink, arguments)
        if context is None:
            raise RuntimeError(f"{self.cli_command} needs a collector context")
        return cast(ContextualCollectorFactory, factory)(sink, context, arguments)


def _load_factory(reference: str) -> Callable[..., object]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"collector factory {reference!r} is not module:callable")
    value: object = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise RuntimeError(f"collector factory {reference} is not callable")
    return cast(Callable[..., object], value)


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
    ),
    CollectorKind.HMA_CLOUDWATCH: CollectorKindSpec(
        kind=CollectorKind.HMA_CLOUDWATCH,
        producer="CLOUDWATCH_HMA_COLLECTOR",
        systemd_unit=None,
        silent_threshold=None,
    ),
}

HMA_KUBERNETES_NODE_PATH = "/v1/provider-events/hyperpod-hma/kubernetes-node"
HMA_CLOUDWATCH_PATH = "/v1/provider-events/hyperpod-hma/cloudwatch"

COLLECTOR_REGISTRY: dict[str, CollectorDescriptor] = {
    "kernel": CollectorDescriptor(
        cli_command="kernel",
        kinds=(CollectorKind.NVIDIA_KERNEL,),
        channel_paths=(NVIDIA_KERNEL_PATH, COLLECTOR_HEALTH_PATH),
        export_name="KernelLogCollector",
        factory="gpu_fault.collectors.logs.kernel:build_from_environment",
        runs_in="node",
        needs_product_discovery=True,
    ),
    "kubernetes-hma": CollectorDescriptor(
        cli_command="kubernetes-hma",
        kinds=(CollectorKind.HMA_NODE,),
        channel_paths=(HMA_KUBERNETES_NODE_PATH,),
        export_name="KubernetesHmaNodeCollector",
        factory="gpu_fault.collectors_cli:build_kubernetes_hma",
        runs_in="cluster",
    ),
    "kubernetes-node-resources": CollectorDescriptor(
        cli_command="kubernetes-node-resources",
        kinds=(CollectorKind.HOST_TELEMETRY,),
        channel_paths=(HOST_TELEMETRY_PATH,),
        export_name="KubernetesNodeResourceCollector",
        factory="gpu_fault.collectors_cli:build_kubernetes_node_resources",
        runs_in="cluster",
    ),
    "sqs-hma": CollectorDescriptor(
        cli_command="sqs-hma",
        kinds=(CollectorKind.HMA_CLOUDWATCH,),
        channel_paths=(HMA_CLOUDWATCH_PATH,),
        export_name="SqsHmaConsumer",
        factory="gpu_fault.collectors_cli:build_sqs_hma",
        runs_in="cluster",
        needs_context=False,
    ),
    "dcgm": CollectorDescriptor(
        cli_command="dcgm",
        kinds=(CollectorKind.GPU_METRICS, CollectorKind.GPU_INVENTORY),
        channel_paths=(GPU_METRICS_PATH, GPU_INVENTORY_PATH),
        export_name="DcgmMetricsCollector",
        factory="gpu_fault.collectors.gpu.dcgm:build_from_environment",
        runs_in="node",
        needs_product_discovery=True,
    ),
    "nvidia-smi": CollectorDescriptor(
        cli_command="nvidia-smi",
        kinds=(CollectorKind.GPU_METRICS, CollectorKind.GPU_INVENTORY),
        channel_paths=(GPU_METRICS_PATH, GPU_INVENTORY_PATH),
        export_name="NvidiaSmiMetricsCollector",
        factory="gpu_fault.collectors.gpu.nvidia_smi:build_from_environment",
        runs_in="node",
        needs_product_discovery=True,
    ),
    "host": CollectorDescriptor(
        cli_command="host",
        kinds=(CollectorKind.HOST_TELEMETRY,),
        channel_paths=(HOST_TELEMETRY_PATH,),
        export_name="HostTelemetryCollector",
        factory="gpu_fault.collectors.host.collector:build_from_environment",
        runs_in="node",
        needs_product_discovery=True,
    ),
    "logs": CollectorDescriptor(
        cli_command="logs",
        kinds=(CollectorKind.NODE_LOGS,),
        channel_paths=(NODE_LOG_PATH,),
        export_name="NodeLogCollector",
        factory="gpu_fault.collectors.logs.node:build_from_environment",
        runs_in="node",
        needs_product_discovery=True,
    ),
    "fabric-manager": CollectorDescriptor(
        cli_command="fabric-manager",
        kinds=(CollectorKind.FABRIC_MANAGER_LOG,),
        channel_paths=(FABRIC_MANAGER_PATH, COLLECTOR_HEALTH_PATH),
        export_name="FabricManagerLogCollector",
        factory="gpu_fault.collectors.logs.fabric_manager:build_from_environment",
        runs_in="node",
        needs_product_discovery=True,
    ),
    "training-progress": CollectorDescriptor(
        cli_command="training-progress",
        kinds=(),
        channel_paths=(TRAINING_PROGRESS_PATH,),
        export_name="TrainingProgressCollector",
        factory="gpu_fault.collectors.training_progress:build_from_environment",
        runs_in="workload",
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


def _known_channel(path: str) -> bool:
    return path in CHANNEL_REGISTRY or path.startswith(DEVICE_EVENT_PATH_PREFIXES)


def validate_collector_registry(
    registry: Mapping[str, CollectorDescriptor] | None = None,
    kinds: Mapping[CollectorKind, CollectorKindSpec] | None = None,
) -> None:
    registry = COLLECTOR_REGISTRY if registry is None else registry
    kinds = COLLECTOR_KINDS if kinds is None else kinds
    declared = frozenset(kinds)
    if declared != frozenset(CollectorKind):
        missing = sorted(item.value for item in frozenset(CollectorKind) - declared)
        extra = sorted(item.value for item in declared - frozenset(CollectorKind))
        raise RuntimeError(
            f"collector kind registry mismatch: missing={missing} extra={extra}"
        )
    for kind, spec in kinds.items():
        if spec.kind is not kind:
            raise RuntimeError(f"collector kind {kind.value} row declares {spec.kind}")
    producer_by_unit: dict[str, str] = {}
    for spec in kinds.values():
        if spec.systemd_unit is None:
            continue
        previous = producer_by_unit.setdefault(spec.systemd_unit, spec.producer)
        if previous != spec.producer:
            raise RuntimeError(
                f"systemd unit {spec.systemd_unit} is claimed by producers "
                f"{previous} and {spec.producer}"
            )
    exports = frozenset(collectors_package.__all__)
    served: set[CollectorKind] = set()
    for command, descriptor in registry.items():
        if descriptor.cli_command != command:
            raise RuntimeError(
                f"collector registry key {command!r} names a descriptor for "
                f"{descriptor.cli_command!r}"
            )
        if len(set(descriptor.kinds)) != len(descriptor.kinds):
            raise RuntimeError(f"{command}: repeats a collector kind")
        if not descriptor.channel_paths:
            raise RuntimeError(f"{command}: declares no channel path")
        for path in descriptor.channel_paths:
            if not _known_channel(path):
                raise RuntimeError(f"{command}: unknown collector channel {path}")
        if descriptor.export_name is not None and descriptor.export_name not in exports:
            raise RuntimeError(
                f"{command}: export {descriptor.export_name} is not published by "
                "gpu_fault.collectors"
            )
        if ":" not in descriptor.factory:
            raise RuntimeError(f"{command}: factory must be module:callable")
        if descriptor.needs_product_discovery and (
            not descriptor.needs_context or descriptor.runs_in != "node"
        ):
            raise RuntimeError(
                f"{command}: GPU product discovery needs a node collector context"
            )
        units = {
            kinds[kind].systemd_unit
            for kind in descriptor.kinds
            if kinds[kind].systemd_unit is not None
        }
        if descriptor.runs_in == "node" and len(units) > 1:
            raise RuntimeError(
                f"{command}: one node process cannot serve systemd units "
                f"{sorted(unit for unit in units if unit is not None)}"
            )
        served.update(descriptor.kinds)
    unserved = sorted(item.value for item in frozenset(CollectorKind) - served)
    if unserved:
        raise RuntimeError(
            f"collector registry mismatch: no CLI command produces {unserved}"
        )


def collector_registry_with_plugins() -> dict[str, CollectorDescriptor]:
    """The built-in table plus every installed ``gpu_fault.collectors`` plugin."""

    registry = dict(COLLECTOR_REGISTRY)
    for name, entry_point in discover_plugins(PluginGroup.COLLECTORS).items():
        descriptor: object = entry_point.load()
        if not isinstance(descriptor, CollectorDescriptor):
            raise RuntimeError(
                f"collector plugin {name} must be a CollectorDescriptor, "
                f"got {type(descriptor).__name__}"
            )
        if descriptor.cli_command != name:
            raise RuntimeError(
                f"collector plugin {name} declares cli_command "
                f"{descriptor.cli_command!r}; the two must agree"
            )
        if name in registry:
            raise RuntimeError(f"collector plugin {name} collides with a CLI command")
        registry[name] = descriptor
    validate_collector_registry(registry)
    return registry


validate_collector_registry()
