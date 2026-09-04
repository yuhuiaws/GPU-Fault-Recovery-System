from importlib import import_module
from typing import Any

_EXPORTS = {
    "CONNECTION_POOL": ("gpu_fault.transport.http_client", "CONNECTION_POOL"),
    "CloudWatchHmaCollector": (
        "gpu_fault.collectors.cloud.cloudwatch",
        "CloudWatchHmaCollector",
    ),
    "CollectorContext": ("gpu_fault.collectors.models", "CollectorContext"),
    "CollectorError": ("gpu_fault.collectors.sinks", "CollectorError"),
    "DcgmMetricsCollector": ("gpu_fault.collectors.gpu.dcgm", "DcgmMetricsCollector"),
    "EventSink": ("gpu_fault.collectors.sinks", "EventSink"),
    "FabricManagerLogCollector": (
        "gpu_fault.collectors.logs.fabric_manager",
        "FabricManagerLogCollector",
    ),
    "HostTelemetryCollector": (
        "gpu_fault.collectors.host.collector",
        "HostTelemetryCollector",
    ),
    "HttpEventSink": ("gpu_fault.collectors.sinks", "HttpEventSink"),
    "KernelLogCollector": ("gpu_fault.collectors.logs.kernel", "KernelLogCollector"),
    "KubernetesHmaNodeCollector": (
        "gpu_fault.collectors.cloud.kubernetes",
        "KubernetesHmaNodeCollector",
    ),
    "KubernetesNodeResourceCollector": (
        "gpu_fault.collectors.cloud.kubernetes",
        "KubernetesNodeResourceCollector",
    ),
    "NVIDIA_SMI_CORE_FIELDS": (
        "gpu_fault.collectors.gpu.nvidia_smi",
        "NVIDIA_SMI_CORE_FIELDS",
    ),
    "NVIDIA_SMI_REMAP_FIELDS": (
        "gpu_fault.collectors.gpu.nvidia_smi",
        "NVIDIA_SMI_REMAP_FIELDS",
    ),
    "NodeLogCollector": ("gpu_fault.collectors.logs.node", "NodeLogCollector"),
    "NvidiaSmiMetricsCollector": (
        "gpu_fault.collectors.gpu.nvidia_smi",
        "NvidiaSmiMetricsCollector",
    ),
    "SqsEventSink": ("gpu_fault.collectors.sinks", "SqsEventSink"),
    "SqsHmaConsumer": ("gpu_fault.collectors.cloud.cloudwatch", "SqsHmaConsumer"),
    "TrainingProgressCollector": (
        "gpu_fault.collectors.training_progress",
        "TrainingProgressCollector",
    ),
    "context_from_environment": (
        "gpu_fault.collectors.context",
        "context_from_environment",
    ),
    "discover_gpu_product": (
        "gpu_fault.collectors.gpu.discovery",
        "discover_gpu_product",
    ),
    "discover_gpu_software_versions": (
        "gpu_fault.collectors.gpu.discovery",
        "discover_gpu_software_versions",
    ),
    "next_stable_phase": ("gpu_fault.collectors.scheduling", "next_stable_phase"),
    "normalize_gpu_product": (
        "gpu_fault.collectors.gpu.discovery",
        "normalize_gpu_product",
    ),
    "query_nvidia_temperature_limits": (
        "gpu_fault.collectors.gpu.discovery",
        "query_nvidia_temperature_limits",
    ),
    "sink_from_environment": ("gpu_fault.collectors.context", "sink_from_environment"),
    "stable_phase_seconds": ("gpu_fault.collectors.scheduling", "stable_phase_seconds"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
