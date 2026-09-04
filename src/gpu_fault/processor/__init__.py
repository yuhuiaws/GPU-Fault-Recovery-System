from importlib import import_module
from typing import Any

_EXPORTS = {
    "PeriodicTaskLease": ("gpu_fault.processor.models", "PeriodicTaskLease"),
    "ProcessorCompletionSignals": (
        "gpu_fault.processor.completion_signals",
        "ProcessorCompletionSignals",
    ),
    "ProcessorCoordinator": ("gpu_fault.processor.coordinator", "ProcessorCoordinator"),
    "ProcessorLaneLease": ("gpu_fault.processor.models", "ProcessorLaneLease"),
    "ProcessorLeadership": ("gpu_fault.processor.models", "ProcessorLeadership"),
    "ProcessorLanePolicy": ("gpu_fault.processor.models", "ProcessorLanePolicy"),
    "ProcessorRequest": ("gpu_fault.processor.models", "ProcessorRequest"),
    "ProcessorRequestStatus": ("gpu_fault.processor.models", "ProcessorRequestStatus"),
    "deferred_strict_processor_lanes": (
        "gpu_fault.processor.models",
        "deferred_strict_processor_lanes",
    ),
    "processor_request_claimable": (
        "gpu_fault.processor.models",
        "processor_request_claimable",
    ),
    "processor_internal_token_valid": (
        "gpu_fault.processor.auth",
        "processor_internal_token_valid",
    ),
    "processor_partition_id": ("gpu_fault.processor.models", "processor_partition_id"),
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
