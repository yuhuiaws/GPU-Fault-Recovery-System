from gpu_fault.lazy_exports import lazy_module

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
    "ProcessorLeaseSettings": (
        "gpu_fault.processor.settings",
        "ProcessorLeaseSettings",
    ),
    "ProcessorPoolSettings": ("gpu_fault.processor.settings", "ProcessorPoolSettings"),
    "ProcessorRequest": ("gpu_fault.processor.models", "ProcessorRequest"),
    "ProcessorRequestStatus": ("gpu_fault.processor.models", "ProcessorRequestStatus"),
    "ProcessorSpoolSettings": (
        "gpu_fault.processor.settings",
        "ProcessorSpoolSettings",
    ),
    "ProcessorStaleSettings": (
        "gpu_fault.processor.settings",
        "ProcessorStaleSettings",
    ),
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

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
