from importlib import import_module

_EXPORTS = {
    "PeriodicTaskLease": ("gpu_fault.processor.models", "PeriodicTaskLease"),
    "ProcessorCoordinator": ("gpu_fault.processor.coordinator", "ProcessorCoordinator"),
    "ProcessorLaneLease": ("gpu_fault.processor.models", "ProcessorLaneLease"),
    "ProcessorLeadership": ("gpu_fault.processor.models", "ProcessorLeadership"),
    "ProcessorRequest": ("gpu_fault.processor.models", "ProcessorRequest"),
    "ProcessorRequestStatus": ("gpu_fault.processor.models", "ProcessorRequestStatus"),
    "processor_internal_token_valid": (
        "gpu_fault.processor.auth",
        "processor_internal_token_valid",
    ),
    "processor_partition_id": ("gpu_fault.processor.models", "processor_partition_id"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
