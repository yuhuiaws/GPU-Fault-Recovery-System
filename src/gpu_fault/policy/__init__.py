from importlib import import_module
from typing import Any

_EXPORTS = {
    "ActionDisposition": ("gpu_fault.policy.models", "ActionDisposition"),
    "ActionSource": ("gpu_fault.policy.models", "ActionSource"),
    "CatalogRule": ("gpu_fault.policy.models", "CatalogRule"),
    "Containment": ("gpu_fault.policy.models", "Containment"),
    "DistributedXidBatch": ("gpu_fault.policy.models", "DistributedXidBatch"),
    "DistributedXidIngestionResult": (
        "gpu_fault.policy.models",
        "DistributedXidIngestionResult",
    ),
    "DynamicRecoveryAction": ("gpu_fault.policy.models", "DynamicRecoveryAction"),
    "FaultPolicyDecision": ("gpu_fault.policy.models", "FaultPolicyDecision"),
    "GpuFaultPolicyEngine": ("gpu_fault.policy.engine", "GpuFaultPolicyEngine"),
    "NVIDIA_ALWAYS_FATAL_SXIDS": (
        "gpu_fault.policy.models",
        "NVIDIA_ALWAYS_FATAL_SXIDS",
    ),
    "NVIDIA_CODE_SPECIFIC_FULL_RESET_SXIDS": (
        "gpu_fault.policy.models",
        "NVIDIA_CODE_SPECIFIC_FULL_RESET_SXIDS",
    ),
    "Nvlink74BitOccurrenceState": (
        "gpu_fault.policy.models",
        "Nvlink74BitOccurrenceState",
    ),
    "NvlinkDecodeRule": ("gpu_fault.policy.models", "NvlinkDecodeRule"),
    "SxidClassification": ("gpu_fault.policy.models", "SxidClassification"),
    "SxidEvent": ("gpu_fault.policy.models", "SxidEvent"),
    "SxidLinkScope": ("gpu_fault.policy.models", "SxidLinkScope"),
    "XidCorrelationRecord": ("gpu_fault.policy.models", "XidCorrelationRecord"),
    "XidCorrelationStatus": ("gpu_fault.policy.models", "XidCorrelationStatus"),
    "XidEvent": ("gpu_fault.policy.models", "XidEvent"),
    "catalog_product_family": (
        "gpu_fault.policy.product_families",
        "catalog_product_family",
    ),
    "catalog_supports_product": (
        "gpu_fault.policy.product_families",
        "catalog_supports_product",
    ),
    "load_sxid_policy": ("gpu_fault.policy.catalog", "load_sxid_policy"),
    "load_xid_policy": ("gpu_fault.policy.catalog", "load_xid_policy"),
    "parse_xid154_action": ("gpu_fault.policy.models", "parse_xid154_action"),
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
