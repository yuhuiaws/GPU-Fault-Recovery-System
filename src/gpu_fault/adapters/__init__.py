from importlib import import_module
from typing import Any

_EXPORTS = {
    "ControlPlaneEvidenceAdapter": (
        "gpu_fault.adapters.evidence",
        "ControlPlaneEvidenceAdapter",
    ),
    "ANNOTATION_MECHANICAL_INSPECTION_COMPLETE": (
        "gpu_fault.adapters.common",
        "ANNOTATION_MECHANICAL_INSPECTION_COMPLETE",
    ),
    "GpuValidationAdapter": (
        "gpu_fault.adapters.gpu_validation",
        "GpuValidationAdapter",
    ),
    "HyperPodLifecycleStepAdapter": (
        "gpu_fault.adapters.hyperpod.lifecycle",
        "HyperPodLifecycleStepAdapter",
    ),
    "KubernetesWorkflowAdapter": (
        "gpu_fault.adapters.kubernetes",
        "KubernetesWorkflowAdapter",
    ),
    "ManagedRecoveryObserverAdapter": (
        "gpu_fault.adapters.managed_recovery",
        "ManagedRecoveryObserverAdapter",
    ),
    "NodeActionPending": ("gpu_fault.adapters.common", "NodeActionPending"),
    "NodeActionWorkflowAdapter": (
        "gpu_fault.adapters.node_action",
        "NodeActionWorkflowAdapter",
    ),
    "SimulatedRecoveryExecutor": (
        "gpu_fault.adapters.simulated",
        "SimulatedRecoveryExecutor",
    ),
    "SupportEscalationAdapter": (
        "gpu_fault.adapters.support_escalation",
        "SupportEscalationAdapter",
    ),
    "quarantine_taint_value": ("gpu_fault.adapters.common", "quarantine_taint_value"),
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
