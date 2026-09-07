from gpu_fault.lazy_exports import lazy_module

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

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
