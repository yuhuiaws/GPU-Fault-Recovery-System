from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "DagBrancher": ("gpu_fault.orchestration.dag_branching", "DagBrancher"),
    "HardwareEscalationService": (
        "gpu_fault.orchestration.escalation",
        "HardwareEscalationService",
    ),
    "RecoveryArbiter": ("gpu_fault.orchestration.arbitration", "RecoveryArbiter"),
    "SxidIngestionCallbacks": (
        "gpu_fault.orchestration.ingest",
        "SxidIngestionCallbacks",
    ),
    "SxidIngestionService": ("gpu_fault.orchestration.ingest", "SxidIngestionService"),
    "WorkflowBuilder": ("gpu_fault.orchestration.workflow_builder", "WorkflowBuilder"),
    "IncidentOrchestrator": (
        "gpu_fault.orchestration.coordinator",
        "IncidentOrchestrator",
    ),
    "OPERATION_CAPABILITY": (
        "gpu_fault.orchestration.coordinator",
        "OPERATION_CAPABILITY",
    ),
    "WorkflowFencingError": (
        "gpu_fault.orchestration.coordinator",
        "WorkflowFencingError",
    ),
    "failure_takes_no_rung": (
        "gpu_fault.orchestration.escalation",
        "failure_takes_no_rung",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
