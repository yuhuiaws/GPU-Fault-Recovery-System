from importlib import import_module
from typing import Any

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
