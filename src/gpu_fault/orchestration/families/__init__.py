from gpu_fault.lazy_exports import lazy_module

# Resolved on first access, never at package import. The coordinator is the
# only composer of these services, but the node and executor collectors reach
# this package through ``training_health -> families.identity``; an eager
# ``__init__`` shipped all eleven families in the node-runtime and executor
# wheels, and a control-plane-only change to one of them re-rolled every GPU
# node (each deploy through #21, 2026-09-19).
_EXPORTS = {
    "NodeConflictService": (
        "gpu_fault.orchestration.families.conflicts",
        "NodeConflictService",
    ),
    "DrainOperationCallbacks": (
        "gpu_fault.orchestration.families.drain",
        "DrainOperationCallbacks",
    ),
    "DrainOperationService": (
        "gpu_fault.orchestration.families.drain",
        "DrainOperationService",
    ),
    "EvidenceOperationService": (
        "gpu_fault.orchestration.families.evidence",
        "EvidenceOperationService",
    ),
    "NodeScopedFaultCallbacks": (
        "gpu_fault.orchestration.families.faults",
        "NodeScopedFaultCallbacks",
    ),
    "NodeScopedFaultService": (
        "gpu_fault.orchestration.families.faults",
        "NodeScopedFaultService",
    ),
    "GroupedFaultCallbacks": (
        "gpu_fault.orchestration.families.grouped_faults",
        "GroupedFaultCallbacks",
    ),
    "GroupedFaultService": (
        "gpu_fault.orchestration.families.grouped_faults",
        "GroupedFaultService",
    ),
    "GroupedHealthCallbacks": (
        "gpu_fault.orchestration.families.grouped_health",
        "GroupedHealthCallbacks",
    ),
    "GroupedHealthService": (
        "gpu_fault.orchestration.families.grouped_health",
        "GroupedHealthService",
    ),
    "NodeHealthCallbacks": (
        "gpu_fault.orchestration.families.health",
        "NodeHealthCallbacks",
    ),
    "NodeHealthIngestionService": (
        "gpu_fault.orchestration.families.health",
        "NodeHealthIngestionService",
    ),
    "NodeHealthPlanBuilder": (
        "gpu_fault.orchestration.families.health",
        "NodeHealthPlanBuilder",
    ),
    "NodeLifecycleCallbacks": (
        "gpu_fault.orchestration.families.node_lifecycle",
        "NodeLifecycleCallbacks",
    ),
    "NodeLifecycleOperationService": (
        "gpu_fault.orchestration.families.node_lifecycle",
        "NodeLifecycleOperationService",
    ),
    "ResetOperationService": (
        "gpu_fault.orchestration.families.reset",
        "ResetOperationService",
    ),
    "ValidationOperationService": (
        "gpu_fault.orchestration.families.validation",
        "ValidationOperationService",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
