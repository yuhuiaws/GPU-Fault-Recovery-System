from importlib import import_module
from typing import Any

_EXPORTS = {
    "ProductionExecutorConfig": (
        "gpu_fault.execution.config",
        "ProductionExecutorConfig",
    ),
    "ProductionWorkflowExecutor": (
        "gpu_fault.execution.executor",
        "ProductionWorkflowExecutor",
    ),
    "WorkflowDispatcher": ("gpu_fault.execution.dispatcher", "WorkflowDispatcher"),
    "WorkflowDispatcherConfig": (
        "gpu_fault.execution.config",
        "WorkflowDispatcherConfig",
    ),
    "WorkflowExecutionError": ("gpu_fault.execution.models", "WorkflowExecutionError"),
    "WorkflowExecutionRequest": ("gpu_fault.models", "WorkflowExecutionRequest"),
    "WorkflowStepAdapter": ("gpu_fault.execution.models", "WorkflowStepAdapter"),
    "WorkflowStepContext": ("gpu_fault.execution.models", "WorkflowStepContext"),
    "WorkflowStepOutcome": ("gpu_fault.execution.models", "WorkflowStepOutcome"),
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
