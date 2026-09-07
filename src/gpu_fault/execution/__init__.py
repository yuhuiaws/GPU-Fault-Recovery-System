from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "managed_recovery_timeout_seconds": (
        "gpu_fault.execution.config",
        "managed_recovery_timeout_seconds",
    ),
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

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
