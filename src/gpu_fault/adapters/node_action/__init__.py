from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "NodeActionWorkflowAdapter": (
        "gpu_fault.adapters.node_action.adapter",
        "NodeActionWorkflowAdapter",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
