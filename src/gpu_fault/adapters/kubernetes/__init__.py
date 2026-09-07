from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "KubernetesWorkflowAdapter": (
        "gpu_fault.adapters.kubernetes.adapter",
        "KubernetesWorkflowAdapter",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
