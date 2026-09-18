from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "KubernetesNodeResourceCollector": (
        "gpu_fault.collectors.cloud.kubernetes",
        "KubernetesNodeResourceCollector",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
