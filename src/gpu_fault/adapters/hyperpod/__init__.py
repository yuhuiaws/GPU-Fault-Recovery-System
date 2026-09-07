from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "HyperPodLifecycleStepAdapter": (
        "gpu_fault.adapters.hyperpod.lifecycle",
        "HyperPodLifecycleStepAdapter",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
