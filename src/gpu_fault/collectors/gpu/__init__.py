from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "DcgmMetricsCollector": ("gpu_fault.collectors.gpu.dcgm", "DcgmMetricsCollector"),
    "NvidiaSmiMetricsCollector": (
        "gpu_fault.collectors.gpu.nvidia_smi",
        "NvidiaSmiMetricsCollector",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
