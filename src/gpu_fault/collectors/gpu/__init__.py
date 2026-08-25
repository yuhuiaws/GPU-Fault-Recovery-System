from importlib import import_module

_EXPORTS = {
    "DcgmMetricsCollector": ("gpu_fault.collectors.gpu.dcgm", "DcgmMetricsCollector"),
    "NvidiaSmiMetricsCollector": (
        "gpu_fault.collectors.gpu.nvidia_smi",
        "NvidiaSmiMetricsCollector",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
