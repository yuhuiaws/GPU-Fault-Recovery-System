from importlib import import_module
from typing import Any

_EXPORTS = {
    "DcgmMetricsCollector": ("gpu_fault.collectors.gpu.dcgm", "DcgmMetricsCollector"),
    "NvidiaSmiMetricsCollector": (
        "gpu_fault.collectors.gpu.nvidia_smi",
        "NvidiaSmiMetricsCollector",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
