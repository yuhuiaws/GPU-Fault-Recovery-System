from importlib import import_module
from typing import Any

_EXPORTS = {
    "FabricManagerLogCollector": (
        "gpu_fault.collectors.logs.fabric_manager",
        "FabricManagerLogCollector",
    ),
    "KernelLogCollector": ("gpu_fault.collectors.logs.kernel", "KernelLogCollector"),
    "NodeLogCollector": ("gpu_fault.collectors.logs.node", "NodeLogCollector"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
