from importlib import import_module

_EXPORTS = {
    "FabricManagerLogCollector": (
        "gpu_fault.collectors.logs.fabric_manager",
        "FabricManagerLogCollector",
    ),
    "KernelLogCollector": ("gpu_fault.collectors.logs.kernel", "KernelLogCollector"),
    "NodeLogCollector": ("gpu_fault.collectors.logs.node", "NodeLogCollector"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
