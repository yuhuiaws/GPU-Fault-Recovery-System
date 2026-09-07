from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "FabricManagerLogCollector": (
        "gpu_fault.collectors.logs.fabric_manager",
        "FabricManagerLogCollector",
    ),
    "KernelLogCollector": ("gpu_fault.collectors.logs.kernel", "KernelLogCollector"),
    "NodeLogCollector": ("gpu_fault.collectors.logs.node", "NodeLogCollector"),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
