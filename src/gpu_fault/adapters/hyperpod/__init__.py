from importlib import import_module
from typing import Any

__all__ = ["HyperPodLifecycleStepAdapter"]


def __getattr__(name: str) -> Any:
    if name != "HyperPodLifecycleStepAdapter":
        raise AttributeError(name)
    value = import_module(
        "gpu_fault.adapters.hyperpod.lifecycle"
    ).HyperPodLifecycleStepAdapter
    globals()[name] = value
    return value
