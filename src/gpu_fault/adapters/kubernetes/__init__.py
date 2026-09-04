from importlib import import_module
from typing import Any

__all__ = ["KubernetesWorkflowAdapter"]


def __getattr__(name: str) -> Any:
    if name != "KubernetesWorkflowAdapter":
        raise AttributeError(name)
    value = import_module(
        "gpu_fault.adapters.kubernetes.adapter"
    ).KubernetesWorkflowAdapter
    globals()[name] = value
    return value
