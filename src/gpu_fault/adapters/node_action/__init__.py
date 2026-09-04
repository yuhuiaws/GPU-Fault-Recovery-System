from importlib import import_module
from typing import Any

__all__ = ["NodeActionWorkflowAdapter"]


def __getattr__(name: str) -> Any:
    if name != "NodeActionWorkflowAdapter":
        raise AttributeError(name)
    value = import_module(
        "gpu_fault.adapters.node_action.adapter"
    ).NodeActionWorkflowAdapter
    globals()[name] = value
    return value
