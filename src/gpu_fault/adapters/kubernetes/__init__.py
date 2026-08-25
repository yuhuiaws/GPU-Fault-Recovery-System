from importlib import import_module

__all__ = ["KubernetesWorkflowAdapter"]


def __getattr__(name: str):
    if name != "KubernetesWorkflowAdapter":
        raise AttributeError(name)
    value = import_module(
        "gpu_fault.adapters.kubernetes.adapter"
    ).KubernetesWorkflowAdapter
    globals()[name] = value
    return value
