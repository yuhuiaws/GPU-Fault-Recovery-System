from importlib import import_module
from typing import Any

_EXPORTS = {
    "CloudWatchHmaCollector": (
        "gpu_fault.collectors.cloud.cloudwatch",
        "CloudWatchHmaCollector",
    ),
    "KubernetesHmaNodeCollector": (
        "gpu_fault.collectors.cloud.kubernetes",
        "KubernetesHmaNodeCollector",
    ),
    "KubernetesNodeResourceCollector": (
        "gpu_fault.collectors.cloud.kubernetes",
        "KubernetesNodeResourceCollector",
    ),
    "SqsHmaConsumer": ("gpu_fault.collectors.cloud.cloudwatch", "SqsHmaConsumer"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
