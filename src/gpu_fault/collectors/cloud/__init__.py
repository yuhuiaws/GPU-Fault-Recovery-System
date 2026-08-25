from importlib import import_module

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


def __getattr__(name):
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value
