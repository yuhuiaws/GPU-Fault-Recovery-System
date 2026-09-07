from gpu_fault.lazy_exports import lazy_module

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

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
