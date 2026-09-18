"""Scoped Kubernetes GET responses for release orchestration test doubles."""

import json
from collections.abc import Sequence


def resource_probe_result(
    arguments: Sequence[str], *, present: bool = True
) -> tuple[int, str, str]:
    assert "get" in arguments
    index = arguments.index("get")
    resource, name = arguments[index + 1 : index + 3]
    kind = {
        "configmap": "ConfigMap",
        "secret": "Secret",
        "deployment": "Deployment",
        "cronjob": "CronJob",
        "pod": "Pod",
        "job": "Job",
    }[resource]
    namespace = arguments[arguments.index("-n") + 1]
    return (
        0,
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": kind,
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "uid": f"uid-{name}",
                    "resourceVersion": "1",
                },
            }
        )
        if present
        else "",
        "",
    )
