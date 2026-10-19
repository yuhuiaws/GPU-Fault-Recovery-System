"""Which GPUs a managed attempt holds is read the same way on both providers.

The host collector names a GPU by the UUID nvidia-smi prints for it
(``--query-gpu=uuid,utilization.gpu``). The completion watcher fills a
container's ``gpu_uuids`` from the ``gpu-fault.io/gpu-uuids`` annotation or, on
HyperPod and plain EKS alike, by running ``nvidia-smi --query-gpu=uuid`` inside
the running container (``GPU_FAULT_DISCOVER_POD_GPU_UUIDS``, on by default) --
the same UUIDs, so the grading cannot depend on the provider. These cases run
real Pod documents of both shapes through the watcher's own parser, publish the
observation it produces and grade host batches against it. Node names and GPU
UUIDs are spelled as on HyperPod ``ml.p5en.48xlarge`` and plain EKS
``p5en.48xlarge`` nodes.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation
from tests._builders import build_store, host_telemetry_batch
from tests.completion._support import NOW, FakeCoreApi, FakeSink

WINDOW = 300
HYPERPOD = {
    "cluster_id": "hp-cluster",
    "environment": Environment.HYPERPOD_EKS,
    "profile": "hyperpod-v1",
    "node": "hyperpod-i-00000000000000001",
    "gpus": tuple(f"GPU-{index:08x}-965e-fc68-40e4-225aaa3315b6" for index in range(8)),
}
PLAIN_EKS = {
    "cluster_id": "gpu-eks-cluster",
    "environment": Environment.EKS,
    "profile": "eks-v1",
    "node": "ip-10-0-0-1.us-west-2.compute.internal",
    "gpus": tuple(f"GPU-{index:08x}-11d3-5c96-fdb5-76df542ca80e" for index in range(8)),
}
JOB = "gpu-fault-partial-1792428951-1"


def _pytorchjob_pod(site, gpu_count):
    """A managed PyTorchJob master Pod as the submit tool leaves it."""

    return {
        "metadata": {
            "name": f"{JOB}-master-0",
            "namespace": "gpu-fault-system",
            "uid": f"uid-{site['node']}",
            "labels": {
                "gpu-fault.io/managed": "true",
                "gpu-fault.io/job-id": JOB,
                "gpu-fault.io/attempt-id": f"{JOB}-a001",
                "gpu-fault.io/critical": "true",
                "gpu-fault.io/role": "master",
                "training.kubeflow.org/job-name": JOB,
                "training.kubeflow.org/replica-index": "0",
                "training.kubeflow.org/replica-type": "master",
            },
            "annotations": {
                "gpu-fault.io/expected-critical-ranks": "1",
                "gpu-fault.io/rank-offset": "0",
                "gpu-fault.io/restart-budget": "1",
                "gpu-fault.io/runtime-profile-version": site["profile"],
                "gpu-fault.io/training-container": "pytorch",
            },
        },
        "spec": {
            "nodeName": site["node"],
            "containers": [
                {
                    "name": "pytorch",
                    "resources": {
                        "requests": {"cpu": "8", "nvidia.com/gpu": str(gpu_count)},
                        "limits": {"cpu": "16", "nvidia.com/gpu": str(gpu_count)},
                    },
                }
            ],
        },
        "status": {
            "phase": "Running",
            "startTime": NOW.isoformat(),
            "containerStatuses": [
                {
                    "name": "pytorch",
                    "containerID": "containerd://1c2915b04c2",
                    "restartCount": 0,
                    "state": {"running": {"startedAt": NOW.isoformat()}},
                }
            ],
        },
    }


def _watched_attempt(site, held):
    """The observation the watcher publishes for a Pod holding ``held``."""

    sink = FakeSink()
    discovered = []

    def nvidia_smi_in_container(pod, container_name):
        discovered.append((pod["spec"]["nodeName"], container_name))
        return [f"{uuid}\n" for uuid in held]

    KubernetesCompletionController(
        FakeCoreApi([_pytorchjob_pod(site, len(held))]),
        sink,
        cluster_id=site["cluster_id"],
        environment=site["environment"],
        publish_observations=True,
        gpu_uuid_resolver=nvidia_smi_in_container,
        now=lambda: NOW,
    ).run_once()
    assert discovered == [(site["node"], "pytorch")], discovered
    (payload,) = [body for path, body in sink.posts if path.endswith("observations")]
    return AttemptObservation.model_validate(payload)


def _host_batch(site, offset, workload_ids):
    observed_at = NOW + timedelta(seconds=offset)
    return host_telemetry_batch(
        f"host-{site['node']}-{offset}",
        observed_at,
        [
            HostMetricSample(
                name="host_gpu_utilization_percent",
                value=0.0,
                unit="percent",
                device=uuid,
            )
            for uuid in site["gpus"]
        ],
        cluster_id=site["cluster_id"],
        node_id=site["node"],
        workload_state="ACTIVE",
        affected_workload_ids=workload_ids,
        received_at=observed_at,
    )


@pytest.mark.parametrize("site", [HYPERPOD, PLAIN_EKS], ids=["hyperpod", "plain-eks"])
@pytest.mark.parametrize("held_count", [1, 8], ids=["one-gpu", "full-node"])
def test_the_watchers_gpu_uuids_select_the_graded_gpus_on_both_providers(
    site, held_count
):
    held = site["gpus"][-held_count:]
    observation = _watched_attempt(site, held)
    assert observation.environment is site["environment"]
    assert [item.node_id for item in observation.containers] == [site["node"]]
    assert observation.containers[0].gpu_uuids == list(held)
    store = build_store()
    policy = NodeHealthPolicy(store)

    fired = {}
    for offset in range(0, WINDOW + 61, 30):
        store.save_attempt_observation(
            observation.model_copy(
                update={"observed_at": NOW + timedelta(seconds=offset)}
            )
        )
        fired[offset] = policy.evaluate_metrics(
            _host_batch(site, offset, observation.workload_ids)
        )

    graded = sorted({item.device for items in fired.values() for item in items})
    assert graded == sorted(held), (
        f"{site['cluster_id']}: graded {graded}, the attempt holds {list(held)}"
    )
    assert {item.attempt_id for items in fired.values() for item in items} == {
        observation.attempt_id
    }
