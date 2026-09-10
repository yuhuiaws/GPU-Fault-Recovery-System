"""Fakes and builders shared by the completion controller test modules."""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.models import Environment

NOW = datetime(2026, 7, 20, 13, 30, tzinfo=timezone.utc)


class FakeCoreApi:
    def __init__(self, pods=None) -> None:
        self.pods = pods or []
        self.list_calls = 0
        self.pod_patches = []
        self.config_map_version = 1
        # One entry per real object: the write-ahead log and the routine attempt
        # state live in separate ConfigMaps so routine state cannot fill the one
        # a terminal event is written to (F6).
        self.config_maps = {
            "gpu-fault-completion-watcher-outbox": {"events.json": "[]"},
            "gpu-fault-completion-watcher-outbox-active": {
                "active-attempts.json": "{}"
            },
        }

    @property
    def config_map_data(self):
        """Both objects merged, so a test can assert on either key."""

        return {
            key: value
            for item in self.config_maps.values()
            for key, value in item.items()
        }

    def list_pod_for_all_namespaces(self, **_kwargs):
        self.list_calls += 1
        return {"metadata": {"resourceVersion": "100"}, "items": self.pods}

    def patch_namespaced_pod(self, name, namespace, body):
        self.pod_patches.append((namespace, name, body))

    def read_namespaced_pod_log(self, *_args, **_kwargs):
        return "training log line\n"

    def read_namespaced_config_map(self, name, _namespace):
        assert name in self.config_maps, f"unexpected ConfigMap read: {name}"
        return {
            "metadata": {"resourceVersion": str(self.config_map_version)},
            "data": dict(self.config_maps[name]),
        }

    def replace_namespaced_config_map(self, name, _namespace, body):
        assert body["metadata"]["resourceVersion"] == str(self.config_map_version)
        assert name in self.config_maps, f"unexpected ConfigMap write: {name}"
        self.config_map_version += 1
        self.config_maps[name] = dict(body["data"])


class FakeWatch:
    def __init__(self, events) -> None:
        self.events = events
        self.arguments = None
        self.stopped = False

    def stream(self, method, **kwargs):
        self.arguments = (method.__name__, kwargs)
        yield from self.events

    def stop(self):
        self.stopped = True


class ExpiredWatch(FakeWatch):
    def stream(self, method, **kwargs):
        self.arguments = (method.__name__, kwargs)
        error = RuntimeError("resource version expired")
        error.status = 410
        raise error
        yield


class FakeSink:
    def __init__(self) -> None:
        self.posts = []

    def post(self, path, payload):
        self.posts.append((path, payload))
        return {"accepted": True}


class FailingSink(FakeSink):
    def post(self, path, payload):
        self.posts.append((path, payload))
        raise RuntimeError("control plane unavailable")


class FakeStopper:
    def __init__(self, snapshots=None) -> None:
        self.calls = []
        self.snapshots = snapshots or []

    def stop(self, workload_ids, attempt_id, incident_id=None):
        self.calls.append((workload_ids, attempt_id, incident_id))
        return list(self.snapshots)


class Clock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self):
        return self.value


def pod(
    rank: int | None,
    *,
    exit_code: int | None = None,
    attempt_id: str = "train-a1",
    expected_ranks: int = 1,
    initiator: str | None = None,
    workload_ids: list[str] | None = None,
    completion_index: int | None = None,
    namespace: str = "default",
    owner_references: list[dict] | None = None,
    extra_labels: dict[str, str] | None = None,
    restart_budget: int | None = None,
    include_gpu_uuids: bool = True,
    gpu_count: int = 0,
    start_time: datetime | None = None,
    creation_time: datetime | None = None,
    deletion_time: datetime | None = None,
):
    annotations = {
        "gpu-fault.io/expected-critical-ranks": str(expected_ranks),
        "gpu-fault.io/training-container": "trainer",
        "gpu-fault.io/runtime-profile-version": "hyperpod-v1",
    }
    if include_gpu_uuids:
        annotations["gpu-fault.io/gpu-uuids"] = f'["GPU-{rank or 0}"]'
    if rank is not None:
        annotations["gpu-fault.io/rank"] = str(rank)
    if completion_index is not None:
        annotations["batch.kubernetes.io/job-completion-index"] = str(completion_index)
    if initiator:
        annotations["gpu-fault.io/termination-initiator-incident-id"] = initiator
    if workload_ids:
        import json

        annotations["gpu-fault.io/workload-ids"] = json.dumps(workload_ids)
    if restart_budget is not None:
        annotations["gpu-fault.io/restart-budget"] = str(restart_budget)
    status = {"phase": "Running", "containerStatuses": [{"name": "trainer"}]}
    if start_time is not None:
        status["startTime"] = start_time.isoformat()
    if exit_code is not None:
        status["containerStatuses"][0].update(
            {
                "state": {
                    "terminated": {"exitCode": exit_code, "finishedAt": NOW.isoformat()}
                },
                "restartCount": 0,
            }
        )
    else:
        status["containerStatuses"][0]["state"] = {"running": {}}
    labels = {
        "gpu-fault.io/managed": "true",
        "gpu-fault.io/job-id": "train",
        "gpu-fault.io/attempt-id": attempt_id,
        "gpu-fault.io/role": "worker",
        "gpu-fault.io/critical": "true",
        **(extra_labels or {}),
    }
    metadata = {
        "name": f"worker-{rank if rank is not None else completion_index}",
        "namespace": namespace,
        "uid": f"pod-{rank if rank is not None else completion_index}",
        "labels": labels,
        "annotations": annotations,
    }
    if creation_time is not None:
        metadata["creationTimestamp"] = creation_time.isoformat()
    if deletion_time is not None:
        metadata["deletionTimestamp"] = deletion_time.isoformat()
    if owner_references is not None:
        metadata["ownerReferences"] = owner_references
    effective_rank = rank if rank is not None else completion_index
    return {
        "metadata": {**metadata},
        "spec": {
            "nodeName": f"node-{effective_rank}",
            "containers": [
                {
                    "name": "trainer",
                    "resources": {
                        "requests": {"nvidia.com/gpu": str(gpu_count)},
                        "limits": {"nvidia.com/gpu": str(gpu_count)},
                    },
                }
            ],
        },
        "status": status,
    }


def controller(core, sink, clock=None, cleanup_timeout=60):
    return KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        cleanup_timeout_seconds=cleanup_timeout,
        now=clock or (lambda: NOW),
    )
