from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timedelta, timezone

from gpu_fault.collectors import CollectorError
from gpu_fault.completion_controller import (
    KubernetesCompletionController,
    KubernetesWorkloadStopper,
)
from gpu_fault.models import Environment
from gpu_fault.watcher import failure_containment_ids

NOW = datetime(2026, 7, 20, 13, 30, tzinfo=timezone.utc)


class FakeCoreApi:
    def __init__(self, pods=None) -> None:
        self.pods = pods or []
        self.list_calls = 0
        self.pod_patches = []
        self.config_map_version = 1
        self.config_map_data = {"active-attempts.json": "{}", "events.json": "[]"}

    def list_pod_for_all_namespaces(self, **_kwargs):
        self.list_calls += 1
        return {"metadata": {"resourceVersion": "100"}, "items": self.pods}

    def patch_namespaced_pod(self, name, namespace, body):
        self.pod_patches.append((namespace, name, body))

    def read_namespaced_pod_log(self, *_args, **_kwargs):
        return "training log line\n"

    def read_namespaced_config_map(self, _name, _namespace):
        return {
            "metadata": {"resourceVersion": str(self.config_map_version)},
            "data": dict(self.config_map_data),
        }

    def replace_namespaced_config_map(self, _name, _namespace, body):
        assert body["metadata"]["resourceVersion"] == str(self.config_map_version)
        self.config_map_version += 1
        self.config_map_data = dict(body["data"])


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


def test_running_pod_does_not_submit_terminal() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()

    assert controller(core, sink).run_once() == []
    assert sink.posts == []


def test_disappeared_running_attempt_emits_stopped_tombstone_after_grace() -> None:
    clock = Clock()
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        now=clock,
        publish_observations=True,
    )

    subject.run_once()
    core.pods = []
    subject.run_once()
    clock.value += timedelta(seconds=29)
    subject.run_once()
    clock.value += timedelta(seconds=2)
    subject.run_once()

    observations = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ]
    assert observations[-2]["workload_phase"] == "RUNNING"
    assert observations[-1]["workload_phase"] == "STOPPED"
    assert observations[-1]["containers"] == []
    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "STOPPED"
    assert [item["rank"] for item in terminal["allocation"]] == [0]


def test_restart_restores_attempt_and_terminalizes_missing_pods() -> None:
    from gpu_fault.completion_outbox import KubernetesCompletionOutbox

    clock = Clock()
    core = FakeCoreApi([pod(0)])
    delivered = FakeSink()
    first_sink = KubernetesCompletionOutbox(core, delivered)
    first = KubernetesCompletionController(
        core,
        first_sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        now=clock,
        publish_observations=True,
    )
    first.run_once()
    assert json.loads(core.config_map_data["active-attempts.json"]), (
        "running attempt was not persisted for Watcher restart"
    )

    core.pods = []
    restarted_sink = KubernetesCompletionOutbox(core, delivered)
    restarted = KubernetesCompletionController(
        core,
        restarted_sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        now=clock,
        publish_observations=True,
    )
    restarted.run_once()
    clock.value += timedelta(seconds=31)
    restarted.run_once()

    observations = [
        value for path, value in delivered.posts if path == "/v1/workload-observations"
    ]
    assert observations[-1]["workload_phase"] == "STOPPED"
    assert json.loads(core.config_map_data["active-attempts.json"]) == {}


def test_observation_uses_earliest_pod_start_with_creation_fallback() -> None:
    subject = controller(FakeCoreApi([]), FakeSink())
    pods = [
        pod(
            0,
            expected_ranks=2,
            start_time=NOW - timedelta(seconds=20),
            creation_time=NOW - timedelta(seconds=30),
        ),
        pod(1, expected_ranks=2, creation_time=NOW - timedelta(seconds=10)),
    ]

    result = subject._observation("train-a1", pods, NOW)

    assert result.started_at == NOW - timedelta(seconds=20)


def test_observation_preserves_declared_gpu_count() -> None:
    subject = controller(FakeCoreApi([]), FakeSink())

    result = subject._observation("train-a1", [pod(0, gpu_count=8)], NOW)

    assert result.containers[0].gpu_count == 8


def test_terminal_rejection_does_not_block_other_attempts() -> None:
    class RejectFirstTerminalSink(FakeSink):
        def post(self, path, payload):
            self.posts.append((path, payload))
            if path == "/v1/attempts/terminal" and payload["attempt_id"] == "attempt-a":
                raise RuntimeError("containment is pending")
            return {"accepted": True}

    sink = RejectFirstTerminalSink()
    subject = controller(
        FakeCoreApi(
            [
                pod(0, exit_code=0, attempt_id="attempt-a"),
                pod(0, exit_code=0, attempt_id="attempt-b"),
            ]
        ),
        sink,
    )

    results = subject.run_once()

    terminal_attempts = [
        payload["attempt_id"]
        for path, payload in sink.posts
        if path == "/v1/attempts/terminal"
    ]
    assert terminal_attempts == ["attempt-a", "attempt-b"]
    assert results == [{"accepted": True}]


def test_running_pod_discovers_and_caches_gpu_uuids() -> None:
    core = FakeCoreApi([pod(0, include_gpu_uuids=False)])
    sink = FakeSink()
    calls = []

    def resolve(pod_body, container_name):
        calls.append((pod_body["metadata"]["uid"], container_name))
        return ["GPU-real-0", "GPU-real-1"]

    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        publish_observations=True,
        gpu_uuid_resolver=resolve,
    )

    subject.run_once()
    subject.run_once()

    assert calls == [("pod-0", "trainer")]
    observations = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ]
    assert observations[-1]["containers"][0]["gpu_uuids"] == [
        "GPU-real-0",
        "GPU-real-1",
    ]


def test_gpu_uuid_discovery_failure_uses_backoff() -> None:
    core = FakeCoreApi([pod(0, include_gpu_uuids=False)])
    clock = Clock()
    calls = []

    def resolve(_pod_body, _container_name):
        calls.append(clock.value)
        raise RuntimeError("pod exec unavailable")

    subject = KubernetesCompletionController(
        core, FakeSink(), cluster_id="hp-cluster", gpu_uuid_resolver=resolve, now=clock
    )

    subject.run_once()
    subject.run_once()
    assert len(calls) == 1

    clock.value += timedelta(seconds=3)
    subject.run_once()
    assert len(calls) == 2


def test_attempt_accepts_incident_termination_marker() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", publish_observations=True
    )

    subject.run_once()
    core.pods = [pod(0, initiator="incident-xid-11")]
    subject.run_once()

    observations = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ]
    assert observations[-1]["termination_initiator_incident_id"] == "incident-xid-11"


def test_restart_budget_is_propagated_to_terminal_event() -> None:
    core = FakeCoreApi([pod(0, exit_code=17, restart_budget=3)])
    sink = FakeSink()

    controller(core, sink).run_once()

    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["restart_budget"] == 3


def test_nonzero_exit_submits_terminal_once() -> None:
    core = FakeCoreApi([pod(0, exit_code=17)])
    sink = FakeSink()
    subject = controller(core, sink)

    subject.run_once()
    subject.run_once()

    terminals = [item for item in sink.posts if item[0] == "/v1/attempts/terminal"]
    assert len(terminals) == 1
    path, event = terminals[0]
    assert path == "/v1/attempts/terminal"
    assert event["terminal_status"] == "FAILED"
    assert event["rank_exit_status"][0]["exit_code"] == 17
    assert event["allocation"][0]["node_id"] == "node-0"


def test_multi_rank_waits_for_all_critical_ranks() -> None:
    core = FakeCoreApi(
        [pod(0, exit_code=1, expected_ranks=2), pod(1, expected_ranks=2)]
    )
    sink = FakeSink()
    subject = controller(core, sink)

    subject.run_once()
    assert sink.posts[0][0] == "/v1/attempts/failure-detected"

    core.pods = [
        pod(0, exit_code=1, expected_ranks=2),
        pod(1, exit_code=143, expected_ranks=2),
    ]
    subject.run_once()
    terminals = [
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    ]
    assert len(terminals) == 1
    exits = terminals[0]["rank_exit_status"]
    assert {item["exit_code"] for item in exits} == {1, 143}


def test_first_failed_rank_submits_control_plane_containment() -> None:
    workload_ids = ["training/job/rank-0", "training/job/rank-1"]
    core = FakeCoreApi(
        [
            pod(0, exit_code=1, expected_ranks=2, workload_ids=workload_ids),
            pod(1, expected_ranks=2, workload_ids=workload_ids),
        ]
    )
    sink = FakeSink()
    stopper = FakeStopper()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", workload_stopper=stopper
    )

    subject.run_once()
    subject.run_once()

    assert stopper.calls == []
    failures = [
        payload
        for path, payload in sink.posts
        if path == "/v1/attempts/failure-detected"
    ]
    assert len(failures) == 1
    assert failures[0]["workload_ids"] == workload_ids


def test_normal_failure_captures_logs_before_containment_post() -> None:
    workload_ids = ["training/pytorchjob/distributed-training"]
    core = FakeCoreApi(
        [
            pod(0, exit_code=1, expected_ranks=2, workload_ids=workload_ids),
            pod(1, expected_ranks=2, workload_ids=workload_ids),
        ]
    )

    class OrderingSink(FakeSink):
        def post(self, path, payload):
            if path == "/v1/attempts/failure-detected":
                assert len(core.pod_patches) == 2
                assert all(
                    (
                        "gpu-fault.io/workload-log-snapshot"
                        in patch[2]["metadata"]["annotations"]
                    )
                    for patch in core.pod_patches
                ), (
                    'expected all( ( "gpu-fault.io/workload-log-snapshot" in patch[2]["metadata"]["annotations"] ) for patch in core.pod_patches ) to be truthy'
                )
            return super().post(path, payload)

    sink = OrderingSink()
    stopper = KubernetesWorkloadStopper(batch_api=None, custom_api=None, core_api=core)
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", workload_stopper=stopper
    )

    subject.run_once()

    failures = [
        payload
        for path, payload in sink.posts
        if path == "/v1/attempts/failure-detected"
    ]
    assert len(failures) == 1
    snapshots = failures[0]["workload_log_snapshots"]
    assert len(snapshots) == 2
    assert all(snapshot["tail"] == "training log line\n" for snapshot in snapshots), (
        'expected all(snapshot["tail"] == "training log line\\n" for snapshot in snapshots) to be truthy'
    )
    assert all(snapshot["record_id"] for snapshot in snapshots), (
        'expected all(snapshot["record_id"] for snapshot in snapshots) to be truthy'
    )


def test_workload_log_s3_failure_keeps_tail_evidence() -> None:
    core = FakeCoreApi([pod(0, exit_code=1)])

    def fail_upload(_destination, _payload):
        raise RuntimeError("S3 unavailable")

    stopper = KubernetesWorkloadStopper(
        batch_api=None,
        custom_api=None,
        core_api=core,
        workload_log_s3_uri="s3://bucket/workload-logs",
        workload_log_uploader=fail_upload,
    )

    snapshots = stopper.capture_logs("train-a1", "incident-a", pods=core.pods)

    assert len(snapshots) == 1
    assert snapshots[0]["tail"] == "training log line\n"
    assert snapshots[0]["s3_uri"] is None
    assert "S3 unavailable" in snapshots[0]["archive_error"]
    assert snapshots[0]["record_id"]


def test_control_plane_timeout_uses_emergency_stop_fallback() -> None:
    workload_ids = ["training/pytorchjob/distributed-training"]
    clock = Clock()
    core = FakeCoreApi(
        [
            pod(0, exit_code=1, expected_ranks=2, workload_ids=workload_ids),
            pod(1, expected_ranks=2, workload_ids=workload_ids),
        ]
    )
    sink = FailingSink()
    snapshot = {
        "record_id": "workload-log/fallback",
        "node_id": "node-0",
        "captured_at": NOW.isoformat(),
        "tail": "training output",
    }
    stopper = FakeStopper([snapshot])
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        workload_stopper=stopper,
        emergency_fallback_seconds=30,
        now=clock,
    )

    subject.run_once()
    assert stopper.calls == []
    clock.value += timedelta(seconds=31)
    subject.run_once()
    subject.run_once()

    event_key = "hp-cluster/train-a1/TrainingAttemptFailureDetected"
    incident_id, _ = failure_containment_ids(event_key)
    assert stopper.calls == [(tuple(workload_ids), "train-a1", incident_id)]
    assert (
        subject._attempt_specs["train-a1"].termination_initiator_incident_id
        == incident_id
    )
    assert subject._failure_events["train-a1"].workload_log_snapshots == [snapshot]


def test_incident_termination_does_not_suspend_workload() -> None:
    workload_ids = ["training/pytorchjob/distributed-training"]
    core = FakeCoreApi(
        [pod(0, exit_code=143, initiator="incident-xid-11", workload_ids=workload_ids)]
    )
    sink = FakeSink()
    stopper = FakeStopper()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", workload_stopper=stopper
    )

    subject.run_once()

    assert stopper.calls == []
    assert [path for path, _ in sink.posts] == ["/v1/attempts/terminal"]
    assert sink.posts[0][1]["termination_initiator_incident_id"] == "incident-xid-11"


def test_passive_fallback_replays_failure_after_watcher_restart() -> None:
    event_key = "hp-cluster/train-a1/TrainingAttemptFailureDetected"
    incident_id, _ = failure_containment_ids(event_key)
    core = FakeCoreApi(
        [
            pod(
                0,
                exit_code=1,
                expected_ranks=2,
                initiator=incident_id,
                workload_ids=["training/pytorchjob/distributed-training"],
                gpu_count=1,
            ),
            pod(
                1,
                expected_ranks=2,
                initiator=incident_id,
                workload_ids=["training/pytorchjob/distributed-training"],
                gpu_count=1,
            ),
        ]
    )
    sink = FakeSink()
    subject = KubernetesCompletionController(core, sink, cluster_id="hp-cluster")

    subject.run_once()

    failures = [
        payload
        for path, payload in sink.posts
        if path == "/v1/attempts/failure-detected"
    ]
    assert len(failures) == 1
    assert failures[0]["attempt_id"] == "train-a1"


def test_indexed_job_reports_single_owning_workload() -> None:
    owner = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": "distributed-training",
            "controller": True,
        }
    ]
    core = FakeCoreApi(
        [
            pod(
                None,
                completion_index=0,
                exit_code=17,
                expected_ranks=2,
                namespace="training",
                owner_references=owner,
            ),
            pod(
                None,
                completion_index=1,
                expected_ranks=2,
                namespace="training",
                owner_references=owner,
            ),
        ]
    )
    sink = FakeSink()
    stopper = FakeStopper()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", workload_stopper=stopper
    )

    subject.run_once()
    subject.run_once()

    assert stopper.calls == []
    failures = [
        payload
        for path, payload in sink.posts
        if path == "/v1/attempts/failure-detected"
    ]
    assert failures[0]["workload_ids"] == ["training/job/distributed-training"]
    observation = subject._observation("train-a1", core.pods, NOW)
    assert {item.rank for item in observation.containers} == {0, 1}


def test_workload_inference_supports_jobset_and_pytorchjob() -> None:
    subject = controller(FakeCoreApi(), FakeSink())

    jobset_pod = pod(
        0,
        namespace="training",
        extra_labels={"jobset.sigs.k8s.io/jobset-name": "set-a"},
    )
    pytorch_pod = pod(
        0,
        namespace="training",
        extra_labels={"training.kubeflow.org/job-name": "torch-a"},
    )

    assert subject._workload_ids(jobset_pod) == ["training/jobset/set-a"]
    assert subject._workload_ids(pytorch_pod) == ["training/pytorchjob/torch-a"]


def test_unmanaged_jobset_observation_mode_never_contains_workload() -> None:
    value = pod(
        0,
        extra_labels={"jobset.sigs.k8s.io/jobset-name": "set-a"},
        owner_references=[
            {"kind": "JobSet", "name": "set-a", "uid": "jobset-uid", "controller": True}
        ],
    )
    labels = value["metadata"]["labels"]
    labels.pop("gpu-fault.io/managed")
    labels.pop("gpu-fault.io/attempt-id")
    labels.pop("gpu-fault.io/job-id")
    sink = FakeSink()
    subject = KubernetesCompletionController(
        FakeCoreApi([value]),
        sink,
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        now=lambda: NOW,
        publish_observations=True,
        observe_unmanaged_workloads=True,
        observation_runtime_profile_version="hyperpod-v1",
        workload_stopper=object(),
    )

    result = subject.run_once()

    assert result[0]["observation_only"] is True
    assert [path for path, _payload in sink.posts] == ["/v1/workload-observations"]


def test_observation_only_attempt_is_pruned_after_missed_cycles() -> None:
    value = pod(0, extra_labels={"jobset.sigs.k8s.io/jobset-name": "set-a"})
    labels = value["metadata"]["labels"]
    labels.pop("gpu-fault.io/managed")
    labels.pop("gpu-fault.io/attempt-id")
    labels.pop("gpu-fault.io/job-id")
    subject = KubernetesCompletionController(
        FakeCoreApi([value]),
        FakeSink(),
        cluster_id="hp-cluster",
        observe_unmanaged_workloads=True,
        observation_runtime_profile_version="hyperpod-v1",
        observation_only_retention_cycles=2,
    )
    subject.run_once()
    assert subject.observation_only.attempts, (
        "expected subject.observation_only.attempts to be truthy"
    )

    subject._reconcile([])
    subject._reconcile([])

    assert not subject.observation_only.attempts, (
        "expected subject.observation_only.attempts to be falsy"
    )
    assert not subject._attempt_specs, "expected subject._attempt_specs to be falsy"
    assert not subject._last_observations, (
        "expected subject._last_observations to be falsy"
    )


def test_rank_uses_kubeflow_replica_index_and_injected_offset() -> None:
    subject = controller(FakeCoreApi(), FakeSink())
    target = pod(None, extra_labels={"training.kubeflow.org/replica-index": "2"})
    target["metadata"]["annotations"]["gpu-fault.io/rank-offset"] = "1"

    assert subject._container(target).rank == 3


def test_rank_combines_jobset_job_and_completion_indexes() -> None:
    subject = controller(FakeCoreApi(), FakeSink())
    target = pod(
        None, completion_index=2, extra_labels={"jobset.sigs.k8s.io/job-index": "3"}
    )
    target["metadata"]["annotations"].update(
        {"gpu-fault.io/rank-offset": "1", "gpu-fault.io/rank-job-stride": "4"}
    )

    assert subject._container(target).rank == 15


def test_explicit_workloads_override_owner_and_are_deduplicated() -> None:
    subject = controller(FakeCoreApi(), FakeSink())
    explicit = ["training/job/explicit", "training/job/explicit"]
    target = pod(
        0,
        namespace="training",
        workload_ids=explicit,
        owner_references=[{"kind": "Job", "name": "inferred", "controller": True}],
    )

    assert subject._workload_ids(target) == explicit
    spec = subject._spec_from_pods("train-a1", [target])
    assert spec.workload_ids == ("training/job/explicit",)


def test_cleanup_timeout_submits_when_remaining_pod_disappears() -> None:
    clock = Clock()
    core = FakeCoreApi(
        [pod(0, exit_code=1, expected_ranks=2), pod(1, expected_ranks=2)]
    )
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        cleanup_timeout_seconds=30,
        now=clock,
        publish_observations=True,
    )

    subject.run_once()
    core.pods = []
    clock.value += timedelta(seconds=31)
    subject.run_once()

    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    observation = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ][-1]
    assert terminal["terminal_status"] == "TIMED_OUT"
    assert observation["workload_phase"] == "FAILED"
    assert [item["rank"] for item in observation["containers"]] == [0]
    assert observation["containers"][0]["terminated"] is True


def test_missing_required_metadata_fails_closed() -> None:
    invalid = pod(0, exit_code=1)
    del invalid["metadata"]["annotations"]["gpu-fault.io/runtime-profile-version"]
    sink = FakeSink()

    assert controller(FakeCoreApi([invalid]), sink).run_once() == []
    assert sink.posts == []


def test_consistent_new_attempt_spec_takes_over_after_mixed_generation() -> None:
    old = pod(0, expected_ranks=1)
    core = FakeCoreApi([old])
    sink = FakeSink()
    subject = controller(core, sink)
    subject.run_once()

    new_zero = pod(0, expected_ranks=2)
    new_one = pod(1, expected_ranks=2)
    mixed_one = pod(1, expected_ranks=1)
    core.pods = [new_zero, mixed_one]
    subject.run_once()

    assert subject.metadata_takeovers_total == 0
    assert subject._attempt_specs["train-a1"].expected_critical_ranks == 1

    core.pods = [new_zero, new_one]
    subject.run_once()

    assert subject.metadata_takeovers_total == 1
    assert subject._attempt_specs["train-a1"].expected_critical_ranks == 2
    assert subject._last_observations["train-a1"].expected_critical_ranks == 2


def test_successful_job_submits_no_failure_terminal() -> None:
    sink = FakeSink()
    controller(FakeCoreApi([pod(0, exit_code=0)]), sink).run_once()

    assert sink.posts[0][1]["terminal_status"] == "SUCCEEDED"


def test_deleted_successful_pods_do_not_regress_observation() -> None:
    core = FakeCoreApi(
        [pod(0, exit_code=0, expected_ranks=2), pod(1, exit_code=0, expected_ranks=2)]
    )
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", publish_observations=True, now=lambda: NOW
    )

    subject.run_once()
    core.pods = [pod(1, exit_code=0, expected_ranks=2)]
    subject.run_once()
    core.pods = []
    subject.run_once()

    observations = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ]
    assert len(observations) == 3
    assert {item["workload_phase"] for item in observations} == {"SUCCEEDED"}
    assert {len(item["containers"]) for item in observations} == {2}
    terminals = [
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    ]
    assert len(terminals) == 1


def test_incident_initiated_termination_is_stopped() -> None:
    sink = FakeSink()
    controller(
        FakeCoreApi([pod(0, exit_code=143, initiator="incident-1")]), sink
    ).run_once()

    event = sink.posts[0][1]
    assert event["terminal_status"] == "STOPPED"
    assert event["termination_initiator_incident_id"] == "incident-1"


def test_watch_uses_list_resource_version_and_modified_event() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    watched = FakeWatch([{"type": "MODIFIED", "object": pod(0, exit_code=17)}])
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", watch_factory=lambda: watched
    )

    subject._run_watch_cycle()

    assert core.list_calls == 1
    assert watched.stopped, "expected watched.stopped to be truthy"
    assert watched.arguments[0] == "list_pod_for_all_namespaces"
    assert watched.arguments[1]["resource_version"] == "100"
    assert watched.arguments[1]["allow_watch_bookmarks"] is True
    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "FAILED"


def test_watch_burst_reconciles_large_attempt_once_after_debounce() -> None:
    pods = [pod(index, expected_ranks=100) for index in range(100)]
    core = FakeCoreApi(pods)
    sink = FakeSink()
    watched = FakeWatch([{"type": "MODIFIED", "object": item} for item in pods])
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        watch_factory=lambda: watched,
        publish_observations=True,
        reconcile_debounce_seconds=10,
    )

    subject._run_watch_cycle()

    observations = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ]
    assert len(observations) == 2
    assert [len(item["containers"]) for item in observations] == [100, 100]
    assert subject.reconcile_runs_total == 2
    assert subject.reconciled_attempts_total == 2


def test_watch_event_only_reconciles_affected_attempt() -> None:
    attempt_a = pod(0, attempt_id="attempt-a")
    attempt_b = pod(0, attempt_id="attempt-b")
    attempt_b["metadata"]["name"] = "worker-b"
    attempt_b["metadata"]["uid"] = "pod-b"
    attempt_b["spec"]["nodeName"] = "node-b"
    core = FakeCoreApi([attempt_a, attempt_b])
    sink = FakeSink()
    watched = FakeWatch([{"type": "MODIFIED", "object": attempt_a}])
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        watch_factory=lambda: watched,
        publish_observations=True,
        reconcile_debounce_seconds=10,
    )

    subject._run_watch_cycle()

    attempts = [
        payload["attempt_id"]
        for path, payload in sink.posts
        if path == "/v1/workload-observations"
    ]
    assert attempts.count("attempt-a") == 2
    assert attempts.count("attempt-b") == 1


def test_watch_deleted_event_removes_pod_from_cache() -> None:
    core = FakeCoreApi([pod(0, expected_ranks=2), pod(1, expected_ranks=2)])
    sink = FakeSink()
    watched = FakeWatch([{"type": "DELETED", "object": pod(0)}])
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", watch_factory=lambda: watched
    )

    subject._run_watch_cycle()

    assert sink.posts == []
    assert watched.stopped, "expected watched.stopped to be truthy"


def test_expired_resource_version_stops_cycle_for_relist() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    watched = FakeWatch(
        [
            {
                "type": "ERROR",
                "object": {"kind": "Status", "code": 410, "reason": "Expired"},
            }
        ]
    )
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", watch_factory=lambda: watched
    )

    subject._run_watch_cycle()

    assert watched.stopped, "expected watched.stopped to be truthy"
    assert sink.posts == []


def test_http_410_watch_exception_stops_cycle_for_relist() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    watched = ExpiredWatch([])
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", watch_factory=lambda: watched
    )

    subject._run_watch_cycle()

    assert watched.stopped, "expected watched.stopped to be truthy"
    assert sink.posts == []


def test_kubernetes_workload_stopper_suspends_supported_kinds() -> None:
    class Batch:
        calls = []

        def patch_namespaced_job(self, name, namespace, body):
            self.calls.append((namespace, name, body))

    class Custom:
        calls = []

        def patch_namespaced_custom_object(
            self, group, version, namespace, plural, name, body
        ):
            self.calls.append((group, version, namespace, plural, name, body))

    batch = Batch()
    custom = Custom()
    stopper = KubernetesWorkloadStopper(batch, custom)

    stopper.stop(
        (
            "train/job-a",
            "train/job-a",
            "train/pytorchjob/torch-a",
            "train/jobset/set-a",
        ),
        "attempt-a",
    )

    assert batch.calls[0][2]["spec"] == {"suspend": True}
    assert (
        batch.calls[0][2]["metadata"]["annotations"]["gpu-fault.io/passive-stop-mode"]
        == "emergency-fallback"
    )
    assert len(batch.calls) == 1
    assert custom.calls[0][5]["spec"] == {"runPolicy": {"suspend": True}}
    assert custom.calls[1][5]["spec"] == {"suspend": True}


def test_emergency_stopper_annotates_pods_before_suspend() -> None:
    class Batch:
        calls = []

        def patch_namespaced_job(self, name, namespace, body):
            self.calls.append((namespace, name, body))

    class Custom:
        calls = []

        def patch_namespaced_custom_object(
            self, group, version, namespace, plural, name, body
        ):
            self.calls.append((group, version, namespace, plural, name, body))

    core = FakeCoreApi([pod(0, workload_ids=["train/job-a"])])
    batch = Batch()
    uploaded = {}

    def upload(destination, value):
        uploaded["destination"] = destination
        uploaded["value"] = value
        return destination

    stopper = KubernetesWorkloadStopper(
        batch,
        Custom(),
        core,
        workload_log_s3_uri="s3://audit-bucket/workload-logs",
        workload_log_uploader=upload,
    )

    snapshots = stopper.stop(("train/job-a",), "train-a1", "inc-attempt-failure-1")

    assert len(core.pod_patches) == 1
    namespace, pod_name, patch = core.pod_patches[0]
    assert (namespace, pod_name) == ("default", "worker-0")
    annotations = patch["metadata"]["annotations"]
    assert (
        annotations["gpu-fault.io/termination-initiator-incident-id"]
        == "inc-attempt-failure-1"
    )
    assert annotations["gpu-fault.io/passive-stop-attempt"] == ("train-a1")
    annotated_snapshot = json.loads(annotations["gpu-fault.io/workload-log-snapshot"])
    assert annotated_snapshot["record_id"] == snapshots[0]["record_id"]
    assert annotated_snapshot["s3_uri"] == snapshots[0]["s3_uri"]
    assert len(batch.calls) == 1
    assert snapshots[0]["tail"] == "training log line\n"
    assert snapshots[0]["s3_uri"].startswith("s3://audit-bucket/workload-logs/"), (
        'expected snapshots[0]["s3_uri"].startswith("s3://audit-bucket/workload-logs/") to be truthy'
    )
    assert gzip.decompress(uploaded["value"]) == (b"training log line\n")


def test_controller_can_publish_running_workload_observation() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", publish_observations=True
    )

    subject.run_once()

    assert sink.posts[0][0] == "/v1/workload-observations"
    assert sink.posts[0][1]["containers"][0]["rank"] == 0


def test_unknown_runtime_profile_404_is_reported_actionably(caplog) -> None:
    """A 404 on the terminal event names the unregistered profile.

    The control plane rejects a terminal event whose runtime profile was
    never registered. Retrying is correct, but the generic "will retry"
    message hides the one thing an operator has to fix, so the log line
    has to name the profile and the annotation that declared it.
    """

    class UnknownProfileSink(FakeSink):
        def post(self, path, payload):
            self.posts.append((path, payload))
            if path == "/v1/attempts/terminal":
                raise CollectorError(
                    "collector event rejected (404): "
                    '{"detail":"resource not found: '
                    'hyperpod-control-plane-recovery-v1"}'
                )
            return {"accepted": True}

    sink = UnknownProfileSink()
    subject = controller(FakeCoreApi([pod(0, exit_code=0)]), sink)

    with caplog.at_level(logging.ERROR):
        results = subject.run_once()

    assert results == []
    message = caplog.text
    assert "hyperpod-v1" in message
    assert "gpu-fault.io/runtime-profile-version" in message
    assert "POST /v1/runtime-profiles" in message
    assert "will not clear on its own" in message


def test_unknown_profile_matcher_ignores_other_failures() -> None:
    """Only a rendered 404-not-found counts as an unknown profile.

    A 404 without the NotFoundError detail, or any other status, must
    keep the original traceback-style logging; misclassifying a
    transient outage as a misconfiguration would tell operators to go
    register a profile that already exists.
    """
    matcher = KubernetesCompletionController._is_unknown_profile_rejection

    assert matcher(
        CollectorError(
            "collector event rejected (404): "
            '{"detail":"resource not found: hyperpod-v1"}'
        )
    ), (
        'expected matcher( CollectorError( "collector event rejected (404): " \'{"detail":"resource not found: hyperpod-v1"}\' ) ) to be truthy'
    )
    assert not matcher(RuntimeError("control plane unavailable")), (
        'expected matcher(RuntimeError("control plane unavailable")) to be falsy'
    )
    assert not matcher(CollectorError("collector event rejected (404): {}")), (
        'expected matcher(CollectorError("collector event rejected (404): {}")) to be falsy'
    )
    assert not matcher(
        CollectorError(
            "collector event rejected (409): "
            '{"detail":"resource not found: hyperpod-v1"}'
        )
    ), (
        'expected matcher( CollectorError( "collector event rejected (409): " \'{"detail":"resource not found: hyperpod-v1"}\' ) ) to be falsy'
    )
