"""Reconcile and observation: one attempt through ``run_once`` / the watch cycle.

Pod -> spec/container parsing, workload phases, GPU UUID discovery, failure
detection and containment on the reconcile path, metadata takeover, and the
list/watch cycle. Missing-Pod tombstones and restore live in
``test_completion_controller_restore``; sink/outbox interaction in
``test_completion_controller_delivery``; the emergency stopper in
``test_completion_workload_stopper``.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.completion_controller import (
    KubernetesCompletionController,
    KubernetesWorkloadStopper,
)
from gpu_fault.models import Environment
from gpu_fault.watcher import failure_containment_ids
from tests.completion._support import (
    NOW,
    Clock,
    ExpiredWatch,
    FailingSink,
    FakeCoreApi,
    FakeSink,
    FakeStopper,
    FakeWatch,
    controller,
    pod,
)


def test_running_pod_does_not_submit_terminal() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()

    assert controller(core, sink).run_once() == []
    assert sink.posts == []


def test_pending_pod_without_statuses_is_pending() -> None:
    """F10: an unscheduled Pod is not a running attempt."""

    unscheduled = pod(0)
    unscheduled["status"] = {"phase": "Pending"}
    del unscheduled["spec"]["nodeName"]
    sink = FakeSink()
    subject = KubernetesCompletionController(
        FakeCoreApi([unscheduled]),
        sink,
        cluster_id="hp-cluster",
        now=lambda: NOW,
        publish_observations=True,
    )

    subject.run_once()

    published = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ][-1]
    assert published["workload_phase"] == "PENDING", published
    assert published["started_at"] is None, (
        f"an unscheduled Pod has no start time: {published}"
    )
    assert published["containers"][0]["node_id"] is None, (
        f"an unscheduled Pod must not claim a node: {published}"
    )


def test_crashlooping_pod_is_running_not_pending() -> None:
    """F10 must not swing the other way: a restarting container has started.

    A Pod in CrashLoopBackOff reports ``state.waiting`` with the exit recorded
    in ``lastState``; reading that as PENDING would hide the attempt from hang
    detection for the whole backoff window.
    """

    crashlooping = pod(0)
    crashlooping["status"]["containerStatuses"][0].update(
        {
            "state": {"waiting": {"reason": "CrashLoopBackOff"}},
            "lastState": {"terminated": {"exitCode": 1, "finishedAt": NOW.isoformat()}},
            "restartCount": 2,
        }
    )
    sink = FakeSink()
    subject = KubernetesCompletionController(
        FakeCoreApi([crashlooping]),
        sink,
        cluster_id="hp-cluster",
        now=lambda: NOW,
        publish_observations=True,
    )

    subject.run_once()

    published = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ][-1]
    assert published["workload_phase"] == "RUNNING", (
        f"a restarting container reads as never started: {published}"
    )


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

    subject.run_watch_cycle()

    assert core.list_calls == 1
    assert watched.stopped, "expected watched.stopped to be truthy"
    assert watched.arguments[0] == "list_pod_for_all_namespaces"
    assert watched.arguments[1]["resource_version"] == "100"
    assert watched.arguments[1]["allow_watch_bookmarks"] is True
    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "FAILED"


def test_watch_stream_passes_a_client_read_timeout() -> None:
    """F3: without ``_request_timeout`` the watch blocks for ever.

    ``kubernetes/client/rest.py`` leaves ``timeout=None`` when the kwarg is
    absent, so an API-server endpoint that dies without an RST -- or a NAT
    that drops the idle stream -- parks the watch thread with no relist, no
    reconcile and no observations until someone deletes the Pod.
    """
    core = FakeCoreApi([pod(0)])
    watched = FakeWatch([])
    subject = KubernetesCompletionController(
        core,
        FakeSink(),
        cluster_id="hp-cluster",
        watch_factory=lambda: watched,
        watch_timeout_seconds=30,
    )

    subject.run_watch_cycle()

    assert watched.arguments[1]["_request_timeout"] == (5, 45), (
        "stream must carry a (connect, read) client timeout above the server "
        f"timeout: {watched.arguments[1]!r}"
    )


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

    subject.run_watch_cycle()

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

    subject.run_watch_cycle()

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

    subject.run_watch_cycle()

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

    subject.run_watch_cycle()

    assert watched.stopped, "expected watched.stopped to be truthy"
    assert sink.posts == []


def test_http_410_watch_exception_stops_cycle_for_relist() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    watched = ExpiredWatch([])
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", watch_factory=lambda: watched
    )

    subject.run_watch_cycle()

    assert watched.stopped, "expected watched.stopped to be truthy"
    assert sink.posts == []


def test_controller_can_publish_running_workload_observation() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core, sink, cluster_id="hp-cluster", publish_observations=True
    )

    subject.run_once()

    assert sink.posts[0][0] == "/v1/workload-observations"
    assert sink.posts[0][1]["containers"][0]["rank"] == 0
