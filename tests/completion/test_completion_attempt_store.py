"""Capacity and fidelity of the persisted active-attempt state (F6).

The Completion Watcher keeps one record per running attempt in a ConfigMap so a
restart does not lose the attempts it was watching. The record used to be the
whole published observation -- every container with its GPU UUIDs, container id
and cgroup path, ~1 KB per rank -- in the same ConfigMap as the critical
write-ahead log, so a large fleet filled the object and the terminal event of a
failing attempt could no longer be written ahead at all.
"""

from __future__ import annotations

import json

from gpu_fault.completion_attempt_state import active_pass_counts
from gpu_fault.completion_attempt_store import POD_FIELDS, PODS_KEY, SPEC_FIELDS
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from gpu_fault.models import Environment
from tests.completion.test_completion_controller import (
    NOW,
    Clock,
    FakeCoreApi,
    FakeSink,
    pod,
)

ACTIVE_KEY = "active-attempts.json"
#: Eight full-length GPU UUIDs, the way a p5 rank reports them: the single
#: biggest field of a container observation and the one that changes most often.
GPU_UUIDS = [f"GPU-{index:08d}-1234-5678-9abc-def012345678" for index in range(8)]


def training_pod(attempt_id: str, rank: int) -> dict:
    """One realistic managed training Pod of ``attempt_id``."""

    item = pod(rank, attempt_id=attempt_id, expected_ranks=8, gpu_count=8)
    metadata = item["metadata"]
    metadata["uid"] = f"pod-{attempt_id}-{rank}"
    metadata["name"] = f"{attempt_id}-worker-{rank}"
    metadata["annotations"]["gpu-fault.io/gpu-uuids"] = json.dumps(GPU_UUIDS)
    metadata["annotations"]["gpu-fault.io/cgroup-path"] = (
        "/kubepods.slice/kubepods-burstable.slice/"
        f"kubepods-burstable-pod{attempt_id}_{rank}.slice/cri-containerd-{rank:064d}"
    )
    item["status"]["containerStatuses"][0]["containerID"] = f"containerd://{rank:064d}"
    item["spec"]["nodeName"] = f"ip-10-0-{rank // 8}-{rank % 8}.ec2.internal"
    return item


def watcher(core: FakeCoreApi, sink, clock) -> KubernetesCompletionController:
    return KubernetesCompletionController(
        core,
        KubernetesCompletionOutbox(core, sink),
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        cleanup_timeout_seconds=30,
        attempt_missing_grace_seconds=30,
        now=clock,
        publish_observations=True,
    )


def test_a_thousand_running_pods_fit_the_active_state_and_a_terminal_still_delivers() -> (
    None
):
    """F6: routine attempt state must never starve critical delivery.

    A measured container observation is ~977 B, so ~920 managed Pods filled the
    900 000 B outbox and every further write raised: the attempt state stopped
    being persisted *and* the write-ahead copy of the next terminal could not be
    made, which is the durability the outbox exists for.
    """

    clock = Clock()
    attempts = [f"train-{index:03d}" for index in range(125)]
    pods = [
        training_pod(attempt_id, rank) for attempt_id in attempts for rank in range(8)
    ]
    assert len(pods) == 1000, f"the fleet under test must be 1000 Pods, got {len(pods)}"
    core = FakeCoreApi(pods)
    sink = FakeSink()
    subject = watcher(core, sink, clock)

    subject.run_once()

    persisted = json.loads(core.config_map_data[ACTIVE_KEY])
    assert sorted(key.rsplit("/", 1)[-1] for key in persisted) == attempts, (
        f"every running attempt must be persisted, got {len(persisted)} of "
        f"{len(attempts)}"
    )
    assert subject.sink.append_failures_total == 0, (
        "no outbox write may have failed while persisting routine state, got "
        f"{subject.sink.append_failures_total}"
    )

    for item in core.pods:
        if item["metadata"]["labels"]["gpu-fault.io/attempt-id"] == attempts[0]:
            item["status"]["containerStatuses"][0]["state"] = {
                "terminated": {"exitCode": 1, "finishedAt": NOW.isoformat()}
            }
    subject.run_once()

    terminals = [
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    ]
    assert [item["attempt_id"] for item in terminals] == [attempts[0]], (
        f"the failing attempt's terminal must be delivered, got {terminals}"
    )
    assert subject.sink.append_failures_total == 0, (
        "the terminal must have been written ahead before it was delivered, "
        f"got {subject.sink.append_failures_total} outbox write failures"
    )


def test_a_restored_attempt_posts_the_same_payload_as_before_the_restart() -> None:
    """The persisted record is smaller; what reaches the control plane is not.

    The restored attempt is rebuilt from its live Pods, so the observation the
    next pass publishes has to be byte-for-byte what the process before the
    restart published for the same Pods at the same instant.
    """

    clock = Clock()
    pods = [training_pod("train-a1", rank) for rank in range(8)]
    core = FakeCoreApi(pods)
    before = FakeSink()
    watcher(core, before, clock).run_once()

    after = FakeSink()
    restarted = watcher(core, after, clock)
    assert active_pass_counts(restarted, []) == (0, 1), (
        "the restarted watcher must count the persisted attempt as active "
        "before it has listed a single Pod"
    )
    restarted.run_once()

    def observation(sink: FakeSink) -> str:
        posted = [
            payload
            for path, payload in sink.posts
            if path == "/v1/workload-observations"
        ]
        return json.dumps(posted[-1], sort_keys=True)

    assert observation(after) == observation(before), (
        "the payload of a restored attempt must not change:\n"
        f"before={observation(before)}\nafter ={observation(after)}"
    )


def test_the_persisted_record_holds_the_spec_and_pod_identity_only() -> None:
    """The bytes that go into the ConfigMap are the ones a restart needs.

    Everything else -- the GPU UUIDs, the container id, the cgroup path, the log
    snapshot, the restart count -- is re-read from the live Pod on the very next
    pass, and persisting it both filled the object and rewrote the whole document
    every time one of those changed.
    """

    core = FakeCoreApi([training_pod("train-a1", rank) for rank in range(8)])
    watcher(core, FakeSink(), Clock()).run_once()

    persisted = json.loads(core.config_map_data[ACTIVE_KEY])
    record = persisted["hp-cluster/train-a1"]
    assert set(record) <= {*SPEC_FIELDS, PODS_KEY}, (
        f"nothing outside the spec and the Pod list may be persisted: {record}"
    )
    assert [item["rank"] for item in record[PODS_KEY]] == list(range(8)), (
        f"every rank must be remembered: {record[PODS_KEY]}"
    )
    for pod_record in record[PODS_KEY]:
        assert set(pod_record) <= {key for key, _field in POD_FIELDS}, (
            f"nothing outside Pod identity may be persisted: {pod_record}"
        )
    document = core.config_map_data[ACTIVE_KEY]
    for field in ("gpu_uuids", "cgroup_path", "container_id", "restart_count"):
        assert field not in document, (
            f"{field} is re-read from the Pod and must not be persisted: {document}"
        )
