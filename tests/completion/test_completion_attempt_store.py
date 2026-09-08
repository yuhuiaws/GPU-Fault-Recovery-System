"""Capacity and fidelity of the persisted active-attempt state (F6).

The Completion Watcher keeps one record per running attempt in a ConfigMap so a
restart does not lose the attempts it was watching. The record used to be the
whole published observation -- ~1 KB per rank, log tail and all -- in the same
ConfigMap as the critical write-ahead log, so a large fleet filled the object
and the terminal event of a failing attempt could no longer be written ahead at
all. It is now the spec plus each Pod's identity, in its own object.

What that identity has to include is the subject of the C1 case below: the GPU
UUIDs, the container id, the cgroup path and the host pid are how a fault is
attributed to a workload, so dropping them from the record made a restored
attempt whose Pods are gone resolve as IDLE. What is dropped is the log
snapshot, the restart count, the exit signal and the fabric partition.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

from gpu_fault.completion_attempt_state import active_pass_counts
from gpu_fault.completion_attempt_store import POD_FIELDS, PODS_KEY, SPEC_FIELDS
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_metrics_server import render_completion_metrics
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from gpu_fault.models import Environment
from gpu_fault.telemetry import WorkloadContext, WorkloadTopologyService
from gpu_fault.watcher import AttemptObservation
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


def watcher(
    core: FakeCoreApi, sink, clock, *, max_bytes: int = 900_000, monotonic=None
) -> KubernetesCompletionController:
    outbox = (
        KubernetesCompletionOutbox(core, sink, max_bytes=max_bytes)
        if monotonic is None
        else KubernetesCompletionOutbox(
            core, sink, max_bytes=max_bytes, monotonic=monotonic
        )
    )
    return KubernetesCompletionController(
        core,
        outbox,
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

    The compact record measures **833 B per Pod** on this fixture (8 full GPU
    UUIDs, a 140-character cgroup path and a 64-hex container id per rank -- the
    attribution identity C1 put back), so the 900 000 B bound holds ~1080
    managed Pods of attempt state in its own object, where before it held ~920
    *and* had to leave room for a terminal event. The per-Pod bound below is the
    ratchet: a field added to ``POD_FIELDS`` has to be worth its bytes.
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
    per_pod = len(core.config_map_data[ACTIVE_KEY].encode()) / len(pods)
    assert per_pod < 900, (
        f"the persisted record must stay under 900 B per Pod, got {per_pod:.0f} "
        "-- at 900 B the 900 000 B bound holds only 1000 managed Pods"
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

    The Pod's attribution identity -- its GPU UUIDs, container id, cgroup path
    and host pid -- is part of that, because a restart whose Pods are gone can
    only republish what it persisted and a fault is attributed by exactly those
    fields (C1). What is left out is what the next pass re-reads from a live Pod
    and no fault is resolved by: the log snapshot, the restart count (which
    rewrote the whole document every time a container restarted), the exit
    signal and the fabric partition.
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
    for field in ("restart_count", "workload_log_snapshot", "signal"):
        assert field not in document, (
            f"{field} is re-read from the Pod and must not be persisted: {document}"
        )
    fields = {field for _key, field in POD_FIELDS}
    for field in ("gpu_uuids", "container_id", "cgroup_path", "host_pid"):
        assert field in fields, (
            f"{field} is how a fault is attributed and must be persisted: {fields}"
        )
    assert GPU_UUIDS[0] in document, (
        "the persisted record must carry the GPU UUIDs a fault names, not just "
        f"a field called after them: {document[:200]}"
    )
    identity = record[PODS_KEY][0]
    persisted = {field for key, field in POD_FIELDS if key in identity}
    for field in ("gpu_uuids", "container_id", "cgroup_path"):
        assert field in persisted, (
            f"the fixture's Pod reports {field}, so the record must hold it: {identity}"
        )


def test_a_restored_attempt_whose_pods_are_gone_still_resolves_active() -> None:
    """The persisted record has to carry the fault-attribution identity (C1).

    A restart whose Pods are already gone can only republish what was persisted,
    and inside the missing-attempt grace it republishes it as RUNNING -- which
    *replaces* the stored observation on the control plane. Fault ingest then
    resolves the node by GPU UUID (and by cgroup path, container id, host pid),
    so a container observation without those fields matches nothing while
    looking perfectly fresh: the answer flips from ACTIVE to IDLE and a
    RESET/REBOOT plan proceeds without STOP_WORKLOADS.
    """

    clock = Clock()
    pods = [training_pod("train-a1", rank) for rank in range(8)]
    core = FakeCoreApi(pods)
    before = FakeSink()
    watcher(core, before, clock).run_once()
    node_id = pods[0]["spec"]["nodeName"]

    def resolved(sink: FakeSink) -> WorkloadContext:
        posted = [
            AttemptObservation.model_validate(payload)
            for path, payload in sink.posts
            if path == "/v1/workload-observations"
        ]
        return WorkloadTopologyService(
            SimpleNamespace(get_workload_coverage_heartbeat=lambda _cluster: None)
        ).resolve(
            "hp-cluster",
            node_id,
            clock(),
            observations=posted[-1:],
            target_gpu_uuids={GPU_UUIDS[0]},
        )

    assert resolved(before).workload_state == "ACTIVE", (
        "the pre-restart observation must resolve the faulted GPU to its attempt"
    )

    # The Pods are gone before the new process lists anything, which is the
    # window this exists for: nothing but the persisted record describes them.
    core.pods = []
    after = FakeSink()
    restarted = watcher(core, after, clock)
    restarted.run_once()

    def observation(sink: FakeSink) -> str:
        posted = [
            payload
            for path, payload in sink.posts
            if path == "/v1/workload-observations"
        ]
        return json.dumps(posted[-1], sort_keys=True)

    assert observation(after) == observation(before), (
        "a restored attempt with no Pods left may only republish what it "
        f"persisted:\nbefore={observation(before)}\nafter ={observation(after)}"
    )
    context = resolved(after)
    assert context.workload_state == "ACTIVE", (
        "the republished observation must still resolve the faulted GPU to its "
        f"attempt, got {context.workload_state} {context.attempt_ids}"
    )
    assert context.attempt_ids == ["train-a1"], (
        f"the attempt must be named: {context.attempt_ids}"
    )

    # And when the grace does expire, the tombstone's terminal still says which
    # GPUs the attempt held: the control plane's containment reads them.
    clock.value = NOW + timedelta(seconds=120)
    restarted.run_once()
    terminals = [
        payload for path, payload in after.posts if path == "/v1/attempts/terminal"
    ]
    assert terminals, "the expired grace must terminalize the restored attempt"
    allocated = [entry["gpu_uuids"] for entry in terminals[-1]["allocation"]]
    assert allocated == [GPU_UUIDS] * 8, (
        f"the tombstone terminal must keep every rank's GPU UUIDs: {allocated}"
    )


def refuse_active_state(core: FakeCoreApi, status: int) -> None:
    """Make ``<name>-active`` -- and only that object -- refuse every read."""

    original = core.read_namespaced_config_map

    def read(name, namespace):
        if name.endswith("-active"):
            error = RuntimeError(f"configmaps {name} is refused")
            error.status = status
            raise error
        return original(name, namespace)

    core.read_namespaced_config_map = read
    core.refused_read = original


def restore_active_state(core: FakeCoreApi) -> None:
    """Apply the manifest: the object answers again from now on."""

    core.read_namespaced_config_map = core.refused_read


def test_a_refused_active_state_is_a_gauge_the_watcher_exports() -> None:
    """I1: an object that is not there yet must be visible, not logged once.

    The watcher keeps delivering -- attempt state is a restart optimisation --
    so nothing in the counters, the depth gauges or the reconcile timestamps
    moves. Without this gauge the only trace is one ERROR line at the instant
    the outage started.
    """

    core = FakeCoreApi([training_pod("train-a1", rank) for rank in range(8)])
    sink = FakeSink()
    subject = watcher(core, sink, Clock())
    assert subject.active_state_unavailable == 0, (
        f"a healthy watcher must report zero, got {subject.active_state_unavailable}"
    )

    refuse_active_state(core, 403)
    subject.run_once()

    assert subject.active_state_unavailable == 1, (
        "a refused attempt-state object must be reported, got "
        f"{subject.active_state_unavailable}"
    )
    exposition = render_completion_metrics(subject)
    assert "gpu_fault_completion_active_state_unavailable 1" in exposition, (
        "the gauge must reach /metrics: "
        f"{[line for line in exposition.splitlines() if 'active_state' in line]}"
    )
    assert [path for path, _payload in sink.posts] == ["/v1/workload-observations"], (
        f"delivery must be unaffected: {[path for path, _ in sink.posts]}"
    )


def test_an_unpersistable_attempt_is_reported_once_not_once_per_pass(caplog) -> None:
    """M1: a record that does not fit is refused on every pass, for ever.

    ``save_attempt_observation`` raises ``CompletionOutboxFull`` at the byte
    bound and the caller swallows it, so the only cost was the traceback -- once
    per over-bound attempt per reconcile pass, which is the same log storm F9
    exists to stop.

    The real exception names the measurement (``... exceeds its byte bound
    (13245 > 7000 bytes)``), and that number moves whenever any *other* attempt
    in the same document changes size. Keying the already-reported set on the
    message text therefore reported a "new" failure on every pass, which is why
    the bound here is the real one and the attempt that fits keeps growing.
    """

    clock = Clock()
    fitting = [training_pod("train-a1", rank) for rank in range(2)]
    over_bound = [training_pod("train-b2", rank) for rank in range(2)]
    probe = FakeCoreApi(deepcopy(fitting))
    watcher(probe, FakeSink(), Clock()).run_once()
    # A bound the attempt that fits stays inside as it grows, and the second
    # attempt cannot possibly reach.
    fits = len(ACTIVE_KEY.encode()) + len(probe.config_map_data[ACTIVE_KEY].encode())
    core = FakeCoreApi(deepcopy(fitting) + deepcopy(over_bound))
    sink = FakeSink()
    subject = watcher(core, sink, clock, max_bytes=fits + 120)

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.completion_attempt_state"):
        for _ in range(3):
            subject.run_once()
            for item in core.pods:
                if item["metadata"]["labels"]["gpu-fault.io/attempt-id"] == "train-a1":
                    # One byte more of cgroup path per pass: the document the
                    # over-bound attempt is measured against is never the same
                    # size twice, which is what the live cluster does too.
                    item["metadata"]["annotations"]["gpu-fault.io/cgroup-path"] += "0"

    reported = [
        record
        for record in caplog.records
        if "persist the active state of" in record.getMessage()
    ]
    levels = [record.levelname for record in reported]
    assert levels == ["ERROR", "DEBUG", "DEBUG"], (
        f"one ERROR then DEBUG per repetition, got {levels}: "
        f"{[record.getMessage() for record in reported]}"
    )
    repeats = [
        record.getMessage() for record in reported if record.levelname == "DEBUG"
    ]
    assert repeats[0] != repeats[1], (
        "this test only proves anything while the reported size keeps moving: "
        f"{repeats}"
    )
    persisted = json.loads(core.config_map_data[ACTIVE_KEY])
    assert list(persisted) == ["hp-cluster/train-a1"], (
        f"the attempt that fits must still be persisted: {list(persisted)}"
    )
    posted = {
        payload["attempt_id"]
        for path, payload in sink.posts
        if path == "/v1/workload-observations"
    }
    assert posted == {"train-a1", "train-b2"}, (
        f"delivery must be unaffected by the refused record: {sorted(posted)}"
    )


def test_a_recovered_active_state_clears_the_gauge_on_an_idle_pass() -> None:
    """The watcher itself has to notice that the manifest was applied (I1).

    Nothing but a load/save/remove clears the gauge, and a cluster with no
    managed Pods makes none of them: the operator applied
    ``deploy/dataplane/completion-watcher.yaml`` and ``/metrics`` went on
    reporting the outage until someone restarted the watcher.
    """

    core = FakeCoreApi([])
    monotonic = [1_000.0]
    refuse_active_state(core, 404)
    subject = watcher(core, FakeSink(), Clock(), monotonic=lambda: monotonic[0])
    assert subject.active_state_unavailable == 1, (
        "this test needs the attempt-state object to be refusing at start-up"
    )

    restore_active_state(core)
    monotonic[0] += 61.0
    subject.run_once()

    assert subject.active_state_unavailable == 0, (
        "an idle pass after the object was applied must clear the gauge, got "
        f"{subject.active_state_unavailable}"
    )
