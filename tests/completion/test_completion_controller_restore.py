"""Missing Pods, tombstones and restore from persisted attempt state (F2/F7/C1)."""

from __future__ import annotations

import json
import logging
from datetime import timedelta

from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from tests.completion._support import Clock, FakeCoreApi, FakeSink, FakeStopper, pod


def test_disappeared_running_attempt_emits_stopped_tombstone_after_grace() -> None:
    clock = Clock()
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        attempt_missing_grace_seconds=30,
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


def test_disappeared_attempt_takes_its_initiator_from_the_workload_object() -> None:
    """DESTR-015 live: STOP_WORKLOADS annotated the Pods and suspended the
    PyTorchJob; the Pods exited within one poll, so the watcher never saw the
    annotation and the STOPPED tombstone read as a user stop. The stop step
    also annotates the workload object, which outlives the Pods."""

    class FakeCustom:
        def get_namespaced_custom_object(
            self, group, version, namespace, plural, name, **kwargs
        ):
            assert kwargs.get("_request_timeout") is not None, (
                "the workload read on the observation path must be bounded"
            )
            assert (group, plural, namespace, name) == (
                "kubeflow.org",
                "pytorchjobs",
                "default",
                "trainer",
            )
            return {
                "metadata": {
                    "annotations": {
                        "gpu-fault.io/termination-initiator-incident-id": "inc-stop"
                    }
                }
            }

    stopper = FakeStopper()
    stopper.custom = FakeCustom()
    stopper.batch = None
    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/pytorchjob/trainer"])])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        attempt_missing_grace_seconds=30,
        now=clock,
        workload_stopper=stopper,
    )

    subject.run_once()
    core.pods = []
    subject.run_once()
    clock.value += timedelta(seconds=31)
    subject.run_once()

    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "STOPPED"
    assert terminal["termination_initiator_incident_id"] == "inc-stop"


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
        attempt_missing_grace_seconds=30,
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
        attempt_missing_grace_seconds=30,
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


def test_resumed_attempt_after_tombstone_reports_its_failure() -> None:
    """F2: a missing-Pod tombstone is a guess, not an observation.

    Every Pod of an attempt can be absent for minutes -- a ``spec.suspend``
    toggle, Kueue preemption and readmission, an operator recreating Pods
    after node loss -- and come back under the same attempt-id. The cached
    STOPPED observation used to short-circuit every later pass, so the
    returning ranks were invisible forever and their failure was never
    reported.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        attempt_missing_grace_seconds=30,
        now=clock,
        publish_observations=True,
    )

    subject.run_once()
    core.pods = []
    subject.run_once()
    clock.value += timedelta(seconds=31)
    subject.run_once()

    statuses = [
        payload["terminal_status"]
        for path, payload in sink.posts
        if path == "/v1/attempts/terminal"
    ]
    assert statuses == ["STOPPED"], f"expected one tombstone, got {statuses}"

    core.pods = [pod(0, exit_code=1)]
    clock.value += timedelta(seconds=5)
    subject.run_once()

    failures = [
        payload
        for path, payload in sink.posts
        if path == "/v1/attempts/failure-detected"
    ]
    assert len(failures) == 1, f"resumed attempt reported no failure: {failures}"
    assert failures[0]["exit_code"] == 1, failures[0]
    statuses = [
        payload["terminal_status"]
        for path, payload in sink.posts
        if path == "/v1/attempts/terminal"
    ]
    assert statuses == ["STOPPED", "FAILED"], (
        f"the tombstone still shadows the resumed attempt: {statuses}"
    )
    published = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ][-1]
    assert published["workload_phase"] == "FAILED", published
    assert subject.resumed_attempts_total == 1, "the resumed attempt was not counted"


def test_missing_grace_is_independent_of_cleanup_timeout() -> None:
    """F2: ``cleanup_timeout_seconds`` is a container-cleanup budget.

    Reusing it as "the whole attempt is gone" declared a user stop after
    30 s. The whole-attempt grace is minutes, and it must not push out the
    TIMED_OUT deadline of an attempt that already reported a failure.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        attempt_missing_grace_seconds=300,
        now=clock,
        publish_observations=True,
    )

    subject.run_once()
    core.pods = []
    clock.value += timedelta(seconds=31)
    subject.run_once()

    terminals = [path for path, _ in sink.posts if path == "/v1/attempts/terminal"]
    assert terminals == [], f"tombstoned on the cleanup budget: {terminals}"
    published = [
        payload for path, payload in sink.posts if path == "/v1/workload-observations"
    ][-1]
    assert published["workload_phase"] == "RUNNING", published

    clock.value += timedelta(seconds=300)
    subject.run_once()

    terminal = next(
        payload for path, payload in sink.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "STOPPED", terminal

    failing_clock = Clock()
    failing_core = FakeCoreApi(
        [pod(0, exit_code=1, expected_ranks=2), pod(1, expected_ranks=2)]
    )
    failing_sink = FakeSink()
    failing = KubernetesCompletionController(
        failing_core,
        failing_sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        now=failing_clock,
    )
    failing.run_once()
    failing_core.pods = []
    failing_clock.value += timedelta(seconds=31)
    failing.run_once()

    timed_out = next(
        payload
        for path, payload in failing_sink.posts
        if path == "/v1/attempts/terminal"
    )
    assert timed_out["terminal_status"] == "TIMED_OUT", (
        f"the longer missing grace delayed the cleanup timeout: {timed_out}"
    )


class FakeJobApi:
    """A ``BatchV1Api`` stand-in that serves one Job object."""

    def __init__(self, status, annotations=None) -> None:
        self.status = status
        self.annotations = annotations or {}
        self.reads = []

    def read_namespaced_job(self, name, namespace, **kwargs):
        self.reads.append((name, namespace, kwargs.get("_request_timeout")))
        return {
            "metadata": {"name": name, "annotations": dict(self.annotations)},
            "status": dict(self.status),
        }


def restored_watcher(core, delivered, clock, batch):
    """A second Watcher generation that restored one attempt from the
    persisted state and can no longer see any of its Pods (F7)."""

    first = KubernetesCompletionController(
        core,
        KubernetesCompletionOutbox(core, delivered),
        cluster_id="hp-cluster",
        attempt_missing_grace_seconds=30,
        now=clock,
        publish_observations=True,
    )
    first.run_once()
    core.pods = []
    stopper = FakeStopper()
    stopper.batch = batch
    stopper.custom = None
    restarted = KubernetesCompletionController(
        core,
        KubernetesCompletionOutbox(core, delivered),
        cluster_id="hp-cluster",
        attempt_missing_grace_seconds=30,
        now=clock,
        publish_observations=True,
        workload_stopper=stopper,
    )
    return restarted, stopper


def test_restored_attempt_with_gc_pods_and_succeeded_job_is_succeeded() -> None:
    """F7: a finish during watcher downtime is not a user stop.

    The Pods of an attempt that finished while the watcher was down are
    garbage-collected (``ttlSecondsAfterFinished``), so the restored RUNNING
    attempt sees no Pods at all. The workload object still records the
    outcome and outlives its Pods.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/job/trainer"])])
    delivered = FakeSink()
    batch = FakeJobApi({"succeeded": 1, "active": 0})
    restarted, _ = restored_watcher(core, delivered, clock, batch)

    restarted.run_once()
    clock.value += timedelta(seconds=31)
    restarted.run_once()

    terminal = next(
        payload for path, payload in delivered.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "SUCCEEDED", (
        f"a job that finished during downtime was recorded as a stop: {terminal}"
    )
    assert batch.reads, "the workload object was never read"
    assert batch.reads[0][2] is not None, (
        f"the workload read is unbounded: {batch.reads[0]}"
    )


def test_unreadable_workload_object_keeps_the_restored_attempt_stopped() -> None:
    """The workload-object read fails closed: unknown is a stop, not a success."""

    class BrokenBatch:
        def read_namespaced_job(self, name, namespace, **_kwargs):
            raise RuntimeError("api server unavailable")

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/job/trainer"])])
    delivered = FakeSink()
    restarted, _ = restored_watcher(core, delivered, clock, BrokenBatch())

    restarted.run_once()
    clock.value += timedelta(seconds=31)
    restarted.run_once()

    terminal = next(
        payload for path, payload in delivered.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "STOPPED", (
        f"an unreadable workload object must not invent an outcome: {terminal}"
    )


def test_restored_attempt_stopped_by_us_keeps_its_initiator() -> None:
    """C1: our own STOP_WORKLOADS leaves ``status.failed`` behind.

    STOP_WORKLOADS annotates the workload object and deletes the Pods, so the
    Job reports ``failed>=1`` afterwards. Deriving the outcome from that count
    turned a stop we initiated into an unattributed FAILED, which the control
    plane answers with a fresh incident and another STOP_WORKLOADS -- on
    workload IDs that are stable across attempts, so a job that has since
    restarted gets suspended. The annotation decides first.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/job/trainer"])])
    delivered = FakeSink()
    batch = FakeJobApi(
        {"failed": 1, "active": 0},
        annotations={"gpu-fault.io/termination-initiator-incident-id": "inc-stop"},
    )
    restarted, stopper = restored_watcher(core, delivered, clock, batch)

    restarted.run_once()
    clock.value += timedelta(seconds=31)
    restarted.run_once()

    terminal = next(
        payload for path, payload in delivered.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "STOPPED", (
        f"a stop we initiated was reported as a failure: {terminal}"
    )
    assert terminal["termination_initiator_incident_id"] == "inc-stop", (
        f"the initiator recorded on the workload object was dropped: {terminal}"
    )
    failures = [
        path for path, _ in delivered.posts if path.endswith("failure-detected")
    ]
    assert failures == [], f"a stop we initiated raised containment: {failures}"
    assert stopper.calls == [], (
        f"a stop we initiated was stopped again: {stopper.calls}"
    )


def test_restored_failed_attempt_posts_a_terminal_but_no_containment(caplog) -> None:
    """C1: a verdict recovered from the workload object is not per-rank evidence.

    There are no Pods left, so there is no failed rank, no node and no exit
    code to attribute. Publishing a synthesized failure-detected would open an
    incident and a STOP_WORKLOADS against workload IDs that outlive the
    attempt. The terminal is still reported, so the job can be restarted.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/job/trainer"])])
    delivered = FakeSink()
    batch = FakeJobApi(
        {"failed": 1, "active": 0, "conditions": [{"type": "Failed", "status": "True"}]}
    )
    restarted, stopper = restored_watcher(core, delivered, clock, batch)

    restarted.run_once()
    clock.value += timedelta(seconds=31)
    with caplog.at_level(logging.WARNING):
        restarted.run_once()

    terminal = next(
        payload for path, payload in delivered.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "FAILED", (
        f"a failure during downtime was recorded as a stop: {terminal}"
    )
    assert [item["exit_code"] for item in terminal["rank_exit_status"]] == [0], (
        f"a per-rank exit code was invented: {terminal}"
    )
    failures = [
        path for path, _ in delivered.posts if path.endswith("failure-detected")
    ]
    assert failures == [], f"a recovered verdict raised containment: {failures}"
    assert stopper.calls == [], (
        f"a recovered verdict stopped live workloads: {stopper.calls}"
    )
    assert "recovered from workload object; no per-rank evidence" in caplog.text, (
        f"the recovered verdict was not reported as unattributed: {caplog.text}"
    )


def test_mixed_succeeded_and_failed_counts_are_a_failure() -> None:
    """I1: a true ``Failed`` condition wins over any success count.

    ``{"succeeded": 3, "failed": 5}`` used to read as SUCCEEDED because the
    success count was checked first, so a failed run was recorded as a success
    and never recovered.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/job/trainer"])])
    delivered = FakeSink()
    batch = FakeJobApi(
        {
            "succeeded": 3,
            "failed": 5,
            "active": 0,
            "conditions": [{"type": "Failed", "status": "True"}],
        }
    )
    restarted, _ = restored_watcher(core, delivered, clock, batch)

    restarted.run_once()
    clock.value += timedelta(seconds=31)
    restarted.run_once()

    terminal = next(
        payload for path, payload in delivered.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "FAILED", (
        f"a failed run was recorded as a success: {terminal}"
    )


def test_restored_attempt_with_a_retrying_job_is_not_tombstoned_yet() -> None:
    """I2: ``failed>0`` on a Job that is still retrying is not a verdict.

    ``backoffLimit`` retries take longer than the missing grace, and an
    elastic PyTorchJob reports failed replicas while it is alive. Declaring
    FAILED there ends an attempt that is about to heal itself; the attempt
    stays in the missing state until the workload object goes quiet.
    """

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=["default/job/trainer"])])
    delivered = FakeSink()
    batch = FakeJobApi({"failed": 1, "active": 1})
    restarted, _ = restored_watcher(core, delivered, clock, batch)

    restarted.run_once()
    clock.value += timedelta(seconds=31)
    restarted.run_once()

    terminals = [path for path, _ in delivered.posts if path == "/v1/attempts/terminal"]
    assert terminals == [], f"a retrying job was terminalized: {terminals}"
    published = [
        payload
        for path, payload in delivered.posts
        if path == "/v1/workload-observations"
    ][-1]
    assert published["workload_phase"] == "RUNNING", (
        f"a retrying job was taken off the active list: {published}"
    )

    batch.status = {"failed": 1, "active": 0}
    clock.value += timedelta(seconds=1)
    restarted.run_once()

    terminal = next(
        payload for path, payload in delivered.posts if path == "/v1/attempts/terminal"
    )
    assert terminal["terminal_status"] == "FAILED", (
        f"the failure was not reported once the job went quiet: {terminal}"
    )
