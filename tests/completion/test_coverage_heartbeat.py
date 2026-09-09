"""The watcher says "I watched everything and nothing is running" (F4).

Completion-watcher F4: the loop posted observations only for the attempts it
found, so a cluster with no managed job for ten minutes published nothing at
all. The control plane cannot tell that apart from a watcher that died, so it
answers UNKNOWN and every node-mutating plan is BLOCKED -- which is what
happened live to DESTR-016 attempt 4 on an idle cluster.

The heartbeat is deliberately weak evidence, and it is deliberately narrow:
it rides the ordinary sink (never the critical outbox), a delivery failure is
counted and dropped, only a *completed* full pass produces one, only a pass
that saw nothing running at all, only a watcher that watches every namespace,
and at most one per ``coverage_heartbeat_interval_seconds``. Every one of those
silences fails closed -- UNKNOWN blocks a plan, IDLE lets it reboot a node
without a checkpoint.
"""

from __future__ import annotations

import copy
import logging
from datetime import timedelta

import pytest

from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_metrics_server import render_completion_metrics
from gpu_fault.models import datetime_json_text
from gpu_fault.telemetry import ATTEMPT_COVERAGE_PATH
from tests.completion._support import NOW, Clock, FakeCoreApi, FakeSink, FakeWatch, pod

INSTANCE = "completion-watcher-0"


class RefusingSink(FakeSink):
    def post(self, path, payload):
        self.posts.append((path, payload))
        if path == ATTEMPT_COVERAGE_PATH:
            raise RuntimeError("control plane unavailable")
        return {"accepted": True}


class NamespacedCoreApi(FakeCoreApi):
    def list_namespaced_pod(self, _namespace, **kwargs):
        return self.list_pod_for_all_namespaces(**kwargs)


def finished_pod(rank: int, *, exit_code: int = 0) -> dict:
    """A Pod whose containers are done, the way Kubernetes leaves it behind."""

    value = copy.deepcopy(pod(rank, exit_code=exit_code))
    value["status"]["phase"] = "Succeeded" if exit_code == 0 else "Failed"
    return value


def _controller(
    core,
    sink,
    *,
    clock=None,
    publish=True,
    serializer=None,
    watch=None,
    namespace=None,
    interval=120,
    missing_grace=300,
):
    return KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        now=clock or (lambda: NOW),
        publish_observations=publish,
        watcher_instance=INSTANCE,
        reconcile_debounce_seconds=0,
        namespace=namespace,
        coverage_heartbeat_interval_seconds=interval,
        attempt_missing_grace_seconds=missing_grace,
        **({"serializer": serializer} if serializer is not None else {}),
        **({"watch_factory": watch} if watch is not None else {}),
    )


def _heartbeats(sink) -> list[dict]:
    return [payload for path, payload in sink.posts if path == ATTEMPT_COVERAGE_PATH]


def test_an_idle_cluster_still_reports_full_coverage_after_a_pass() -> None:
    core = FakeCoreApi([])
    sink = FakeSink()
    subject = _controller(core, sink)

    subject.run_once()

    heartbeats = _heartbeats(sink)
    assert len(heartbeats) == 1, (
        f"a completed pass over an idle cluster must publish coverage: {sink.posts}"
    )
    assert heartbeats[0] == {
        "cluster_id": "hp-cluster",
        "observed_at": datetime_json_text(NOW),
        "watched_pods": 0,
        "watched_attempts": 0,
        "resource_version": "100",
        "watcher_instance": INSTANCE,
    }, heartbeats[0]
    assert subject.coverage_heartbeats_total == 1, subject.coverage_heartbeats_total
    assert subject.last_coverage_heartbeat_at == NOW, subject.last_coverage_heartbeat_at


def test_a_cluster_of_finished_pods_is_idle() -> None:
    """The live case: the job ended, its Pods are not collected yet."""

    core = FakeCoreApi([finished_pod(0)])
    sink = FakeSink()
    subject = _controller(core, sink)

    subject.run_once()

    heartbeat = _heartbeats(sink)[-1] if _heartbeats(sink) else None
    assert heartbeat is not None, (
        "a Succeeded Pod holds no GPU and can linger for days; counting it "
        f"would keep the cluster UNKNOWN for ever: {sink.posts}"
    )
    assert (heartbeat["watched_pods"], heartbeat["watched_attempts"]) == (0, 0), (
        f"nothing was running, and the heartbeat must say so: {heartbeat}"
    )


def test_a_pass_that_watched_a_running_pod_claims_no_coverage() -> None:
    core = FakeCoreApi([pod(0), pod(1, attempt_id="train-b1")])
    sink = FakeSink()
    subject = _controller(core, sink)

    subject.run_once()

    assert _heartbeats(sink) == [], (
        "a busy cluster is answered by its observations; a heartbeat from a "
        "pass that saw running work would let a lost observation read as an "
        f"idle cluster: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total


@pytest.mark.parametrize(
    "phase",
    ["Unknown", "Terminating", None],
    ids=["kubelet-lost", "a-phase-this-release-does-not-know", "no-status-at-all"],
)
def test_a_managed_pod_that_has_not_finished_blocks_the_claim(phase) -> None:
    """Only Succeeded and Failed mean "not running".

    ``watched_pods`` exists for the managed Pod that could not be grouped -- no
    attempt-id label, the warn-and-skip case -- because such a Pod produces no
    observation at all, so the resolver has nothing else to read. Phase
    ``Unknown`` is the kubelet losing contact, which correlates with the GPU
    faults this system reacts to, and its containers may well still be running;
    a Pod whose status has not been written yet is the same unknown. Treating
    either as finished would let the heartbeat vouch for a cluster that is
    training.
    """

    ungrouped = copy.deepcopy(pod(0))
    ungrouped["metadata"]["labels"].pop("gpu-fault.io/attempt-id")
    if phase is None:
        ungrouped.pop("status")
    else:
        ungrouped["status"]["phase"] = phase
    sink = FakeSink()
    subject = _controller(FakeCoreApi([ungrouped]), sink)

    subject.run_once()

    assert _heartbeats(sink) == [], (
        "a managed Pod that has not reached Succeeded or Failed is running "
        "work nobody can attribute, which is precisely what the coverage "
        f"claim must not paper over: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total


def test_an_attempt_with_no_listed_pods_still_blocks_the_claim() -> None:
    """Restored state: this process believes an attempt runs, sees no Pod."""

    observation = {
        "cluster_id": "hp-cluster",
        "environment": "hyperpod-eks",
        "job_id": "train",
        "attempt_id": "train-a1",
        "observed_at": datetime_json_text(NOW),
        "workload_phase": "RUNNING",
        "expected_critical_ranks": 1,
        "runtime_profile_version": "hyperpod-v1",
        "cleanup_timeout_seconds": 120,
        "containers": [],
        "workload_ids": ["default/job/train"],
    }

    class RestoringSink(FakeSink):
        def load_attempt_observations(self):
            return [observation]

    sink = RestoringSink()
    subject = _controller(FakeCoreApi([]), sink)

    subject.run_once()

    assert _heartbeats(sink) == [], (
        "an attempt this watcher still believes is RUNNING is exactly the case "
        f"where IDLE would reboot a training node: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total


def test_a_pass_that_failed_on_an_attempt_claims_no_coverage() -> None:
    # Same shape as the finished-Pods case above -- nothing running, so the
    # pass would otherwise vouch for the cluster -- except that handling this
    # attempt raised, so no observation was published at all. The failure is
    # swallowed per attempt on purpose (P1-47A), which is exactly why the
    # heartbeat has to see it: "no observation" is then a lost write.
    unreadable = finished_pod(0)
    unreadable["metadata"]["annotations"].pop("gpu-fault.io/rank")
    sink = FakeSink()
    subject = _controller(FakeCoreApi([unreadable]), sink)

    subject.run_once()

    assert subject.reconcile_failures_total == 1, subject.reconcile_failures_total
    assert _heartbeats(sink) == [], (
        "a pass that could not publish one attempt's state must not then tell "
        f"the control plane there is nothing to publish: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total


def test_a_namespace_scoped_watcher_never_claims_cluster_coverage(caplog) -> None:
    core = NamespacedCoreApi([])
    sink = FakeSink()
    subject = _controller(core, sink, namespace="team-a")

    with caplog.at_level(logging.INFO, logger="gpu_fault.completion_attempt_state"):
        subject.run_once()
        subject.run_once()

    assert _heartbeats(sink) == [], (
        "coverage is a statement about every node of the cluster, and a "
        "watcher that lists one namespace knows nothing about a node running "
        f"a job in another one: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total
    scope_records = [
        record for record in caplog.records if "publishes no coverage" in record.message
    ]
    assert len(scope_records) == 1, (
        f"the reason belongs in the log once, not once per pass: {caplog.records}"
    )


def test_coverage_is_restated_once_per_interval_not_once_per_pass() -> None:
    clock = Clock()
    core = FakeCoreApi([])
    sink = FakeSink()
    subject = _controller(core, sink, clock=clock, interval=120)

    subject.run_once()
    clock.value += timedelta(seconds=30)
    subject.run_once()
    assert len(_heartbeats(sink)) == 1, (
        "the poll interval is seconds and the coverage window is minutes, so a "
        f"heartbeat per pass is pure load on a strict lane: {sink.posts}"
    )

    clock.value += timedelta(seconds=120)
    subject.run_once()

    stamps = [item["observed_at"] for item in _heartbeats(sink)]
    assert len(stamps) == 2, f"coverage must still be restated: {sink.posts}"
    assert stamps[0] != stamps[1], f"a restatement carries its own clock: {stamps}"
    assert subject.coverage_heartbeats_total == 2, subject.coverage_heartbeats_total


def test_a_refused_heartbeat_is_not_retried_on_the_next_pass() -> None:
    clock = Clock()
    sink = RefusingSink()
    subject = _controller(FakeCoreApi([]), sink, clock=clock)

    subject.run_once()
    clock.value += timedelta(seconds=30)
    subject.run_once()

    assert subject.coverage_heartbeat_failures_total == 1, (
        "the rate limit counts attempts, so a control plane that is refusing "
        f"this write is not asked again every pass: {sink.posts}"
    )


def test_an_aborted_pass_publishes_no_coverage() -> None:
    def explode(value):
        raise RuntimeError("unserializable Pod")

    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = _controller(core, sink, serializer=explode)

    with pytest.raises(RuntimeError, match="unserializable Pod"):
        subject.run_once()

    assert _heartbeats(sink) == [], (
        f"a pass that never finished must not claim coverage: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total


def test_only_the_cycles_full_pass_publishes_coverage() -> None:
    core = FakeCoreApi([])
    sink = FakeSink()
    watch = FakeWatch([{"type": "MODIFIED", "object": pod(0)}])
    subject = _controller(core, sink, watch=lambda: watch)

    subject.run_watch_cycle()

    assert len(_heartbeats(sink)) == 1, (
        "only the cycle's full pass proves coverage; the debounced pass over "
        f"one attempt does not: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 1, subject.coverage_heartbeats_total


def test_a_watcher_that_does_not_publish_observations_claims_no_coverage() -> None:
    core = FakeCoreApi([pod(0)])
    sink = FakeSink()
    subject = _controller(core, sink, publish=False)

    subject.run_once()

    assert _heartbeats(sink) == [], (
        "a watcher whose observations the control plane never sees must not "
        f"tell it the cluster is covered: {sink.posts}"
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total


def test_a_lost_heartbeat_is_counted_and_never_fails_the_pass() -> None:
    core = FakeCoreApi([])
    sink = RefusingSink()
    subject = _controller(core, sink)

    assert subject.run_once() == [], "a refused heartbeat must not fail the pass"
    assert subject.coverage_heartbeat_failures_total == 1, (
        subject.coverage_heartbeat_failures_total
    )
    assert subject.coverage_heartbeats_total == 0, subject.coverage_heartbeats_total
    assert subject.last_coverage_heartbeat_at is None, (
        "an undelivered heartbeat must not look delivered"
    )
    assert subject.last_cycle_completed_at == NOW, (
        "the pass itself completed and must still be recorded"
    )


def test_the_coverage_counters_are_scrapable() -> None:
    core = FakeCoreApi([])
    sink = FakeSink()
    subject = _controller(core, sink)
    subject.run_once()

    body = render_completion_metrics(subject)

    for name, value in (
        ("gpu_fault_completion_watcher_coverage_heartbeats_total", 1),
        ("gpu_fault_completion_watcher_coverage_heartbeat_failures_total", 0),
    ):
        assert f"# TYPE {name} counter" in body, f"{name} is not a counter: {body!r}"
        assert f"{name} {value}" in body.splitlines(), f"{name} missing from {body!r}"
    stamp = "gpu_fault_completion_watcher_last_coverage_heartbeat_timestamp"
    assert f"{stamp} {int(NOW.timestamp())}" in body.splitlines(), (
        f"{stamp} missing from {body!r}"
    )
