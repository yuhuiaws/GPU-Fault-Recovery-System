"""The watcher says "I watched everything and saw nothing" after a full pass.

Completion-watcher F4: the loop posted observations only for the attempts it
found, so a cluster with no managed job for ten minutes published nothing at
all. The control plane cannot tell that apart from a watcher that died, so it
answers UNKNOWN and every node-mutating plan is BLOCKED -- which is what
happened live to DESTR-016 attempt 4 on an idle cluster.

The heartbeat is deliberately weak evidence: it rides the ordinary sink (never
the critical outbox), a delivery failure is counted and dropped, and only a
*completed* full pass produces one. A pass that aborted half way through has
not watched the cluster, and a debounced pass over one attempt never did.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_metrics_server import render_completion_metrics
from gpu_fault.telemetry import ATTEMPT_COVERAGE_PATH
from tests.completion.test_completion_controller import (
    NOW,
    Clock,
    FakeCoreApi,
    FakeSink,
    FakeWatch,
    pod,
)

INSTANCE = "completion-watcher-0"


class RefusingSink(FakeSink):
    def post(self, path, payload):
        self.posts.append((path, payload))
        if path == ATTEMPT_COVERAGE_PATH:
            raise RuntimeError("control plane unavailable")
        return {"accepted": True}


def _controller(core, sink, *, clock=None, publish=True, serializer=None, watch=None):
    return KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        now=clock or (lambda: NOW),
        publish_observations=publish,
        watcher_instance=INSTANCE,
        reconcile_debounce_seconds=0,
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
        "observed_at": NOW.isoformat(timespec="microseconds")[:-6] + "Z",
        "watched_pods": 0,
        "watched_attempts": 0,
        "resource_version": "100",
        "watcher_instance": INSTANCE,
    }, heartbeats[0]
    assert subject.coverage_heartbeats_total == 1, subject.coverage_heartbeats_total
    assert subject.last_coverage_heartbeat_at == NOW, subject.last_coverage_heartbeat_at


def test_every_completed_pass_publishes_its_own_heartbeat() -> None:
    clock = Clock()
    core = FakeCoreApi([])
    sink = FakeSink()
    subject = _controller(core, sink, clock=clock)

    subject.run_once()
    clock.value += timedelta(seconds=30)
    subject.run_once()

    stamps = [item["observed_at"] for item in _heartbeats(sink)]
    assert len(stamps) == 2, f"the heartbeat interval is the pass: {sink.posts}"
    assert stamps[0] != stamps[1], (
        f"a second pass must restate coverage with its own clock: {stamps}"
    )
    assert subject.coverage_heartbeats_total == 2, subject.coverage_heartbeats_total


def test_the_heartbeat_counts_what_the_pass_actually_watched() -> None:
    core = FakeCoreApi([pod(0), pod(1, attempt_id="train-b1")])
    sink = FakeSink()

    _controller(core, sink).run_once()

    heartbeat = _heartbeats(sink)[-1]
    assert (heartbeat["watched_pods"], heartbeat["watched_attempts"]) == (2, 2), (
        f"the heartbeat must describe the pass, not an empty cluster: {heartbeat}"
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


def test_a_debounced_single_attempt_pass_publishes_no_coverage() -> None:
    core = FakeCoreApi([pod(0)])
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
