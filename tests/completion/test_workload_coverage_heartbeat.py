"""The completion watcher reports a coverage heartbeat after every full scan.

An idle cluster has no managed Pod, so the watcher used to say nothing and the
control plane could not tell "no attempt" from "watcher dead". The heartbeat
is the watcher's statement that it scanned at ``scanned_at`` and found
``attempt_count`` attempts; it rides the observation publish switch, is sent
only by full passes (a filtered pass has not seen the whole cluster), is
throttled, and is fire-and-forget: a failed heartbeat is dropped, never
buffered, because the next scan supersedes it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.channel_registry import WORKLOAD_COVERAGE_PATH
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from gpu_fault.models import Environment
from tests.completion.test_completion_controller import (
    FailingSink,
    FakeCoreApi,
    FakeSink,
    pod,
)

NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        return self.value


def _controller(core, sink, clock, **overrides) -> KubernetesCompletionController:
    arguments = dict(
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        cleanup_timeout_seconds=60,
        now=clock,
        publish_observations=True,
    )
    arguments.update(overrides)
    return KubernetesCompletionController(core, sink, **arguments)


def _heartbeats(sink) -> list[dict]:
    return [payload for path, payload in sink.posts if path == WORKLOAD_COVERAGE_PATH]


def test_an_empty_full_pass_reports_zero_attempts() -> None:
    clock = Clock()
    sink = FakeSink()

    _controller(FakeCoreApi([]), sink, clock).run_once()

    [heartbeat] = _heartbeats(sink)
    assert heartbeat["cluster_id"] == "hp-cluster"
    assert datetime.fromisoformat(heartbeat["scanned_at"]) == NOW
    assert heartbeat["attempt_count"] == 0
    assert heartbeat["observe_unmanaged"] is False


def test_a_full_pass_counts_the_attempts_it_saw() -> None:
    sink = FakeSink()

    _controller(FakeCoreApi([pod(0)]), sink, Clock()).run_once()

    assert [item["attempt_count"] for item in _heartbeats(sink)] == [1]


def test_heartbeats_are_throttled_between_full_passes() -> None:
    clock = Clock()
    sink = FakeSink()
    subject = _controller(FakeCoreApi([]), sink, clock, coverage_heartbeat_seconds=15)

    subject.run_once()
    clock.value = NOW + timedelta(seconds=5)
    subject.run_once()
    clock.value = NOW + timedelta(seconds=15)
    subject.run_once()

    assert [
        datetime.fromisoformat(item["scanned_at"]) for item in _heartbeats(sink)
    ] == [NOW, NOW + timedelta(seconds=15)]


def test_a_filtered_pass_does_not_claim_cluster_coverage() -> None:
    sink = FakeSink()
    subject = _controller(FakeCoreApi([pod(0)]), sink, Clock())

    subject._reconcile([pod(0)], attempt_filter={"job-1-attempt-1"})

    assert _heartbeats(sink) == []


def test_the_heartbeat_rides_the_observation_publish_switch() -> None:
    sink = FakeSink()

    _controller(FakeCoreApi([]), sink, Clock(), publish_observations=False).run_once()

    assert _heartbeats(sink) == []


def test_a_failed_heartbeat_is_dropped_not_buffered() -> None:
    core = FakeCoreApi([])
    failing = FailingSink()
    outbox = KubernetesCompletionOutbox(core, failing)
    subject = _controller(core, outbox, Clock())

    subject.run_once()  # must not raise

    assert _heartbeats(failing), "the attempt was made"
    assert outbox.depth() == 0, "a heartbeat is superseded by the next one"
