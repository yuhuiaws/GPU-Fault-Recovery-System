"""Completion controller memory and loop hygiene (FINAL-建议汇总 F-G4).

Four things the reconcile loop did not do: isolate a failure in the second
half of its body, evict the attempts the watcher core already pruned, bound
the time a Pod log capture may hold the loop, and start when one persisted
attempt record is not its own.
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest

from gpu_fault.completion_controller import (
    KubernetesCompletionController,
    KubernetesWorkloadStopper,
)
from gpu_fault.models import Environment
from tests._builders import attempt_observation
from tests.completion.test_completion_controller import (
    NOW,
    Clock,
    FakeCoreApi,
    FakeSink,
    FakeWatch,
    controller,
    pod,
)


def test_a_malformed_pod_does_not_stop_other_attempts_from_reconciling() -> None:
    """One attempt's bad data must cost that attempt, not the whole pass.

    ``status.startTime`` as an integer is not one of the two error types the
    loop caught, so the ``AttributeError`` escaped ``run_once`` and attempt-b's
    terminal was never submitted.
    """
    broken = pod(0, exit_code=0, attempt_id="attempt-a")
    broken["status"]["startTime"] = 12345
    healthy = pod(0, exit_code=0, attempt_id="attempt-b")
    healthy["metadata"]["uid"] = "pod-b"
    healthy["metadata"]["name"] = "worker-b"
    sink = FakeSink()
    subject = controller(FakeCoreApi([broken, healthy]), sink)

    results = subject.run_once()

    terminal_attempts = [
        payload["attempt_id"]
        for path, payload in sink.posts
        if path == "/v1/attempts/terminal"
    ]
    assert terminal_attempts == ["attempt-b"]
    assert results == [{"accepted": True}]
    assert subject.reconcile_failures_total == 1


class PersistingSink(FakeSink):
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        super().__init__()
        self.payloads = payloads

    def load_attempt_observations(self) -> list[dict[str, object]]:
        return list(self.payloads)


def test_unknown_persisted_attempt_record_is_skipped_not_fatal() -> None:
    """A foreign or unparsable persisted record must not keep the process down.

    Restart state is a ConfigMap shared by every Watcher generation; one record
    written by a differently configured generation used to raise out of the
    constructor and the process crash-looped until an operator edited JSON.
    """

    def payload(attempt_id: str, cluster_id: str) -> dict[str, object]:
        return attempt_observation(
            "train",
            attempt_id,
            NOW,
            cluster_id=cluster_id,
            environment=Environment.HYPERPOD_EKS,
            runtime_profile_version="hyperpod-v1",
        ).model_dump(mode="json")

    sink = PersistingSink(
        [
            payload("train-good", "hp-cluster"),
            payload("train-foreign", "other-cluster"),
            {"attempt_id": "train-garbage"},
        ]
    )

    subject = KubernetesCompletionController(
        FakeCoreApi([]),
        sink,
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        now=lambda: NOW,
    )

    assert set(subject._attempt_specs) == {"train-good"}
    assert subject.restore_skipped_total == 2


def test_terminal_attempt_state_is_evicted_with_the_watcher() -> None:
    """When the watcher prunes a terminal attempt, the controller lets go too.

    The controller kept ``_attempt_specs`` / ``_terminal_observations`` for
    every attempt it ever saw and re-fed the cached terminal observation each
    pass, so the watcher re-created the state it had just pruned: unbounded
    memory on one side, create/prune churn on the other.
    """
    clock = Clock()
    core = FakeCoreApi([pod(0, exit_code=0, attempt_id="old-attempt")])
    subject = KubernetesCompletionController(
        core,
        FakeSink(),
        cluster_id="hp-cluster",
        now=clock,
        terminal_retention_seconds=60,
    )

    subject.run_once()
    assert "old-attempt" in subject._terminal_observations

    core.pods = [pod(0, attempt_id="new-attempt")]
    clock.value += timedelta(seconds=61)
    subject.run_once()

    assert "old-attempt" not in subject._attempt_specs
    assert "old-attempt" not in subject._terminal_observations
    assert "old-attempt" not in subject._last_observations
    assert not any("/old-attempt/" in key for key in subject._terminal_sent), (
        'expected any("/old-attempt/" in key for key in subject._terminal_sent) to be false'
    )
    assert "old-attempt" not in subject.watcher._attempts
    assert subject.evicted_attempts_total == 1

    subject.run_once()

    assert "old-attempt" not in subject.watcher._attempts
    assert subject.evicted_attempts_total == 1


def test_log_capture_is_bounded_by_a_timeout() -> None:
    """A hung ``read_namespaced_pod_log`` must not hold the reconcile loop.

    The capture ran inside the loop with no request timeout and no overall
    budget, so one unresponsive kubelet stalled every other attempt's failure
    containment for as long as the API client cared to wait.
    """
    release = threading.Event()

    class SlowCore(FakeCoreApi):
        def __init__(self, pods: list[dict[str, object]]) -> None:
            super().__init__(pods)
            self.request_timeouts: list[object] = []

        def read_namespaced_pod_log(self, name: str, _namespace: str, **kwargs):
            self.request_timeouts.append(kwargs.get("_request_timeout"))
            if name == "worker-1":
                release.wait(5)
            return "training log line\n"

    core = SlowCore([pod(0, expected_ranks=2), pod(1, expected_ranks=2)])
    stopper = KubernetesWorkloadStopper(
        batch_api=None, custom_api=None, core_api=core, workload_log_timeout_seconds=0.2
    )

    started = time.monotonic()
    try:
        snapshots = stopper.capture_logs("train-a1", "inc-a", pods=core.pods)
    finally:
        release.set()

    assert time.monotonic() - started < 3
    by_pod = {snapshot["pod_name"]: snapshot for snapshot in snapshots}
    assert by_pod["worker-0"]["record_id"]
    assert "timed out" in by_pod["worker-1"]["capture_error"]
    assert set(core.request_timeouts) == {0.2}


class OverlapRecordingSink(FakeSink):
    """Records the thread and the time window of every POST (F5).

    The sink is a constructor argument, so this instruments the reconcile path
    through a public seam: no private attribute of the controller is patched.
    """

    def __init__(self, hold_seconds: float) -> None:
        super().__init__()
        self.hold_seconds = hold_seconds
        self.windows: list[tuple[str, float, float]] = []
        self.timer_post_started = threading.Event()
        self._lock = threading.Lock()

    def post(self, path, payload):
        thread = threading.current_thread()
        if thread is not threading.main_thread():
            self.timer_post_started.set()
        started = time.monotonic()
        time.sleep(self.hold_seconds)
        finished = time.monotonic()
        with self._lock:
            self.windows.append((thread.name, started, finished))
        return {"accepted": True}


class HandOffWatch(FakeWatch):
    """Ends the stream only once the debounce timer is inside the sink."""

    def __init__(self, events, started: threading.Event) -> None:
        super().__init__(events)
        self.started = started
        self.handed_off = False

    def stream(self, method, **kwargs):
        self.arguments = (method.__name__, kwargs)
        yield from self.events
        self.handed_off = self.started.wait(10)


def _overlapping(windows: list[tuple[str, float, float]]) -> tuple[str, str] | None:
    for index, (name, start, end) in enumerate(windows):
        for other_name, other_start, other_end in windows[index + 1 :]:
            if name == other_name:
                continue
            if start < other_end and other_start < end:
                return (name, other_name)
    return None


def test_timer_reconcile_never_overlaps_the_full_reconcile() -> None:
    """F5: the debounce timer must not still be reconciling in the next cycle.

    ``reconcile_lock`` was a per-cycle local, so a timer thread mid-reconcile
    (a terminal POST can take four retries plus a 120 s receipt poll) ran
    concurrently with the next cycle's full pass: duplicate POSTs, a
    ``_terminal_sent`` set reassigned under a concurrent ``add``, two threads
    passing the ``_workloads_stopped`` check, and racing outbox CAS writes.
    """
    sink = OverlapRecordingSink(0.3)
    core = FakeCoreApi([pod(0)])
    watches = [
        HandOffWatch([{"type": "MODIFIED", "object": pod(0)}], sink.timer_post_started),
        FakeWatch([]),
    ]
    subject = KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        watch_factory=lambda: watches.pop(0),
        publish_observations=True,
        reconcile_debounce_seconds=0.02,
        now=lambda: NOW,
    )

    subject.run_watch_cycle()
    subject.run_watch_cycle()

    assert len(sink.windows) >= 3, (
        f"expected an initial, a debounced and a second full pass POST: {sink.windows}"
    )
    assert {name for name, _, _ in sink.windows} != {threading.main_thread().name}, (
        "the debounced reconcile never ran on a timer thread, so the test "
        f"cannot observe an overlap: {sink.windows}"
    )
    overlap = _overlapping(sink.windows)
    assert overlap is None, (
        f"reconcile passes overlapped across threads {overlap}: {sink.windows}"
    )


class ArmedSerializer:
    """A serializer that starts failing once the watch stream has ended."""

    def __init__(self) -> None:
        self.armed = False
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        if self.armed:
            raise RuntimeError("serializer failed after the watch stream ended")
        return value if isinstance(value, dict) else value.to_dict()


class ArmingWatch(FakeWatch):
    def __init__(self, events, serializer: ArmedSerializer) -> None:
        super().__init__(events)
        self.serializer = serializer

    def stream(self, method, **kwargs):
        self.arguments = (method.__name__, kwargs)
        yield from self.events
        self.serializer.armed = True


def test_watcher_stops_even_when_flush_raises() -> None:
    """F11: a raising final flush must not leak the watch connection.

    ``finally: flush_pending(); watcher.stop()`` skipped ``stop()`` whenever
    the flush raised, so every such cycle left the API-server stream and its
    thread behind while ``_run_forever`` opened another one.
    """
    serializer = ArmedSerializer()
    watched = ArmingWatch([{"type": "MODIFIED", "object": pod(0)}], serializer)
    subject = KubernetesCompletionController(
        FakeCoreApi([pod(0)]),
        FakeSink(),
        cluster_id="hp-cluster",
        watch_factory=lambda: watched,
        serializer=serializer,
        reconcile_debounce_seconds=10,
        now=lambda: NOW,
    )

    with pytest.raises(RuntimeError, match="serializer failed"):
        subject.run_watch_cycle()

    assert watched.stopped, "watcher.stop() must run even when the flush raises"


def test_a_completed_full_pass_records_its_timestamp() -> None:
    """F3/F12: liveness needs a value that only a completed pass moves."""
    clock = Clock()
    subject = controller(FakeCoreApi([pod(0)]), FakeSink(), clock)

    assert subject.started_at == NOW, subject.started_at
    assert subject.last_cycle_completed_at is None, (
        "no full pass has completed yet, so there is no cycle timestamp"
    )

    clock.value += timedelta(seconds=5)
    subject.run_once()

    assert subject.last_cycle_completed_at == clock.value, (
        f"a completed full pass must stamp the clock: {subject.last_cycle_completed_at}"
    )
