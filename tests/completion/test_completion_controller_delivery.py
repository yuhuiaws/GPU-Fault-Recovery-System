"""Delivery and outbox interaction: what the controller does with the sink's answer.

Rejected terminals, the unknown-runtime-profile 404, the write-ahead outbox
(F1/F8), an unwritable outbox and a deferred terminal that must not count as
sent.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta

from gpu_fault.collectors import CollectorError
from gpu_fault.completion_controller import (
    KubernetesCompletionController,
    KubernetesWorkloadStopper,
)
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from gpu_fault.watcher import failure_containment_ids
from tests.completion._support import (
    Clock,
    FailingSink,
    FakeCoreApi,
    FakeSink,
    FakeStopper,
    controller,
    pod,
)


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


class BigLogCoreApi(FakeCoreApi):
    """A kubelet whose training log is a realistic 260 KB, not one line.

    ``workload_log_max_bytes`` is 262 144, so a normal 2000-line tail fills
    it; four ranks of it is ~1 MB of failure event.
    """

    LINE = "2026-07-20T13:30:00.000000000Z " + ("y" * 98) + "\n"

    def read_namespaced_pod_log(self, *_args, **_kwargs):
        return self.LINE * 2000


class BrokenOutboxCoreApi(FakeCoreApi):
    """The write-ahead ConfigMap cannot be written at all."""

    def replace_namespaced_config_map(self, _name, _namespace, body):
        error = RuntimeError('configmaps "…-outbox" is forbidden')
        error.status = 500
        raise error


class RecordingStopper(KubernetesWorkloadStopper):
    """A real stopper for log capture, but its emergency ``stop`` is recorded."""

    def __init__(self, core_api) -> None:
        super().__init__(batch_api=None, custom_api=None, core_api=core_api)
        self.stop_calls = []

    def stop(self, workload_ids, attempt_id, incident_id=None, *, capture_logs=True):
        self.stop_calls.append((tuple(workload_ids), attempt_id, incident_id))
        return []


class BufferSnapshotSink(FakeSink):
    """Records what the write-ahead ConfigMap held at each critical POST."""

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core
        self.buffered = []

    def post(self, path, payload):
        if path == "/v1/attempts/failure-detected":
            self.buffered.append(json.loads(self.core.config_map_data["events.json"]))
        return super().post(path, payload)


def test_failure_detected_with_large_log_tails_reaches_the_sink() -> None:
    """F1 (P0): a 4-rank attempt with normal logs used to deliver nothing.

    ``_append`` raised ``CompletionOutboxFull`` on a ~918 KB payload before
    the live POST, so no failure-detected ever reached the control plane and
    the 30 s fallback suspended the job with no incident recorded.
    """

    workload_ids = ["training/pytorchjob/distributed-training"]
    clock = Clock()
    core = BigLogCoreApi(
        [
            pod(
                rank,
                exit_code=1 if rank == 0 else None,
                expected_ranks=4,
                workload_ids=workload_ids,
            )
            for rank in range(4)
        ]
    )
    inner = BufferSnapshotSink(core)
    outbox = KubernetesCompletionOutbox(core, inner)
    stopper = RecordingStopper(core)
    subject = KubernetesCompletionController(
        core,
        outbox,
        cluster_id="hp-cluster",
        workload_stopper=stopper,
        emergency_fallback_seconds=30,
        now=clock,
    )

    subject.run_once()
    clock.value += timedelta(seconds=31)
    subject.run_once()

    failures = [
        payload
        for path, payload in inner.posts
        if path == "/v1/attempts/failure-detected"
    ]
    assert len(failures) == 1, (
        f"failure-detected must reach the sink exactly once, got {len(failures)}"
    )
    tails = [
        len(item["tail"].encode()) for item in failures[0]["workload_log_snapshots"]
    ]
    assert tails == [260_000] * 4, (
        f"the live POST must carry every full log tail, got {tails}"
    )
    assert stopper.stop_calls == [], (
        f"a delivered failure must not trigger the emergency stop: {stopper.stop_calls}"
    )
    assert outbox.append_failures_total == 0, (
        "the pointer-sized write-ahead copy must fit the outbox bound "
        f"(append_failures_total={outbox.append_failures_total})"
    )
    assert subject.outbox_append_failures_total == 0, (
        "the controller must export the outbox counter for /metrics"
    )
    assert len(inner.buffered) == 1, f"expected one WAL snapshot: {inner.buffered!r}"
    wal_tails = [
        len(item["tail"].encode())
        for item in inner.buffered[0][0]["payload"]["workload_log_snapshots"]
    ]
    assert all(size <= 8192 for size in wal_tails), (
        f"the buffered copy must be pointer-sized, tails were {wal_tails}"
    )
    assert json.loads(core.config_map_data["events.json"]) == [], (
        "a delivered record must be removed from the outbox: "
        f"{core.config_map_data['events.json']!r}"
    )


def test_buffered_failure_is_left_to_replay_but_keeps_the_stop_armed() -> None:
    """F8 must not disarm the last line of defence.

    Leaving a buffered record to ``replay`` removes the *second* live POST of
    the same pass, not the fact that the control plane has accepted nothing:
    the containment clock keeps running and the emergency workload stop still
    fires at ``emergency_fallback_seconds``.
    """

    workload_ids = ["training/pytorchjob/distributed-training"]
    clock = Clock()
    core = FakeCoreApi(
        [
            pod(0, exit_code=1, expected_ranks=2, workload_ids=workload_ids),
            pod(1, expected_ranks=2, workload_ids=workload_ids),
        ]
    )
    inner = FailingSink()
    outbox = KubernetesCompletionOutbox(core, inner)
    stopper = FakeStopper()
    subject = KubernetesCompletionController(
        core,
        outbox,
        cluster_id="hp-cluster",
        workload_stopper=stopper,
        emergency_fallback_seconds=30,
        now=clock,
    )

    subject.run_once()
    assert stopper.calls == [], f"not overdue yet: {stopper.calls!r}"
    clock.value += timedelta(seconds=31)
    subject.run_once()

    attempts = [path for path, _ in inner.posts]
    assert attempts == ["/v1/attempts/failure-detected"] * 2, (
        "one live attempt on the first pass and one replay attempt on the "
        f"second; the live path must not retry a buffered record: {attempts!r}"
    )
    event_key = "hp-cluster/train-a1/TrainingAttemptFailureDetected"
    incident_id, _ = failure_containment_ids(event_key)
    assert stopper.calls == [(tuple(workload_ids), "train-a1", incident_id)], (
        f"the emergency stop must still fire when nothing was delivered: "
        f"{stopper.calls!r}"
    )
    buffered = [item["key"] for item in json.loads(core.config_map_data["events.json"])]
    assert buffered == ["hp-cluster/train-a1/failure-detected"], (
        f"the undelivered event must still be buffered for replay: {buffered!r}"
    )


def test_unwritable_outbox_still_delivers_and_is_counted() -> None:
    core = BrokenOutboxCoreApi([pod(0, exit_code=1)])
    inner = FakeSink()
    outbox = KubernetesCompletionOutbox(core, inner)
    subject = KubernetesCompletionController(core, outbox, cluster_id="hp-cluster")

    subject.run_once()

    assert [path for path, _ in inner.posts] == [
        "/v1/attempts/failure-detected",
        "/v1/attempts/terminal",
    ], f"both critical events must be delivered live: {inner.posts!r}"
    assert subject.outbox_append_failures_total == 2, (
        "each write-ahead failure must be counted and exported, got "
        f"{subject.outbox_append_failures_total}"
    )


class TogglingSink(FakeSink):
    """Rejects every POST with a non-retryable 422 until ``fail`` is cleared."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def post(self, path, payload):
        self.posts.append((path, payload))
        if self.fail:
            raise CollectorError("collector event rejected (422)", status_code=422)
        return {"accepted": True}


def test_a_deferred_terminal_is_not_recorded_as_sent() -> None:
    """ "The buffered record carries it" is not a delivery guarantee.

    Marking ``_terminal_sent`` on a ``deferred_to_replay`` answer retires the
    live path for the lifetime of the process. Once ``replay`` quarantines the
    record -- which it does on a non-retryable status -- nothing but the live
    path would ever deliver the terminal event again. The live path has to
    stay the owner until the control plane really accepts it.
    """

    core = FakeCoreApi([pod(0, exit_code=0)])
    inner = TogglingSink()
    outbox = KubernetesCompletionOutbox(core, inner)
    subject = KubernetesCompletionController(core, outbox, cluster_id="hp-cluster")

    subject.run_once()  # live POST fails, the record is buffered
    subject.run_once()  # replay is rejected and quarantines it; live POST fails too
    assert outbox.quarantined_depth() == 1, (
        "a 422 verdict must quarantine on the first replay, "
        f"{core.config_map_data['events.json']!r}"
    )

    inner.fail = False
    subject.run_once()

    assert [path for path, _ in inner.posts] == ["/v1/attempts/terminal"] * 4, (
        "one live attempt per pass plus the single replay attempt; the third "
        f"pass must still try: {inner.posts!r}"
    )
    assert outbox.depth() == 0, (
        "the recovered delivery must clear the buffered terminal, "
        f"{core.config_map_data['events.json']!r} is left"
    )
