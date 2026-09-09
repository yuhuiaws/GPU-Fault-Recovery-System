"""The completion outbox must not let one rejected record block the rest.

FINAL-建议汇总 F-G1 (P0-47C / P0-64A / P0-64B / P1-64C / P1-64F / P1-64G).
``replay()`` walked the ConfigMap in order and raised on the first failure, so
a terminal event the control plane permanently rejects (unknown runtime
profile, 409 from a stuck containment workflow) stopped every later record of
the whole GPU cluster from ever being delivered. The collector outbox in
``gpu_fault.collectors.sinks`` already isolates per record, keeps a budget and
marks records non-replayable; this file pins the same behaviour here and pins
that the two layers classify a status code the same way.
"""

from __future__ import annotations

import json
import logging

import pytest

from gpu_fault.collectors.sinks import CollectorError, is_retryable_delivery_status
from gpu_fault.completion_outbox import (
    KubernetesCompletionOutbox,
    completion_delivery_disposition,
    replay_quarantined_once,
)
from tests.completion.test_completion_outbox import ConfigMapCore, payload

TERMINAL = "/v1/attempts/terminal"


class ScriptedSink:
    """Fails specific attempts with a given error; accepts everything else."""

    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.failures = dict(failures or {})
        self.posts: list[tuple[str, str]] = []

    def post(self, path, payload):
        attempt = payload["attempt_id"]
        self.posts.append((path, attempt))
        error = self.failures.get(attempt)
        if error is not None:
            raise error
        return {"accepted": True}


def _buffered(core: ConfigMapCore, *attempts: str) -> None:
    """Leave one undelivered record per attempt in the ConfigMap."""
    outbox = KubernetesCompletionOutbox(
        core, ScriptedSink({a: OSError("down") for a in attempts})
    )
    for attempt in attempts:
        with pytest.raises(OSError):
            outbox.post(TERMINAL, payload(attempt))


def _records(core: ConfigMapCore) -> dict[str, dict]:
    return {item["key"]: item for item in json.loads(core.data["events.json"])}


def _rejected(status: int, text: str = "rejected") -> CollectorError:
    return CollectorError(
        f"collector event rejected ({status}): {text}", status_code=status
    )


def test_replay_skips_a_permanently_rejected_record_and_delivers_the_rest() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a", "attempt-b", "attempt-c")
    sink = ScriptedSink({"attempt-a": _rejected(404, "resource not found")})
    outbox = KubernetesCompletionOutbox(core, sink)

    replayed = outbox.replay()

    assert replayed == 2
    assert [attempt for _, attempt in sink.posts] == [
        "attempt-a",
        "attempt-b",
        "attempt-c",
    ]
    records = _records(core)
    assert set(records) == {"cluster-a/attempt-a/terminal"}
    assert records["cluster-a/attempt-a/terminal"]["quarantined"] is True
    assert records["cluster-a/attempt-a/terminal"]["last_status"] == 404
    assert outbox.depth() == 1
    assert outbox.quarantined_depth() == 1


def test_a_quarantined_record_is_not_replayed_every_cycle() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a")
    sink = ScriptedSink({"attempt-a": _rejected(422, "workload_ids is required")})
    outbox = KubernetesCompletionOutbox(core, sink)

    outbox.replay()
    outbox.replay()
    outbox.replay()

    assert len(sink.posts) == 1
    assert outbox.last_replay["quarantined"] == 0  # already quarantined, untouched
    assert outbox.quarantined_depth() == 1


def test_a_transient_failure_keeps_the_record_and_continues_with_the_next() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a", "attempt-b")
    sink = ScriptedSink(
        {"attempt-a": CollectorError("delivery failed", replayable=True)}
    )
    outbox = KubernetesCompletionOutbox(core, sink)

    replayed = outbox.replay()

    assert replayed == 1
    records = _records(core)
    assert set(records) == {"cluster-a/attempt-a/terminal"}
    assert records["cluster-a/attempt-a/terminal"].get("quarantined", False) is False
    assert records["cluster-a/attempt-a/terminal"]["attempts"] == 1
    assert outbox.last_replay == {
        "replayed": 1,
        "deferred": 1,
        "quarantined": 0,
        "expired": 0,
    }


def test_repeated_transient_failures_never_quarantine_by_count() -> None:
    """Final review C1: an outage is not a verdict on the record.

    Twenty failed attempts used to quarantine a record whose every failure
    was retryable -- about ten minutes of control-plane outage at one replay
    per 30 s pass -- after which ``replay`` skipped it for ever.
    """

    core = ConfigMapCore()
    _buffered(core, "attempt-a")
    sink = ScriptedSink({"attempt-a": OSError("down")})
    outbox = KubernetesCompletionOutbox(core, sink)

    for _ in range(25):
        outbox.replay()

    assert outbox.quarantined_depth() == 0, (
        "a retryable failure must never quarantine a record however often it "
        f"repeats: {_records(core)}"
    )
    assert _records(core)["cluster-a/attempt-a/terminal"]["attempts"] == 25
    assert outbox.expired_total == 0


def test_a_retry_disposition_older_than_a_day_expires_out_of_the_wal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The bound on a retryable record is its age, not its attempt count.

    Nothing that has failed for a whole day is a transient outage any more.
    The record is *removed* (R5 (a)): a day-old retry disposition kept in the
    WAL only consumed one of the 256 slots a critical event needs, and the
    live path of a still-held attempt keeps re-posting from its cache anyway.
    The expiry is counted and named at ERROR -- key, payload digest, and the
    fact that the one-shot cannot deliver what is gone.
    """

    now = [1_000_000.0]
    core = ConfigMapCore()
    outbox = KubernetesCompletionOutbox(
        core, ScriptedSink({"attempt-a": OSError("down")}), now=lambda: now[0]
    )
    with pytest.raises(OSError):
        outbox.post(TERMINAL, payload("attempt-a"))

    now[0] += 24 * 3600 - 1
    outbox.replay()
    assert outbox.depth() == 1, (
        f"one second short of a day is still a retry: {_records(core)}"
    )
    assert outbox.expired_total == 0

    now[0] += 1
    with caplog.at_level(logging.ERROR, logger="gpu_fault.completion_outbox"):
        outbox.replay()

    assert _records(core) == {}, (
        f"a day-old retry must be removed from the WAL, not kept: {_records(core)}"
    )
    assert outbox.expired_total == 1, (
        f"the expiry must be counted, expired_total={outbox.expired_total}"
    )
    assert outbox.last_replay == {
        "replayed": 0,
        "deferred": 0,
        "quarantined": 0,
        "expired": 1,
    }, outbox.last_replay
    assert (outbox.last_depth, outbox.last_quarantined_depth) == (0, 0), (
        "the depth gauges must drop with the removal, not one pass later"
    )
    assert outbox.buffered_attempt_ids == frozenset(), (
        "the WAL no longer names the attempt, so the controller may evict it: "
        f"{outbox.buffered_attempt_ids}"
    )
    errors = [
        message.getMessage()
        for message in caplog.records
        if message.levelno == logging.ERROR
    ]
    named = [text for text in errors if "cluster-a/attempt-a/terminal" in text]
    assert named, f"the expiry must be logged at ERROR with the record key: {errors}"
    assert "digest=" in named[0], named[0]
    assert "--replay-quarantined" in named[0] and "cannot" in named[0], (
        f"the ERROR must say the one-shot cannot revive a removed record: {named[0]}"
    )


def test_an_expired_record_whose_removal_fails_is_not_counted_yet() -> None:
    """Count the loss only once it happened; the next pass tries again."""

    from tests.completion.test_completion_outbox import FakeApiException, UnwritableCore

    now = [1_000_000.0]
    core = UnwritableCore()
    outbox = KubernetesCompletionOutbox(
        core, ScriptedSink({"attempt-a": OSError("down")}), now=lambda: now[0]
    )
    with pytest.raises(OSError):
        outbox.post(TERMINAL, payload("attempt-a"))
    now[0] += 24 * 3600
    core.writable = False

    with pytest.raises(FakeApiException):
        outbox.replay()

    assert outbox.expired_total == 0, "nothing was removed, so nothing expired"
    assert _records(core)["cluster-a/attempt-a/terminal"]["quarantined"] is False
    core.writable = True
    outbox.replay()
    assert _records(core) == {}
    assert outbox.expired_total == 1


def test_a_transient_failure_in_the_one_shot_keeps_a_day_old_rejected_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R5 fix 1, C1: the age bound is for live retry dispositions only.

    A rejected record is almost always older than a day by the time an
    operator runs ``--replay-quarantined``. R5 let a transient failure in that
    pass (a 503, a connection reset) read as "retryable and older than a day":
    the record was expired, removed, counted, and the one-shot exited 0 over an
    empty log -- the operator's evidence gone in the very command meant to
    recover it. A quarantined record keeps its quarantine, its reason and the
    422 evidence of the verdict; the transient failure is recorded beside it.
    """

    now = [1_000_000.0]
    core = ConfigMapCore()
    outbox = KubernetesCompletionOutbox(
        core,
        ScriptedSink({"attempt-a": _rejected(422, "workload_ids is required")}),
        now=lambda: now[0],
    )
    with pytest.raises(CollectorError):
        outbox.post(TERMINAL, payload("attempt-a"))
    outbox.replay()
    assert _records(core)["cluster-a/attempt-a/terminal"]["quarantined"] is True

    now[0] += 25 * 3600
    one_shot = KubernetesCompletionOutbox(
        core,
        ScriptedSink({"attempt-a": OSError("connection reset")}),
        now=lambda: now[0],
    )
    with caplog.at_level(logging.INFO, logger="test.one_shot"):
        rc = replay_quarantined_once(one_shot, logging.getLogger("test.one_shot"))

    record = _records(core).get("cluster-a/attempt-a/terminal")
    assert record is not None, (
        "a transient failure in the one-shot must not remove a rejected record: "
        f"{_records(core)}"
    )
    assert (
        record["quarantined"] is True and record["quarantine_reason"] == "rejected"
    ), f"the quarantine and its reason must survive a transient failure: {record}"
    assert record["last_status"] == 422 and "workload_ids" in record["last_error"], (
        f"the evidence of the verdict must not be overwritten by the outage: {record}"
    )
    assert record["attempts"] == 2, record
    assert record["last_transient_error"] == "connection reset", (
        f"the transient failure is recorded beside the verdict, not over it: {record}"
    )
    assert record["last_transient_status"] is None, record
    assert record["last_transient_at"] == now[0], record
    assert one_shot.expired_total == 0, (
        f"a rejected record never expires: expired_total={one_shot.expired_total}"
    )
    assert one_shot.last_replay == {
        "replayed": 0,
        "deferred": 1,
        "quarantined": 0,
        "expired": 0,
    }, one_shot.last_replay
    assert rc == 1, (
        f"the record is still quarantined, so the one-shot must say so: {rc}"
    )
    closing = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
        and "still quarantined" in record.getMessage()
    ]
    assert closing and "transiently" in closing[-1], (
        "the closing ERROR must blame this run's transient failure, not the 422 "
        f"verdict in last_error: {closing}"
    )


def test_the_one_shot_exits_nonzero_when_it_expired_a_live_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R5 fix 1, M2: a removal during the one-shot is a loss, not a success.

    The operator ran the command to recover records. A live retry disposition
    older than a day that fails transiently in that pass is expired and
    removed exactly as the loop would have done it; the one-shot must not then
    log ``expired=1`` at INFO and exit 0 over the empty log.
    """

    now = [1_000_000.0]
    core = ConfigMapCore()
    outbox = KubernetesCompletionOutbox(
        core, ScriptedSink({"attempt-a": OSError("down")}), now=lambda: now[0]
    )
    with pytest.raises(OSError):
        outbox.post(TERMINAL, payload("attempt-a"))
    now[0] += 25 * 3600
    one_shot = KubernetesCompletionOutbox(
        core,
        ScriptedSink({"attempt-a": OSError("connection reset")}),
        now=lambda: now[0],
    )

    with caplog.at_level(logging.INFO, logger="test.one_shot"):
        rc = replay_quarantined_once(one_shot, logging.getLogger("test.one_shot"))

    assert _records(core) == {}, f"a day-old live retry still expires: {_records(core)}"
    assert one_shot.expired_total == 1
    assert rc == 1, f"an expiry in the one-shot is a loss and must exit 1, got {rc}"
    errors = [
        message.getMessage()
        for message in caplog.records
        if message.levelno == logging.ERROR
    ]
    assert any("expired" in text and "1" in text for text in errors), (
        f"the loss must be summarised at ERROR, not INFO: {caplog.text}"
    )


def test_a_non_retryable_status_quarantines_on_first_sight() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a")
    outbox = KubernetesCompletionOutbox(
        core, ScriptedSink({"attempt-a": _rejected(422, "workload_ids is required")})
    )

    outbox.replay()

    record = _records(core)["cluster-a/attempt-a/terminal"]
    assert record["quarantined"] is True, record
    assert record["quarantine_reason"] == "rejected", record
    assert record["attempts"] == 1
    assert outbox.expired_total == 0, "a verdict is not an expiry"


def test_replay_respects_a_time_budget() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a", "attempt-b", "attempt-c", "attempt-d")
    clock = [0.0]

    class SlowSink(ScriptedSink):
        def post(self, path, payload):
            clock[0] += 3.0
            return super().post(path, payload)

    outbox = KubernetesCompletionOutbox(
        core, SlowSink(), replay_budget_seconds=5.0, monotonic=lambda: clock[0]
    )

    assert outbox.replay() == 2
    assert outbox.depth() == 2
    assert outbox.last_replay["deferred"] == 2


def test_a_quarantined_record_recovers_when_retried_explicitly() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a")
    KubernetesCompletionOutbox(
        core, ScriptedSink({"attempt-a": _rejected(404, "resource not found")})
    ).replay()
    recovered = ScriptedSink()
    outbox = KubernetesCompletionOutbox(core, recovered)

    assert outbox.replay() == 0  # quarantined records are left alone by default
    assert outbox.replay(include_quarantined=True) == 1
    assert outbox.depth() == 0
    assert recovered.posts == [(TERMINAL, "attempt-a")]


def test_stats_expose_depth_quarantine_and_oldest_age() -> None:
    core = ConfigMapCore()
    now = [1000.0]
    outbox = KubernetesCompletionOutbox(
        core, ScriptedSink({"attempt-a": OSError("down")}), now=lambda: now[0]
    )
    with pytest.raises(OSError):
        outbox.post(TERMINAL, payload("attempt-a"))
    now[0] += 90.0

    stats = outbox.stats()

    assert stats["depth"] == 1
    assert stats["quarantined"] == 0
    assert stats["oldest_age_seconds"] == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        (400, "quarantine"),
        (401, "quarantine"),
        (403, "quarantine"),
        (404, "quarantine"),
        (408, "retry"),
        (409, "quarantine"),
        (422, "quarantine"),
        (425, "retry"),
        (429, "retry"),
        (500, "retry"),
        (502, "retry"),
        (503, "retry"),
        (None, "retry"),
    ],
)
def test_both_layers_classify_a_status_code_the_same_way(status, disposition) -> None:
    error = CollectorError("delivery", status_code=status, replayable=status is None)

    assert completion_delivery_disposition(error) == disposition
    assert is_retryable_delivery_status(status) is (disposition == "retry")


def test_unknown_exceptions_are_retried_not_quarantined_on_first_sight() -> None:
    assert completion_delivery_disposition(OSError("socket")) == "retry"
    assert completion_delivery_disposition(RuntimeError("bug")) == "retry"
