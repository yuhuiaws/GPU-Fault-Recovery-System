"""The completion outbox must not let one rejected record block the rest.

FINAL-建议汇总 F-G1 (P0-47C / P0-64A / P0-64B / P1-64C / P1-64F / P1-64G).
``replay()`` walked the ConfigMap in order and raised on the first failure, so
a terminal event the control plane permanently rejects (an unknown runtime
profile, or -- before terminals were decided immediately and chained behind
their containment workflow -- a 409 from a containment still open) stopped
every later record of the whole GPU cluster from ever being delivered. The collector outbox in
``gpu_fault.collectors.sinks`` already isolates per record, keeps a budget and
marks records non-replayable; this file pins the same behaviour here and pins
that the two layers classify a status code the same way.
"""

from __future__ import annotations

import json

import pytest

from gpu_fault.collectors.sinks import CollectorError, is_retryable_delivery_status
from gpu_fault.completion_outbox import (
    KubernetesCompletionOutbox,
    completion_delivery_disposition,
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
    assert outbox.last_replay == {"replayed": 1, "deferred": 1, "quarantined": 0}


def test_repeated_transient_failures_eventually_quarantine() -> None:
    core = ConfigMapCore()
    _buffered(core, "attempt-a")
    sink = ScriptedSink({"attempt-a": OSError("down")})
    outbox = KubernetesCompletionOutbox(core, sink, max_replay_attempts=3)

    for _ in range(3):
        outbox.replay()

    assert outbox.quarantined_depth() == 1
    assert _records(core)["cluster-a/attempt-a/terminal"]["attempts"] == 3


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
