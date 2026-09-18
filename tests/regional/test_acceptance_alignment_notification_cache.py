from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from threading import Event, Thread

import pytest

from scripts.e2e.regional.notification_evidence import NotificationAcceptanceError
from scripts.e2e.regional.notification_probe_cache import cached_drill
from scripts.e2e.regional.probes import notification_drill
from tests.regional._cov95_notify_harness import Notifier


def test_completed_observation_can_be_rejudged_without_sending_again(tmp_path) -> None:
    calls = []
    path = tmp_path / "drill.json"
    identity = {"source_digest": "a", "release_id": "r", "attempt": 1}

    def execute():
        calls.append(True)
        return {"notifier_calls": 1}

    assert cached_drill(path, identity, execute) == {"notifier_calls": 1}
    assert cached_drill(path, identity, execute) == {"notifier_calls": 1}
    assert calls == [True]
    with pytest.raises(NotificationAcceptanceError, match="identity changed"):
        cached_drill(path, {**identity, "release_id": "other"}, execute)
    assert calls == [True]


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"identity": {"attempt": 1}, "state": "STARTED"},
        {"identity": {"attempt": 1}, "state": "COMPLETED", "result": None},
    ],
)
def test_unconfirmed_or_changed_observation_does_not_retry_provider(
    tmp_path, document
) -> None:
    path = tmp_path / "drill.json"
    path.write_text(json.dumps(document))
    calls = []
    with pytest.raises(NotificationAcceptanceError):
        cached_drill(path, {"attempt": 1}, lambda: calls.append(True))
    assert calls == []


def test_missing_ack_keeps_started_intent_and_blocks_duplicate_send(tmp_path) -> None:
    path = tmp_path / "drill.json"
    calls = []

    def lost_ack():
        calls.append(True)
        raise OSError("isolated ACK loss")

    with pytest.raises(OSError):
        cached_drill(path, {}, lost_ack)
    with pytest.raises(NotificationAcceptanceError, match="unconfirmed"):
        cached_drill(path, {}, lost_ack)
    assert calls == [True]
    assert json.loads(path.read_text())["state"] == "STARTED"


def test_symlink_and_nonobject_result_never_create_a_successful_receipt(
    tmp_path,
) -> None:
    path = tmp_path / "drill.json"
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(NotificationAcceptanceError, match="symlink"):
        cached_drill(path, {}, lambda: {})
    path.unlink()
    with pytest.raises(NotificationAcceptanceError, match="no observation"):
        cached_drill(path, {}, lambda: None)


def test_concurrent_observers_cannot_both_send_the_same_drill(tmp_path) -> None:
    started, finish = Event(), Event()
    calls, errors = [], []
    path = tmp_path / "drill.json"

    def execute():
        calls.append(True)
        started.set()
        assert finish.wait(5), "the concurrent caller must release the first drill"
        return {}

    def first():
        try:
            cached_drill(path, {}, execute)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=first)
    thread.start()
    try:
        assert started.wait(5), "the first drill must own the cache before contention"
        with pytest.raises(NotificationAcceptanceError, match="already running"):
            cached_drill(path, {}, execute)
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and errors == [] and calls == [True]


@pytest.mark.parametrize("delay", [-1, 91, True, 0.5])
def test_duplicate_period_must_be_bounded_before_sending(delay) -> None:
    notifier = Notifier()
    with pytest.raises(ValueError, match="duplicate delay"):
        notification_drill.replay_completion(
            "gpu-reset", "d", "c", notifier, duplicate_delay_seconds=delay
        )
    assert notifier.sent == []


def test_duplicate_window_waits_a_full_metric_period_but_never_past_deadline(
    monkeypatch,
) -> None:
    now = [datetime.now(timezone.utc)]
    waits = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    def sleep(delay):
        waits.append(delay)
        now[0] += timedelta(seconds=delay)

    monkeypatch.setattr(notification_drill, "datetime", Clock)
    monkeypatch.setattr(notification_drill.time, "sleep", sleep)
    notifier = Notifier()
    result = notification_drill.replay_completion(
        "gpu-reset", "d", "c", notifier, duplicate_delay_seconds=65
    )
    assert waits == [65] and len(notifier.sent) == 1
    assert datetime.fromisoformat(
        result["duplicate_window_start"]
    ) - datetime.fromisoformat(result["initial_completed_at"]) == timedelta(seconds=65)
    notifier.sent.clear()
    with pytest.raises(RuntimeError, match="maintenance deadline"):
        notification_drill.replay_completion(
            "gpu-reset",
            "next",
            "c",
            notifier,
            duplicate_delay_seconds=65,
            deadline=now[0] + timedelta(seconds=30),
        )
    assert notifier.sent == []


def test_drill_identity_survives_operator_evidence_and_unrelated_source(
    tmp_path,
) -> None:
    """A completed drill is re-judged with later receipts: its identity binds the
    plan details and the notification-runner modules, not the plan's argument
    digest (which the evidence flags change) nor the whole tree."""
    import json

    from scripts.e2e.regional import notification_probe_cache as cache
    from scripts.e2e.regional import run_notification_acceptance as runner

    plan = tmp_path / "plan.json"
    base = {
        "details_sha256": "d" * 64,
        "arguments_sha256": "a" * 64,
        "source_digest": "s" * 64,
    }
    plan.write_text(json.dumps(base), encoding="utf-8")
    first = runner.drill_plan_binding(plan)
    plan.write_text(
        json.dumps({**base, "arguments_sha256": "b" * 64, "source_digest": "t" * 64}),
        encoding="utf-8",
    )
    assert runner.drill_plan_binding(plan) == first == "d" * 64
    plan.write_text(json.dumps({**base, "details_sha256": "e" * 64}), encoding="utf-8")
    assert runner.drill_plan_binding(plan) != first
    assert runner.drill_plan_binding(tmp_path / "missing.json") is None
    assert cache.DRILL_SOURCE_MODULES == (
        "run_notification_acceptance.py",
        "notification_probe_cache.py",
        "notification_evidence.py",
    )
    assert len(runner.drill_source_digest()) == 64
    assert runner.drill_source_digest() == runner.drill_source_digest()
