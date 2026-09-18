"""Terminal failure event time is independent of retained result populations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    control_loop_metric_lines,
)
from gpu_fault.app.metric_scan_cache import MetricScanCache
from gpu_fault.app.process_metrics import parse_lines
from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.store import WorkflowLeaseError
from tests._builders import build_context, build_store

STAMP = "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds"


class Notifier:
    def __init__(self, *outcomes: NotificationStatus | Exception):
        self.outcomes = iter(outcomes)
        self.calls: list[str] = []

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        self.calls.append(notification.notification_id)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return NotificationResult(
            notification_id=notification.notification_id, status=outcome
        )


def notification(store, *, name: str = "notification-a") -> AdvisoryNotification:
    record = AdvisoryNotification(
        notification_id=name,
        deduplication_key=name,
        cluster_name="cluster-a",
        incident_id=f"incident-{name}",
        subject="Recovery needs an operator",
        body_text="Synthetic notification fixture",
        support_case_draft="Synthetic draft",
    )
    return store.save_notification_if_absent(record)


def metric_value(lines: list[str], name: str) -> float:
    (sample,) = parse_lines(lines).samples[name]
    return float(sample.value)


def stamp(service: AdvisoryNotificationService) -> float:
    return metric_value(
        control_loop_metric_lines(
            SimpleNamespace(context=SimpleNamespace(advisory_notifications=service))
        ),
        STAMP,
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    now = [datetime.now(timezone.utc) + timedelta(seconds=1)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr("gpu_fault.notification_service.datetime", Clock)
    return now


def service(store, notifier: Notifier, *, asynchronous: bool):
    return AdvisoryNotificationService(
        store,
        notifier,
        async_delivery=asynchronous,
        ttl_seconds=0,
        deliver_backlog=True,
        deliver_drills=True,
    )


@pytest.mark.parametrize(
    "outcome", [NotificationStatus.FAILED, RuntimeError("synthetic provider failure")]
)
def test_inline_terminal_failure_is_recorded_once_and_keeps_event_time(
    clock, outcome: NotificationStatus | Exception
) -> None:
    store = build_store()
    record = notification(store)
    sender = service(store, Notifier(outcome), asynchronous=False)
    assert stamp(sender) == 0
    result = sender.send(record.notification_id)
    assert result.status is NotificationStatus.FAILED
    assert store.get_notification_result(record.notification_id) == result
    assert stamp(sender) == pytest.approx(clock[0].timestamp(), abs=0.001, rel=0)


@pytest.mark.parametrize(
    "outcome",
    [NotificationStatus.SENT, NotificationStatus.SKIPPED, NotificationStatus.QUEUED],
)
def test_other_inline_outcomes_do_not_create_terminal_failure_events(
    clock, outcome: NotificationStatus
) -> None:
    store = build_store()
    record = notification(store)
    sender = service(store, Notifier(outcome), asynchronous=False)
    assert sender.send(record.notification_id).status is outcome
    assert stamp(sender) == 0


def test_outbox_retries_do_not_stamp_a_terminal_failure_and_success_stays_quiet(
    clock,
) -> None:
    store = build_store()
    record = notification(store)
    sender = service(
        store,
        Notifier(NotificationStatus.FAILED, NotificationStatus.SENT),
        asynchronous=True,
    )
    sender.dispatch_outbox("owner", max_attempts=2)
    assert store.get_notification_delivery(record.notification_id).status is (
        NotificationDeliveryStatus.RETRY
    )
    assert store.get_notification_result(record.notification_id) is None
    assert stamp(sender) == 0
    clock[0] += timedelta(seconds=60)
    sender.dispatch_outbox("owner", max_attempts=2)
    assert store.get_notification_result(record.notification_id).status is (
        NotificationStatus.SENT
    )
    assert stamp(sender) == 0


def test_dead_letter_has_an_event_but_is_not_pending_and_reads_do_not_restamp(clock):
    store = build_store()
    record = notification(store)
    notifier = Notifier(NotificationStatus.FAILED)
    sender = service(store, notifier, asynchronous=True)
    sender.dispatch_outbox("owner", max_attempts=1)
    first = stamp(sender)
    assert first == pytest.approx(clock[0].timestamp(), abs=0.001, rel=0)
    assert store.notification_delivery_stats()["pending"] == 0
    context = build_context(store=store)
    lines = closed_loop_metric_lines(
        SimpleNamespace(
            context=context, metric_scan_cache=MetricScanCache(store, ttl_seconds=0)
        )
    )
    assert metric_value(lines, "gpu_fault_notification_outbox_depth") == 0
    clock[0] += timedelta(seconds=60)
    assert sender.send(record.notification_id).status is NotificationStatus.FAILED
    assert sender.dispatch_outbox("owner").attempted == 0
    assert stamp(sender) == first
    assert notifier.calls == [record.notification_id]
    sender.requeue(record.notification_id)
    assert store.notification_delivery_stats()["pending"] == 1
    assert stamp(sender) == first


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_result_write_does_not_claim_a_recorded_terminal_event(
    clock, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    store = build_store()
    record = notification(store)
    sender = service(
        store, Notifier(NotificationStatus.FAILED), asynchronous=asynchronous
    )

    def failed(*_args, **_kwargs):
        raise RuntimeError("synthetic result write failure")

    if asynchronous:
        monkeypatch.setattr(store, "complete_notification_delivery", failed)
        sender.dispatch_outbox("owner", max_attempts=1)
        assert sender.delivery_errors_total == 1
    else:
        monkeypatch.setattr(store, "save_notification_result", failed)
        with pytest.raises(RuntimeError, match="result write failure"):
            sender.send(record.notification_id)
    assert stamp(sender) == 0
    assert store.get_notification_result(record.notification_id) is None


@pytest.mark.parametrize("terminal", [False, True])
def test_lease_failure_stamps_only_a_terminal_result_that_the_fallback_records(
    clock, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    store = build_store()
    record = notification(store)
    sender = service(store, Notifier(NotificationStatus.FAILED), asynchronous=True)

    def fenced(*_args, **_kwargs):
        raise WorkflowLeaseError("synthetic expired lease")

    monkeypatch.setattr(store, "complete_notification_delivery", fenced)
    sender.dispatch_outbox("owner", max_attempts=1 if terminal else 2)
    assert (stamp(sender) > 0) is terminal
    assert (
        store.get_notification_result(record.notification_id) is not None
    ) is terminal


def test_clock_regression_cannot_overwrite_a_newer_recorded_failure(clock):
    store = build_store()
    first = notification(store)
    second = notification(store, name="notification-b")
    sender = service(
        store,
        Notifier(NotificationStatus.FAILED, NotificationStatus.FAILED),
        asynchronous=False,
    )
    sender.send(first.notification_id)
    newest = stamp(sender)
    clock[0] -= timedelta(seconds=60)
    sender.send(second.notification_id)
    assert stamp(sender) == newest
