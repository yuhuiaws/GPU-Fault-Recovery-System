"""The notification outbox exports its real state machine (ARCH-E E1).

``complete_notification_delivery`` wrote a ``FAILED`` result row on every
attempt, so one provider hiccup on a notification the outbox was about to
retry looked, on ``/metrics`` and to the critical alert, exactly like a
notification that had been given up on. The result row now means what the
alert reads it as: terminal. A retry keeps its reason on the delivery row
(``last_error``, ``attempts``) and nowhere else.

The delivery table itself was invisible: nothing exported PENDING/LEASED/
RETRY/SENT/DEAD or how long the oldest undelivered notification had waited.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    control_loop_metric_lines,
)
from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.store import SqliteStore, WorkflowLeaseError
from tests._builders import build_store
from tests.notifications._support import _watermark_already_drawn
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

# Delivery rows are stamped with the wall clock when they are saved, so every
# instant here is relative to it rather than to a fixed date, and far enough
# ahead that a slow Postgres schema check cannot put a row after it.
NOW = datetime.now(timezone.utc) + timedelta(hours=1)
LEASE = timedelta(seconds=120)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        instance = SqliteStore(str(tmp_path / "delivery-state.db"))
        try:
            yield instance
        finally:
            instance.close()
        return
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


def _notification(store, suffix: str, **values) -> AdvisoryNotification:
    return store.save_notification_if_absent(
        AdvisoryNotification(
            notification_id=f"notification-state-{suffix}",
            deduplication_key=f"notification-state-{suffix}",
            cluster_name="cluster-a",
            incident_id=f"incident-state-{suffix}",
            subject="subject",
            body_text="body",
            support_case_draft="",
            **values,
        )
    )


def _failed(notification_id: str) -> NotificationResult:
    return NotificationResult(
        notification_id=notification_id,
        status=NotificationStatus.FAILED,
        reason="RuntimeError: SES unavailable",
    )


def test_a_retryable_failure_keeps_its_reason_on_the_delivery_not_as_a_result(
    store,
) -> None:
    notification = _notification(store, "retry")
    [delivery] = store.claim_notification_deliveries(
        "pod-a", now=NOW, lease_duration=LEASE, limit=5
    )

    value = store.complete_notification_delivery(
        notification.notification_id,
        owner_id="pod-a",
        lease_epoch=delivery.lease_epoch,
        result=_failed(notification.notification_id),
        now=NOW + timedelta(seconds=1),
        retry_at=NOW + timedelta(seconds=30),
        terminal=False,
    )

    assert value.status is NotificationDeliveryStatus.RETRY, value
    assert value.attempts == 1, value
    assert value.last_error == "RuntimeError: SES unavailable", value
    assert store.get_notification_result(notification.notification_id) is None, (
        "a retry is not a terminal outcome and must not be recorded as one"
    )
    counts = store.notification_status_counts()
    assert counts[NotificationStatus.FAILED] == 0, counts
    assert counts[NotificationStatus.QUEUED] == 1, counts


def test_a_terminal_failure_is_recorded_as_failed(store) -> None:
    notification = _notification(store, "dead")
    [delivery] = store.claim_notification_deliveries(
        "pod-a", now=NOW, lease_duration=LEASE, limit=5
    )

    value = store.complete_notification_delivery(
        notification.notification_id,
        owner_id="pod-a",
        lease_epoch=delivery.lease_epoch,
        result=_failed(notification.notification_id),
        now=NOW + timedelta(seconds=1),
        terminal=True,
    )

    assert value.status is NotificationDeliveryStatus.DEAD, value
    result = store.get_notification_result(notification.notification_id)
    assert result is not None and result.status is NotificationStatus.FAILED, result
    assert store.notification_status_counts()[NotificationStatus.FAILED] == 1


def test_a_stale_lease_cannot_complete_a_delivery_someone_else_reclaimed(store) -> None:
    """The epoch check is the only thing between two dispatchers and one row."""
    notification = _notification(store, "reclaimed")
    [first] = store.claim_notification_deliveries(
        "pod-a", now=NOW, lease_duration=LEASE, limit=5
    )
    # pod-a's lease lapses; pod-b re-claims the expired LEASED row.
    later = NOW + LEASE + timedelta(seconds=1)
    [second] = store.claim_notification_deliveries(
        "pod-b", now=later, lease_duration=LEASE, limit=5
    )
    assert second.lease_epoch == first.lease_epoch + 1, second

    with pytest.raises(WorkflowLeaseError):
        store.complete_notification_delivery(
            notification.notification_id,
            owner_id="pod-a",
            lease_epoch=first.lease_epoch,
            result=NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
            ),
            now=later + timedelta(seconds=1),
        )
    released = store.release_notification_delivery(
        notification.notification_id,
        owner_id="pod-a",
        lease_epoch=first.lease_epoch,
        now=later + timedelta(seconds=1),
        retry_at=later + timedelta(seconds=30),
    )
    assert released is None, "a stale releaser must not touch the fresh lease"
    stats = store.notification_delivery_stats(now=later + timedelta(seconds=2))
    assert stats["by_status"][NotificationDeliveryStatus.LEASED.value] == 1, stats


def test_delivery_stats_report_the_effective_state_and_the_oldest_age(store) -> None:
    queued_at = datetime.now(timezone.utc)
    pending = _notification(store, "pending")
    retrying = _notification(store, "retrying")
    sent_inline = _notification(store, "sent-inline")
    drill_like = _notification(store, "skipped-inline")
    dead = _notification(store, "dead")

    # A retry: claimed, failed once, handed back with a backoff.
    claimed = {
        item.notification_id: item
        for item in store.claim_notification_deliveries(
            "pod-a", now=NOW, lease_duration=LEASE, limit=10
        )
    }
    store.complete_notification_delivery(
        retrying.notification_id,
        owner_id="pod-a",
        lease_epoch=claimed[retrying.notification_id].lease_epoch,
        result=_failed(retrying.notification_id),
        now=NOW + timedelta(seconds=1),
        retry_at=NOW + timedelta(minutes=15),
    )
    # Dead-lettered after exhausting its attempts.
    store.complete_notification_delivery(
        dead.notification_id,
        owner_id="pod-a",
        lease_epoch=claimed[dead.notification_id].lease_epoch,
        result=_failed(dead.notification_id),
        now=NOW + timedelta(seconds=1),
        terminal=True,
    )
    # The rest go back to the outbox untouched.
    for notification_id in (
        pending.notification_id,
        sent_inline.notification_id,
        drill_like.notification_id,
    ):
        store.release_notification_delivery(
            notification_id,
            owner_id="pod-a",
            lease_epoch=claimed[notification_id].lease_epoch,
            now=NOW + timedelta(seconds=1),
            retry_at=NOW + timedelta(seconds=1),
        )
    # Inline paths record a result without ever touching the delivery row.
    store.save_notification_result(
        NotificationResult(
            notification_id=sent_inline.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="ses-1",
        )
    )
    store.save_notification_result(
        NotificationResult(
            notification_id=drill_like.notification_id,
            status=NotificationStatus.SKIPPED,
            reason="drill",
        )
    )

    observed_at = NOW + timedelta(minutes=45)
    stats = store.notification_delivery_stats(now=observed_at)

    by_status = stats["by_status"]
    assert set(by_status) == {status.value for status in NotificationDeliveryStatus}
    # pending and retrying are the two undelivered rows; the inline SENT and
    # the inline SKIPPED rows are counted as the outbox would treat them.
    assert by_status[NotificationDeliveryStatus.RETRY.value] == 2, by_status
    assert by_status[NotificationDeliveryStatus.SENT.value] == 1, by_status
    assert by_status[NotificationDeliveryStatus.DEAD.value] == 2, by_status
    assert by_status[NotificationDeliveryStatus.PENDING.value] == 0, by_status
    assert by_status[NotificationDeliveryStatus.LEASED.value] == 0, by_status
    assert stats["pending"] == 2, stats
    # Every row was queued at ``queued_at``; the age is the undelivered rows'
    # age, and a retry backoff does not restart it.
    assert stats["oldest_pending_age_seconds"] == pytest.approx(
        (observed_at - queued_at).total_seconds(), abs=30
    ), stats


def test_delivery_stats_are_empty_zeros_without_deliveries(store) -> None:
    stats = store.notification_delivery_stats(now=NOW)

    assert stats["pending"] == 0, stats
    assert stats["oldest_pending_age_seconds"] == 0.0, stats
    assert all(count == 0 for count in stats["by_status"].values()), stats


def test_delivery_stats_anchor_a_requeued_row_on_the_requeue(store) -> None:
    """Re-queueing restarts the shelf life, so it restarts the age too."""
    notification = _notification(store, "requeued")
    requeued_at = NOW + timedelta(hours=2)
    store.enqueue_notification_delivery(notification.notification_id, now=requeued_at)

    stats = store.notification_delivery_stats(now=requeued_at + timedelta(minutes=2))

    assert stats["oldest_pending_age_seconds"] == pytest.approx(120, abs=2), stats


def test_metrics_export_the_delivery_state_machine() -> None:
    store = build_store()
    notification = _notification(store, "metrics")
    store.claim_notification_deliveries("pod-a", now=NOW, lease_duration=LEASE, limit=5)

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    assert 'gpu_fault_notification_delivery_total{status="LEASED"} 1' in lines
    for status in NotificationDeliveryStatus:
        assert any(
            line.startswith(
                f'gpu_fault_notification_delivery_total{{status="{status.value}"}}'
            )
            for line in lines
        ), status
    ages = [
        line
        for line in lines
        if line.startswith("gpu_fault_notification_oldest_pending_age_seconds ")
    ]
    assert len(ages) == 1, lines
    assert float(ages[0].split()[-1]) >= 0.0, ages
    assert notification.notification_id not in "\n".join(lines), (
        "the gauges are aggregates, never per-notification series"
    )


class _AlwaysFailingNotifier:
    def send(self, notification):
        raise RuntimeError("SES unavailable")


def test_dispatch_outbox_counts_dead_letters_and_expiries_and_stamps_its_cycle(
    monkeypatch,
) -> None:
    store = build_store()
    _watermark_already_drawn(store)
    service = AdvisoryNotificationService(
        store, _AlwaysFailingNotifier(), async_delivery=True, ttl_seconds=3600
    )
    fresh = _notification(store, "fresh")
    # Saved three hours ago and never re-queued since: saving already created
    # its delivery row, and ``send`` would restart its shelf life.
    _notification(
        store, "stale", created_at=datetime.now(timezone.utc) - timedelta(hours=3)
    )
    service.send(fresh.notification_id)

    assert service.last_cycle_timestamp_seconds == 0.0
    report = service.dispatch_outbox("pod-a", max_attempts=1)

    assert report.expired == 1, report
    assert report.failed == 1, report
    assert service.expired_total == 1
    assert service.expired_last_seen_timestamp_seconds > 0.0
    assert service.dead_lettered_total == 1
    assert service.last_cycle_timestamp_seconds > 0.0
    stats = store.notification_delivery_stats(now=datetime.now(timezone.utc))
    assert stats["by_status"][NotificationDeliveryStatus.DEAD.value] == 2, stats

    lines = control_loop_metric_lines(
        SimpleNamespace(
            context=SimpleNamespace(store=store, advisory_notifications=service)
        )
    )
    assert "gpu_fault_notification_expired_total 1" in lines
    assert any(
        line.startswith("gpu_fault_notification_expired_last_seen_timestamp_seconds ")
        and float(line.split()[-1]) > 0
        for line in lines
    ), lines
    assert "gpu_fault_notification_dead_lettered_total 1" in lines
    assert any(
        line.startswith("gpu_fault_notification_dispatch_last_cycle_timestamp_seconds ")
        and float(line.split()[-1]) > 0
        for line in lines
    ), lines


def test_a_retry_whose_lease_lapsed_is_not_recorded_as_a_terminal_failure(
    monkeypatch,
) -> None:
    """The stale-lease fallback persists SENT so mail is not sent twice; a
    FAILED attempt has nothing to protect and would masquerade as a dead letter."""
    store = build_store()
    _watermark_already_drawn(store)
    service = AdvisoryNotificationService(
        store, _AlwaysFailingNotifier(), async_delivery=True
    )
    notification = _notification(store, "lapsed")
    service.send(notification.notification_id)

    def stale(*args, **kwargs):
        raise WorkflowLeaseError("notification delivery lease is stale")

    monkeypatch.setattr(store, "complete_notification_delivery", stale)

    service.dispatch_outbox("pod-a", max_attempts=8)

    assert store.get_notification_result(notification.notification_id) is None
