"""``send`` on a notification already in the outbox is idempotent; only an
explicit operator requeue revives a dead letter.

Control-plane review 2026-09-08, F-2 (CP-6). ``collector_silence`` called
``send`` every 60 s for the same still-silent collector, and ``send`` called
``enqueue_notification_delivery`` unconditionally, which put a RETRY row back
to PENDING with ``available_at=now`` and revived DEAD rows. ``attempts`` was
never reset, so a revived row died on its next attempt, ``dead_lettered_total``
grew by one a minute and ``GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS`` meant nothing:
eleven attempts and nine dead letters for one notification in the repro.
"""

from __future__ import annotations

from gpu_fault.models import NotificationDeliveryStatus, NotificationStatus
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import HyperPodAdvisoryEmailBuilder
from tests._builders import build_store
from tests.notifications._support import (
    RecordingNotifier,
    _delivery,
    _watermark_already_drawn,
    advisory,
)

MAX_ATTEMPTS = 3


class _AlwaysFailingNotifier:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, notification):
        self.calls += 1
        raise RuntimeError("SES unavailable")


def _saved(store, incident_id: str = "incident-silent"):
    return store.save_notification_if_absent(
        HyperPodAdvisoryEmailBuilder().build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id=incident_id,
            node_ids=["worker-1"],
            issue_summary="collector silent",
        )
    )


def test_send_on_a_retrying_delivery_keeps_its_backoff_and_attempts() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    service = AdvisoryNotificationService(
        store, _AlwaysFailingNotifier(), async_delivery=True
    )
    notification = _saved(store)
    service.send(notification.notification_id)
    service.dispatch_outbox("pod-a", max_attempts=MAX_ATTEMPTS)
    retrying = _delivery(store, notification.notification_id)
    assert retrying.status is NotificationDeliveryStatus.RETRY
    assert retrying.attempts == 1

    again = service.send(notification.notification_id)

    assert again.status is NotificationStatus.QUEUED
    after = _delivery(store, notification.notification_id)
    assert after.status is NotificationDeliveryStatus.RETRY
    assert after.attempts == 1
    assert after.available_at == retrying.available_at, (
        "send() reset the retry backoff of a delivery already in the outbox"
    )
    assert after.last_error == retrying.last_error


def test_repeated_send_cannot_push_a_notification_past_max_attempts() -> None:
    """The F-2 repro: alternate ``dispatch_outbox`` and ``send`` ten times.
    The delivery must die once, after exactly ``max_attempts`` provider calls,
    and stay dead."""

    store = build_store()
    _watermark_already_drawn(store)
    notifier = _AlwaysFailingNotifier()
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    notification = _saved(store)
    service.send(notification.notification_id)

    for _ in range(10):
        # Back-off is real time; make every RETRY claimable at once.
        delivery = _delivery(store, notification.notification_id)
        if delivery.status is NotificationDeliveryStatus.RETRY:
            store._notification_deliveries[notification.notification_id] = (
                delivery.model_copy(update={"available_at": delivery.updated_at})
            )
        service.dispatch_outbox("pod-a", max_attempts=MAX_ATTEMPTS)
        service.send(notification.notification_id)

    final = _delivery(store, notification.notification_id)
    assert final.status is NotificationDeliveryStatus.DEAD, final
    assert final.attempts == MAX_ATTEMPTS, final
    assert notifier.calls == MAX_ATTEMPTS
    assert service.dead_lettered_total == 1


def test_send_on_a_dead_letter_reports_failure_without_reviving_it() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    service = AdvisoryNotificationService(
        store, _AlwaysFailingNotifier(), async_delivery=True
    )
    notification = _saved(store)
    service.send(notification.notification_id)
    service.dispatch_outbox("pod-a", max_attempts=1)
    assert _delivery(store, notification.notification_id).status is (
        NotificationDeliveryStatus.DEAD
    )

    result = service.send(notification.notification_id)

    assert result.status is NotificationStatus.FAILED
    assert "requeue" in (result.reason or "")
    assert _delivery(store, notification.notification_id).status is (
        NotificationDeliveryStatus.DEAD
    )


def test_an_explicit_requeue_revives_a_dead_letter_with_fresh_attempts() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    failing = _AlwaysFailingNotifier()
    service = AdvisoryNotificationService(store, failing, async_delivery=True)
    notification = _saved(store)
    service.send(notification.notification_id)
    service.dispatch_outbox("pod-a", max_attempts=1)
    dead = _delivery(store, notification.notification_id)
    assert dead.status is NotificationDeliveryStatus.DEAD and dead.attempts == 1

    requeued = service.requeue(notification.notification_id)

    assert requeued.status is NotificationStatus.QUEUED
    revived = _delivery(store, notification.notification_id)
    assert revived.status is NotificationDeliveryStatus.PENDING
    assert revived.attempts == 0, "an operator requeue starts the attempt budget over"
    assert revived.requeued_at is not None

    # And the provider is asked again, once, by the next cycle.
    service.notifier = RecordingNotifier()
    report = service.dispatch_outbox("pod-a", max_attempts=1)
    assert report.sent == 1
    assert _delivery(store, notification.notification_id).status is (
        NotificationDeliveryStatus.SENT
    )
