"""One delivery's bookkeeping failure neither loses a SENT verdict nor stalls
the rest of the claimed batch.

Control-plane review 2026-09-08, F-3. ``_record_delivery_outcome`` caught only
``WorkflowLeaseError``; a pool timeout or ``OperationalError`` after SES had
accepted the mail escaped ``dispatch_outbox`` entirely. The delivery stayed
LEASED with no result row, so 120 s later another replica claimed it and mailed
it again, and the other <=24 deliveries claimed in the same cycle sat LEASED for
the whole lease instead of being handled or released.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


def _saved(store, incident_id: str):
    return store.save_notification_if_absent(
        HyperPodAdvisoryEmailBuilder().build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id=incident_id,
            node_ids=["worker-1"],
            issue_summary=incident_id,
        )
    )


def _service_with_flaky_completion(store, notifier, failing_ids: set[str]):
    """``complete_notification_delivery`` fails like a pool timeout for the
    given notifications; everything else in the store works."""

    complete = store.complete_notification_delivery

    def flaky(notification_id, **kwargs):
        if notification_id in failing_ids:
            raise TimeoutError("couldn't get a connection after 5.00 sec")
        return complete(notification_id, **kwargs)

    store.complete_notification_delivery = flaky
    return AdvisoryNotificationService(store, notifier, async_delivery=True)


def test_a_sent_outcome_survives_a_store_error_after_the_provider_accepted() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    notifier = RecordingNotifier()
    first = _saved(store, "incident-first")
    service = _service_with_flaky_completion(store, notifier, {first.notification_id})

    report = service.dispatch_outbox("pod-a")

    recorded = store.get_notification_result(first.notification_id)
    assert recorded is not None and recorded.status is NotificationStatus.SENT, (
        "the SENT verdict was lost; the next lease holder will mail it again"
    )
    assert service.delivery_errors_total == 1
    assert service.delivery_error_last_seen_timestamp_seconds > 0.0
    assert report.attempted == 1

    # The recorded SENT is what keeps the provider from being asked twice,
    # whichever replica claims the row after the lease lapses.
    store.complete_notification_delivery = (
        store.__class__.complete_notification_delivery.__get__(store)
    )
    later = service.dispatch_outbox("pod-a")
    assert later.attempted <= 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-first"]


def test_the_rest_of_the_batch_is_handled_in_the_same_cycle() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    notifier = RecordingNotifier()
    first = _saved(store, "incident-first")
    second = _saved(store, "incident-second")
    third = _saved(store, "incident-third")
    service = _service_with_flaky_completion(store, notifier, {first.notification_id})

    report = service.dispatch_outbox("pod-a")

    assert report.attempted == 3, report
    assert report.sent == 3, report
    for notification in (second, third):
        assert _delivery(store, notification.notification_id).status is (
            NotificationDeliveryStatus.SENT
        ), notification.incident_id
    assert sorted(item.incident_id for item in notifier.notifications) == [
        "incident-first",
        "incident-second",
        "incident-third",
    ]


def test_a_failed_attempt_whose_bookkeeping_fails_is_released_not_left_leased() -> None:
    """Nothing was sent, so there is no verdict to protect -- but the row
    must not sit LEASED for the whole lease with no owner working on it."""

    class _Failing:
        def send(self, notification):
            raise RuntimeError("SES unavailable")

    store = build_store()
    _watermark_already_drawn(store)
    notification = _saved(store, "incident-failing")
    service = _service_with_flaky_completion(
        store, _Failing(), {notification.notification_id}
    )
    before = datetime.now(timezone.utc)

    service.dispatch_outbox("pod-a", max_attempts=8)

    delivery = _delivery(store, notification.notification_id)
    assert delivery.status is NotificationDeliveryStatus.RETRY, delivery
    assert delivery.available_at >= before
    assert store.get_notification_result(notification.notification_id) is None
    assert service.delivery_errors_total == 1
