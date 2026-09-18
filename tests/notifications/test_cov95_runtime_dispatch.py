from __future__ import annotations

from datetime import timedelta

import pytest
from botocore.exceptions import ClientError

from gpu_fault.models import (
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.store import WorkflowLeaseError
from tests.notifications._cov95_runtime_support import NOW, saved
from tests.notifications._cov95_runtime_support import (
    runtime_store_fixture as runtime_store_fixture,
)
from tests.notifications._support import RecordingNotifier


def test_throttled_release_failure_does_not_strand_other_claimed_rows(
    runtime_store, monkeypatch, caplog
) -> None:
    store, _ = runtime_store
    notifications = [saved(store, index) for index in range(3)]
    calls, releases = [], []
    release = store.release_notification_delivery

    class Throttled:
        def send(self, notification):
            calls.append(notification.notification_id)
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "unit throttle"}},
                "Publish",
            )

    def flaky_release(notification_id, **kwargs):
        releases.append(notification_id)
        if notification_id == notifications[0].notification_id:
            raise TimeoutError("unit release transport failure")
        return release(notification_id, **kwargs)

    monkeypatch.setattr(store, "release_notification_delivery", flaky_release)
    service = AdvisoryNotificationService(store, Throttled(), async_delivery=True)
    report = service.dispatch_outbox("owner")
    assert report.throttled == 3
    assert report.attempted == 0
    assert calls == [notifications[0].notification_id]
    assert releases == [item.notification_id for item in notifications]
    rows = [
        store.get_notification_delivery(item.notification_id) for item in notifications
    ]
    assert [row.status for row in rows] == [
        NotificationDeliveryStatus.LEASED,
        NotificationDeliveryStatus.RETRY,
        NotificationDeliveryStatus.RETRY,
    ]
    assert [row.attempts for row in rows] == [0, 0, 0]
    assert service.delivery_errors_total == 1
    assert "could not be released" in caplog.text
    assert (
        store.claim_notification_deliveries(
            "other", now=NOW, lease_duration=timedelta(seconds=120), limit=3
        )
        == []
    )
    reclaimed = store.claim_notification_deliveries(
        "other",
        now=NOW + timedelta(seconds=121),
        lease_duration=timedelta(seconds=120),
        limit=3,
    )
    assert len(reclaimed) == 3
    assert {row.lease_epoch for row in reclaimed} == {2}


@pytest.mark.parametrize(
    "response", [{"Error": "invalid"}, {"Error": {"Code": "Rejected"}}, []]
)
def test_non_throttle_provider_errors_charge_one_attempt(runtime_store, response):
    store, _ = runtime_store
    notification = saved(store)

    class ProviderFailure(RuntimeError):
        pass

    class Provider:
        def send(self, notification):
            error = ProviderFailure("unit rejection")
            error.response = response
            raise error

    service = AdvisoryNotificationService(store, Provider(), async_delivery=True)
    report = service.dispatch_outbox("owner")
    assert report.failed == 1
    assert report.throttled == 0
    row = store.get_notification_delivery(notification.notification_id)
    assert row.status is NotificationDeliveryStatus.RETRY
    assert row.attempts == 1
    assert row.available_at == NOW + timedelta(seconds=15)
    assert store.get_notification_result(notification.notification_id) is None


def test_lookup_and_release_failures_are_isolated_from_the_rest_of_the_batch(
    runtime_store, monkeypatch, caplog
) -> None:
    store, _ = runtime_store
    first, second = saved(store), saved(store, 1)
    original_get = store.get_notification
    original_release = store.release_notification_delivery

    def get(notification_id):
        if notification_id == first.notification_id:
            raise TimeoutError("unit read unavailable")
        return original_get(notification_id)

    def release(notification_id, **kwargs):
        if notification_id == first.notification_id:
            raise TimeoutError("unit release unavailable")
        return original_release(notification_id, **kwargs)

    monkeypatch.setattr(store, "get_notification", get)
    monkeypatch.setattr(store, "release_notification_delivery", release)
    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    report = service.dispatch_outbox("owner")
    assert report.sent == 1
    assert [item.notification_id for item in notifier.notifications] == [
        second.notification_id
    ]
    assert service.delivery_errors_total == 1
    assert "could not be released either" in caplog.text
    assert store.get_notification_delivery(first.notification_id).status is (
        NotificationDeliveryStatus.LEASED
    )


def test_accepted_but_unrecordable_result_exposes_replay_ambiguity(
    runtime_store, monkeypatch, caplog
) -> None:
    store, clock = runtime_store
    notification = saved(store)
    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    def unavailable(*args, **kwargs):
        raise TimeoutError("unit commit transport unavailable")

    with monkeypatch.context() as boundary:
        boundary.setattr(store, "complete_notification_delivery", unavailable)
        boundary.setattr(store, "save_notification_result", unavailable)
        report = service.dispatch_outbox("owner")
    assert report.sent == 1
    assert service.delivery_errors_total == 1
    assert store.get_notification_result(notification.notification_id) is None
    assert "it may be sent again" in caplog.text
    clock.value += timedelta(seconds=16)
    assert service.dispatch_outbox("new-owner").sent == 1
    assert len(notifier.notifications) == 2, (
        "unrecorded provider acceptance cannot guarantee exactly-once delivery"
    )
    assert (
        store.get_notification_result(notification.notification_id).status
        is NotificationStatus.SENT
    )


@pytest.mark.parametrize("provider_id", ["provider-accepted", None])
def test_duplicate_requires_provider_acceptance_to_normalize_as_sent(
    runtime_store, provider_id
):
    store, _ = runtime_store
    notification = saved(store)

    class Provider:
        def send(self, item):
            return NotificationResult(
                notification_id=item.notification_id,
                status=NotificationStatus.DUPLICATE,
                provider_message_id=provider_id,
            )

    service = AdvisoryNotificationService(store, Provider(), async_delivery=True)
    report = service.dispatch_outbox("owner", max_attempts=1)
    row = store.get_notification_delivery(notification.notification_id)
    assert report.attempted == 1
    if provider_id:
        assert report.sent == 1
        assert report.results[0].reason == "provider already accepted this notification"
        assert row.status is NotificationDeliveryStatus.SENT
        assert (
            service.requeue(notification.notification_id).status
            is NotificationStatus.DUPLICATE
        )
    else:
        assert report.sent == 0
        assert row.status is NotificationDeliveryStatus.DEAD
        assert service.dead_lettered_total == 1


@pytest.mark.parametrize("retirement", ["expired", "drill"])
def test_retirement_lease_loss_preserves_the_skipped_verdict(
    runtime_store, monkeypatch, retirement
):
    store, _ = runtime_store
    notification = saved(
        store,
        created_at=NOW - timedelta(hours=1),
        not_before=None,
        drill_id="local-drill" if retirement == "drill" else None,
    )

    def stale(*args, **kwargs):
        raise WorkflowLeaseError("unit lease changed")

    monkeypatch.setattr(store, "complete_notification_delivery", stale)
    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, ttl_seconds=60
    )
    report = service.dispatch_outbox("owner")
    result = store.get_notification_result(notification.notification_id)
    assert result.status is NotificationStatus.SKIPPED
    assert notifier.notifications == []
    assert report.expired == (1 if retirement == "expired" else 0)
    assert report.suppressed_drills == (1 if retirement == "drill" else 0)


@pytest.mark.parametrize(
    "kind,options",
    [
        ("outbox", {"limit": 0}),
        ("outbox", {"limit": 101}),
        ("outbox", {"lease_seconds": 29}),
        ("pending", {"limit": 0}),
        ("pending", {"limit": 101}),
    ],
)
def test_dispatch_rejects_invalid_limits_before_claim(runtime_store, kind, options):
    store, _ = runtime_store
    notification = saved(store)
    service = AdvisoryNotificationService(
        store, RecordingNotifier(), async_delivery=True
    )
    call = (
        service.dispatch_pending
        if kind == "pending"
        else lambda **kw: service.dispatch_outbox("owner", **kw)
    )
    with pytest.raises(ValueError, match="limit|lease"):
        call(**options)
    assert (
        store.get_notification_delivery(notification.notification_id).status
        is NotificationDeliveryStatus.PENDING
    )


def test_negative_backlog_grace_and_category_ttl_are_refused(
    runtime_store, monkeypatch
):
    store, _ = runtime_store
    with pytest.raises(ValueError, match="backlog grace"):
        AdvisoryNotificationService(
            store, RecordingNotifier(), backlog_grace_seconds=-1
        )
    service = AdvisoryNotificationService(store, RecordingNotifier())
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_TTL_SECONDS_GPU_RESET", "-1")
    with pytest.raises(ValueError, match="ttl for gpu.reset"):
        service.category_ttl_seconds("gpu.reset")
