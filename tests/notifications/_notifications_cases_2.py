from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest
from fastapi.testclient import TestClient

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.lifespan_workers import start_notification_worker
from gpu_fault.models import (
    NotificationDeliveryStatus,
    NotificationDispatchReport,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import HyperPodAdvisoryEmailBuilder
from gpu_fault.store import SqliteStore, WorkflowLeaseError
from tests._builders import build_store, copy_model
from tests.notifications._support import (
    LeaseLosingStore,
    RecordingNotifier,
    ThrottlingNotifier,
    _aged_notification,
    _delivery,
    _drill_notification,
    _watermark_already_drawn,
    advisory,
)

LIFESPAN_TOKEN = "notification-lifespan-token-" + "x" * 40


@pytest.mark.parametrize("durable", [False, True])
def test_async_outbox_claim_is_fenced_and_prioritized(tmp_path, durable: bool) -> None:
    store = (
        SqliteStore(str(tmp_path / "notifications.db")) if durable else build_store()
    )
    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    builder = HyperPodAdvisoryEmailBuilder()
    low = copy_model(
        builder.build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id="incident-low",
            node_ids=["worker-1"],
            issue_summary="health trend",
        ),
        priority=200,
    )
    high = copy_model(
        builder.build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id="incident-high",
            node_ids=["worker-2"],
            issue_summary="XID",
        ),
        priority=0,
    )
    store.save_notification_if_absent(low)
    store.save_notification_if_absent(high)

    assert service.send(low.notification_id).status is (NotificationStatus.QUEUED)
    assert notifier.notifications == []
    report = service.dispatch_outbox("pod-a", limit=1)

    assert report.sent == 1
    assert notifier.notifications[0].incident_id == "incident-high"
    now = datetime.now(timezone.utc)
    claimed = store.claim_notification_deliveries(
        "pod-a", now=now, lease_duration=timedelta(seconds=60), limit=1
    )[0]
    assert not store.claim_notification_deliveries(
        "pod-b", now=now, lease_duration=timedelta(seconds=60), limit=1
    )
    replacement = store.claim_notification_deliveries(
        "pod-b",
        now=now + timedelta(seconds=61),
        lease_duration=timedelta(seconds=60),
        limit=1,
    )[0]
    with pytest.raises(
        WorkflowLeaseError, match="notification delivery lease is stale"
    ):
        store.complete_notification_delivery(
            claimed.notification_id,
            owner_id="pod-a",
            lease_epoch=claimed.lease_epoch,
            result=NotificationResult(
                notification_id=claimed.notification_id, status=NotificationStatus.SENT
            ),
            now=now + timedelta(seconds=61),
        )
    assert replacement.lease_epoch == claimed.lease_epoch + 1


@pytest.mark.parametrize("durable", [False, True])
def test_first_dispatch_suppresses_the_pre_enable_backlog(
    tmp_path, durable: bool
) -> None:
    store = SqliteStore(str(tmp_path / "watermark.db")) if durable else build_store()
    notifier = RecordingNotifier()
    builder = HyperPodAdvisoryEmailBuilder()
    long_ago = datetime.now(timezone.utc) - timedelta(days=2)
    backlog = []
    for index in range(3):
        notification = copy_model(
            builder.build(
                advisory(),
                cluster_name="hp-cluster",
                incident_id=f"incident-old-{index}",
                node_ids=[f"worker-{index}"],
                issue_summary="queued while the dispatcher was off",
            ),
            created_at=long_ago + timedelta(minutes=index),
        )
        backlog.append(store.save_notification_if_absent(notification))
    fresh = store.save_notification_if_absent(
        builder.build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id="incident-live",
            node_ids=["worker-9"],
            issue_summary="raised after the dispatcher came up",
        )
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.suppressed_backlog == 3
    assert report.sent == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-live"]
    for notification in backlog:
        result = store.get_notification_result(notification.notification_id)
        assert result is not None
        assert result.status is NotificationStatus.SKIPPED
        assert "pre-watermark backlog" in (result.reason or "")

    # A second cycle must not re-suppress or re-count anything, and a
    # later replica must not move the line forward.
    again = service.dispatch_outbox("pod-b")

    assert again.suppressed_backlog == 0
    assert again.attempted == 0
    watermark = store.get_notification_watermark()
    assert watermark is not None
    assert watermark.established_by == "pod-a"
    assert watermark.suppressed == 3
    assert (
        store.get_notification_result(fresh.notification_id).status
        is NotificationStatus.SENT
    )


def test_suppressed_backlog_can_be_requeued_on_demand() -> None:
    store = build_store()
    notifier = RecordingNotifier()
    stale = store.save_notification_if_absent(
        copy_model(
            HyperPodAdvisoryEmailBuilder().build(
                advisory(),
                cluster_name="hp-cluster",
                incident_id="incident-old",
                node_ids=["worker-1"],
                issue_summary="queued while the dispatcher was off",
            ),
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    assert service.dispatch_outbox("pod-a").suppressed_backlog == 1
    assert notifier.notifications == []

    assert service.send(stale.notification_id).status is (NotificationStatus.QUEUED)
    report = service.dispatch_outbox("pod-a")

    assert report.sent == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-old"]


def test_backlog_delivery_can_be_opted_into() -> None:
    store = build_store()
    notifier = RecordingNotifier()
    store.save_notification_if_absent(
        copy_model(
            HyperPodAdvisoryEmailBuilder().build(
                advisory(),
                cluster_name="hp-cluster",
                incident_id="incident-old",
                node_ids=["worker-1"],
                issue_summary="queued while the dispatcher was off",
            ),
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, deliver_backlog=True
    )

    report = service.dispatch_outbox("pod-a")

    assert report.suppressed_backlog == 0
    assert report.sent == 1


def test_recent_notifications_survive_the_backlog_grace_window() -> None:
    store = build_store()
    notifier = RecordingNotifier()
    store.save_notification_if_absent(
        copy_model(
            HyperPodAdvisoryEmailBuilder().build(
                advisory(),
                cluster_name="hp-cluster",
                incident_id="incident-just-now",
                node_ids=["worker-1"],
                issue_summary="raised while the process was starting",
            ),
            created_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
    )
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, backlog_grace_seconds=300
    )

    report = service.dispatch_outbox("pod-a")

    assert report.suppressed_backlog == 0
    assert report.sent == 1


@pytest.mark.parametrize("durable", [False, True])
def test_stale_notifications_are_not_mailed_when_the_outbox_drains(
    tmp_path, durable: bool
) -> None:
    """The outbox has no upper bound on how long a row can sit in it.

    Everything queued during a blockage becomes deliverable in the same
    instant it clears, so without a deadline the operator is mailed a
    day's history at the batch rate -- indistinguishable from an attack
    to them and to their mail provider, and it puts the live notification
    behind the archive in the claim order.
    """

    store = SqliteStore(str(tmp_path / "ttl.db")) if durable else build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    stale = [
        _aged_notification(
            store,
            age=timedelta(days=2, minutes=index),
            incident_id=f"incident-stale-{index}",
        )
        for index in range(3)
    ]
    _aged_notification(store, age=timedelta(seconds=30), incident_id="incident-live")
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 3
    assert report.sent == 1
    assert report.attempted == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-live"]
    for notification in stale:
        result = store.get_notification_result(notification.notification_id)
        assert result is not None
        assert result.status is NotificationStatus.SKIPPED
        assert "expired" in (result.reason or "")

    # Terminal, or the same rows are re-claimed ahead of live traffic on
    # every cycle for the rest of the process's life.
    again = service.dispatch_outbox("pod-a")
    assert again.expired == 0
    assert again.attempted == 0


def test_expiry_is_measured_from_the_event_not_from_the_last_attempt() -> None:
    """A retry chain must not be able to extend the window.

    ``available_at`` moves forward with every backoff, so anchoring there
    would make a notification that keeps failing immortal -- exactly the
    one least worth mailing late.
    """

    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    _aged_notification(store, age=timedelta(seconds=30), incident_id="incident-1")
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, ttl_seconds=5
    )

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 1
    assert notifier.notifications == []


def test_each_category_expires_on_its_own_clock() -> None:
    """A trend sample and a fault advisory do not age at the same rate.

    The next cooldown bucket restates a host-resource trend, so an
    undelivered one is worth less than the mail it costs. A fault
    advisory is the only record the operator has that a node was rebooted
    on their behalf, so it keeps the longer default.
    """

    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    _aged_notification(
        store,
        age=timedelta(hours=2),
        incident_id="incident-trend",
        category="HEALTH_TREND",
    )
    _aged_notification(
        store,
        age=timedelta(hours=2),
        incident_id="incident-fault",
        category="FAULT_DETECTED",
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-fault"]


def test_a_category_shelf_life_can_be_widened_on_a_live_process(monkeypatch) -> None:
    """Read from the environment per call, on purpose.

    An operator who discovers mail is being dropped needs to be able to
    widen the window on the process that is dropping it, without waiting
    for a rollout to restore the notifications it already retired.
    """

    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    _aged_notification(
        store,
        age=timedelta(hours=2),
        incident_id="incident-trend",
        category="HEALTH_TREND",
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    assert service.category_ttl_seconds("HEALTH_TREND") == 3600

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_TTL_SECONDS_HEALTH_TREND", "86400")

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 0
    assert report.sent == 1


def test_the_shelf_life_can_be_turned_off_entirely() -> None:
    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    _aged_notification(store, age=timedelta(days=30), incident_id="incident-ancient")
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, ttl_seconds=0
    )

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 0
    assert report.sent == 1


def test_asking_for_the_backlog_opts_out_of_the_shelf_life() -> None:
    """``DELIVER_BACKLOG=true`` already means "mail me the history".

    Applying a deadline over that switch would make it a no-op for almost
    everything it exists to deliver.
    """

    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    _aged_notification(store, age=timedelta(days=2), incident_id="incident-stale")
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, deliver_backlog=True
    )
    assert "off" in service.describe_delivery_mode()

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 0
    assert report.sent == 1


def test_a_deferred_notification_is_not_expired_before_it_is_due() -> None:
    """``not_before`` moves the deadline, not just the availability.

    A notification held back on purpose would otherwise be born past its
    shelf life and be retired the first time it became claimable.
    """

    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    _aged_notification(
        store,
        age=timedelta(days=2),
        incident_id="incident-deferred",
        not_before=datetime.now(timezone.utc) - timedelta(seconds=30),
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 0
    assert report.sent == 1


@pytest.mark.parametrize("durable", [False, True])
def test_asking_for_a_stale_notification_restarts_its_shelf_life(
    tmp_path, durable: bool
) -> None:
    """An explicit request outranks the deadline.

    The deadline suppresses mail nobody asked for. An operator reading
    the persisted notification and pressing send has asked for it, and
    the window restarts rather than being removed -- so a stale
    notification asked for once still stops being retried eventually.
    """

    store = SqliteStore(str(tmp_path / "requeue.db")) if durable else build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    stale = _aged_notification(
        store, age=timedelta(days=2), incident_id="incident-stale"
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    assert service.send(stale.notification_id).status is (NotificationStatus.QUEUED)
    report = service.dispatch_outbox("pod-a")

    assert report.expired == 0
    assert report.sent == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-stale"]


def test_bulk_resend_does_not_mail_what_expiry_just_retired() -> None:
    """Expiry records SKIPPED, and SKIPPED is what this path re-sends.

    Without the deadline here, one call to the dispatch endpoint in
    synchronous mode mails back the entire archive the outbox path had
    already decided was too old to mail.
    """

    store = build_store()
    notifier = RecordingNotifier()
    stale = _aged_notification(
        store, age=timedelta(days=2), incident_id="incident-stale"
    )
    store.save_notification_result(
        NotificationResult(
            notification_id=stale.notification_id,
            status=NotificationStatus.SKIPPED,
            reason="expired earlier",
        )
    )
    _aged_notification(store, age=timedelta(seconds=30), incident_id="incident-live")
    service = AdvisoryNotificationService(store, notifier, async_delivery=False)

    report = service.dispatch_pending()

    assert report.expired == 1
    assert report.sent == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-live"]


def test_an_expired_delivery_is_retired_rather_than_left_pending() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    stale = _aged_notification(
        store, age=timedelta(days=2), incident_id="incident-stale"
    )
    service = AdvisoryNotificationService(
        store, RecordingNotifier(), async_delivery=True
    )

    service.dispatch_outbox("pod-a")

    delivery = store.claim_notification_deliveries(
        "pod-b",
        now=datetime.now(timezone.utc) + timedelta(hours=1),
        lease_duration=timedelta(seconds=60),
        limit=10,
    )
    assert delivery == []
    assert store.get_notification(stale.notification_id) is not None


def test_a_send_is_recorded_even_if_its_lease_expired_first() -> None:
    """Otherwise one notification becomes an unbounded stream of mail.

    The mail is already out when the bookkeeping fails. Dropping the
    record leaves the row claimable, so the next replica sends the same
    message and fails to record it the same way -- forever, with the
    attempt counter never advancing because incrementing it is exactly the
    write that failed.
    """

    store = LeaseLosingStore()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    stale = _aged_notification(
        store, age=timedelta(seconds=5), incident_id="incident-1"
    )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    first = service.dispatch_outbox("pod-a")

    assert first.sent == 1
    assert store.completions == 1
    result = store.get_notification_result(stale.notification_id)
    assert result is not None
    assert result.status is NotificationStatus.SENT

    # The claim query skips anything already recorded as SENT, so the
    # second cycle finds nothing and the mail is not sent again.
    second = service.dispatch_outbox("pod-b")

    assert second.attempted == 0
    assert len(notifier.notifications) == 1


def test_one_items_bookkeeping_failure_does_not_abandon_the_batch() -> None:
    """The exception used to escape the loop.

    Everything already claimed behind the failing item stayed LEASED with
    nothing coming back for it, so it was re-claimed after the lease
    expired and re-sent -- the same duplicate engine, one batch at a time.
    """

    store = LeaseLosingStore()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    for index in range(5):
        _aged_notification(
            store, age=timedelta(seconds=5 + index), incident_id=f"incident-{index}"
        )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.attempted == 5
    assert report.sent == 5
    assert len(notifier.notifications) == 5


def test_a_throttled_batch_is_returned_rather_than_burnt() -> None:
    """A rate limit says nothing about the notification in hand.

    Charging it an attempt is how a storm gets mailed a few times and then
    declared undeliverable: eight attempts against a provider that is
    throttling *because* of the storm exhausts the budget without ever
    being about this message.
    """

    store = build_store()
    notifier = ThrottlingNotifier(throttle_after=2)
    _watermark_already_drawn(store)
    for index in range(6):
        _aged_notification(
            store, age=timedelta(seconds=5 + index), incident_id=f"incident-{index}"
        )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.sent == 2
    assert report.throttled == 4
    assert report.failed == 0
    # Stopped at the first refusal instead of asking four more times.
    assert len(notifier.notifications) == 2

    returned = [
        delivery
        for delivery in store._notification_deliveries.values()
        if delivery.status is NotificationDeliveryStatus.RETRY
    ]
    assert len(returned) == 4
    for delivery in returned:
        assert delivery.attempts == 0
        assert delivery.lease_owner is None
        # No result written either: a throttle is not an outcome for this
        # notification, and a FAILED record here is what the bulk resend
        # path would later mail again.
        assert store.get_notification_result(delivery.notification_id) is None


def test_a_throttled_notification_still_expires() -> None:
    """The interlock that keeps "do not charge an attempt" bounded.

    A notification the provider never accepts would otherwise be retried
    forever now that throttling is free. The shelf life is what ends it,
    which is why the two changes belong together.
    """

    store = build_store()
    notifier = ThrottlingNotifier(throttle_after=0)
    _watermark_already_drawn(store)
    _aged_notification(store, age=timedelta(days=2), incident_id="incident-stale")
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.expired == 1
    assert report.throttled == 0
    assert notifier.notifications == []


@pytest.mark.parametrize("durable", [False, True])
def test_releasing_a_lease_someone_else_holds_is_a_no_op(
    tmp_path, durable: bool
) -> None:
    """Release is fenced on the lease, and says so by returning nothing.

    A replica that was throttled after its lease expired must not drag the
    row out of whatever the next owner is doing with it -- clearing a lease
    it no longer holds would hand the same notification to two senders.
    """

    store = SqliteStore(str(tmp_path / "release.db")) if durable else build_store()
    _watermark_already_drawn(store)
    notification = _aged_notification(
        store, age=timedelta(seconds=5), incident_id="incident-1"
    )
    at = datetime.now(timezone.utc)
    store.enqueue_notification_delivery(notification.notification_id, now=at)
    claimed = store.claim_notification_deliveries(
        "pod-a", limit=10, now=at, lease_duration=timedelta(seconds=30)
    )
    assert len(claimed) == 1

    # The lease runs out and another replica picks the row up.
    reclaimed = store.claim_notification_deliveries(
        "pod-b",
        limit=10,
        now=at + timedelta(seconds=31),
        lease_duration=timedelta(seconds=30),
    )
    assert len(reclaimed) == 1

    stale = store.release_notification_delivery(
        notification.notification_id,
        owner_id="pod-a",
        lease_epoch=claimed[0].lease_epoch,
        now=at + timedelta(seconds=32),
        retry_at=at + timedelta(seconds=60),
    )

    assert stale is None
    current = _delivery(store, notification.notification_id)
    assert current is not None
    assert current.lease_owner == "pod-b"
    assert current.status is NotificationDeliveryStatus.LEASED


@pytest.mark.parametrize("durable", [False, True])
def test_a_release_does_not_spend_an_attempt(tmp_path, durable: bool) -> None:
    """The one property that makes throttling free.

    ``complete_notification_delivery`` is what charges an attempt, because
    an attempt means "this notification was offered to the provider and
    the provider had an opinion about it". A rate limit is not that.
    """

    store = SqliteStore(str(tmp_path / "attempts.db")) if durable else build_store()
    _watermark_already_drawn(store)
    notification = _aged_notification(
        store, age=timedelta(seconds=5), incident_id="incident-1"
    )
    at = datetime.now(timezone.utc)
    store.enqueue_notification_delivery(notification.notification_id, now=at)

    for round_index in range(3):
        claimed = store.claim_notification_deliveries(
            "pod-a", limit=10, now=at, lease_duration=timedelta(seconds=30)
        )
        assert len(claimed) == 1, round_index
        released = store.release_notification_delivery(
            notification.notification_id,
            owner_id="pod-a",
            lease_epoch=claimed[0].lease_epoch,
            now=at,
            retry_at=at,
        )
        assert released is not None, round_index
        assert released.attempts == 0, round_index
        assert released.status is NotificationDeliveryStatus.RETRY
        assert released.lease_owner is None
        assert released.last_error is None


def test_the_dispatch_loop_backs_off_while_it_is_being_throttled() -> None:
    """Releasing a throttled batch is only half of not making it worse.

    The other half is the loop: a fixed 2s poll re-offers one notification
    per replica per poll for as long as the throttle lasts, which is how
    17,564 refusals were produced by six replicas that were each behaving
    correctly.
    """

    from gpu_fault.app import notification_throttle_delay

    delay = notification_throttle_delay(0.0, interval=2.0, cap=300.0)
    assert delay == 4.0
    for expected in (8.0, 16.0, 32.0):
        delay = notification_throttle_delay(delay, interval=2.0, cap=300.0)
        assert delay == expected

    # Capped rather than growing without bound: the outbox is durable and
    # the shelf life is what retires what it outlived, so a long hold-off
    # costs nothing but has to stay within a poll of usefully recovering.
    assert notification_throttle_delay(256.0, interval=2.0, cap=300.0) == 300.0
    assert notification_throttle_delay(300.0, interval=2.0, cap=300.0) == 300.0


def test_notification_worker_reclaims_expired_lease_and_stops(monkeypatch) -> None:
    store = build_store()
    notifier = RecordingNotifier()
    now = datetime.now(timezone.utc)
    notification = store.save_notification_if_absent(
        copy_model(
            HyperPodAdvisoryEmailBuilder().build(
                advisory(),
                cluster_name="hp-cluster",
                incident_id="incident-worker-takeover",
                node_ids=["worker-1"],
                issue_summary="notification worker takeover",
            ),
            created_at=now - timedelta(seconds=60),
        )
    )
    claimed_at = datetime.now(timezone.utc)
    claimed = store.claim_notification_deliveries(
        "owner-old", now=claimed_at, lease_duration=timedelta(seconds=30), limit=1
    )
    assert [item.notification_id for item in claimed] == [notification.notification_id]
    with store._lock:
        store._notification_deliveries[notification.notification_id] = copy_model(
            claimed[0], lease_expires_at=claimed_at - timedelta(seconds=1)
        )
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    context = type("Context", (), {"advisory_notifications": service})()
    stop = Event()
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_POLL_SECONDS", "0.01")
    worker = start_notification_worker(
        context=context,
        stop=stop,
        owner="owner-new",
        throttle_delay=lambda previous, **_kwargs: previous,
    )
    deadline = time.monotonic() + 2
    while (
        store.get_notification_result(notification.notification_id) is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    stop.set()
    worker.join(timeout=2)

    result = store.get_notification_result(notification.notification_id)
    assert result is not None
    assert result.status is NotificationStatus.SENT
    assert len(notifier.notifications) == 1
    assert not worker.is_alive()


def test_a_drill_is_recorded_but_never_re_enqueued() -> None:
    """A drill describes a fault that did not happen.

    The load suites raise real notifications -- 16,384 of them in one run --
    and every one of them was mailed. Saving a notification already creates
    its delivery row, so what ``send`` has to avoid is queueing the drill a
    second time: that restarts its shelf life and revives a row the
    dispatcher has already retired.
    """

    store = build_store()
    notifier = RecordingNotifier()
    drill = _drill_notification(store, incident_id="incident-drill")
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    result = service.send(drill.notification_id)

    assert result.status is NotificationStatus.SKIPPED
    assert "perf-burst" in (result.reason or "")
    assert notifier.notifications == []
    assert _delivery(store, drill.notification_id).requeued_at is None
    # Recorded, so the run is still auditable: the drill is in the store
    # with a verdict, which is what distinguishes suppression from a drop.
    assert (
        store.get_notification_result(drill.notification_id).status
        is NotificationStatus.SKIPPED
    )


def test_a_drill_already_in_the_outbox_is_retired_not_re_claimed() -> None:
    """Rows enqueued before the switch was set still have to drain.

    Leaving them for the deadline to retire means every dispatch cycle
    claims, leases and releases the same backlog until its shelf life runs
    out -- with the live notification behind it in the claim order.
    """

    store = build_store()
    notifier = RecordingNotifier()
    _watermark_already_drawn(store)
    drill = _drill_notification(store, incident_id="incident-drill")
    store.enqueue_notification_delivery(drill.notification_id)
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)

    report = service.dispatch_outbox("pod-a")

    assert report.suppressed_drills == 1
    assert report.attempted == 0
    assert notifier.notifications == []
    assert (
        store.claim_notification_deliveries(
            "pod-b",
            now=datetime.now(timezone.utc) + timedelta(hours=1),
            lease_duration=timedelta(seconds=60),
            limit=10,
        )
        == []
    )


def test_a_drill_does_not_consume_the_bulk_resend_budget() -> None:
    """``dispatch_pending`` re-sends SKIPPED, and drills are SKIPPED."""

    store = build_store()
    notifier = RecordingNotifier()
    _drill_notification(store, incident_id="incident-drill")
    _aged_notification(store, age=timedelta(seconds=5), incident_id="incident-live")
    service = AdvisoryNotificationService(store, notifier, async_delivery=False)

    report = service.dispatch_pending()

    assert report.suppressed_drills == 1
    assert report.sent == 1
    assert [item.incident_id for item in notifier.notifications] == ["incident-live"]


def test_a_drill_can_be_mailed_on_purpose(monkeypatch) -> None:
    """Rehearsing the mail path is a legitimate reason to run a drill."""

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_DELIVER_DRILLS", "true")
    store = build_store()
    notifier = RecordingNotifier()
    drill = _drill_notification(store, incident_id="incident-drill")
    service = AdvisoryNotificationService(store, notifier, async_delivery=False)

    result = service.send(drill.notification_id)

    assert result.status is NotificationStatus.SENT
    assert [item.notification_id for item in notifier.notifications] == [
        drill.notification_id
    ]


def test_delivery_mode_states_whether_drills_are_mailed(monkeypatch) -> None:
    """The switch decides whether a load run mails the operator."""

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", "true")
    assert (
        "drills=suppressed"
        in AdvisoryNotificationService(
            build_store(), RecordingNotifier(), async_delivery=True
        ).describe_delivery_mode()
    )

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_DELIVER_DRILLS", "true")
    assert (
        "drills=delivered"
        in AdvisoryNotificationService(
            build_store(), RecordingNotifier(), async_delivery=True
        ).describe_delivery_mode()
    )


def _live_dispatcher_threads() -> int:
    return sum(
        thread.name == "gpu-fault-notification-dispatcher"
        for thread in threading.enumerate()
    )


def test_only_the_worker_role_dispatches_the_outbox_through_the_lifespan(
    monkeypatch,
) -> None:
    """The service role decides who drains the outbox, not the switch.

    ``dispatch_outbox`` has unit coverage, and the manifests have coverage that
    they set ``GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED``; neither shows that
    the ``worker`` deployment actually starts the background dispatcher, or that
    ``ingress`` and ``spool-worker`` refrain from starting one while carrying the
    identical switch. That gap is the whole point of splitting the roles: two
    extra dispatchers would mean the same notification is claimed by three
    replicas, which is how a throttled provider turns one advisory into a retry
    storm.

    So the three roles are assembled through the real lifespan, against one
    shared store holding one PENDING notification, with the delivery switches
    set identically for all three. Shutdown is asserted as well: a dispatcher
    that outlived its lifespan would keep claiming from a process Kubernetes has
    already been told is gone.
    """

    store = build_store()
    notifier = RecordingNotifier()
    notification = store.save_notification_if_absent(
        HyperPodAdvisoryEmailBuilder().build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id="incident-role-assembly",
            node_ids=["worker-1"],
            issue_summary="notification dispatcher role assembly",
        )
    )
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY", "true")
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", "true")
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_POLL_SECONDS", "0.01")
    # Counted as a delta: another test in this worker process may legitimately
    # hold a dispatcher thread, and blaming this test for that would hide the
    # residue this test exists to detect.
    baseline = _live_dispatcher_threads()
    observed: dict[str, dict[str, int]] = {}
    for role in ("ingress", "spool-worker", "worker"):
        monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
        monkeypatch.setenv("POD_UID", f"pod-{role}")
        if role == "spool-worker":
            # The role refuses to start without them, and that refusal is not
            # what is under test here.
            monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
            monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
        else:
            monkeypatch.delenv("GPU_FAULT_PROCESSOR_MODE", raising=False)
            monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)
        app = create_app(
            ApplicationContext(notifier, store=store, execution_token=LIFESPAN_TOKEN)
        )
        with TestClient(app):
            deadline = time.monotonic() + 2
            while (
                store.get_notification_result(notification.notification_id) is None
                and time.monotonic() < deadline
                and role == "worker"
            ):
                time.sleep(0.01)
            if role != "worker":
                # Long enough for a dispatcher that had been started to have
                # polled the outbox many times at a 10ms interval.
                time.sleep(0.2)
            observed[role] = {
                "dispatchers": _live_dispatcher_threads() - baseline,
                "sent": len(notifier.notifications),
            }
        assert _live_dispatcher_threads() == baseline, (
            f"{role} left a notification dispatcher behind after shutdown"
        )

    assert observed["ingress"]["dispatchers"] == 0
    assert observed["spool-worker"]["dispatchers"] == 0
    assert observed["worker"]["dispatchers"] >= 1
    assert observed["ingress"]["sent"] == 0
    assert observed["spool-worker"]["sent"] == 0
    # Sent once, by the one role that runs a dispatcher: the shared store means
    # the two earlier roles had the same claimable row in front of them.
    assert len(notifier.notifications) == 1
    result = store.get_notification_result(notification.notification_id)
    assert result is not None
    assert result.status is NotificationStatus.SENT


class _RecordingStop(Event):
    """An ``Event`` that records what the dispatch loop waited for.

    The delay is computed inside the loop and handed straight to
    ``stop.wait``, so it is otherwise unobservable -- and it is the delay,
    not the backoff function, that decides whether two replicas come back
    together.
    """

    def __init__(self, *, stop_after: int) -> None:
        super().__init__()
        self.delays: list[float | None] = []
        self._stop_after = stop_after

    def wait(self, timeout: float | None = None) -> bool:
        self.delays.append(timeout)
        if len(self.delays) >= self._stop_after:
            self.set()
        return super().wait(0)


def test_throttled_replicas_back_off_on_independent_schedules(monkeypatch) -> None:
    """A shared backoff curve is still a synchronized retry storm.

    ``notification_throttle_delay`` has coverage that the curve grows and caps,
    which stops one replica from hammering a throttled provider. It says nothing
    about six replicas that started together: they compute the same curve from
    the same throttle, so without jitter they return in lockstep and the
    provider sees the same burst it was already refusing, just spaced out.

    The loop multiplies the backoff by a per-cycle random factor for exactly
    that reason. This asserts the applied delay stays inside the declared jitter
    band and that two independently running workers do not produce the same
    schedule.
    """

    from gpu_fault.app import notification_throttle_delay

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_POLL_SECONDS", "2")
    # A store-driven throttle only produces one throttled cycle: the release
    # puts the delivery back 15s out, so the next cycle finds nothing to claim
    # and the loop resets. What has to be exercised here is a throttle that
    # persists across cycles, which is the case the storm came from.
    throttled = NotificationDispatchReport(
        attempted=1, sent=0, skipped=0, failed=0, results=[], throttled=1
    )

    schedules = []
    for owner in ("owner-a", "owner-b"):
        service = type(
            "ThrottledService",
            (),
            {
                "dispatch_outbox": lambda _self, _owner, **_kwargs: throttled,
                "owner": owner,
            },
        )()
        context = type("Context", (), {"advisory_notifications": service})()
        stop = _RecordingStop(stop_after=4)
        worker = start_notification_worker(
            context=context,
            stop=stop,
            owner=owner,
            throttle_delay=notification_throttle_delay,
        )
        worker.join(timeout=5)
        assert not worker.is_alive(), (
            f"the {owner} dispatch loop did not stop when its stop event was set"
        )
        schedules.append([item for item in stop.delays if item is not None])

    for delays in schedules:
        assert len(delays) == 4
        for index, expected in enumerate((4.0, 8.0, 16.0, 32.0)):
            assert 0.8 * expected <= delays[index] <= 1.2 * expected, (
                f"cycle {index} waited {delays[index]}s, outside the "
                f"jitter band around {expected}s"
            )
    # Same curve, different schedule: identical sequences would mean the two
    # replicas come back to a throttled provider at the same moment.
    assert schedules[0] != schedules[1]
