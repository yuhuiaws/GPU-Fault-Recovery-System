from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import DisabledNotificationNotifier
from tests._builders import build_store
from tests.notifications._support import (
    RecordingNotifier,
    _aged_notification,
    _delivery_mode,
    _watermark_already_drawn,
)


def test_the_load_suites_label_every_synthetic_fault_as_a_drill() -> None:
    """The label has to survive normalization to reach the notification.

    Both fault paths a burst run exercises parse the drill id out of the
    message text, so a suite that stamps it anywhere else raises real
    notifications for faults that did not happen.
    """

    from gpu_fault.hma import HyperPodHmaNormalizer
    from scripts.perf import benchmark_mixed_control_plane as suite

    for kind in ("NVIDIA_KERNEL", "FABRIC_MANAGER_LOG"):
        payload = suite.stamp({}, kind, "cluster-a", "node-a", 1)
        assert HyperPodHmaNormalizer._drill_id(payload["message"]) == suite.DRILL_ID


def test_the_bulk_resend_scan_is_bounded_and_says_when_it_truncated() -> None:
    """The synchronous path used to read the whole notification history.

    Nothing leaves that table, and nothing leaves the pending set either --
    drills and expired advisories are both recorded SKIPPED, which is picked up
    again -- so the read grew forever and carried one extra result lookup per
    row.
    """

    store = build_store()
    _watermark_already_drawn(store)
    for index in range(4):
        _aged_notification(
            store, age=timedelta(seconds=40 - index), incident_id=f"incident-{index}"
        )
    service = AdvisoryNotificationService(
        store, RecordingNotifier(), async_delivery=False, dispatch_scan_limit=2
    )

    report = service.dispatch_pending()

    assert report.scan_truncated is True
    # The newest two are the ones still deliverable, and they are attempted
    # oldest-first within that window.
    assert report.sent == 2
    assert [item.incident_id for item in service.notifier.notifications] == [
        "incident-2",
        "incident-3",
    ]


def test_a_bulk_resend_within_budget_reports_no_truncation() -> None:
    store = build_store()
    _watermark_already_drawn(store)
    _aged_notification(store, age=timedelta(seconds=30), incident_id="incident-live")
    service = AdvisoryNotificationService(
        store, RecordingNotifier(), async_delivery=False, dispatch_scan_limit=50
    )

    report = service.dispatch_pending()

    assert report.scan_truncated is False
    assert report.sent == 1


def test_a_non_positive_dispatch_scan_budget_is_refused() -> None:
    """Zero would read nothing and report an empty backlog as a healthy one."""

    with pytest.raises(ValueError, match="dispatch scan limit"):
        AdvisoryNotificationService(
            build_store(),
            RecordingNotifier(),
            async_delivery=False,
            dispatch_scan_limit=0,
        )


def test_an_unbounded_notification_read_is_unchanged() -> None:
    """Only the dispatch path is bounded.

    Every other caller reads the whole table oldest-first and must keep doing
    so; the acceptance scripts count on it.
    """

    store = build_store()
    for index in range(3):
        _aged_notification(
            store, age=timedelta(seconds=30 - index), incident_id=f"incident-{index}"
        )

    assert [item.incident_id for item in store.list_notifications()] == [
        "incident-0",
        "incident-1",
        "incident-2",
    ]
    assert [
        item.incident_id
        for item in store.list_notifications(limit=2, newest_first=True)
    ] == ["incident-2", "incident-1"]


def test_a_negative_shelf_life_is_refused() -> None:
    with pytest.raises(ValueError, match="ttl"):
        AdvisoryNotificationService(
            build_store(), RecordingNotifier(), async_delivery=True, ttl_seconds=-1
        )


@pytest.mark.parametrize(
    "email,async_delivery,dispatcher,delivered",
    [
        (True, True, True, True),
        (True, False, True, True),
        (True, False, False, True),
        # The production misconfiguration: mail is allowed and queued,
        # but no replica drains the outbox, so nothing is ever sent.
        (True, True, False, False),
        (False, True, True, False),
        (False, False, False, False),
    ],
)
def test_delivery_mode_states_whether_notifications_reach_anyone(
    monkeypatch, email, async_delivery, dispatcher, delivered
) -> None:
    """Three switches multiply into states no single one explains.

    ALLOW_EMAIL, DISPATCHER_ENABLED and ASYNC_DELIVERY each read as
    sensible alone, but their product decides whether an operator is
    paged. Without this summary the silent failure mode -- every
    notification sitting in the store as QUEUED -- looks identical to a
    healthy system from the logs.
    """

    description = _delivery_mode(
        monkeypatch, email=email, async_delivery=async_delivery, dispatcher=dispatcher
    )
    assert f"email={'on' if email else 'off'}" in description
    assert f"dispatcher={'on' if dispatcher else 'off'}" in description
    assert f"async={'on' if async_delivery else 'off'}" in description
    assert ("NOT DELIVERED" in description) is not delivered


def test_delivery_mode_names_the_switch_that_blocks_delivery(monkeypatch) -> None:
    """A summary that says "broken" without saying "fix this" is noise."""

    queued_but_undrained = _delivery_mode(
        monkeypatch, email=True, async_delivery=True, dispatcher=False
    )
    assert "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED" in queued_but_undrained

    ses_disabled = _delivery_mode(
        monkeypatch, email=False, async_delivery=True, dispatcher=True
    )
    assert "GPU_FAULT_ALLOW_EMAIL" in ses_disabled

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", "true")
    no_channel = AdvisoryNotificationService(
        build_store(), DisabledNotificationNotifier(), async_delivery=True
    ).describe_delivery_mode()
    assert "NOT DELIVERED" in no_channel
    assert "GPU_FAULT_EMAIL_SENDER" in no_channel
