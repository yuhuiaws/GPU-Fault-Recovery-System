from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store import NotFoundError, WorkflowLeaseError
from tests._builders import fault_incident
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)


def notification(name, *, created_at=NOW, available_at=NOW, **values):
    return AdvisoryNotification(
        notification_id=name,
        deduplication_key=f"dedup/{name}",
        incident_id=f"incident/{name}",
        cluster_name="cluster-local",
        subject="Local compatibility notice",
        body_text="No provider transport is used.",
        support_case_draft="",
        created_at=created_at,
        not_before=available_at,
        **values,
    )


def finish_delivery(store, lease, status, *, terminal=True):
    return store.complete_notification_delivery(
        lease.notification_id,
        owner_id="worker",
        lease_epoch=lease.lease_epoch,
        result=NotificationResult(
            notification_id=lease.notification_id, status=status, reason="local result"
        ),
        now=NOW,
        terminal=terminal,
        retry_at=None if terminal else NOW + timedelta(days=1),
    )


def test_retention_preserves_pins_and_open_work_and_releases_deleted_keys(compat_store):
    cutoff = NOW - timedelta(days=1)
    created = {
        "pinned": NOW - timedelta(days=4),
        "pending": NOW - timedelta(days=4),
        "retry": NOW - timedelta(days=3),
        "leased": NOW - timedelta(days=3),
        "a-dead": NOW - timedelta(days=2),
        "b-sent": NOW - timedelta(days=2),
        "boundary": cutoff,
        "recent": cutoff + timedelta(microseconds=1),
    }
    compat_store.save_incident(
        fault_incident("incident/pinned", "pinned-event", cluster_id="cluster-local")
    )
    notices = {
        name: compat_store.save_notification_if_absent(
            notification(
                name,
                created_at=at,
                available_at=NOW + timedelta(days=1) if name == "pending" else NOW,
            )
        )
        for name, at in created.items()
    }
    claims = compat_store.claim_notification_deliveries(
        "worker", now=NOW, lease_duration=timedelta(minutes=5), limit=20
    )
    assert {item.notification_id for item in claims} == set(created) - {"pending"}, (
        "the future pending row must not acquire a lease during fixture setup"
    )
    for lease in claims:
        if lease.notification_id == "leased":
            continue
        failed = lease.notification_id in {"a-dead", "retry"}
        finish_delivery(
            compat_store,
            lease,
            NotificationStatus.FAILED if failed else NotificationStatus.SENT,
            terminal=lease.notification_id != "retry",
        )
    survivors = {"pinned", "pending", "retry", "leased", "recent"}
    before = {
        name: (
            compat_store.get_notification_delivery(name),
            compat_store.get_notification_result(name),
        )
        for name in survivors
    }

    assert (
        compat_store.cleanup_terminal_notifications(older_than=cutoff, limit=1) == 1
    ), "cleanup must advance past older pinned and unfinished deliveries"
    with pytest.raises(NotFoundError, match="a-dead"):
        compat_store.get_notification("a-dead")
    assert compat_store.get_notification("b-sent") == notices["b-sent"], (
        "equal timestamps must be broken by notification ID before applying the limit"
    )
    assert (
        compat_store.cleanup_terminal_notifications(older_than=cutoff, limit=20) == 2
    ), "the cutoff is inclusive, but a row one microsecond newer must survive"
    assert {
        item.notification_id for item in compat_store.list_notifications()
    } == survivors, (
        "only unpinned terminal rows within the retention window may disappear"
    )
    for name in survivors:
        assert compat_store.get_notification(name) == notices[name], (
            f"retention changed surviving notification {name}"
        )
        assert (
            compat_store.get_notification_delivery(name),
            compat_store.get_notification_result(name),
        ) == before[name], f"retention changed delivery state for {name}"
    for name in ("a-dead", "b-sent", "boundary"):
        assert compat_store.get_notification_delivery(name) is None, (
            f"the delivery for removed notification {name} must also be removed"
        )
        assert compat_store.get_notification_result(name) is None, (
            f"the result for removed notification {name} must also be removed"
        )
        replacement = notices[name].model_copy(
            update={"notification_id": f"replacement/{name}", "created_at": NOW}
        )
        assert compat_store.save_notification_if_absent(replacement) == replacement, (
            f"the deleted deduplication key for {name} must not remain dangling"
        )
        assert compat_store.save_notification_if_absent(notices[name]) == replacement, (
            "a replay must resolve to the replacement and cannot resurrect the old row"
        )
    assert (
        compat_store.cleanup_terminal_notifications(older_than=cutoff, limit=20) == 0
    ), "a repeated cleanup must leave pending replacements and retained work alone"


def test_watermark_suppresses_only_unfinished_history_and_fences_its_leases(
    compat_store,
):
    names = (
        "pending",
        "leased",
        "retry",
        "dead",
        "sent",
        "sent-result",
        "boundary",
        "future",
    )
    for name in names:
        created_at = {"boundary": NOW, "future": NOW + timedelta(seconds=1)}.get(
            name, NOW - timedelta(days=1)
        )
        available_at = (
            NOW
            if name in {"leased", "retry", "dead", "sent"}
            else NOW + timedelta(days=1)
        )
        compat_store.save_notification_if_absent(
            notification(name, created_at=created_at, available_at=available_at)
        )
    claims = {
        item.notification_id: item
        for item in compat_store.claim_notification_deliveries(
            "worker", now=NOW, lease_duration=timedelta(minutes=5), limit=20
        )
    }
    assert set(claims) == {"leased", "retry", "dead", "sent"}, (
        "the watermark fixture must have distinct pending, leased and terminal states"
    )
    finish_delivery(
        compat_store, claims["retry"], NotificationStatus.FAILED, terminal=False
    )
    finish_delivery(compat_store, claims["dead"], NotificationStatus.FAILED)
    finish_delivery(compat_store, claims["sent"], NotificationStatus.SENT)
    compat_store.save_notification_result(
        NotificationResult(
            notification_id="sent-result", status=NotificationStatus.SENT
        )
    )
    unchanged = {
        name: (
            compat_store.get_notification_delivery(name),
            compat_store.get_notification_result(name),
        )
        for name in ("dead", "sent", "sent-result", "boundary", "future")
    }
    assert compat_store.get_notification_watermark() is None, (
        "an uninitialized dispatcher must not have an implicit responsibility watermark"
    )

    watermark = compat_store.establish_notification_watermark(
        established_at=NOW, established_by="worker"
    )

    assert watermark.suppressed == 3, (
        "only pending, leased and retry history is suppressed"
    )
    assert compat_store.get_notification_watermark() == watermark, (
        "the persisted responsibility boundary must equal the winning proposal"
    )
    for name in ("pending", "leased", "retry"):
        delivery = compat_store.get_notification_delivery(name)
        result = compat_store.get_notification_result(name)
        assert delivery is not None and result is not None, (
            f"suppression must preserve a delivery and attributed result for {name}"
        )
        assert delivery.status is NotificationDeliveryStatus.DEAD, (
            f"suppressed history {name} must not remain claimable"
        )
        assert delivery.lease_owner is None and delivery.lease_expires_at is None, (
            f"suppression must revoke the previous delivery lease for {name}"
        )
        assert result.status is NotificationStatus.SKIPPED, (
            "retired backlog is not a successful provider delivery"
        )
        assert result.reason == "suppressed as pre-watermark backlog by worker", (
            "the suppression result must retain who established the boundary"
        )
    for name, expected in unchanged.items():
        assert (
            compat_store.get_notification_delivery(name),
            compat_store.get_notification_result(name),
        ) == expected, f"watermark suppression changed ineligible notification {name}"
    with pytest.raises(WorkflowLeaseError, match="lease"):
        finish_delivery(compat_store, claims["leased"], NotificationStatus.SENT)
    assert (
        compat_store.get_notification_result("leased").status
        is NotificationStatus.SKIPPED
    ), "a stale worker must not replace the suppression result with a delivery claim"
    assert (
        compat_store.establish_notification_watermark(
            established_at=NOW + timedelta(days=2), established_by="later-worker"
        )
        == watermark
    ), "a later replica must not move the boundary over fresh notifications"
    assert (
        compat_store.claim_notification_deliveries(
            "later-worker", now=NOW, lease_duration=timedelta(seconds=30), limit=20
        )
        == []
    ), "suppressed and completed history must not reenter the dispatch queue"


def test_backlog_opt_in_is_sticky_and_does_not_consume_delivery_state(compat_store):
    saved = compat_store.save_notification_if_absent(
        notification("backlog", created_at=NOW - timedelta(days=1))
    )
    before = compat_store.get_notification_delivery(saved.notification_id)
    watermark = compat_store.establish_notification_watermark(
        owner_id="selected-dispatcher",
        established_at=NOW,
        established_by="first-worker",
        suppress_backlog=False,
    )

    assert watermark.suppressed == 0, (
        "explicit backlog delivery must not retire any row"
    )
    assert compat_store.get_notification_watermark() is None, (
        "a named dispatcher must not silently establish the default owner"
    )
    assert (
        compat_store.establish_notification_watermark(
            owner_id="selected-dispatcher",
            established_at=NOW + timedelta(days=1),
            established_by="second-worker",
            suppress_backlog=True,
        )
        == watermark
    ), "a second replica cannot reverse the first responsibility decision"
    assert compat_store.get_notification_delivery(saved.notification_id) == before, (
        "neither watermark call may charge attempts or change the pending delivery"
    )
    assert compat_store.get_notification_result(saved.notification_id) is None, (
        "opting into backlog must not invent a final notification result"
    )
    (lease,) = compat_store.claim_notification_deliveries(
        "worker", now=NOW, lease_duration=timedelta(seconds=30), limit=1
    )
    assert lease.notification_id == saved.notification_id, (
        "the historical notification must remain available for a real dispatcher"
    )
