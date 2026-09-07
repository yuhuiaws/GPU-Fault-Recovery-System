"""A notification completer holds the row lock the claim query respects (ARCH-E E2).

``claim_notification_deliveries`` locks candidate rows with ``FOR UPDATE SKIP
LOCKED``. The inherited completer serialised itself against other completers
with an advisory lock the claim never takes, then read the row unlocked and
wrote it back whole -- so a claim for an expired lease could land between that
read and that write, and the write put the completer's stale epoch and status
over the fresh claim. Postgres-only: the race is between two connections.

Run alone with ``GPU_FAULT_TEST_POSTGRES_URL`` set; never under xdist.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store import PostgresStore, WorkflowLeaseError
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

LEASE = timedelta(seconds=120)


@pytest.fixture
def store():
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


@pytest.fixture
def sibling(store):
    """A second replica: its own pool, so its claim is a separate transaction."""
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()


def _saved(store) -> AdvisoryNotification:
    return store.save_notification_if_absent(
        AdvisoryNotification(
            notification_id="notification-lock",
            deduplication_key="notification-lock",
            cluster_name="cluster-a",
            incident_id="incident-lock",
            subject="subject",
            body_text="body",
            support_case_draft="",
        )
    )


def test_a_claim_for_an_expired_lease_cannot_slip_under_a_running_completion(
    store, sibling, monkeypatch
) -> None:
    notification = _saved(store)
    start = datetime.now(timezone.utc)
    [lease] = store.claim_notification_deliveries(
        "pod-a", now=start, lease_duration=LEASE, limit=1
    )
    # pod-a's provider call took longer than its lease. From pod-b's point of
    # view the LEASED row has expired and is claimable again.
    late = start + LEASE + timedelta(seconds=5)
    claimed_meanwhile: list = []
    original_put = store._put

    def put_with_a_concurrent_claim(kind, key, value, **kwargs):
        if kind == "notification_delivery" and not claimed_meanwhile:
            claimed_meanwhile.append(
                sibling.claim_notification_deliveries(
                    "pod-b", now=late, lease_duration=LEASE, limit=5
                )
            )
        return original_put(kind, key, value, **kwargs)

    monkeypatch.setattr(store, "_put", put_with_a_concurrent_claim)

    # pod-a completes against a clock where its own lease is still valid: the
    # row it read says so, and nothing but the row lock can tell it otherwise.
    value = store.complete_notification_delivery(
        notification.notification_id,
        owner_id="pod-a",
        lease_epoch=lease.lease_epoch,
        result=NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="ses-a",
        ),
        now=start + timedelta(seconds=1),
    )

    assert claimed_meanwhile == [[]], (
        "the row pod-a is completing is locked; pod-b's claim must skip it "
        "rather than take a lease the completion is about to overwrite"
    )
    assert value.status is NotificationDeliveryStatus.SENT, value
    stored = store.get_notification_result(notification.notification_id)
    assert stored is not None and stored.status is NotificationStatus.SENT, stored
    # Once the completion has committed, the row is SENT and no longer a claim
    # candidate for anyone.
    assert (
        sibling.claim_notification_deliveries(
            "pod-b", now=late, lease_duration=LEASE, limit=5
        )
        == []
    ), "a SENT row must not be re-claimed"


def test_a_completer_that_lost_the_row_sees_the_fresh_epoch_not_its_own(
    store, sibling
) -> None:
    notification = _saved(store)
    start = datetime.now(timezone.utc)
    [first] = store.claim_notification_deliveries(
        "pod-a", now=start, lease_duration=LEASE, limit=1
    )
    late = start + LEASE + timedelta(seconds=5)
    [second] = sibling.claim_notification_deliveries(
        "pod-b", now=late, lease_duration=LEASE, limit=1
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
            now=late + timedelta(seconds=1),
        )
    assert (
        store.release_notification_delivery(
            notification.notification_id,
            owner_id="pod-a",
            lease_epoch=first.lease_epoch,
            now=late + timedelta(seconds=1),
            retry_at=late + timedelta(seconds=30),
        )
        is None
    ), "a stale releaser must not touch the fresh lease"
    stats = store.notification_delivery_stats(now=late + timedelta(seconds=2))
    assert stats["by_status"][NotificationDeliveryStatus.LEASED.value] == 1, stats
    assert store.get_notification_result(notification.notification_id) is None
