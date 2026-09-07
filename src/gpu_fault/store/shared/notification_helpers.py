from __future__ import annotations

from datetime import datetime
from typing import Any

from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    NotificationDelivery,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError

PERFORMANCE_CLUSTER_PREFIX = "perf-cap-"
PERFORMANCE_DRILL_ID = "perf-capacity"


def bound_notifications(
    notifications: list[AdvisoryNotification],
    *,
    limit: int | None,
    newest_first: bool,
) -> list[AdvisoryNotification]:
    """Order and truncate a read of the notification table.

    Nothing ever leaves the notification table, and nothing ever leaves the set
    ``dispatch_pending`` treats as pending either: an entry qualifies when it has
    no result or a ``SKIPPED``/``FAILED`` one, and both a drill and an expired
    advisory are recorded as ``SKIPPED``. The synchronous dispatch path therefore
    read the entire history on every call and issued one extra result lookup per
    row. This is how it asks for a bounded slice instead, in the same shape
    ``list_workflows`` and ``list_attempt_observation_states`` already offer.

    Ordering is left exactly as it was for the unbounded ascending case, which is
    what every other caller uses: sorted on ``created_at`` alone, ties in
    whatever order the storage produced them. A bounded or reversed read has to
    be deterministic to be a stable page, so it breaks ties on
    ``notification_id``.
    """

    if limit is not None and limit < 0:
        raise ValueError("notification scan limit must not be negative")
    if limit is None and not newest_first:
        return notifications
    ordered = sorted(
        notifications,
        key=lambda item: (item.created_at, item.notification_id),
        reverse=newest_first,
    )
    return ordered if limit is None else ordered[:limit]


def with_incident_drill_label(
    notification: AdvisoryNotification,
    incident: FaultIncident | None,
) -> AdvisoryNotification:
    if notification.drill_id is not None:
        return notification
    drill_id = incident.drill_id if incident is not None else None
    if drill_id is None and notification.cluster_name.startswith(
        PERFORMANCE_CLUSTER_PREFIX
    ):
        drill_id = PERFORMANCE_DRILL_ID
    if drill_id is None:
        return notification
    return notification.model_copy(
        update={
            "drill_id": drill_id,
            "subject": f"[DRILL:{drill_id}] {notification.subject}",
            "body_text": (
                "【演练通知 / DRILL - 非真实故障】\n"
                f"Drill ID：{drill_id}\n\n" + notification.body_text
            ),
        }
    )


# Delivery states the outbox still owes a send for; everything else is over.
UNDELIVERED_STATUSES = frozenset(
    {
        NotificationDeliveryStatus.PENDING,
        NotificationDeliveryStatus.RETRY,
        NotificationDeliveryStatus.LEASED,
    }
)


def check_delivery_lease(
    current: NotificationDelivery | None,
    *,
    owner_id: str,
    lease_epoch: int,
    now: datetime | None,
) -> NotificationDelivery:
    """The row is ours to finish only while our lease on it is the live one.

    ``now`` is ``None`` for a release, which does not care whether the lease
    has run out -- the caller is giving the row back either way.
    """

    if (
        current is None
        or current.status is not NotificationDeliveryStatus.LEASED
        or current.lease_owner != owner_id
        or current.lease_epoch != lease_epoch
        or (
            now is not None
            and (current.lease_expires_at is None or current.lease_expires_at <= now)
        )
    ):
        raise WorkflowLeaseError("notification delivery lease is stale")
    return current


def completed_delivery(
    current: NotificationDelivery,
    *,
    result: NotificationResult,
    now: datetime,
    retry_at: datetime | None,
    terminal: bool,
) -> tuple[NotificationDelivery, bool]:
    """The row after one attempt, and whether the attempt is worth a result row.

    A result row is the notification's terminal verdict: ``SENT``, or the
    status the outbox gave up with. A retryable failure is neither -- it used
    to be written as ``FAILED`` anyway, which made one provider hiccup on a
    row the outbox was about to retry indistinguishable, on ``/metrics`` and to
    the critical alert, from a notification nobody will ever deliver. The
    attempt keeps its reason on the delivery row (``last_error``, ``attempts``)
    and nowhere else.
    """

    if result.status is NotificationStatus.SENT:
        status = NotificationDeliveryStatus.SENT
    elif terminal:
        status = NotificationDeliveryStatus.DEAD
    else:
        status = NotificationDeliveryStatus.RETRY
    value = current.model_copy(
        update={
            "status": status,
            "attempts": current.attempts + 1,
            "available_at": retry_at or now,
            "lease_owner": None,
            "lease_expires_at": None,
            "last_error": result.reason,
            "updated_at": now,
        }
    )
    return value, status is not NotificationDeliveryStatus.RETRY


def released_delivery(
    current: NotificationDelivery, *, now: datetime, retry_at: datetime
) -> NotificationDelivery:
    return current.model_copy(
        update={
            "status": NotificationDeliveryStatus.RETRY,
            "available_at": retry_at,
            "lease_owner": None,
            "lease_expires_at": None,
            "updated_at": now,
        }
    )


def effective_delivery_status(
    delivery: NotificationDelivery, result: NotificationResult | None
) -> NotificationDeliveryStatus:
    """The delivery state as the outbox treats the row, not as it was written.

    An inline ``send`` records its result without touching the delivery row,
    so a row can read PENDING forever under a SENT verdict; the claim query
    already skips those. A SKIPPED verdict on an undelivered row (a drill, an
    expired advisory judged inline) is what the dispatcher retires as DEAD the
    moment it claims it. Counting either as undelivered would make the outbox
    look permanently backed up on a deployment that delivers inline.
    """

    if result is not None and result.status is NotificationStatus.SENT:
        return NotificationDeliveryStatus.SENT
    if (
        result is not None
        and result.status is NotificationStatus.SKIPPED
        and delivery.status in UNDELIVERED_STATUSES
    ):
        return NotificationDeliveryStatus.DEAD
    return delivery.status


def delivery_age_anchor(delivery: NotificationDelivery) -> datetime:
    """When the wait for this delivery started: its creation, or the last time an
    operator re-queued it -- re-queueing restarts the shelf life, so it restarts
    the age as well."""

    anchor = delivery.created_at
    if delivery.requeued_at is not None and delivery.requeued_at > anchor:
        anchor = delivery.requeued_at
    return anchor


def delivery_stats_from_rows(
    deliveries: list[NotificationDelivery],
    results: dict[str, NotificationResult],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Fold delivery rows into the outbox state machine (ARCH-E E1)."""

    by_status = {status.value: 0 for status in NotificationDeliveryStatus}
    pending = 0
    oldest_anchor: datetime | None = None
    for delivery in deliveries:
        status = effective_delivery_status(
            delivery, results.get(delivery.notification_id)
        )
        by_status[status.value] += 1
        if status not in UNDELIVERED_STATUSES:
            continue
        pending += 1
        anchor = delivery_age_anchor(delivery)
        if oldest_anchor is None or anchor < oldest_anchor:
            oldest_anchor = anchor
    return {
        "by_status": by_status,
        "pending": pending,
        "oldest_pending_age_seconds": (
            max(0.0, (now - oldest_anchor).total_seconds())
            if oldest_anchor is not None
            else 0.0
        ),
    }
