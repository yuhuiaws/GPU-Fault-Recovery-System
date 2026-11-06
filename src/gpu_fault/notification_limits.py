"""How many notification deliveries one dispatch cycle may claim.

The bound is shared, so the dispatcher and the administrator config cannot
disagree on it. ``AdvisoryNotificationService.dispatch_outbox`` claims at most
this many deliveries per cycle. They all sit under one delivery lease (120 s by
default) that is not renewed while they are sent one after another, so a larger
batch only stretches the cycle against that lease. ``dispatch_pending`` and the
dispatch API request (``models.NotificationDispatchRequest``) take the same
bound.

``spec.notificationDelivery.batchSize`` is the dispatcher's per-cycle claim, so
the admin config refuses any value the dispatcher would refuse on every cycle.

A leaf module: the admin config imports it without the notification runtime.
"""

from __future__ import annotations

NOTIFICATION_DISPATCH_BATCH_MIN = 1
NOTIFICATION_DISPATCH_BATCH_MAX = 100


def require_dispatch_batch(limit: int) -> None:
    """Refuse a dispatch cycle that would claim outside the shared bound."""

    if not NOTIFICATION_DISPATCH_BATCH_MIN <= limit <= NOTIFICATION_DISPATCH_BATCH_MAX:
        raise ValueError(
            "dispatch limit must be between "
            f"{NOTIFICATION_DISPATCH_BATCH_MIN} and {NOTIFICATION_DISPATCH_BATCH_MAX}"
        )
