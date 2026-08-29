from __future__ import annotations

from typing import Any

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDelivery,
    NotificationDeliveryStatus,
    NotificationDispatchWatermark,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store.shared.errors import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.shared.notification_helpers import (
    with_incident_drill_label as _with_incident_drill_label,
)


class MemoryNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    _notification_deliveries: Any
    _notification_results: Any
    _notification_watermarks: Any
    _notifications: Any

    _incidents: Any
    _lock: Any
    _notification_by_deduplication_key: Any

    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification:
        with self._lock:
            notification = _with_incident_drill_label(
                notification,
                self._incidents.get(notification.incident_id),
            )
            existing_id = self._notification_by_deduplication_key.get(
                notification.deduplication_key
            )
            if existing_id:
                return self._notifications[existing_id]
            self._notifications[notification.notification_id] = notification
            self._notification_by_deduplication_key[notification.deduplication_key] = (
                notification.notification_id
            )
            now = datetime.now(timezone.utc)
            self._notification_deliveries[notification.notification_id] = (
                NotificationDelivery(
                    notification_id=notification.notification_id,
                    available_at=notification.not_before or now,
                    created_at=now,
                    updated_at=now,
                )
            )
            return notification

    def claim_notification_deliveries(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[NotificationDelivery]:
        with self._lock:
            candidates = []
            for notification in self._notifications.values():
                result = self._notification_results.get(notification.notification_id)
                if result is not None and result.status is NotificationStatus.SENT:
                    continue
                delivery = self._notification_deliveries.get(
                    notification.notification_id
                )
                if delivery is None:
                    continue
                eligible = (
                    delivery.status
                    in {
                        NotificationDeliveryStatus.PENDING,
                        NotificationDeliveryStatus.RETRY,
                    }
                    and delivery.available_at <= now
                ) or (
                    delivery.status is NotificationDeliveryStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at <= now
                )
                if eligible:
                    candidates.append((notification, delivery))
            candidates.sort(
                key=lambda item: (
                    item[0].priority,
                    item[1].available_at,
                    item[0].created_at,
                    item[0].notification_id,
                )
            )
            claimed = []
            for _, delivery in candidates[:limit]:
                value = delivery.model_copy(
                    update={
                        "status": NotificationDeliveryStatus.LEASED,
                        "lease_owner": owner_id,
                        "lease_epoch": delivery.lease_epoch + 1,
                        "lease_expires_at": now + lease_duration,
                        "updated_at": now,
                    }
                )
                self._notification_deliveries[delivery.notification_id] = value
                claimed.append(value)
            return claimed

    def get_notification_watermark(
        self, owner_id: str = "default"
    ) -> NotificationDispatchWatermark | None:
        with self._lock:
            return self._notification_watermarks.get(owner_id)

    def establish_notification_watermark(
        self,
        *,
        owner_id: str = "default",
        established_at: datetime,
        established_by: str,
        suppress_backlog: bool = True,
    ) -> NotificationDispatchWatermark:
        """Claim responsibility from ``established_at`` onwards, once.

        The first caller wins: a second replica starting later must not
        move the line forward, or notifications raised in between would be
        suppressed as if they were history.
        """
        with self._lock:
            existing = self._notification_watermarks.get(owner_id)
            if existing is not None:
                return existing
            suppressed = (
                self._suppress_notification_backlog(established_at, established_by)
                if suppress_backlog
                else 0
            )
            watermark = NotificationDispatchWatermark(
                owner_id=owner_id,
                established_at=established_at,
                established_by=established_by,
                suppressed=suppressed,
            )
            self._notification_watermarks[owner_id] = watermark
            return watermark

    def _suppress_notification_backlog(
        self, established_at: datetime, established_by: str
    ) -> int:
        """Retire undelivered pre-watermark rows without sending them."""
        suppressed = 0
        for notification in self._notifications.values():
            if notification.created_at >= established_at:
                continue
            notification_id = notification.notification_id
            result = self._notification_results.get(notification_id)
            if result is not None and result.status is NotificationStatus.SENT:
                continue
            delivery = self._notification_deliveries.get(notification_id)
            if delivery is None or delivery.status in {
                NotificationDeliveryStatus.SENT,
                NotificationDeliveryStatus.DEAD,
            }:
                continue
            reason = f"suppressed as pre-watermark backlog by {established_by}"
            self._notification_results[notification_id] = NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.SKIPPED,
                reason=reason,
            )
            self._notification_deliveries[notification_id] = delivery.model_copy(
                update={
                    "status": NotificationDeliveryStatus.DEAD,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "last_error": reason,
                    "updated_at": established_at,
                }
            )
            suppressed += 1
        return suppressed

    def enqueue_notification_delivery(
        self,
        notification_id: str,
        *,
        now: datetime | None = None,
    ) -> NotificationDelivery:
        with self._lock:
            notification = self.get_notification(notification_id)
            current = self._notification_deliveries.get(notification_id)
            if current is not None:
                queued_at = now or datetime.now(timezone.utc)
                # Asking for a delivery again restarts its shelf life:
                # otherwise the dispatcher expires it unsent for exactly
                # the reason it was asked for a second time.
                update: dict = {
                    "requeued_at": queued_at,
                    "updated_at": queued_at,
                }
                if current.status in {
                    NotificationDeliveryStatus.RETRY,
                    NotificationDeliveryStatus.DEAD,
                }:
                    update |= {
                        "status": NotificationDeliveryStatus.PENDING,
                        "available_at": queued_at,
                        "lease_owner": None,
                        "lease_expires_at": None,
                        "last_error": None,
                    }
                current = current.model_copy(update=update)
                self._notification_deliveries[notification_id] = current
                return current
            queued_at = now or datetime.now(timezone.utc)
            value = NotificationDelivery(
                notification_id=notification_id,
                available_at=max(
                    queued_at,
                    notification.not_before or queued_at,
                ),
                created_at=queued_at,
                updated_at=queued_at,
            )
            self._notification_deliveries[notification_id] = value
            return value

    def complete_notification_delivery(
        self,
        notification_id: str,
        *,
        owner_id: str,
        lease_epoch: int,
        result: NotificationResult,
        now: datetime,
        retry_at: datetime | None = None,
        terminal: bool = False,
    ) -> NotificationDelivery:
        with self._lock:
            current = self._notification_deliveries.get(notification_id)
            if (
                current is None
                or current.status is not NotificationDeliveryStatus.LEASED
                or current.lease_owner != owner_id
                or current.lease_epoch != lease_epoch
                or current.lease_expires_at is None
                or current.lease_expires_at <= now
            ):
                raise WorkflowLeaseError("notification delivery lease is stale")
            attempts = current.attempts + 1
            if result.status is NotificationStatus.SENT:
                status = NotificationDeliveryStatus.SENT
            elif terminal:
                status = NotificationDeliveryStatus.DEAD
            else:
                status = NotificationDeliveryStatus.RETRY
            value = current.model_copy(
                update={
                    "status": status,
                    "attempts": attempts,
                    "available_at": retry_at or now,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "last_error": result.reason,
                    "updated_at": now,
                }
            )
            self._notification_results[notification_id] = result
            self._notification_deliveries[notification_id] = value
            return value

    def release_notification_delivery(
        self,
        notification_id: str,
        *,
        owner_id: str,
        lease_epoch: int,
        now: datetime,
        retry_at: datetime,
    ) -> NotificationDelivery | None:
        """Hand a lease back without spending one of its attempts.

        Used when the failure is the provider's rate limit rather than
        anything about the notification: charging an attempt would walk a
        whole throttled batch to its terminal state and discard it,
        deciding that mail is undeliverable because too much of it was
        deliverable at once. Returns ``None`` rather than raising if the
        lease has already moved on -- the caller is giving up, so there is
        nothing to protect.
        """

        with self._lock:
            current = self._notification_deliveries.get(notification_id)
            if (
                current is None
                or current.status is not NotificationDeliveryStatus.LEASED
                or current.lease_owner != owner_id
                or current.lease_epoch != lease_epoch
            ):
                return None
            value = current.model_copy(
                update={
                    "status": NotificationDeliveryStatus.RETRY,
                    "available_at": retry_at,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._notification_deliveries[notification_id] = value
            return value

    def get_notification(self, notification_id: str) -> AdvisoryNotification:
        with self._lock:
            notification = self._notifications.get(notification_id)
            if notification is None:
                raise NotFoundError(notification_id)
            return notification

    def list_notifications(
        self,
    ) -> list[AdvisoryNotification]:
        with self._lock:
            return sorted(
                self._notifications.values(),
                key=lambda item: item.created_at,
            )

    def notification_status_counts(self) -> dict[NotificationStatus, int]:
        with self._lock:
            counts = {status: 0 for status in NotificationStatus}
            for notification_id in self._notifications:
                result = self._notification_results.get(notification_id)
                status = (
                    result.status if result is not None else NotificationStatus.QUEUED
                )
                counts[status] += 1
            return counts

    def save_notification_result(self, result: NotificationResult) -> None:
        with self._lock:
            self._notification_results[result.notification_id] = result

    def get_notification_result(
        self, notification_id: str
    ) -> NotificationResult | None:
        with self._lock:
            return self._notification_results.get(notification_id)
