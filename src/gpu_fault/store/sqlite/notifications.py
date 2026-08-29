from __future__ import annotations

from typing import Any, Callable

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDelivery,
    NotificationDeliveryStatus,
    NotificationDispatchWatermark,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError
from gpu_fault.store.shared.notification_helpers import (
    with_incident_drill_label as _with_incident_drill_label,
)


class SqliteNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _link: Callable[..., Any]
    _list: Callable[..., Any]
    _lock: Any
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification:
        with self._lock:
            notification = _with_incident_drill_label(
                notification,
                self._get_optional("incident", notification.incident_id),
            )
            existing_id = self._get_link(
                "notification_dedup",
                notification.deduplication_key,
            )
            if existing_id:
                return self._get("notification", existing_id)
            self._put(
                "notification",
                notification.notification_id,
                notification,
            )
            self._link(
                "notification_dedup",
                notification.deduplication_key,
                notification.notification_id,
            )
            now = datetime.now(timezone.utc)
            self._put(
                "notification_delivery",
                notification.notification_id,
                NotificationDelivery(
                    notification_id=notification.notification_id,
                    available_at=notification.not_before or now,
                    created_at=now,
                    updated_at=now,
                ),
            )
            return notification

    def get_notification(self, notification_id: str) -> AdvisoryNotification:
        return self._get("notification", notification_id)

    def list_notifications(
        self,
    ) -> list[AdvisoryNotification]:
        return sorted(
            self._list("notification"),
            key=lambda item: item.created_at,
        )

    def notification_status_counts(self) -> dict[NotificationStatus, int]:
        with self._lock:
            notification_ids = {
                item.notification_id for item in self._list("notification")
            }
            results = {
                item.notification_id: item
                for item in self._list("notification_result")
                if item.notification_id in notification_ids
            }
        counts = {status: 0 for status in NotificationStatus}
        for notification_id in notification_ids:
            result = results.get(notification_id)
            status = result.status if result is not None else NotificationStatus.QUEUED
            counts[status] += 1
        return counts

    def save_notification_result(self, result: NotificationResult) -> None:
        with self._lock:
            self._put(
                "notification_result",
                result.notification_id,
                result,
            )

    def get_notification_result(
        self, notification_id: str
    ) -> NotificationResult | None:
        return self._get_optional("notification_result", notification_id)

    def claim_notification_deliveries(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[NotificationDelivery]:
        with self._state_transaction("notification_delivery/claims"):
            notifications = {
                item.notification_id: item for item in self._list("notification")
            }
            deliveries = {
                item.notification_id: item
                for item in self._list("notification_delivery")
            }
            results = {
                item.notification_id: item for item in self._list("notification_result")
            }
            candidates = []
            for notification in notifications.values():
                result = results.get(notification.notification_id)
                if result is not None and result.status is NotificationStatus.SENT:
                    continue
                delivery = deliveries.get(notification.notification_id)
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
                self._put(
                    "notification_delivery",
                    delivery.notification_id,
                    value,
                )
                claimed.append(value)
            return claimed

    def get_notification_watermark(
        self, owner_id: str = "default"
    ) -> NotificationDispatchWatermark | None:
        return self._get_optional("notification_watermark", owner_id)

    def establish_notification_watermark(
        self,
        *,
        owner_id: str = "default",
        established_at: datetime,
        established_by: str,
        suppress_backlog: bool = True,
    ) -> NotificationDispatchWatermark:
        with self._state_transaction(f"notification_watermark/{owner_id}"):
            existing = self._get_optional("notification_watermark", owner_id)
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
            self._put(
                "notification_watermark",
                owner_id,
                watermark,
            )
            return watermark

    def _suppress_notification_backlog(
        self, established_at: datetime, established_by: str
    ) -> int:
        results = {
            item.notification_id: item for item in self._list("notification_result")
        }
        deliveries = {
            item.notification_id: item for item in self._list("notification_delivery")
        }
        suppressed = 0
        for notification in self._list("notification"):
            if notification.created_at >= established_at:
                continue
            notification_id = notification.notification_id
            result = results.get(notification_id)
            if result is not None and result.status is NotificationStatus.SENT:
                continue
            delivery = deliveries.get(notification_id)
            if delivery is None or delivery.status in {
                NotificationDeliveryStatus.SENT,
                NotificationDeliveryStatus.DEAD,
            }:
                continue
            reason = f"suppressed as pre-watermark backlog by {established_by}"
            self._put(
                "notification_result",
                notification_id,
                NotificationResult(
                    notification_id=notification_id,
                    status=NotificationStatus.SKIPPED,
                    reason=reason,
                ),
            )
            self._put(
                "notification_delivery",
                notification_id,
                delivery.model_copy(
                    update={
                        "status": NotificationDeliveryStatus.DEAD,
                        "lease_owner": None,
                        "lease_expires_at": None,
                        "last_error": reason,
                        "updated_at": established_at,
                    }
                ),
            )
            suppressed += 1
        return suppressed

    def enqueue_notification_delivery(
        self,
        notification_id: str,
        *,
        now: datetime | None = None,
    ) -> NotificationDelivery:
        with self._state_transaction(f"notification_delivery/{notification_id}"):
            notification = self._get("notification", notification_id)
            current = self._get_optional("notification_delivery", notification_id)
            if current is not None:
                queued_at = now or datetime.now(timezone.utc)
                # See the in-memory implementation: asking again restarts
                # the shelf life.
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
                self._put(
                    "notification_delivery",
                    notification_id,
                    current,
                )
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
            self._put(
                "notification_delivery",
                notification_id,
                value,
            )
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
        with self._state_transaction(f"notification_delivery/{notification_id}"):
            current = self._get("notification_delivery", notification_id)
            if (
                current.status is not NotificationDeliveryStatus.LEASED
                or current.lease_owner != owner_id
                or current.lease_epoch != lease_epoch
                or current.lease_expires_at is None
                or current.lease_expires_at <= now
            ):
                raise WorkflowLeaseError("notification delivery lease is stale")
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
            self._put(
                "notification_result",
                notification_id,
                result,
            )
            self._put(
                "notification_delivery",
                notification_id,
                value,
            )
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
        with self._state_transaction(f"notification_delivery/{notification_id}"):
            current = self._get_optional("notification_delivery", notification_id)
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
            self._put(
                "notification_delivery",
                notification_id,
                value,
            )
            return value
