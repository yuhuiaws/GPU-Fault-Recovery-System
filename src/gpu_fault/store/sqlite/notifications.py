from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDelivery,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store.shared.cleanup_log import log_cleanup
from gpu_fault.store.shared.notification_helpers import (
    bound_notifications,
    delivery_stats_from_rows,
)
from gpu_fault.store.shared.notification_helpers import (
    with_incident_drill_label as _with_incident_drill_label,
)


class SqliteNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: Callable[..., Any]
    _delete: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _link: Callable[..., Any]
    _list: Callable[..., Any]
    _db: sqlite3.Connection
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

    def cleanup_terminal_notifications(
        self, *, older_than: datetime, limit: int
    ) -> int:
        with self._state_transaction("notification/cleanup"):
            victims = []
            for notification in sorted(
                self._list("notification"),
                key=lambda item: (item.created_at, item.notification_id),
            ):
                if notification.created_at > older_than:
                    continue
                if self._get_optional("incident", notification.incident_id) is not None:
                    continue
                delivery = self._get_optional(
                    "notification_delivery", notification.notification_id
                )
                if delivery is not None and delivery.status not in {
                    NotificationDeliveryStatus.SENT,
                    NotificationDeliveryStatus.DEAD,
                }:
                    continue
                victims.append(notification)
                if len(victims) >= limit:
                    break
            for notification in victims:
                for kind in (
                    "notification",
                    "notification_delivery",
                    "notification_result",
                ):
                    self._delete(kind, notification.notification_id)
                self._db.execute(
                    "DELETE FROM links WHERE kind='notification_dedup' AND key=? AND value=?",
                    (notification.deduplication_key, notification.notification_id),
                )
            return log_cleanup(
                "notification", [item.notification_id for item in victims]
            )

    def list_notifications(
        self,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[AdvisoryNotification]:
        return bound_notifications(
            sorted(
                self._list("notification"),
                key=lambda item: item.created_at,
            ),
            limit=limit,
            newest_first=newest_first,
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

    def notification_delivery_stats(
        self, *, now: datetime | None = None
    ) -> dict[str, Any]:
        observed_at = now or datetime.now(timezone.utc)
        with self._lock:
            deliveries = list(self._list("notification_delivery"))
            results = {
                item.notification_id: item for item in self._list("notification_result")
            }
        return delivery_stats_from_rows(deliveries, results, now=observed_at)

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
