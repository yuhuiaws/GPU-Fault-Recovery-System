"""Notification-record templates shared by the key/value stores: the
notification, its result, the dispatch watermark and the delivery row's
lease transitions, each one row keyed by notification or owner id."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDelivery,
    NotificationDeliveryStatus,
    NotificationDispatchWatermark,
    NotificationResult,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError
from gpu_fault.store.shared.notification_helpers import (
    check_delivery_lease,
    completed_delivery,
    released_delivery,
)
from gpu_fault.store.shared.primitives import (
    GetOptionalRecord,
    GetRecord,
    PutRecord,
    StatementGuard,
    StateTransaction,
)


class SharedNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    _suppress_notification_backlog: Callable[..., Any]

    _get: GetRecord
    _get_optional: GetOptionalRecord
    _put: PutRecord
    _state_transaction: StateTransaction
    _statement_guard: StatementGuard

    def get_notification(self, notification_id: str) -> AdvisoryNotification:
        return self._get("notification", notification_id)

    def save_notification_result(self, result: NotificationResult) -> None:
        with self._statement_guard():
            self._put(
                "notification_result",
                result.notification_id,
                result,
            )

    def get_notification_result(
        self, notification_id: str
    ) -> NotificationResult | None:
        return self._get_optional("notification_result", notification_id)

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
            current = check_delivery_lease(
                self._get("notification_delivery", notification_id),
                owner_id=owner_id,
                lease_epoch=lease_epoch,
                now=now,
            )
            value, record_result = completed_delivery(
                current,
                result=result,
                now=now,
                retry_at=retry_at,
                terminal=terminal,
            )
            if record_result:
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
            try:
                current = check_delivery_lease(
                    self._get_optional("notification_delivery", notification_id),
                    owner_id=owner_id,
                    lease_epoch=lease_epoch,
                    now=None,
                )
            except WorkflowLeaseError:
                return None
            value = released_delivery(current, now=now, retry_at=retry_at)
            self._put(
                "notification_delivery",
                notification_id,
                value,
            )
            return value
