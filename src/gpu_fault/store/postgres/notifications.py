from __future__ import annotations

from typing import Any, Callable

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDelivery,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store.shared.notification_helpers import (
    with_incident_drill_label as _with_incident_drill_label,
)
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


class PostgresNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _get: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]

    def list_notifications(
        self,
    ) -> list[AdvisoryNotification]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='notification'
                ORDER BY payload->>'created_at', key
                """
            )
            rows = cursor.fetchall()
        return [self._decode("notification", row[0]) for row in rows]

    def claim_notification_deliveries(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[NotificationDelivery]:
        lease_expires_at = now + lease_duration
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH candidates AS (
                        SELECT delivery.key
                        FROM gpu_fault_objects AS delivery
                        JOIN gpu_fault_objects AS notification
                          ON notification.kind='notification'
                         AND notification.key=delivery.key
                        LEFT JOIN gpu_fault_objects AS result
                          ON result.kind='notification_result'
                         AND result.key=delivery.key
                        WHERE delivery.kind='notification_delivery'
                          AND (
                              result.key IS NULL
                              OR result.payload->>'status'<>'SENT'
                          )
                          AND (
                              (
                                  delivery.payload->>'status' IN (
                                      'PENDING', 'RETRY'
                                  )
                                  AND (
                                      delivery.payload->>'available_at'
                                  )::timestamptz <= %s
                              )
                              OR (
                                  delivery.payload->>'status'='LEASED'
                                  AND delivery.payload
                                      ->>'lease_expires_at' IS NOT NULL
                                  AND (
                                      delivery.payload
                                      ->>'lease_expires_at'
                                  )::timestamptz <= %s
                              )
                          )
                        ORDER BY
                            (
                                notification.payload->>'priority'
                            )::integer,
                            (
                                delivery.payload->>'available_at'
                            )::timestamptz,
                            (
                                notification.payload->>'created_at'
                            )::timestamptz,
                            delivery.key
                        LIMIT %s
                        FOR UPDATE OF delivery SKIP LOCKED
                    )
                    UPDATE gpu_fault_objects AS delivery
                    SET payload=delivery.payload || jsonb_build_object(
                        'status', 'LEASED'::text,
                        'lease_owner', %s::text,
                        'lease_epoch',
                            COALESCE(
                                (
                                    delivery.payload->>'lease_epoch'
                                )::integer,
                                0
                            ) + 1,
                        'lease_expires_at', %s::text,
                        'updated_at', %s::text
                    )
                    FROM candidates
                    WHERE delivery.kind='notification_delivery'
                      AND delivery.key=candidates.key
                    RETURNING delivery.payload
                    """,
                    (
                        now,
                        now,
                        limit,
                        owner_id,
                        _utc_text(lease_expires_at),
                        _utc_text(now),
                    ),
                )
                rows = cursor.fetchall()
        return [self._decode("notification_delivery", row[0]) for row in rows]

    def _suppress_notification_backlog(
        self, established_at: datetime, established_by: str
    ) -> int:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT notification.key, delivery.payload
                FROM gpu_fault_objects AS notification
                JOIN gpu_fault_objects AS delivery
                  ON delivery.kind='notification_delivery'
                 AND delivery.key=notification.key
                LEFT JOIN gpu_fault_objects AS result
                  ON result.kind='notification_result'
                 AND result.key=notification.key
                WHERE notification.kind='notification'
                  AND (
                      notification.payload->>'created_at'
                  )::timestamptz < %s
                  AND (
                      result.key IS NULL
                      OR result.payload->>'status'<>'SENT'
                  )
                  AND delivery.payload->>'status'
                      NOT IN ('SENT', 'DEAD')
                ORDER BY notification.key
                """,
                (established_at,),
            )
            rows = cursor.fetchall()
        reason = f"suppressed as pre-watermark backlog by {established_by}"
        for notification_id, delivery_payload in rows:
            delivery = self._decode("notification_delivery", delivery_payload)
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
        return len(rows)

    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification:
        with self._db.transaction():
            notification = _with_incident_drill_label(
                notification,
                self._get_optional("incident", notification.incident_id),
            )
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_links(kind, key, value)
                    VALUES ('notification_dedup', %s, %s)
                    ON CONFLICT(kind, key) DO NOTHING
                    RETURNING value
                    """,
                    (
                        notification.deduplication_key,
                        notification.notification_id,
                    ),
                )
                won = cursor.fetchone()
            if won:
                now = datetime.now(timezone.utc)
                self._put(
                    "notification",
                    notification.notification_id,
                    notification,
                )
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
            existing_id = self._get_link(
                "notification_dedup",
                notification.deduplication_key,
            )
            if existing_id is None:
                raise RuntimeError("notification deduplication link disappeared")
            return self._get("notification", existing_id)
