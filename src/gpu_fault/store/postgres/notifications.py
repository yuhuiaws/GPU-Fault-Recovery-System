from __future__ import annotations

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
from gpu_fault.store.shared.errors import NotFoundError, WorkflowLeaseError
from gpu_fault.store.shared.notification_helpers import (
    UNDELIVERED_STATUSES,
    check_delivery_lease,
    completed_delivery,
    released_delivery,
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
    _get_for_update: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def list_notifications(
        self,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[AdvisoryNotification]:
        if limit is not None and limit < 0:
            raise ValueError("notification scan limit must not be negative")
        direction = "DESC" if newest_first else "ASC"
        query = f"""
            SELECT payload
            FROM gpu_fault_objects
            WHERE kind='notification'
            ORDER BY payload->>'created_at' {direction}, key {direction}
        """
        parameters: list[Any] = []
        if limit is not None:
            # Pushed into SQL rather than sliced after the fetch: the point of
            # the bound is that the whole notification history is never decoded,
            # and slicing in Python would decode all of it first.
            query += " LIMIT %s"
            parameters.append(limit)
        with self._db.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
        return [self._decode("notification", row[0]) for row in rows]

    def notification_status_counts(self) -> dict[NotificationStatus, int]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COALESCE(
                        result.payload->>'status',
                        'QUEUED'
                    ) AS status,
                    COUNT(*)
                FROM gpu_fault_objects AS notification
                LEFT JOIN gpu_fault_objects AS result
                  ON result.kind='notification_result'
                 AND result.key=notification.key
                WHERE notification.kind='notification'
                GROUP BY status
                """
            )
            rows = cursor.fetchall()
        counts = {status: 0 for status in NotificationStatus}
        for status, count in rows:
            counts[NotificationStatus(status)] = int(count)
        return counts

    def notification_delivery_stats(
        self, *, now: datetime | None = None
    ) -> dict[str, Any]:
        observed_at = now or datetime.now(timezone.utc)
        # The CASE is ``effective_delivery_status`` in SQL: a SENT verdict wins
        # over whatever the delivery row says, and a SKIPPED verdict on an
        # undelivered row is the DEAD the dispatcher would write on claiming it.
        # The anchor is ``delivery_age_anchor``: creation, or a later re-queue.
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    effective.status,
                    COUNT(*),
                    MIN(effective.anchor)
                FROM (
                    SELECT
                        CASE
                            WHEN result.payload->>'status'='SENT' THEN 'SENT'
                            WHEN result.payload->>'status'='SKIPPED'
                             AND delivery.payload->>'status' IN (
                                 'PENDING', 'RETRY', 'LEASED'
                             ) THEN 'DEAD'
                            ELSE delivery.payload->>'status'
                        END AS status,
                        GREATEST(
                            (delivery.payload->>'created_at')::timestamptz,
                            COALESCE(
                                (delivery.payload->>'requeued_at')::timestamptz,
                                (delivery.payload->>'created_at')::timestamptz
                            )
                        ) AS anchor
                    FROM gpu_fault_objects AS delivery
                    LEFT JOIN gpu_fault_objects AS result
                      ON result.kind='notification_result'
                     AND result.key=delivery.key
                    WHERE delivery.kind='notification_delivery'
                ) AS effective
                GROUP BY effective.status
                """
            )
            rows = cursor.fetchall()
        by_status = {status.value: 0 for status in NotificationDeliveryStatus}
        pending = 0
        oldest_anchor: datetime | None = None
        for status_value, count, anchor in rows:
            status = NotificationDeliveryStatus(status_value)
            by_status[status.value] = int(count)
            if status not in UNDELIVERED_STATUSES:
                continue
            pending += int(count)
            if anchor is not None and (oldest_anchor is None or anchor < oldest_anchor):
                oldest_anchor = anchor
        return {
            "by_status": by_status,
            "pending": pending,
            "oldest_pending_age_seconds": (
                max(0.0, (observed_at - oldest_anchor).total_seconds())
                if oldest_anchor is not None
                else 0.0
            ),
        }

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
        """Finish an attempt under the row lock the claim query respects.

        The inherited implementation serialised completers against each other
        with an advisory lock, then read the row unlocked and wrote it back
        whole. ``claim_notification_deliveries`` never takes that advisory
        lock -- it locks rows with ``FOR UPDATE SKIP LOCKED`` -- so a claim for
        an expired lease could land between the completer's read and its
        write, and the write then put the completer's stale epoch and status
        over the fresh claim (ARCH-E E2). ``_get_for_update`` takes the same
        row lock the claim skips: a concurrent claim skips this row until the
        completion commits, and a claim that got there first is seen as the
        epoch mismatch it is.
        """

        with self._db.transaction():
            current = check_delivery_lease(
                self._get_for_update("notification_delivery", notification_id),
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
                self._put("notification_result", notification_id, result)
            self._put("notification_delivery", notification_id, value)
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
        with self._db.transaction():
            try:
                current = check_delivery_lease(
                    self._get_for_update("notification_delivery", notification_id),
                    owner_id=owner_id,
                    lease_epoch=lease_epoch,
                    now=None,
                )
            except (NotFoundError, WorkflowLeaseError):
                return None
            value = released_delivery(current, now=now, retry_at=retry_at)
            self._put("notification_delivery", notification_id, value)
            return value

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

    def cleanup_terminal_notifications(
        self, *, older_than: datetime, limit: int
    ) -> int:
        """Drop settled notifications past retention with their delivery,
        result and dedup link (F-8 / G-9).

        A notification whose incident still exists is kept for the archiver
        (F-I1); one whose delivery is still PENDING/RETRY/LEASED is the
        dispatcher's. The dedup link is removed through the notification's
        own ``deduplication_key`` (the links primary key), never by value.
        """

        with self._state_transaction("notification/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH victims AS (
                        SELECT notification.key,
                               notification.payload->>'deduplication_key'
                                   AS deduplication_key
                        FROM gpu_fault_objects AS notification
                        LEFT JOIN gpu_fault_objects AS delivery
                          ON delivery.kind='notification_delivery'
                         AND delivery.key=notification.key
                        WHERE notification.kind='notification'
                          AND notification.payload->>'created_at' <= %s
                          AND (
                              delivery.key IS NULL
                              OR delivery.payload->>'status' IN ('SENT', 'DEAD')
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM gpu_fault_objects AS incident
                              WHERE incident.kind='incident'
                                AND incident.key=notification.payload->>'incident_id'
                          )
                        ORDER BY notification.payload->>'created_at',
                                 notification.key
                        LIMIT %s
                        FOR UPDATE OF notification SKIP LOCKED
                    ),
                    deleted_links AS (
                        DELETE FROM gpu_fault_links AS link
                        USING victims
                        WHERE link.kind='notification_dedup'
                          AND link.key=victims.deduplication_key
                          AND link.value=victims.key
                        RETURNING link.key
                    ),
                    deleted AS (
                        DELETE FROM gpu_fault_objects AS target
                        USING victims
                        WHERE target.kind IN (
                                  'notification',
                                  'notification_delivery',
                                  'notification_result'
                              )
                          AND target.key=victims.key
                        RETURNING target.kind, target.key
                    )
                    SELECT key FROM deleted WHERE kind='notification' ORDER BY key
                    """,
                    (_utc_text(older_than), limit),
                )
                keys = [row[0] for row in cursor.fetchall()]
            return log_cleanup("notification", keys)

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
