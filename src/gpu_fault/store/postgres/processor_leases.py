from __future__ import annotations

from typing import Any, Callable

from datetime import datetime, timedelta, timezone


class PostgresProcessorLeaseMixin:
    # Attributes supplied by the composed concrete implementation.
    get_processor_request: Callable[..., Any]

    _db: Any
    _decode: Callable[..., Any]
    _persist_processor_state: Callable[..., Any]
    _processor_queue_effective_payload: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def validate_processor_lane(
        self,
        ordering_key: str,
        owner_id: str,
        epoch: int,
        lease_token: str,
    ) -> bool:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM gpu_fault_processor_lanes
                    WHERE ordering_key=%s
                      AND owner_id=%s
                      AND epoch=%s
                      AND lease_token=%s
                      AND lease_expires_at > now()
                )
                """,
                (
                    ordering_key,
                    owner_id,
                    epoch,
                    lease_token,
                ),
            )
            return cursor.fetchone()[0]

    def _lock_processor_queue_row(self, request_id: str) -> None:
        """Take the queue row lock before touching its lane row.

        Every lane write below is followed by a write to the queue row
        it fences, while the claim path locks the queue row first and
        the lane second. Without this the two orders cross: a claim
        that picked up a request whose lease had just expired holds the
        queue row and waits for the lane, and the renew or completion
        that still owns the lane waits for the queue row. Locking here
        costs one round trip inside a transaction that is about to
        write the row anyway.
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM gpu_fault_processor_queue
                WHERE request_id=%s
                FOR UPDATE
                """,
                (request_id,),
            )
            cursor.fetchall()

    def renew_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        lease_duration: timedelta,
    ) -> bool:
        with self._state_transaction(f"processor_request/{request_id}"):
            self._lock_processor_queue_row(request_id)
            current = self.get_processor_request(request_id)
            now = datetime.now(timezone.utc)
            if (
                current.lease_owner != owner_id
                or current.leader_epoch != lane_epoch
                or current.lease_token != lease_token
            ):
                return False
            expires_at = now + lease_duration
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE gpu_fault_processor_lanes
                    SET lease_expires_at=%s, updated_at=%s
                    WHERE ordering_key=%s
                      AND owner_id=%s
                      AND epoch=%s
                      AND lease_token=%s
                      AND lease_expires_at > %s
                    """,
                    (
                        expires_at,
                        now,
                        current.ordering_key(),
                        owner_id,
                        lane_epoch,
                        lease_token,
                        now,
                    ),
                )
                renewed = cursor.rowcount == 1
            if not renewed:
                return False
            self._persist_processor_state(
                current.model_copy(
                    update={
                        "lease_expires_at": expires_at,
                        "updated_at": now,
                    }
                )
            )
            return True

    def release_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None:
        from gpu_fault.processor import (
            ProcessorRequestStatus,
        )

        with self._state_transaction(f"processor_request/{request_id}"):
            self._lock_processor_queue_row(request_id)
            current = self.get_processor_request(request_id)
            if (
                current.lease_owner != owner_id
                or current.leader_epoch != lane_epoch
                or current.lease_token != lease_token
            ):
                return
            # Completing does not clear the fencing fields, so without
            # this a COMPLETED request passes the check above and goes
            # back to PENDING - it is claimed again and its side effects
            # run twice. The callers do exactly that: a batch whose
            # result is missing one request raises, and the handler then
            # releases every request in the batch, including the ones the
            # same statement had just committed.
            if current.status != ProcessorRequestStatus.LEASED:
                return
            now = datetime.now(timezone.utc)
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE gpu_fault_processor_lanes
                    SET lease_expires_at=%s, updated_at=%s
                    WHERE ordering_key=%s
                      AND owner_id=%s
                      AND epoch=%s
                      AND lease_token=%s
                    """,
                    (
                        now,
                        now,
                        current.ordering_key(),
                        owner_id,
                        lane_epoch,
                        lease_token,
                    ),
                )
                released = cursor.rowcount == 1
            if not released:
                return
            self._persist_processor_state(
                current.model_copy(
                    update={
                        "status": ProcessorRequestStatus.PENDING,
                        "lease_owner": None,
                        "leader_epoch": None,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "not_before": not_before,
                        "retry_count": (
                            current.retry_count if retry_count is None else retry_count
                        ),
                        "updated_at": now,
                    }
                )
            )

    def release_processor_request(
        self,
        request_id: str,
        owner_id: str,
        leader_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None:
        from gpu_fault.processor import (
            ProcessorRequestStatus,
        )

        with self._state_transaction(f"processor_request/{request_id}"):
            current = self.get_processor_request(request_id)
            if (
                current.lease_owner != owner_id
                or current.leader_epoch != leader_epoch
                or current.lease_token != lease_token
            ):
                return
            # Same guard as ``release_active_processor_request``: completion
            # keeps the fencing fields, so without it a COMPLETED row passes
            # the token check and is reopened as PENDING (F-D9).
            if current.status != ProcessorRequestStatus.LEASED:
                return
            self._persist_processor_state(
                current.model_copy(
                    update={
                        "status": ProcessorRequestStatus.PENDING,
                        "lease_owner": None,
                        "leader_epoch": None,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "not_before": not_before,
                        "retry_count": (
                            current.retry_count if retry_count is None else retry_count
                        ),
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
            )

    def reclaim_expired_processor_leases(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> int:
        """Hand LEASED rows whose lease lapsed back to PENDING (F-D5).

        A lease that expired without a release means its owner is gone.
        The claim window does reclaim such rows, but only when they fall
        inside it; behind the retry horizon, or on a priority nobody is
        claiming, a row could stay LEASED indefinitely with nothing
        counting it. Booked as a retry (``retry_count + 1``) so the second
        execution is visible as one. Queue rows are locked here and the
        lane row is left alone: the lane lease expired with the request
        (renewal moves both in one transaction) and the next claim's
        ``leased_lanes`` upsert takes it over, so this keeps the queue-row
        -> lane lock order documented on ``_lock_processor_queue_row``.
        ``gpu_fault_processor_queue_available`` leads with ``status``, so
        the scan is bounded by the LEASED set, not the history.
        """

        from gpu_fault.processor import (
            ProcessorRequestStatus,
        )

        with self._state_transaction("processor_request/reclaim-expired"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {
                        self._processor_queue_effective_payload(
                            "gpu_fault_processor_queue"
                        )
                    }
                    FROM gpu_fault_processor_queue
                    WHERE status='LEASED'
                      AND lease_expires_at <= %s
                    ORDER BY lease_expires_at, request_id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (now, limit),
                )
                rows = cursor.fetchall()
            reclaimed = 0
            for row in rows:
                current = self._decode("processor_request", row[0])
                self._persist_processor_state(
                    current.model_copy(
                        update={
                            "status": ProcessorRequestStatus.PENDING,
                            "lease_owner": None,
                            "leader_epoch": None,
                            "lease_token": None,
                            "lease_expires_at": None,
                            "not_before": None,
                            "retry_count": current.retry_count + 1,
                            # The state upsert is guarded on ``updated_at``.
                            "updated_at": max(now, current.updated_at),
                        }
                    )
                )
                reclaimed += 1
            return reclaimed

    def cleanup_processor_lanes(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Bounded retention delete for the lane table.

        The outer scan is served by ``gpu_fault_processor_lanes_expiry``
        and the guard by ``gpu_fault_processor_queue_lane``, so no new
        index is required. Resetting a lane epoch by deleting the row is
        safe because every fencing predicate also matches ``lease_token``.
        """

        with self._state_transaction("processor_lane/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH victims AS (
                        SELECT lane.ordering_key
                        FROM gpu_fault_processor_lanes AS lane
                        WHERE lane.lease_expires_at <= %s
                          AND NOT EXISTS (
                              SELECT 1
                              FROM gpu_fault_processor_queue AS queue
                              WHERE queue.ordering_key
                                    = lane.ordering_key
                                AND queue.status IN (
                                    'PENDING', 'LEASED'
                                )
                          )
                        ORDER BY lane.lease_expires_at,
                                 lane.ordering_key
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    ),
                    deleted AS (
                        DELETE FROM gpu_fault_processor_lanes AS target
                        USING victims
                        WHERE target.ordering_key=victims.ordering_key
                        RETURNING target.ordering_key
                    )
                    SELECT count(*) FROM deleted
                    """,
                    (older_than, limit),
                )
                deleted = cursor.fetchone()[0]
            return deleted
