from __future__ import annotations

from typing import Any, Callable

import secrets
from datetime import datetime, timedelta

from gpu_fault.channel_registry import (
    CHANNEL_REGISTRY,
    FAULT_CHANNEL_PATHS,
)
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


# The lane fairness rules below have to know which queue rows are faults.
# Membership comes from the channel registry so that registering a new
# FAULT channel cannot silently leave it out of the correlation checks;
# registry declaration order is kept so the rendered SQL stays stable.
_GPU_EVENT_FAULT_PATHS = (
    "/v1/gpu-events/xid",
    "/v1/gpu-events/sxid",
)
_FAULT_CLAIM_PATHS = (
    *_GPU_EVENT_FAULT_PATHS,
    *(path for path in CHANNEL_REGISTRY if path in FAULT_CHANNEL_PATHS),
)


def _fault_path_list_sql(indent: int) -> str:
    pad = " " * indent
    return ",\n".join(f"{pad}'{path}'" for path in _FAULT_CLAIM_PATHS)


_CLAIM_ACTIVE_PROCESSOR_SQL = f"""\
WITH claim_window AS MATERIALIZED (
    SELECT
        candidate.request_id,
        candidate.ordering_key,
        CASE
            WHEN candidate.priority=100
              AND candidate.created_at <= %s
            THEN 49
            ELSE candidate.priority
        END AS priority,
        candidate.created_at,
        candidate.payload->>'path' AS path,
        coalesce(
            candidate.payload
                ->'correlation_scope_keys',
            '[]'::jsonb
        ) AS scope_keys
    FROM gpu_fault_processor_queue AS candidate
    LEFT JOIN gpu_fault_processor_lanes AS lane
      ON lane.ordering_key=candidate.ordering_key
    WHERE (
        candidate.status='PENDING'
        OR (
            candidate.status='LEASED'
            AND candidate.lease_expires_at <= %s
        )
    )
      AND (
          candidate.not_before IS NULL
          OR candidate.not_before <= %s
      )
      AND NOT EXISTS (
          SELECT 1
          FROM gpu_fault_processor_queue AS retry_barrier
          WHERE retry_barrier.ordering_key=candidate.ordering_key
            AND retry_barrier.status='PENDING'
            AND retry_barrier.lane_policy='STRICT'
            AND retry_barrier.not_before > %s
      )
      AND (
          %s
          OR candidate.payload->>'path'=ANY(%s)
      )
      AND (
          %s
          OR NOT (
              candidate.payload->>'path'=ANY(%s)
          )
      )
      AND (
          lane.ordering_key IS NULL
          OR lane.lease_expires_at <= %s
      )
    ORDER BY
        candidate.priority,
        candidate.created_at,
        candidate.request_id
    LIMIT %s
),
candidate_ids AS MATERIALIZED (
    SELECT DISTINCT ON (claim_window.ordering_key)
        claim_window.request_id,
        claim_window.ordering_key,
        CASE
            WHEN claim_window.path
                 = '/v1/workload-observations'
              AND EXISTS (
                  SELECT 1
                  FROM
                      jsonb_array_elements_text(
                          claim_window.scope_keys
                      ) AS candidate_scope(value)
                  WHERE EXISTS (
                      SELECT 1
                      FROM gpu_fault_processor_queue
                           AS fault
                      WHERE fault.status='PENDING'
                        AND fault.payload->>'path' IN (
{_fault_path_list_sql(28)}
                        )
                        AND coalesce(
                            fault.payload
                                ->'correlation_scope_keys',
                            '[]'::jsonb
                        ) ? candidate_scope.value
                  )
              )
            THEN -1
            ELSE claim_window.priority
        END AS effective_priority,
        claim_window.created_at
    FROM claim_window
    WHERE NOT (
        claim_window.path IN (
{_fault_path_list_sql(12)}
        )
        AND EXISTS (
            SELECT 1
            FROM
                jsonb_array_elements_text(
                    claim_window.scope_keys
                ) AS fault_scope(value)
            WHERE EXISTS (
                SELECT 1
                FROM gpu_fault_processor_queue
                     AS observation
                WHERE observation.status IN (
                          'PENDING', 'LEASED'
                      )
                  AND observation.payload->>'path'
                      = '/v1/workload-observations'
                  AND coalesce(
                      observation.payload
                          ->'correlation_scope_keys',
                      '[]'::jsonb
                  ) ? fault_scope.value
            )
        )
    )
    ORDER BY
        claim_window.ordering_key,
        effective_priority,
        claim_window.created_at,
        claim_window.request_id
),
selected AS MATERIALIZED (
    SELECT
        queue.request_id,
        queue.ordering_key,
        candidate_ids.effective_priority,
        candidate_ids.created_at,
        (
            replace(
                gen_random_uuid()::text, '-', ''
            )
            || replace(
                gen_random_uuid()::text, '-', ''
            )
        ) AS lease_token
    FROM gpu_fault_processor_queue AS queue
    JOIN candidate_ids
      ON candidate_ids.request_id=queue.request_id
    ORDER BY
        candidate_ids.effective_priority,
        candidate_ids.created_at,
        queue.request_id
    LIMIT %s
    FOR UPDATE OF queue SKIP LOCKED
),
leased_lanes AS (
    INSERT INTO gpu_fault_processor_lanes (
        ordering_key,
        owner_id,
        epoch,
        lease_token,
        lease_expires_at,
        updated_at
    )
    SELECT
        selected.ordering_key,
        %s,
        1,
        selected.lease_token,
        %s,
        %s
    FROM selected
    -- selected is ordered by priority, so two
    -- workers claiming an overlapping set of
    -- ordering keys reach the lane rows in
    -- different orders and deadlock on the
    -- unique index ("while inserting index
    -- tuple in relation
    -- gpu_fault_processor_lanes"). Ordering the
    -- feed makes every claim take the lane locks
    -- in the same sequence; the lease that each
    -- request receives is unaffected because the
    -- lease_token is already bound to the row in
    -- selected.
    ORDER BY selected.ordering_key
    ON CONFLICT(ordering_key) DO UPDATE SET
        owner_id=excluded.owner_id,
        epoch=(
            gpu_fault_processor_lanes.epoch + 1
        ),
        lease_token=excluded.lease_token,
        lease_expires_at=excluded.lease_expires_at,
        updated_at=excluded.updated_at
    WHERE
        gpu_fault_processor_lanes.lease_expires_at
        <= %s
    RETURNING ordering_key, epoch, lease_token
),
updated AS (
    UPDATE gpu_fault_processor_queue AS queue
    SET
        status='LEASED',
        lease_owner=%s,
        leader_epoch=leased_lanes.epoch,
        lease_token=leased_lanes.lease_token,
        lease_expires_at=%s,
        updated_at=%s
        {{payload_update}}
    FROM selected
    JOIN leased_lanes
      ON leased_lanes.ordering_key=
         selected.ordering_key
     AND leased_lanes.lease_token=
         selected.lease_token
    WHERE queue.request_id=selected.request_id
    RETURNING
        queue.request_id,
        queue.payload,
        queue.status,
        queue.lease_owner,
        queue.leader_epoch,
        queue.lease_token,
        queue.lease_expires_at,
        queue.response_status,
        queue.response_content_type,
        queue.response_body_base64,
        queue.not_before,
        queue.retry_count,
        queue.lane_policy,
        queue.updated_at
)
SELECT {{effective_payload}}
FROM updated
JOIN selected
  ON selected.request_id=updated.request_id
ORDER BY
    selected.effective_priority,
    selected.created_at,
    selected.request_id
"""


class PostgresProcessorClaimsMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _maybe_reconcile_legacy_processor_requests: Callable[..., Any]
    _persist_processor_state: Callable[..., Any]
    _processor_queue_effective_payload: Callable[..., Any]
    claim_window_multiplier: Any
    get_processor_leadership: Callable[..., Any]
    processor_queue_state_mode: Any

    def claim_processor_requests(
        self,
        owner_id: str,
        leader_epoch: int,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ):
        from gpu_fault.processor import (
            ProcessorRequestStatus,
        )

        with self._db.transaction():
            leadership = self.get_processor_leadership()
            if (
                leadership is None
                or leadership.owner_id != owner_id
                or leadership.epoch != leader_epoch
                or leadership.lease_expires_at <= now
            ):
                return []
            self._maybe_reconcile_legacy_processor_requests()
            with self._db.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT queue.payload
                    FROM gpu_fault_processor_queue AS queue
                    JOIN (
                        SELECT DISTINCT ON (candidate.ordering_key)
                            candidate.request_id,
                            CASE
                                WHEN candidate.payload->>'path'
                                     = '/v1/workload-observations'
                                  AND EXISTS (
                                      SELECT 1
                                      FROM
                                          jsonb_array_elements_text(
                                              coalesce(
                                                  candidate.payload
                                                      ->'correlation_scope_keys',
                                                  '[]'::jsonb
                                              )
                                          ) AS candidate_scope(value)
                                      WHERE EXISTS (
                                          SELECT 1
                                          FROM gpu_fault_processor_queue
                                               AS fault
                                          WHERE fault.status='PENDING'
                                            AND fault.payload->>'path' IN (
{_fault_path_list_sql(48)}
                                            )
                                            AND coalesce(
                                                fault.payload
                                                    ->'correlation_scope_keys',
                                                '[]'::jsonb
                                            ) ? candidate_scope.value
                                      )
                                  )
                                THEN -1
                                ELSE candidate.priority
                            END AS effective_priority,
                            candidate.created_at
                        FROM gpu_fault_processor_queue AS candidate
                        WHERE (
                            candidate.status='PENDING'
                            OR (
                                candidate.status='LEASED'
                                AND candidate.lease_expires_at <= %s
                            )
                        )
                          AND (
                              candidate.not_before IS NULL
                              OR candidate.not_before <= %s
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM gpu_fault_processor_queue
                                   AS retry_barrier
                              WHERE retry_barrier.ordering_key
                                    = candidate.ordering_key
                                AND retry_barrier.status='PENDING'
                                AND retry_barrier.lane_policy='STRICT'
                                AND retry_barrier.not_before > %s
                          )
                          AND NOT (
                              candidate.payload->>'path' IN (
{_fault_path_list_sql(34)}
                              )
                              AND EXISTS (
                                  SELECT 1
                                  FROM
                                      jsonb_array_elements_text(
                                          coalesce(
                                              candidate.payload
                                                  ->'correlation_scope_keys',
                                              '[]'::jsonb
                                          )
                                      ) AS fault_scope(value)
                                  WHERE EXISTS (
                                      SELECT 1
                                      FROM gpu_fault_processor_queue
                                           AS observation
                                      WHERE observation.status IN (
                                                'PENDING', 'LEASED'
                                            )
                                        AND observation.payload->>'path'
                                            = '/v1/workload-observations'
                                        AND coalesce(
                                            observation.payload
                                                ->'correlation_scope_keys',
                                            '[]'::jsonb
                                        ) ? fault_scope.value
                                  )
                              )
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM gpu_fault_processor_queue AS busy
                              WHERE busy.ordering_key
                                    = candidate.ordering_key
                                AND busy.status='LEASED'
                                AND busy.lease_expires_at > %s
                          )
                        ORDER BY
                            candidate.ordering_key,
                            effective_priority,
                            candidate.created_at,
                            candidate.request_id
                    ) AS selected
                      ON selected.request_id=queue.request_id
                    ORDER BY
                        selected.effective_priority,
                        selected.created_at,
                        selected.request_id
                    LIMIT %s
                    FOR UPDATE OF queue SKIP LOCKED
                    """,
                    (now, now, now, now, limit),
                )
                rows = cursor.fetchall()
            claimed = []
            for row in rows:
                item = self._decode("processor_request", row[0])
                value = item.model_copy(
                    update={
                        "status": ProcessorRequestStatus.LEASED,
                        "lease_owner": owner_id,
                        "leader_epoch": leader_epoch,
                        "lease_token": secrets.token_urlsafe(32),
                        "lease_expires_at": now + lease_duration,
                        "updated_at": now,
                    }
                )
                self._persist_processor_state(value)
                claimed.append(value)
            return claimed

    def claim_active_processor_requests(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
        routine_starvation_seconds: float = 30.0,
    ):
        included = sorted(include_paths or ())
        excluded = sorted(exclude_paths or ())
        expires_at = now + lease_duration
        routine_starvation_before = now - timedelta(seconds=routine_starvation_seconds)
        # Bound the claim's cost. Lane dedup and the two
        # correlation-scope interlocks are the expensive parts, and they
        # now run over this window instead of the whole eligible backlog,
        # so claim latency stops scaling with queue depth - that scaling
        # is what let depth feed back into itself at a 65536 cap. A
        # window short on claimable rows just returns fewer than
        # ``limit``; the caller claims again.
        window_limit = max(limit * self.claim_window_multiplier, limit + 64)
        payload_update = ""
        payload_parameters: tuple[object, ...] = ()
        if self.processor_queue_state_mode != "dedicated":
            payload_update = """
                            , payload=queue.payload ||
                                jsonb_build_object(
                                    'status', 'LEASED'::text,
                                    'lease_owner', %s::text,
                                    'leader_epoch',
                                        leased_lanes.epoch,
                                    'lease_token',
                                        leased_lanes.lease_token,
                                    'lease_expires_at', %s::text,
                                    'updated_at', %s::text
                                )
            """
            payload_parameters = (
                owner_id,
                _utc_text(expires_at),
                _utc_text(now),
            )
        with self._db.transaction():
            self._maybe_reconcile_legacy_processor_requests()
            with self._db.cursor() as cursor:
                cursor.execute(
                    _CLAIM_ACTIVE_PROCESSOR_SQL.format(
                        payload_update=payload_update,
                        effective_payload=(
                            self._processor_queue_effective_payload("updated")
                        ),
                    ),
                    (
                        routine_starvation_before,
                        now,
                        now,
                        now,
                        not included,
                        included,
                        not excluded,
                        excluded,
                        now,
                        window_limit,
                        limit,
                        owner_id,
                        expires_at,
                        now,
                        now,
                        owner_id,
                        expires_at,
                        now,
                    )
                    + payload_parameters,
                )
                rows = cursor.fetchall()
            return [self._decode("processor_request", row[0]) for row in rows]

    def active_backlog_is_lane_blocked(
        self,
        *,
        now: datetime,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> bool:
        """One index probe: is the backlog blocked rather than drained?

        The predicate is the claim window's, minus the lane filter and
        plus its inverse - so this answers exactly the question the
        claim's empty result leaves open. ``LIMIT 1`` keeps it O(1) in
        queue depth; the consumer only asks when it is about to sleep.
        """

        included = sorted(include_paths or ())
        excluded = sorted(exclude_paths or ())
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM gpu_fault_processor_queue AS candidate
                    JOIN gpu_fault_processor_lanes AS lane
                      ON lane.ordering_key=candidate.ordering_key
                    WHERE (
                        candidate.status='PENDING'
                        OR (
                            candidate.status='LEASED'
                            AND candidate.lease_expires_at <= %s
                        )
                    )
                      AND (
                          candidate.not_before IS NULL
                          OR candidate.not_before <= %s
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM gpu_fault_processor_queue AS retry_barrier
                          WHERE retry_barrier.ordering_key
                                = candidate.ordering_key
                            AND retry_barrier.status='PENDING'
                            AND retry_barrier.lane_policy='STRICT'
                            AND retry_barrier.not_before > %s
                      )
                      AND (
                          %s
                          OR candidate.payload->>'path'=ANY(%s)
                      )
                      AND (
                          %s
                          OR NOT (
                              candidate.payload->>'path'=ANY(%s)
                          )
                      )
                      AND lane.lease_expires_at > %s
                    LIMIT 1
                )
                """,
                (
                    now,
                    now,
                    now,
                    not included,
                    included,
                    not excluded,
                    excluded,
                    now,
                ),
            )
            return bool(cursor.fetchone()[0])
