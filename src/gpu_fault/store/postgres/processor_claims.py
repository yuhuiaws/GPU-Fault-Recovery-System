from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable

from gpu_fault.channel_registry import (
    CHANNEL_REGISTRY,
    FAULT_CHANNEL_PATHS,
)
from gpu_fault.processor.models import (
    ROUTINE_PRIORITY,
    STARVED_ROUTINE_PRIORITY,
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


_CLAIM_WINDOW_COLUMNS_SQL = f"""\
        candidate.request_id,
        candidate.ordering_key,
        CASE
            WHEN candidate.priority={ROUTINE_PRIORITY}
              AND candidate.created_at
                  <= %(routine_starvation_before)s
            THEN {STARVED_ROUTINE_PRIORITY}
            ELSE candidate.priority
        END AS priority,
        candidate.created_at,
        candidate.payload->>'path' AS path,
        coalesce(
            candidate.payload
                ->'correlation_scope_keys',
            '[]'::jsonb
        ) AS scope_keys
"""

# Shared by both sub-windows. Every predicate here is either implied by the
# partial index predicate of gpu_fault_processor_queue_priority_claim
# (status IN ('PENDING','LEASED')) or a cheap residual filter on the rows
# that index yields in claim order.
_CLAIM_ELIGIBLE_SQL = """\
    FROM gpu_fault_processor_queue AS candidate
    LEFT JOIN gpu_fault_processor_lanes AS lane
      ON lane.ordering_key=candidate.ordering_key
    WHERE (
        candidate.status='PENDING'
        OR (
            candidate.status='LEASED'
            AND candidate.lease_expires_at <= %(now)s
        )
    )
      AND (
          candidate.not_before IS NULL
          OR candidate.not_before <= %(now)s
      )
      AND NOT EXISTS (
          SELECT 1
          FROM gpu_fault_processor_queue AS retry_barrier
          WHERE retry_barrier.ordering_key=candidate.ordering_key
            AND retry_barrier.status='PENDING'
            AND retry_barrier.lane_policy='STRICT'
            AND retry_barrier.not_before > %(now)s
      )
      AND (
          lane.ordering_key IS NULL
          OR lane.lease_expires_at <= %(now)s
      )
{path_filter}"""

# The path filter of a sub-window. A dedicated pool's claim names its path
# (7 of the 8 sub-claims per cycle do), and that path is rendered as one
# equality literal per sub-window so the planner walks
# gpu_fault_processor_queue_path_priority_claim ((payload->>'path'),
# priority, created_at, request_id) in claim order and stops at the LIMIT.
# ``= ANY(array)`` on the leading column gave a Bitmap scan of the whole path
# backlog and a top-N sort instead, so the claim's cost grew with the path's
# depth - the shape F-D2 removed for the unfiltered window (B-3 / G-3). The
# fault stream excludes the dedicated paths; that stays a residual filter on
# the priority_claim walk, which is still an ordered index walk.
_PATH_EQUALS_SQL = """\
      AND candidate.payload->>'path'={path_literal}
"""
_PATH_EXCLUDES_SQL = """\
      AND NOT (
          candidate.payload->>'path'=ANY(%(excluded)s)
      )
"""

# Every path the claim can be asked for is a registered route; anything else
# is rejected before it reaches the statement text.
_PATH_LITERAL_PATTERN = re.compile(r"^/[A-Za-z0-9/_.-]*$")


def _path_literal(path: str) -> str:
    if not _PATH_LITERAL_PATTERN.match(path):
        raise ValueError(f"processor claim path is not a route: {path!r}")
    return "'" + path + "'"


def _claim_sub_windows(path_filter: str) -> str:
    """The two index walks of F-D2 for one path filter: the best rows by raw
    priority, plus the oldest routine rows past the starvation threshold."""

    eligible = _CLAIM_ELIGIBLE_SQL.format(path_filter=path_filter)
    return f"""\
    (
        SELECT
{_CLAIM_WINDOW_COLUMNS_SQL}
{eligible}
        ORDER BY
            candidate.priority,
            candidate.created_at,
            candidate.request_id
        LIMIT %(window_limit)s
    )
    UNION ALL
    (
        SELECT
{_CLAIM_WINDOW_COLUMNS_SQL}
{eligible}
          AND candidate.priority={ROUTINE_PRIORITY}
          AND candidate.created_at <= %(routine_starvation_before)s
        ORDER BY
            candidate.created_at,
            candidate.request_id
        LIMIT %(window_limit)s
    )"""


@lru_cache(maxsize=64)
def _claim_window_sql(included: tuple[str, ...], excluded: bool) -> str:
    if included:
        return "\n    UNION ALL\n".join(
            _claim_sub_windows(
                _PATH_EQUALS_SQL.format(path_literal=_path_literal(path))
            )
            for path in included
        )
    return _claim_sub_windows(_PATH_EXCLUDES_SQL if excluded else "")


# The window is the union of two index walks per path filter (F-D2): the
# best rows by raw priority, plus the oldest routine rows past the starvation
# threshold. The second walk is what lets a starved routine row enter a
# window that fresher evidence rows would otherwise fill on every claim;
# before it, the 49 promotion ran only over rows the first walk had already
# admitted. Every walk stops at the window limit, so the claim's cost stays
# bounded by the window (times the number of included paths) rather than by
# queue depth. Duplicates collapse in candidate_ids. The window text is
# rendered per (include_paths, exclude_paths) by ``_claim_window_sql``.
_CLAIM_ACTIVE_PROCESSOR_SQL = f"""\
WITH claim_window AS MATERIALIZED (
{{claim_window}}
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
                WHERE (
                          observation.status='PENDING'
                          OR (
                              observation.status='LEASED'
                              AND observation.lease_expires_at > %(now)s
                          )
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
    LIMIT %(limit)s
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
        %(owner_id)s,
        1,
        selected.lease_token,
        %(expires_at)s,
        %(now)s
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
        <= %(now)s
    RETURNING ordering_key, epoch, lease_token
),
updated AS (
    UPDATE gpu_fault_processor_queue AS queue
    SET
        status='LEASED',
        lease_owner=%(owner_id)s,
        leader_epoch=leased_lanes.epoch,
        lease_token=leased_lanes.lease_token,
        lease_expires_at=%(expires_at)s,
        updated_at=%(now)s
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
    _persist_processor_state: Callable[..., Any]
    _processor_queue_effective_payload: Callable[..., Any]
    claim_window_multiplier: Any
    get_processor_leadership: Callable[..., Any]
    processor_queue_state_mode: Any

    def count_fault_rows_blocked_by_observation(self, *, now: datetime) -> int:
        """Claimable fault rows an incomplete observation holds back (F-D3)."""

        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT count(*)
                FROM gpu_fault_processor_queue AS fault
                WHERE fault.status='PENDING'
                  AND (fault.not_before IS NULL OR fault.not_before <= %s)
                  AND fault.payload->>'path' IN (
{_fault_path_list_sql(20)}
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(
                          coalesce(
                              fault.payload->'correlation_scope_keys',
                              '[]'::jsonb
                          )
                      ) AS fault_scope(value)
                      WHERE EXISTS (
                          SELECT 1
                          FROM gpu_fault_processor_queue AS observation
                          WHERE observation.payload->>'path'
                                = '/v1/workload-observations'
                            AND (
                                observation.status='PENDING'
                                OR (
                                    observation.status='LEASED'
                                    AND observation.lease_expires_at > %s
                                )
                            )
                            AND coalesce(
                                observation.payload->'correlation_scope_keys',
                                '[]'::jsonb
                            ) ? fault_scope.value
                      )
                  )
                """,
                (now, now),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else 0

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
                                      WHERE (
                                                observation.status='PENDING'
                                                OR (
                                                    observation.status='LEASED'
                                                    AND observation.lease_expires_at
                                                        > %s
                                                )
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
                    (now, now, now, now, now, limit),
                )
                rows = cursor.fetchall()
            claimed = []
            for row in rows:
                item = self._decode("processor_request", row[0])
                # The state write below is the guarded upsert (``excluded.
                # updated_at >= updated_at``); a row admitted after the
                # caller's clock was read would otherwise keep its PENDING
                # row while the leader believes it holds the lease (F-D9).
                value = item.model_copy(
                    update={
                        "status": ProcessorRequestStatus.LEASED,
                        "lease_owner": owner_id,
                        "leader_epoch": leader_epoch,
                        "lease_token": secrets.token_urlsafe(32),
                        "lease_expires_at": now + lease_duration,
                        "updated_at": max(now, item.updated_at),
                    }
                )
                self._persist_processor_state(value)
                claimed.append(value)
            return claimed

    def claim_active_processor_query(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
        routine_starvation_seconds: float = 30.0,
    ) -> tuple[str, dict[str, object]]:
        """The exact statement ``claim_active_processor_requests`` runs.

        Exposed so a test can EXPLAIN it against the real schema (F-J3):
        the window must walk ``gpu_fault_processor_queue_priority_claim``
        rather than sort the eligible backlog.
        """

        included = sorted(include_paths or ())
        excluded = sorted(exclude_paths or ())
        expires_at = now + lease_duration
        # Bound the claim's cost. Lane dedup and the two
        # correlation-scope interlocks are the expensive parts, and they
        # run over this window instead of the whole eligible backlog,
        # so claim latency stops scaling with queue depth - that scaling
        # is what let depth feed back into itself at a 65536 cap. A
        # window short on claimable rows just returns fewer than
        # ``limit``; the caller claims again.
        window_limit = max(limit * self.claim_window_multiplier, limit + 64)
        payload_update = ""
        if self.processor_queue_state_mode != "dedicated":
            payload_update = """
                            , payload=queue.payload ||
                                jsonb_build_object(
                                    'status', 'LEASED'::text,
                                    'lease_owner', %(owner_id)s::text,
                                    'leader_epoch',
                                        leased_lanes.epoch,
                                    'lease_token',
                                        leased_lanes.lease_token,
                                    'lease_expires_at', %(expires_text)s::text,
                                    'updated_at', %(now_text)s::text
                                )
            """
        sql = _CLAIM_ACTIVE_PROCESSOR_SQL.format(
            claim_window=_claim_window_sql(tuple(included), bool(excluded)),
            payload_update=payload_update,
            effective_payload=self._processor_queue_effective_payload("updated"),
        )
        parameters: dict[str, object] = {
            "now": now,
            "routine_starvation_before": (
                now - timedelta(seconds=routine_starvation_seconds)
            ),
            "excluded": excluded,
            "window_limit": window_limit,
            "limit": limit,
            "owner_id": owner_id,
            "expires_at": expires_at,
            "expires_text": _utc_text(expires_at),
            "now_text": _utc_text(now),
        }
        return sql, parameters

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
        sql, parameters = self.claim_active_processor_query(
            owner_id,
            now=now,
            lease_duration=lease_duration,
            limit=limit,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            routine_starvation_seconds=routine_starvation_seconds,
        )
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(sql, parameters)
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
