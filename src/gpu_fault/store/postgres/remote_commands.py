from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable

from gpu_fault.remote_command_models import (
    RemoteCommandStatus,
    lease_deadline,
)
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.store.shared.remote_helpers import (
    LEGACY_EXECUTOR_SAFETY_REJECTION_ERRORS,
    UNCLAIMED_DEADLINE_STATUS_SOURCE,
)
from gpu_fault.store.shared.remote_helpers import (
    unclaimed_expiry_update as _unclaimed_expiry_update,
)
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand


class PostgresRemoteCommandMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _get_for_update: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def get_remote_command(self, command_id: str) -> RemoteActionCommand:
        command = self._get_optional("remote_command", command_id)
        if command is None:
            raise NotFoundError(command_id)
        return command  # type: ignore[no-any-return]

    def list_remote_commands(
        self,
        *,
        workflow_request_ids: Iterable[str] | None = None,
    ) -> list[RemoteActionCommand]:
        query = """
            SELECT payload
            FROM gpu_fault_objects
            WHERE kind='remote_command'
        """
        parameters: list[Any] = []
        if workflow_request_ids is not None:
            # ``build_workflow_reconcile_plan`` reads this table for at most 1000
            # workflows and then filters every row in Python on exactly this
            # field, so the whole command history was decoded to answer a
            # question about a bounded set. Narrowing here is equivalent and
            # keeps the read proportional to the reconcile scope.
            query += " AND payload->>'workflow_request_id' = ANY(%s)"
            parameters.append(sorted(set(workflow_request_ids)))
        query += " ORDER BY payload->>'created_at', key"
        with self._db.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
        return [self._decode("remote_command", row[0]) for row in rows]

    def remote_command_stats(self, *, now: datetime | None = None) -> dict[str, Any]:
        observed_at = now or datetime.now(timezone.utc)
        by_status = {status.value: 0 for status in RemoteCommandStatus}
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    payload->>'cluster_id',
                    payload->>'status',
                    count(*),
                    count(*) FILTER (
                        WHERE payload->>'status_source'=
                              'executor-internal-error'
                          AND NOT (
                              COALESCE(payload->>'error', '')=ANY(%s)
                          )
                    ),
                    max(
                        (payload->>'updated_at')::timestamptz
                    ) FILTER (
                        WHERE payload->>'status_source'=
                              'executor-internal-error'
                          AND NOT (
                              COALESCE(payload->>'error', '')=ANY(%s)
                          )
                    ),
                    min(
                        (payload->>'created_at')::timestamptz
                    ) FILTER (
                        WHERE payload->>'status'='PENDING'
                    ),
                    -- The dead-letter counter has to be aggregated
                    -- here too: /metrics reads it unconditionally, so
                    -- an override that omits the key does not degrade
                    -- one gauge, it makes the whole endpoint 500 and
                    -- takes every other metric with it.
                    count(*) FILTER (
                        WHERE payload->>'status_source'=%s
                    )
                FROM gpu_fault_objects
                WHERE kind='remote_command'
                GROUP BY
                    payload->>'cluster_id',
                    payload->>'status'
                """,
                (
                    sorted(LEGACY_EXECUTOR_SAFETY_REJECTION_ERRORS),
                    sorted(LEGACY_EXECUTOR_SAFETY_REJECTION_ERRORS),
                    UNCLAIMED_DEADLINE_STATUS_SOURCE,
                ),
            )
            rows = cursor.fetchall()
        internal_errors = 0
        internal_error_last_seen = 0.0
        unclaimed_expired = 0
        total = 0
        open_by_cluster: dict[str, int] = {}
        by_cluster_status: dict[str, dict[str, int]] = {}
        oldest_pending_by_cluster = {}
        open_statuses = {
            RemoteCommandStatus.PENDING.value,
            RemoteCommandStatus.WAITING.value,
            RemoteCommandStatus.LEASED.value,
        }
        for (
            cluster_id,
            status_value,
            count,
            error_count,
            latest_internal_error,
            oldest_pending,
            expired_count,
        ) in rows:
            # Rows are grouped by (cluster, status): add them up, do not
            # let the last cluster's count stand for the fleet (F-D12).
            by_status[status_value] += count
            cluster_counts = by_cluster_status.setdefault(cluster_id, {})
            cluster_counts[status_value] = cluster_counts.get(status_value, 0) + count
            total += count
            internal_errors += error_count
            if latest_internal_error is not None:
                internal_error_last_seen = max(
                    internal_error_last_seen,
                    latest_internal_error.timestamp(),
                )
            unclaimed_expired += expired_count
            if status_value in open_statuses:
                open_by_cluster[cluster_id] = open_by_cluster.get(cluster_id, 0) + count
            if oldest_pending is not None:
                oldest_pending_by_cluster[cluster_id] = oldest_pending
        unclaimed_age_by_cluster = {}
        for (
            cluster_id,
            oldest_pending,
        ) in oldest_pending_by_cluster.items():
            unclaimed_age_by_cluster[cluster_id] = max(
                0.0,
                (observed_at - oldest_pending).total_seconds(),
            )
        return {
            "total": total,
            "by_status": by_status,
            "by_cluster_status": by_cluster_status,
            "open_by_cluster": open_by_cluster,
            "oldest_unclaimed_age_seconds_by_cluster": (unclaimed_age_by_cluster),
            "oldest_unclaimed_age_seconds": max(
                unclaimed_age_by_cluster.values(),
                default=0.0,
            ),
            "executor_internal_error_total": internal_errors,
            "executor_internal_error_last_seen_timestamp_seconds": (
                internal_error_last_seen
            ),
            "unclaimed_expired_total": unclaimed_expired,
        }

    def _open_remote_command_candidates(
        self,
        cluster_id: str,
        execution_owners: set[str] | None,
    ):
        """Push the claim's candidate filter into SQL.

        Matches gpu_fault_remote_command_claim so the hot per-poll read
        touches only this cluster's open backlog instead of decoding
        every command ever written. Unadvertised owners are excluded
        here as well: the caller skips them anyway, and fetching them
        only to discard them is what made a large terminal history
        dominate claim latency.
        """

        owners = sorted(execution_owners) if execution_owners is not None else None
        query = """
            SELECT payload FROM gpu_fault_objects
            WHERE kind='remote_command'
              AND payload->>'cluster_id'=%s
              AND payload->>'status' IN (
                  'PENDING', 'WAITING', 'LEASED'
              )
        """
        params: list = [cluster_id]
        if owners is not None:
            query += "  AND payload->'step'->>'execution_owner' = ANY(%s)\n"
            params.append(owners)
        query += "            ORDER BY payload->>'created_at', key\n"
        with self._db.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        return [self._decode("remote_command", row[0]) for row in rows]

    def claim_remote_commands(
        self,
        cluster_id: str,
        executor_id: str,
        *,
        limit: int,
        lease_seconds: int,
        execution_owners: set[str] | None = None,
    ):
        """Lease a batch of commands inside a single transaction.

        The inherited implementation decoded this cluster's whole open
        backlog and then opened one transaction per candidate, so every
        executor poll cost O(backlog) round trips even when it claimed
        nothing. Here eligibility (status, lease expiry, advertised owner
        and workflow fencing) is pushed into SQL so ``LIMIT`` stops the
        scan at the first claimable rows, and all leases share one
        transaction. The per-command advisory lock is still taken so
        completion keeps serialising against the claim; taking them in
        ``command_id`` order keeps two concurrent claimers deadlock free.
        The redundant ``status IN`` conjunct is what lets the planner
        match ``gpu_fault_remote_command_claim``.
        """
        if limit <= 0:
            return []
        now = datetime.now(timezone.utc)
        owners = sorted(execution_owners) if execution_owners is not None else None
        query = """
            SELECT cmd.key
            FROM gpu_fault_objects AS cmd
            WHERE cmd.kind='remote_command'
              AND cmd.payload->>'cluster_id'=%s
              AND cmd.payload->>'status' IN (
                  'PENDING', 'WAITING', 'LEASED'
              )
              AND (
                  cmd.payload->>'status' IN ('PENDING', 'WAITING')
                  OR (cmd.payload->>'lease_expires_at')::timestamptz
                     <= %s
              )
              AND cmd.payload->>'cancellation_requested_at' IS NULL
        """
        params: list = [cluster_id, now]
        if owners is not None:
            query += "  AND cmd.payload->'step'->>'execution_owner' = ANY(%s)\n"
            params.append(owners)
        query += """
              AND NOT EXISTS (
                  SELECT 1 FROM gpu_fault_objects AS flow
                  WHERE flow.kind='workflow'
                    AND flow.key
                        = cmd.payload->>'workflow_request_id'
                    AND flow.payload->>'fencing_token'
                        IS DISTINCT FROM
                        cmd.payload->>'fencing_token'
              )
            ORDER BY cmd.payload->>'created_at', cmd.key
            LIMIT %s
        """
        params.append(limit)
        claimed = []
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(query, tuple(params))
                keys = [row[0] for row in cursor.fetchall()]
                if not keys:
                    return []
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(
                            'remote_command/' || command_id, 0
                        )
                    )
                    FROM (
                        SELECT unnest(%s::text[]) AS command_id
                        ORDER BY command_id
                    ) AS ordered
                    """,
                    (keys,),
                )
                cursor.execute(
                    """
                    SELECT payload FROM gpu_fault_objects
                    WHERE kind='remote_command' AND key=ANY(%s)
                    ORDER BY payload->>'created_at', key
                    """,
                    (keys,),
                )
                rows = cursor.fetchall()
            for row in rows:
                command = self._decode("remote_command", row[0])
                workflow = self._get_optional("workflow", command.workflow_request_id)
                if (
                    workflow is not None
                    and workflow.fencing_token != command.fencing_token
                ):
                    continue
                expired = (
                    command.status is RemoteCommandStatus.LEASED
                    and command.lease_expires_at is not None
                    and command.lease_expires_at <= now
                )
                if (
                    command.cluster_id != cluster_id
                    # A command told to stop is never handed to another
                    # executor, even after its lease lapses (F-D12); the
                    # memory and sqlite claims already filtered this.
                    or command.cancellation_requested_at is not None
                    or (
                        execution_owners is not None
                        and command.step.execution_owner not in execution_owners
                    )
                    or (
                        command.status
                        not in {
                            RemoteCommandStatus.PENDING,
                            RemoteCommandStatus.WAITING,
                        }
                        and not expired
                    )
                ):
                    continue
                command = command.model_copy(
                    update={
                        "status": RemoteCommandStatus.LEASED,
                        "lease_owner": executor_id,
                        "lease_token": secrets.token_urlsafe(32),
                        "lease_expires_at": lease_deadline(lease_seconds),
                        "updated_at": now,
                    }
                )
                self._put(
                    "remote_command",
                    command.command_id,
                    command,
                )
                claimed.append(command)
                if len(claimed) >= limit:
                    break
        return claimed

    def cancel_remote_commands_for_workflow(
        self, workflow_request_id: str, *, reason: str
    ) -> dict[str, int]:
        """Cancel one workflow's open commands without a full scan.

        The inherited implementation listed and decoded every remote
        command ever written to find the handful belonging to one
        workflow, which is the dominant cost of a workflow timeout once
        the terminal history is large. ``gpu_fault_remote_command_workflow``
        indexes exactly the rows this needs; the per-command transaction
        and state machine below are unchanged.
        """
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key FROM gpu_fault_objects
                WHERE kind='remote_command'
                  AND payload->>'workflow_request_id'=%s
                  AND payload->>'status' NOT IN (
                      'SUCCEEDED', 'FAILED'
                  )
                ORDER BY key
                """,
                (workflow_request_id,),
            )
            command_ids = [row[0] for row in cursor.fetchall()]
        result = {
            "cancelled": 0,
            "cancellation_requested": 0,
        }
        for command_id in command_ids:
            with self._state_transaction(f"remote_command/{command_id}/timeout"):
                command = self._get_optional("remote_command", command_id)
                if (
                    command is None
                    or command.workflow_request_id != workflow_request_id
                ):
                    continue
                now = datetime.now(timezone.utc)
                if command.status in {
                    RemoteCommandStatus.PENDING,
                    RemoteCommandStatus.WAITING,
                }:
                    command = command.model_copy(
                        update={
                            "status": RemoteCommandStatus.FAILED,
                            "error": reason,
                            "status_source": "workflow-timeout",
                            "lease_owner": None,
                            "lease_token": None,
                            "lease_expires_at": None,
                            "updated_at": now,
                        }
                    )
                    result["cancelled"] += 1
                elif (
                    command.status is RemoteCommandStatus.LEASED
                    and command.cancellation_requested_at is None
                ):
                    command = command.model_copy(
                        update={
                            "cancellation_requested_at": now,
                            "cancellation_reason": reason,
                            "updated_at": now,
                        }
                    )
                    result["cancellation_requested"] += 1
                else:
                    continue
                self._put("remote_command", command_id, command)
        return result

    def cleanup_terminal_remote_commands(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Set-based retention delete matching the terminal index.

        The parent implementation decodes every command to sort them in
        Python, which defeats the purpose of the cleanup on a large
        backlog. Here the ordering and the limit stay in SQL.
        """

        with self._state_transaction("remote_command/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH victims AS (
                        SELECT key
                        FROM gpu_fault_objects
                        WHERE kind='remote_command'
                          AND payload->>'status' IN (
                              'SUCCEEDED', 'FAILED'
                          )
                          AND payload->>'updated_at' <= %s
                        ORDER BY payload->>'updated_at', key
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    ),
                    deleted AS (
                        DELETE FROM gpu_fault_objects AS objects
                        USING victims
                        WHERE objects.kind='remote_command'
                          AND objects.key=victims.key
                        RETURNING objects.key
                    )
                    SELECT count(*) FROM deleted
                    """,
                    (_utc_text(older_than), limit),
                )
                row = cursor.fetchone()
            return int(row[0]) if row else 0

    def expire_unclaimed_remote_commands(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Same dead-letter transition, candidates selected in SQL.

        The parent decodes every command ever written to find the PENDING
        ones. This runs on the periodic services thread next to the
        retention delete, so it has to stay proportional to the stuck
        backlog instead of the history.

        The candidate scan holds no lock, so each candidate is re-read
        under the per-command advisory lock the claim path takes (in
        ``command_id`` order, like the claim) and its row lock, and is
        only expired if it is still PENDING (F-D12). Before that, a lease
        issued between the scan and the write was overwritten by FAILED.
        """

        now = datetime.now(timezone.utc)
        expired = 0
        with self._state_transaction("remote_command/unclaimed-expiry"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT key FROM gpu_fault_objects
                    WHERE kind='remote_command'
                      AND payload->>'status'='PENDING'
                      AND payload->>'created_at' <= %s
                    ORDER BY payload->>'created_at', key
                    LIMIT %s
                    """,
                    (_utc_text(older_than), limit),
                )
                keys = sorted(row[0] for row in cursor.fetchall())
                if not keys:
                    return 0
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(
                            'remote_command/' || command_id, 0
                        )
                    )
                    FROM (
                        SELECT unnest(%s::text[]) AS command_id
                        ORDER BY command_id
                    ) AS ordered
                    """,
                    (keys,),
                )
            for key in keys:
                try:
                    command = self._get_for_update("remote_command", key)
                except NotFoundError:
                    continue
                if (
                    command.status is not RemoteCommandStatus.PENDING
                    or command.created_at > older_than
                ):
                    continue
                self._put(
                    "remote_command",
                    command.command_id,
                    _unclaimed_expiry_update(command, now),
                )
                expired += 1
        return expired
