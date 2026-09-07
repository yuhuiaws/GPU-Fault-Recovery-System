from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, cast

from gpu_fault.installation_resources import InstallationResource
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    NodeMarker,
    RecoveryAction,
    TerminalEvent,
    WorkflowRequest,
)
from gpu_fault.store.shared.attempt_observation_support import (
    AttemptObservationTerminalSupport,
    reconcile_terminal_attempt_observations,
    terminalize_attempt_observation,
)
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


# Hot-state kind -> (dedicated table, dedicated timestamp column, legacy
# timestamp expression). Shared by the counting status (CLI) and the EXISTS
# gap check (startup) so the two cannot disagree on what a twin is.
_HOT_STATE_MAPPINGS: dict[str, tuple[str, str | None, str | None]] = {
    "gpu_metric_latest": (
        "gpu_fault_gpu_metric_latest",
        "observed_at",
        "legacy.payload->>'observed_at'",
    ),
    "gpu_metrics_batch": (
        "gpu_fault_gpu_metrics_batches",
        None,
        None,
    ),
    "attempt_observation": (
        "gpu_fault_attempt_observations",
        "observed_at",
        "legacy.payload->'observation'->>'observed_at'",
    ),
    "training_progress": (
        "gpu_fault_training_progress",
        "observed_at",
        "legacy.payload->'heartbeat'->>'observed_at'",
    ),
}


def _hot_state_mismatch(
    dedicated_timestamp: str | None, legacy_timestamp: str | None
) -> str:
    """SQL predicate: the dedicated twin is older than (or differs from) the
    legacy row. Kinds without a timestamp compare the whole payload."""

    if dedicated_timestamp is None:
        return "dedicated.payload IS DISTINCT FROM legacy.payload"
    return f"dedicated.{dedicated_timestamp} < ({legacy_timestamp})::timestamptz"


class PostgresControlRecordMixin(AttemptObservationTerminalSupport):
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _link: Callable[..., Any]
    _put: Callable[..., Any]
    _state_key: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    hot_state_mode: Any
    _get_optional: Callable[..., Any]
    _count_by_field: Callable[..., dict[str, int]]

    def list_markers(self) -> list[NodeMarker]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='marker'
                ORDER BY payload->>'observed_at', key
                """
            )
            rows = cursor.fetchall()
        return [self._decode("marker", row[0]) for row in rows]

    def save_installation_resource(
        self,
        resource: InstallationResource,
    ) -> InstallationResource:
        key = f"{resource.site_id}/{resource.resource_key}"
        with self._state_transaction(f"installation_resource/{key}"):
            existing = self._get_optional("installation_resource", key)
            if (
                existing is not None
                and existing.immutable_identity() != resource.immutable_identity()
            ):
                raise ValueError("installation resource identity cannot change")
            self._put("installation_resource", key, resource)
        return resource

    def get_installation_resource(
        self,
        site_id: str,
        resource_key: str,
    ) -> InstallationResource:
        resource = self._get_optional(
            "installation_resource",
            f"{site_id}/{resource_key}",
        )
        if resource is None:
            from gpu_fault.store.shared.errors import NotFoundError

            raise NotFoundError(resource_key)
        return cast(InstallationResource, resource)

    def list_installation_resources(
        self,
        site_id: str | None = None,
    ) -> list[InstallationResource]:
        clauses = ["kind='installation_resource'"]
        parameters: list[object] = []
        if site_id is not None:
            clauses.append("payload->>'site_id'=%s")
            parameters.append(site_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM gpu_fault_objects WHERE "
                + " AND ".join(clauses)
                + " ORDER BY payload->>'site_id', payload->>'resource_key'",
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("installation_resource", row[0]) for row in rows]

    def list_markers_for_incident(self, incident_id: str) -> list[NodeMarker]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='marker'
                  AND payload->>'incident_id'=%s
                ORDER BY payload->>'observed_at', key
                """,
                (incident_id,),
            )
            rows = cursor.fetchall()
        return [self._decode("marker", row[0]) for row in rows]

    def list_hyperpod_node_identities(self, cluster_name: str | None = None):
        clauses = ["kind='hyperpod_node_identity'"]
        parameters: list[object] = []
        if cluster_name is not None:
            clauses.append("payload->>'cluster_name'=%s")
            parameters.append(cluster_name)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY payload->>'cluster_name',
                         payload->>'node_logical_id',
                         key
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("hyperpod_node_identity", row[0]) for row in rows]

    def list_active_markers_for_nodes(
        self,
        node_ids: set[str],
        actions: set[RecoveryAction],
    ) -> list[NodeMarker]:
        if not node_ids or not actions:
            return []
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='marker'
                  AND payload->>'active'='true'
                  AND payload->>'trusted'='true'
                  AND payload->>'recommended_action'=ANY(%s)
                  AND payload->'scope'->'node_ids' ?| %s
                ORDER BY payload->>'observed_at' DESC, key DESC
                """,
                (
                    sorted(action.value for action in actions),
                    sorted(node_ids),
                ),
            )
            rows = cursor.fetchall()
        return [self._decode("marker", row[0]) for row in rows]

    def list_recent_markers_for_nodes(
        self,
        node_ids: set[str],
        observed_after: datetime,
        *,
        source_boot_id: str | None = None,
        limit: int = 1000,
    ) -> list[NodeMarker]:
        if not node_ids or limit < 1:
            return []
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='marker'
                  AND payload->>'active'='true'
                  AND payload->'scope'->'node_ids' ?| %s
                  AND (
                      (payload->>'observed_at')::timestamptz >= %s
                      OR payload->>'source_boot_id'=%s
                  )
                ORDER BY (payload->>'observed_at')::timestamptz DESC,
                         key DESC
                LIMIT %s
                """,
                (
                    sorted(node_ids),
                    observed_after,
                    source_boot_id,
                    limit,
                ),
            )
            rows = cursor.fetchall()
        return [self._decode("marker", row[0]) for row in rows]

    def list_markers_in_scope_window(
        self,
        *,
        node_ids: set[str],
        gpu_uuids: set[str],
        fabric_partitions: set[str],
        observed_from: datetime,
        observed_to: datetime,
        limit: int = 1000,
    ) -> list[NodeMarker]:
        """Actionable markers whose scope touches an allocation, newest first.

        The terminal-event correlator asks this once per completed training
        attempt. It matches on GPU UUID and fabric partition as well as node ID,
        because a marker raised by a fabric-level fault names the partition, not
        the nodes attached to it. An empty scope list is passed as an empty array
        rather than skipped, so ``?|`` simply never matches that branch and the
        SQL stays one statement.
        """
        if limit < 1 or not (node_ids or gpu_uuids or fabric_partitions):
            return []
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='marker'
                  AND payload->>'active'='true'
                  AND payload->>'trusted'='true'
                  AND payload->>'recommended_action' IS NOT NULL
                  AND (payload->>'observed_at')::timestamptz >= %s
                  AND (payload->>'observed_at')::timestamptz <= %s
                  AND (
                      payload->'scope'->'node_ids' ?| %s
                      OR payload->'scope'->'gpu_uuids' ?| %s
                      OR payload->'scope'->'fabric_partitions' ?| %s
                  )
                ORDER BY (payload->>'observed_at')::timestamptz DESC,
                         key DESC
                LIMIT %s
                """,
                (
                    observed_from,
                    observed_to,
                    sorted(node_ids),
                    sorted(gpu_uuids),
                    sorted(fabric_partitions),
                    limit,
                ),
            )
            rows = cursor.fetchall()
        return [self._decode("marker", row[0]) for row in rows]

    def has_workflow_successor(self, predecessor_workflow_id: str) -> bool:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM gpu_fault_objects
                    WHERE kind='workflow'
                      AND payload->>'predecessor_workflow_id'=%s
                )
                """,
                (predecessor_workflow_id,),
            )
            return bool(cursor.fetchone()[0])

    def get_preempting_successor(
        self, predecessor_workflow_id: str
    ) -> WorkflowRequest | None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='workflow'
                  AND payload->>'predecessor_workflow_id'=%s
                  AND payload->>'preempt_predecessor'='true'
                  AND payload->>'status' IN (
                      'PENDING', 'SAFETY_PENDING'
                  )
                ORDER BY payload->>'created_at', key
                LIMIT 1
                """,
                (predecessor_workflow_id,),
            )
            row = cursor.fetchone()
        return self._decode("workflow", row[0]) if row is not None else None

    def save_raw_evidence(self, record, *, max_records_per_node: int) -> None:
        storage_key = self._state_key(
            (
                record.cluster_id,
                record.node_id,
                record.record_id,
            )
        )
        with self._state_transaction(
            f"raw_evidence/{record.cluster_id}/{record.node_id}"
        ):
            self._put("raw_evidence", storage_key, record)
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_objects
                    WHERE kind='raw_evidence'
                      AND key IN (
                          SELECT key
                          FROM gpu_fault_objects
                          WHERE kind='raw_evidence'
                            AND payload->>'cluster_id'=%s
                            AND payload->>'node_id'=%s
                            AND payload->>'expires_at' > %s
                          ORDER BY
                            (payload->>'observed_at')::timestamptz
                            DESC,
                            key DESC
                          OFFSET %s
                      )
                    """,
                    (
                        record.cluster_id,
                        record.node_id,
                        _utc_text(datetime.now(timezone.utc)),
                        max_records_per_node,
                    ),
                )

    def list_raw_evidence(
        self,
        cluster_id: str,
        *,
        node_id: str | None = None,
        attempt_id: str | None = None,
        kind=None,
        limit: int = 100,
    ):
        clauses = [
            "kind='raw_evidence'",
            "payload->>'cluster_id'=%s",
            "payload->>'expires_at' > %s",
        ]
        parameters = [
            cluster_id,
            _utc_text(datetime.now(timezone.utc)),
        ]
        if node_id is not None:
            clauses.append("payload->>'node_id'=%s")
            parameters.append(node_id)
        if attempt_id is not None:
            clauses.append("payload->'attempt_ids' ? %s")
            parameters.append(attempt_id)
        if kind is not None:
            clauses.append("payload->>'kind'=%s")
            parameters.append(kind.value)
        parameters.append(limit)
        query = (
            "SELECT payload FROM gpu_fault_objects WHERE "
            + " AND ".join(clauses)
            + " ORDER BY "
            "(payload->>'observed_at')::timestamptz DESC, key DESC "
            "LIMIT %s"
        )
        with self._db.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
        return [self._decode("raw_evidence", row[0]) for row in rows]

    def cleanup_expired_raw_evidence(
        self,
        *,
        now: datetime | None = None,
        limit: int = 1000,
    ) -> int:
        observed = now or datetime.now(timezone.utc)
        with self._state_transaction("raw_evidence/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH expired AS (
                        SELECT key
                        FROM gpu_fault_objects
                        WHERE kind='raw_evidence'
                          AND payload->>'expires_at' <= %s
                        ORDER BY payload->>'expires_at', key
                        LIMIT %s
                    )
                    DELETE FROM gpu_fault_objects target
                    USING expired
                    WHERE target.kind='raw_evidence'
                      AND target.key=expired.key
                    """,
                    (_utc_text(observed), limit),
                )
                return int(cursor.rowcount)

    def backfill_hot_state_tables(self) -> dict[str, int]:
        counts = {}
        with self._state_transaction("hot_state/backfill"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_gpu_metric_latest(
                        key, cluster_id, node_id,
                        observed_at, payload
                    )
                    SELECT
                        key,
                        payload->>'cluster_id',
                        payload->>'node_id',
                        (payload->>'observed_at')::timestamptz,
                        payload
                    FROM gpu_fault_objects
                    WHERE kind='gpu_metric_latest'
                    ON CONFLICT(key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        node_id=excluded.node_id,
                        observed_at=excluded.observed_at,
                        payload=excluded.payload
                    WHERE excluded.observed_at >=
                          gpu_fault_gpu_metric_latest.observed_at
                    """
                )
                counts["gpu_metric_latest"] = cursor.rowcount
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_gpu_metrics_batches(
                        key, cluster_id, node_id,
                        created_at, payload
                    )
                    SELECT
                        key,
                        key::jsonb->>0,
                        key::jsonb->>1,
                        now(),
                        payload
                    FROM gpu_fault_objects
                    WHERE kind='gpu_metrics_batch'
                    ON CONFLICT(key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        node_id=excluded.node_id,
                        payload=excluded.payload
                    """
                )
                counts["gpu_metrics_batch"] = cursor.rowcount
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_attempt_observations(
                        key, cluster_id, attempt_id,
                        observed_at, payload
                    )
                    SELECT
                        key,
                        payload->'observation'->>'cluster_id',
                        payload->'observation'->>'attempt_id',
                        (
                            payload->'observation'->>'observed_at'
                        )::timestamptz,
                        payload
                    FROM gpu_fault_objects
                    WHERE kind='attempt_observation'
                    ON CONFLICT(key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        attempt_id=excluded.attempt_id,
                        observed_at=excluded.observed_at,
                        payload=excluded.payload
                    WHERE excluded.observed_at >=
                          gpu_fault_attempt_observations.observed_at
                    """
                )
                counts["attempt_observation"] = cursor.rowcount
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_training_progress(
                        key, cluster_id, attempt_id,
                        rank, observed_at, payload
                    )
                    SELECT
                        key,
                        payload->'heartbeat'->>'cluster_id',
                        payload->'heartbeat'->>'attempt_id',
                        (
                            payload->'heartbeat'->>'rank'
                        )::bigint,
                        (
                            payload->'heartbeat'->>'observed_at'
                        )::timestamptz,
                        payload
                    FROM gpu_fault_objects
                    WHERE kind='training_progress'
                    ON CONFLICT(key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        attempt_id=excluded.attempt_id,
                        rank=excluded.rank,
                        observed_at=excluded.observed_at,
                        payload=excluded.payload
                    WHERE excluded.observed_at >=
                          gpu_fault_training_progress.observed_at
                    """
                )
                counts["training_progress"] = cursor.rowcount
        return counts

    def hot_state_migration_status(
        self,
    ) -> dict[str, dict[str, int]]:
        status = {}
        with self._db.cursor() as cursor:
            for kind, (
                table,
                dedicated_timestamp,
                legacy_timestamp,
            ) in _HOT_STATE_MAPPINGS.items():
                mismatch_condition = _hot_state_mismatch(
                    dedicated_timestamp, legacy_timestamp
                )
                cursor.execute(
                    f"""
                    SELECT
                        count(*),
                        count(dedicated.key),
                        count(*) FILTER (
                            WHERE dedicated.key IS NULL
                               OR {mismatch_condition}
                        )
                    FROM gpu_fault_objects AS legacy
                    LEFT JOIN {table} AS dedicated
                      ON dedicated.key=legacy.key
                    WHERE legacy.kind=%s
                    """,
                    (kind,),
                )
                (
                    legacy_count,
                    dedicated_matches,
                    mismatched,
                ) = cursor.fetchone()
                cursor.execute(f"SELECT count(*) FROM {table}")
                dedicated_count = cursor.fetchone()[0]
                status[kind] = {
                    "legacy": legacy_count,
                    "dedicated": dedicated_count,
                    "matched_keys": dedicated_matches,
                    "missing_or_mismatched": mismatched,
                }
        return status

    def hot_state_backfill_gaps(self) -> dict[str, bool]:
        """Per hot-state kind: does any legacy row lack an up-to-date twin?

        The dedicated-mode startup check used to call
        ``hot_state_migration_status`` -- a full LEFT JOIN with three
        aggregates per kind plus ``count(*)`` of each dedicated table -- and
        uvicorn ``--limit-max-requests`` makes process start routine (store
        review 2026-09-07, item J). Startup only needs a yes/no per kind, so
        this stops at the first gap and never counts the dedicated tables;
        the CLI keeps the counting form.
        """

        gaps: dict[str, bool] = {}
        with self._db.cursor() as cursor:
            for kind, (
                table,
                dedicated_timestamp,
                legacy_timestamp,
            ) in _HOT_STATE_MAPPINGS.items():
                cursor.execute(
                    f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM gpu_fault_objects AS legacy
                        LEFT JOIN {table} AS dedicated
                          ON dedicated.key=legacy.key
                        WHERE legacy.kind=%s
                          AND (
                              dedicated.key IS NULL
                              OR {_hot_state_mismatch(dedicated_timestamp, legacy_timestamp)}
                          )
                    )
                    """,
                    (kind,),
                )
                gaps[kind] = bool(cursor.fetchone()[0])
        return gaps

    def purge_legacy_hot_state(self) -> dict[str, int]:
        status = self.hot_state_migration_status()
        unsafe = {
            kind: item["missing_or_mismatched"]
            for kind, item in status.items()
            if item["missing_or_mismatched"]
        }
        if unsafe:
            raise RuntimeError(
                "hot state backfill is incomplete: "
                + json.dumps(unsafe, sort_keys=True)
            )
        deleted = {}
        with self._state_transaction("hot_state/purge_legacy"):
            with self._db.cursor() as cursor:
                for kind in (
                    "gpu_metric_latest",
                    "gpu_metrics_batch",
                    "attempt_observation",
                    "training_progress",
                ):
                    cursor.execute(
                        """
                        DELETE FROM gpu_fault_objects
                        WHERE kind=%s
                        """,
                        (kind,),
                    )
                    deleted[kind] = cursor.rowcount
        return deleted

    def cleanup_hot_state(
        self,
        *,
        now: datetime | None = None,
        batch_retention: timedelta = timedelta(hours=24),
        terminal_retention: timedelta = timedelta(days=30),
        latest_retention: timedelta = timedelta(days=30),
        finding_history_retention: timedelta = timedelta(days=30),
        attempt_observation_max_age: timedelta | None = None,
        limit: int = 1000,
    ) -> dict[str, int]:
        observed = now or datetime.now(timezone.utc)
        # No global lock around the sweep: it takes each observation row's own
        # ``attempt_observation/<key>`` advisory lock, the one every other
        # writer of that row holds (F-G7 / P1-44L).
        terminalized = reconcile_terminal_attempt_observations(self, limit)
        deleted = {"attempt_observation_terminalized": terminalized}
        with self._state_transaction("hot_state/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH expired AS (
                        SELECT key
                        FROM gpu_fault_objects
                        WHERE kind='gpu_finding_history'
                          AND payload->>'observed_at' <= %s
                        ORDER BY payload->>'observed_at', key
                        LIMIT %s
                    )
                    DELETE FROM gpu_fault_objects AS target
                    USING expired
                    WHERE target.kind='gpu_finding_history'
                      AND target.key=expired.key
                    """,
                    (
                        _utc_text(observed - finding_history_retention),
                        limit,
                    ),
                )
                deleted["gpu_finding_history"] = cursor.rowcount
        if self.hot_state_mode != "dedicated":
            # The ``attempt_observation`` kind in gpu_fault_objects had no
            # retention at all, so in legacy and dual mode a terminal row
            # lived forever (F-G7 / P3-50J). Same two rules as the dedicated
            # table below: terminal rows by ``terminal_retention``, any row by
            # ``attempt_observation_max_age``.
            deleted["attempt_observation_legacy"] = self._cleanup_legacy_observations(
                terminal_cutoff=observed - terminal_retention,
                stale_cutoff=(
                    observed - attempt_observation_max_age
                    if attempt_observation_max_age is not None
                    else None
                ),
                limit=limit,
            )
        if self.hot_state_mode == "legacy":
            return deleted
        cutoffs = {
            "gpu_metrics_batch": (
                "gpu_fault_gpu_metrics_batches",
                "created_at",
                observed - batch_retention,
                None,
            ),
            "gpu_metric_latest": (
                "gpu_fault_gpu_metric_latest",
                "observed_at",
                observed - latest_retention,
                None,
            ),
            "training_progress": (
                "gpu_fault_training_progress",
                "observed_at",
                observed - terminal_retention,
                None,
            ),
            "attempt_observation": (
                "gpu_fault_attempt_observations",
                "observed_at",
                observed - terminal_retention,
                (
                    "payload->'observation'->>'workload_phase' "
                    "IN ('SUCCEEDED', 'FAILED', 'STOPPED')"
                ),
            ),
        }
        if attempt_observation_max_age is not None:
            # Every save refreshes observed_at, so a row that stopped
            # moving is one nobody reports on any more. Without this the
            # phase filter above keeps RUNNING/UNKNOWN observations
            # forever, which is the only unbounded hot-state kind left.
            cutoffs["attempt_observation_stale"] = (
                "gpu_fault_attempt_observations",
                "observed_at",
                observed - attempt_observation_max_age,
                None,
            )
        with self._state_transaction("hot_state/cleanup"):
            with self._db.cursor() as cursor:
                for name, (
                    table,
                    timestamp_column,
                    cutoff,
                    extra_condition,
                ) in cutoffs.items():
                    condition = f"{timestamp_column} <= %s"
                    if extra_condition is not None:
                        condition += f" AND {extra_condition}"
                    cursor.execute(
                        f"""
                        WITH expired AS (
                            SELECT key
                            FROM {table}
                            WHERE {condition}
                            ORDER BY {timestamp_column}, key
                            LIMIT %s
                        )
                        DELETE FROM {table} AS target
                        USING expired
                        WHERE target.key=expired.key
                        """,
                        (cutoff, limit),
                    )
                    deleted[name] = cursor.rowcount
        return deleted

    def _cleanup_legacy_observations(
        self,
        *,
        terminal_cutoff: datetime,
        stale_cutoff: datetime | None,
        limit: int,
    ) -> int:
        condition = """
            (
                payload->'observation'->>'workload_phase'
                    IN ('SUCCEEDED', 'FAILED', 'STOPPED')
                AND payload->'observation'->>'observed_at' <= %s
            )
        """
        parameters: list[object] = [_utc_text(terminal_cutoff)]
        if stale_cutoff is not None:
            condition += " OR payload->'observation'->>'observed_at' <= %s"
            parameters.append(_utc_text(stale_cutoff))
        parameters.append(limit)
        with self._state_transaction("hot_state/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH expired AS (
                        SELECT key
                        FROM gpu_fault_objects
                        WHERE kind='attempt_observation'
                          AND ({condition})
                        ORDER BY payload->'observation'->>'observed_at', key
                        LIMIT %s
                    )
                    DELETE FROM gpu_fault_objects AS target
                    USING expired
                    WHERE target.kind='attempt_observation'
                      AND target.key=expired.key
                    """,
                    parameters,
                )
                return int(cursor.rowcount)

    def completion_transaction(self, event_key: str) -> AbstractContextManager[None]:
        # One transaction, one advisory lock per event: the writes the
        # completion service nests inside (event row, incident + workflow,
        # plan, decision) become savepoints of it and commit or roll back as
        # one, and a second replica deciding the same event waits here.
        return cast(
            AbstractContextManager[None],
            self._state_transaction(f"completion/{event_key}"),
        )

    def list_decisions_by_status(
        self,
        status: DecisionStatus,
        *,
        older_than: datetime | None = None,
        limit: int = 100,
    ) -> list[CompletionDecision]:
        if limit < 1:
            return []
        clauses = ["decision.kind='decision'", "decision.payload->>'status'=%s"]
        parameters: list[object] = [status.value]
        if older_than is not None:
            clauses.append(
                "(diagnostic.key IS NULL OR diagnostic.payload->>'created_at' <= %s)"
            )
            parameters.append(_utc_text(older_than))
        parameters.append(limit)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT decision.payload
                FROM gpu_fault_objects AS decision
                LEFT JOIN gpu_fault_objects AS diagnostic
                  ON diagnostic.kind='diagnostic'
                 AND diagnostic.key=decision.payload->>'diagnostic_request_id'
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY diagnostic.payload->>'created_at' NULLS FIRST,
                         decision.key
                LIMIT %s
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [
            cast(CompletionDecision, self._decode("decision", row[0])) for row in rows
        ]

    def decision_status_counts(self) -> dict[DecisionStatus, int]:
        # Per-value counts on ``gpu_fault_decision_status_count`` rather than a
        # GROUP BY over every decision payload; see
        # ``PostgresWorkflowMixin.workflow_status_counts`` (store review
        # 2026-09-07, item G).
        counts = self._count_by_field(
            "decision", "status", [status.value for status in DecisionStatus]
        )
        return {DecisionStatus(value): count for value, count in counts.items()}

    def count_completion_events_without_decision(self) -> int:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM gpu_fault_objects AS event
                WHERE event.kind='event'
                  AND NOT EXISTS (
                      SELECT 1 FROM gpu_fault_objects AS decision
                      WHERE decision.kind='decision'
                        AND decision.key=event.key
                  )
                """
            )
            return int(cursor.fetchone()[0])

    def save_event_if_absent(self, event: TerminalEvent) -> bool:
        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        with self._state_transaction(f"attempt_observation/{storage_key}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    VALUES ('event', %s, %s::jsonb)
                    ON CONFLICT(kind, key) DO NOTHING
                    RETURNING key
                    """,
                    (
                        event.event_key,
                        event.model_dump_json(),
                    ),
                )
                inserted = cursor.fetchone() is not None
            if inserted:
                self._link(
                    "attempt_event",
                    self._state_key((event.cluster_id, event.attempt_id)),
                    event.event_key,
                )
            terminal = (
                event if inserted else self._get_optional("event", event.event_key)
            )
            if terminal is None:
                raise RuntimeError("terminal event disappeared during save")
            terminalize_attempt_observation(self, terminal)
            return inserted
