from __future__ import annotations

from gpu_fault.store.contracts import ACTIVE_WORKFLOW_INCIDENTS_LIMIT

from typing import Any, Callable, Collection

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.time import utc_text as _utc_text
from gpu_fault.store.shared.workflow_scan import dispatch_order_key
from gpu_fault.store.shared.errors import (
    NotFoundError,
    StaleWriteError,
    WorkflowMergedError,
    RemediationBudgetError,
    WorkflowLeaseError,
    StaleFencingTokenError,
)
from gpu_fault.store.shared.transactional_workflows import (
    lease_extension_due,
    stale_workflow_versions,
)
from gpu_fault.store.shared.remediation_budgets import (
    apply_remediation_budget,
    extend_remediation_budget,
    blocked_by_remediation_budget,
)


class PostgresWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _get_for_update: Callable[..., Any]
    _link: Callable[..., Any]
    _put: Callable[..., Any]

    def list_workflows(
        self,
        statuses: set[WorkflowStatus] | None = None,
        *,
        limit: int = 100,
        newest_first: bool = False,
        dispatchable_at: datetime | None = None,
        exclude_request_ids: Collection[str] = (),
        after: WorkflowRequest | None = None,
    ) -> list[WorkflowRequest]:
        if statuses is not None and not statuses:
            return []
        sql, parameters = self.workflow_scan_query(
            statuses,
            limit=limit,
            newest_first=newest_first,
            dispatchable_at=dispatchable_at,
            exclude_request_ids=exclude_request_ids,
            after=after,
        )
        with self._db.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()
        return [self._decode("workflow", row[0]) for row in rows]

    # A predecessor that is still open holds its successor: the three
    # executable statuses, plus a BLOCKED row waiting for an operator or
    # parked by an internal error (F-A4). A settled safety plan and a legacy
    # BLOCKED row without a kind release it. Mirrors ``workflow_is_open``.
    _OPEN_PREDECESSOR_SQL = (
        "(predecessor.payload->>'status' IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')"
        " OR (predecessor.payload->>'status' = 'BLOCKED'"
        " AND predecessor.payload->>'blocked_kind'"
        " IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')))"
    )

    @classmethod
    def _pushdown_clauses(
        cls,
        dispatchable_at: datetime | None,
        exclude_request_ids: Collection[str],
    ) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if dispatchable_at is not None:
            # Text comparison on purpose: payload timestamps are stored in the
            # one UTC form ``utc_text`` renders, and a ``::timestamptz`` cast
            # would defeat any expression index (see the review's note on
            # payload timestamp comparisons).
            clauses.append(
                "(w.payload->>'not_before' IS NULL OR w.payload->>'not_before' <= %s)"
            )
            parameters.append(_utc_text(dispatchable_at))
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM gpu_fault_objects AS predecessor"
                " WHERE predecessor.kind='workflow'"
                " AND predecessor.key=w.payload->>'predecessor_workflow_id'"
                f" AND {cls._OPEN_PREDECESSOR_SQL})"
            )
        if exclude_request_ids:
            clauses.append("NOT (w.key = ANY(%s))")
            parameters.append(sorted(set(exclude_request_ids)))
        return clauses, parameters

    # The dispatch-mode sort key (F-A2a): when the row became eligible. Text
    # on purpose, like every payload timestamp comparison here -- the stored
    # value is ``isoformat()`` and a ``::timestamptz`` cast would defeat the
    # expression index ``gpu_fault_executable_workflow_dispatch_order`` that
    # carries exactly this expression. GREATEST skips a NULL ``not_before``.
    _DISPATCH_ORDER_SQL = "GREATEST(w.payload->>'created_at', w.payload->>'not_before')"

    @classmethod
    def workflow_scan_query(
        cls,
        statuses: set[WorkflowStatus] | None,
        *,
        limit: int,
        newest_first: bool = False,
        dispatchable_at: datetime | None = None,
        exclude_request_ids: Collection[str] = (),
        after: WorkflowRequest | None = None,
    ) -> tuple[str, tuple[object, ...]]:
        """The SQL behind ``list_workflows``, exposed so tests can EXPLAIN it.

        The status list is spelled as literals, not ``= ANY(%s)``: the planner
        cannot prove a partial index's predicate from a bound array, so the
        dispatcher's candidate scan fell back to a sequential scan on every
        tick (P0-73C). Values come from the ``WorkflowStatus`` enum, never from
        callers.

        With ``dispatchable_at`` the order is the eligibility key rather than
        ``updated_at`` (F-A2a) and ``after`` becomes a row-value cursor on
        that key (F-A2c), so a page is one index range scan.
        """

        if after is not None and dispatchable_at is None:
            raise ValueError(
                "the scan cursor (after) is only defined with dispatchable_at"
            )
        clauses = ["w.kind='workflow'"]
        if statuses is not None:
            literals = ", ".join(
                f"'{status.value}'"
                for status in sorted(statuses, key=lambda item: item.value)
            )
            clauses.append(f"w.payload->>'status' IN ({literals})")
        pushdown, parameters = cls._pushdown_clauses(
            dispatchable_at, exclude_request_ids
        )
        clauses.extend(pushdown)
        direction = "DESC" if newest_first else "ASC"
        if dispatchable_at is not None:
            order_key = cls._DISPATCH_ORDER_SQL
            if after is not None:
                eligible_at, request_id = dispatch_order_key(after)
                comparison = "<" if newest_first else ">"
                clauses.append(f"({order_key}, w.key) {comparison} (%s, %s)")
                parameters.extend((eligible_at, request_id))
        else:
            order_key = "w.payload->>'updated_at'"
        sql = (
            "SELECT w.payload FROM gpu_fault_objects AS w WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY {order_key} {direction}, "
            + f"w.key {direction} LIMIT %s"
        )
        return sql, (*parameters, limit)

    def count_held_workflows(
        self,
        statuses: set[WorkflowStatus] | None,
        *,
        dispatchable_at: datetime,
        exclude_request_ids: Collection[str] = (),
    ) -> dict[str, int]:
        if statuses is not None and not statuses:
            return {}
        clauses = ["w.kind='workflow'"]
        if statuses is not None:
            literals = ", ".join(
                f"'{status.value}'"
                for status in sorted(statuses, key=lambda item: item.value)
            )
            clauses.append(f"w.payload->>'status' IN ({literals})")
        excluded = sorted(set(exclude_request_ids))
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT"
                " count(*) FILTER (WHERE w.payload->>'not_before' > %s),"
                " count(*) FILTER (WHERE EXISTS ("
                "SELECT 1 FROM gpu_fault_objects AS predecessor"
                " WHERE predecessor.kind='workflow'"
                " AND predecessor.key=w.payload->>'predecessor_workflow_id'"
                f" AND {self._OPEN_PREDECESSOR_SQL})),"
                " count(*) FILTER (WHERE w.key = ANY(%s))"
                " FROM gpu_fault_objects AS w WHERE " + " AND ".join(clauses),
                (_utc_text(dispatchable_at), excluded),
            )
            not_before, predecessor, retired = cursor.fetchone()
        counts = {
            "not_before": int(not_before),
            "predecessor": int(predecessor),
            "retired": int(retired),
        }
        return {reason: count for reason, count in counts.items() if count}

    def workflow_status_counts(self) -> dict[WorkflowStatus, int]:
        """Count every persisted workflow by status without decoding any.

        The /metrics workflow gauge must stay exact while the detail scan that
        feeds the duration and step families is bounded, so the count is a
        server-side aggregate rather than a by-product of that scan.

        One ``count(*)`` per enum value rather than ``GROUP BY``: Postgres never
        plans an Index Only Scan over an expression index, so the grouped form
        read and detoasted every workflow payload on every scrape, and the
        cost grew with lifetime because control-record retention is off. The
        per-value form is an index range scan on
        ``gpu_fault_workflow_status_count`` with no payload evaluated (store
        review 2026-09-07, item G).
        """

        counts = self._count_by_field(
            "workflow", "status", [status.value for status in WorkflowStatus]
        )
        return {WorkflowStatus(value): count for value, count in counts.items()}

    def incident_state_counts(self) -> dict[IncidentState, int]:
        """Count every persisted incident by state (server-side aggregate
        for the /metrics incident gauge; ESCALATED is the operator queue).
        Per-value counts on ``gpu_fault_incident_state_count`` (item G)."""

        counts = self._count_by_field(
            "incident", "state", [state.value for state in IncidentState]
        )
        return {IncidentState(value): count for value, count in counts.items()}

    def _count_by_field(
        self, kind: str, field: str, values: list[str]
    ) -> dict[str, int]:
        """``{value: count}`` for one kind, one index range scan per value.

        ``kind`` and ``field`` are literals from the callers, never input; the
        values are the enum members. Missing values count as zero; a stored
        value outside the enum is not counted (the gauge is per enum member).
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT s.value, (
                    SELECT count(*) FROM gpu_fault_objects
                    WHERE kind='{kind}' AND payload->>'{field}' = s.value
                )
                FROM unnest(%s::text[]) AS s(value)
                """,
                (values,),
            )
            rows = cursor.fetchall()
        return {value: int(count) for value, count in rows}

    def blocked_workflows_without_verified_restore(self) -> int:
        """Count the BLOCKED workflows whose GPU node is still held.

        See the SQLite implementation for why the lifetime BLOCKED count cannot
        answer this and why the predicate is
        ``workflow_resolution.verified_restore_successor`` negated.
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM gpu_fault_objects w
                WHERE w.kind='workflow'
                  AND w.payload->>'status'='BLOCKED'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM gpu_fault_objects i
                      JOIN gpu_fault_objects s
                        ON s.kind='workflow'
                       AND s.key=i.payload->>'workflow_request_id'
                      WHERE i.kind='incident'
                        AND i.key=w.payload->>'incident_id'
                        AND i.payload->>'state'='RECOVERED'
                        AND s.key<>w.key
                        AND s.payload->>'incident_id'
                            =w.payload->>'incident_id'
                        AND s.payload->>'status'='SUCCEEDED'
                        AND s.payload->>'fencing_token'
                            =w.payload->>'fencing_token'
                        AND i.payload->>'fencing_token'
                            =w.payload->>'fencing_token'
                        AND jsonb_exists(
                            s.payload->'completed_operations',
                            'RESTORE_SCHEDULING'
                        )
                  )
                """
            )
            row = cursor.fetchone()
        return int(row[0])

    def list_orphan_workflows(
        self, *, created_before: datetime, limit: int = 1000
    ) -> list[WorkflowRequest]:
        """See ``ControlPlaneStore.list_orphan_workflows``.

        The same predicate as Q-ORPHAN in
        ``scripts/e2e/regional/audit_stuck_workflow_baseline.py``, widened to
        a workflow whose incident row is gone. ``created_at`` is compared as
        text (no ``::timestamptz``): the stored value is ``isoformat()``
        output, whose text order is time order, and a cast would bypass the
        expression indexes.
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT w.payload
                FROM gpu_fault_objects w
                LEFT JOIN gpu_fault_objects i
                  ON i.kind='incident'
                 AND i.key=w.payload->>'incident_id'
                WHERE w.kind='workflow'
                  AND w.payload->>'status' IN ('PENDING', 'SAFETY_PENDING')
                  AND w.payload->>'created_at' < %s
                  AND (
                      i.key IS NULL
                      OR COALESCE(i.payload->>'workflow_request_id', '') <> w.key
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM gpu_fault_objects s
                      WHERE s.kind='workflow'
                        AND s.payload->>'predecessor_workflow_id'=w.key
                  )
                ORDER BY w.payload->>'created_at', w.key
                LIMIT %s
                """,
                (_utc_text(created_before), limit),
            )
            rows = cursor.fetchall()
        return [self._decode("workflow", row[0]) for row in rows]

    def list_incidents_with_missing_workflow(
        self, *, limit: int = 1000
    ) -> list[FaultIncident]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT i.payload
                FROM gpu_fault_objects i
                WHERE i.kind='incident'
                  AND COALESCE(i.payload->>'workflow_request_id', '') <> ''
                  AND NOT EXISTS (
                      SELECT 1 FROM gpu_fault_objects w
                      WHERE w.kind='workflow'
                        AND w.key=i.payload->>'workflow_request_id'
                  )
                ORDER BY i.payload->>'created_at', i.key
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [self._decode("incident", row[0]) for row in rows]

    def list_unhandled_failed_workflows(
        self, *, limit: int = 1000
    ) -> list[WorkflowRequest]:
        sql, parameters = self.unhandled_failed_workflows_query(limit=limit)
        with self._db.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()
        return [self._decode("workflow", row[0]) for row in rows]

    @staticmethod
    def unhandled_failed_workflows_query(
        *, limit: int
    ) -> tuple[str, tuple[object, ...]]:
        """The SQL behind ``list_unhandled_failed_workflows``; its predicate is
        exactly the one ``gpu_fault_unhandled_failed_workflow_updated`` carries."""

        sql = """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='workflow'
                  AND payload->>'status'='FAILED'
                  AND (
                      payload->>'failure_handled_at' IS NULL
                      OR payload->>'failure_handled_at'=''
                  )
                ORDER BY payload->>'updated_at', key
                LIMIT %s
                """
        return sql, (limit,)

    def list_active_workflow_incidents(
        self,
        cluster_id: str,
        *,
        node_ids: set[str] | None = None,
        job_id: str | None = None,
        limit: int = ACTIVE_WORKFLOW_INCIDENTS_LIMIT,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        clauses = [
            "w.kind='workflow'",
            "i.kind='incident'",
            # Executable rows plus BLOCKED rows that still occupy their node
            # (waiting for an operator / parked by an internal error, F-A4).
            "(w.payload->>'status' IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')"
            " OR (w.payload->>'status' = 'BLOCKED'"
            " AND w.payload->>'blocked_kind' IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')))",
            "i.payload->>'cluster_id'=%s",
        ]
        parameters: list[object] = [cluster_id]
        if job_id is not None:
            clauses.append("i.payload->>'job_id'=%s")
            parameters.append(job_id)
        if node_ids is not None:
            if not node_ids:
                return []
            clauses.append("i.payload->'node_ids' ?| %s")
            parameters.append(sorted(node_ids))
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT i.payload, w.payload
                FROM gpu_fault_objects w
                JOIN gpu_fault_objects i
                  ON i.key=w.payload->>'incident_id'
                WHERE
                """
                + " AND ".join(clauses)
                + " ORDER BY w.payload->>'updated_at' DESC, w.key DESC"
                + " LIMIT %s",
                [*parameters, limit],
            )
            rows = cursor.fetchall()
        return [
            (
                self._decode("incident", incident),
                self._decode("workflow", workflow),
            )
            for incident, workflow in rows
        ]

    def list_job_recovery_workflow_incidents(
        self,
        cluster_id: str,
        job_id: str,
        attempt_id: str,
        *,
        limit: int = 100,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        # Text order on ``updated_at`` (store review 2026-09-07, item H2): the
        # ``::timestamptz`` cast forced a per-row detoast and sort. Text order
        # is time order for rows written after item E (fixed-width
        # ``YYYY-MM-DDTHH:MM:SS.ffffffZ``); legacy rows with a shorter
        # fraction may misorder within one second, which this reader
        # tolerates because it filters by rank afterwards.
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT i.payload, w.payload
                FROM gpu_fault_objects w
                JOIN gpu_fault_objects i
                  ON i.kind='incident'
                 AND i.key=w.payload->>'incident_id'
                WHERE w.kind='workflow'
                  AND i.payload->>'cluster_id'=%s
                  AND i.payload->>'job_id'=%s
                  AND (
                        (
                            w.payload->>'status' IN (
                                'PENDING', 'RUNNING', 'SAFETY_PENDING'
                            )
                            AND i.payload->>'attempt_id'=%s
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM jsonb_array_elements(
                                coalesce(
                                    w.payload->'step_executions',
                                    '[]'::jsonb
                                )
                            ) execution
                            WHERE execution->>'operation'
                                  ='RESTART_WORKLOAD'
                              AND execution->'details'
                                  ->>'restart_attempt_id'=%s
                        )
                  )
                ORDER BY w.payload->>'updated_at' DESC,
                         w.key DESC
                LIMIT %s
                """,
                (
                    cluster_id,
                    job_id,
                    attempt_id,
                    attempt_id,
                    limit,
                ),
            )
            rows = cursor.fetchall()
        return [
            (
                self._decode("incident", incident),
                self._decode("workflow", workflow),
            )
            for incident, workflow in rows
        ]

    def claim_workflow(
        self,
        request_id: str,
        executor_id: str,
        fencing_token: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
        remediation_budget_claims: dict[str, int] | None = None,
    ) -> WorkflowRequest:
        budget_error: RemediationBudgetError | None = None
        scopes = sorted(remediation_budget_claims or {})
        with self._db.transaction():
            if scopes:
                # Lock order rule (F-B2): advisory locks first, sorted, then
                # row locks. The merge paths take a group advisory lock and
                # then the workflow row; taking the row first here made the
                # two families a deadlock pair.
                with self._db.cursor() as cursor:
                    for scope in scopes:
                        cursor.execute(
                            """
                            SELECT pg_advisory_xact_lock(
                                hashtextextended(%s, 0)
                            )
                            """,
                            (f"remediation_budget/{scope}",),
                        )
            workflow = self._get_for_update("workflow", request_id)
            if workflow.fencing_token != fencing_token:
                raise StaleFencingTokenError("stale workflow fencing token")
            claimed_at = now or datetime.now(timezone.utc)
            lease_active = (
                workflow.execution_lease_expires_at is not None
                and workflow.execution_lease_expires_at > claimed_at
            )
            if (
                workflow.execution_owner_id is not None
                and workflow.execution_owner_id != executor_id
                and lease_active
            ):
                raise WorkflowLeaseError("workflow is leased by another executor")
            if remediation_budget_claims is not None:
                if scopes:
                    with self._db.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT payload
                            FROM gpu_fault_objects
                            WHERE kind='workflow'
                              AND key<>%s
                              AND payload->>'status'='RUNNING'
                              AND (payload->>'execution_lease_expires_at')
                                  ::timestamptz > %s
                              AND coalesce(
                                  payload->'remediation_budget_claims',
                                  '[]'::jsonb
                              ) ?| %s
                            """,
                            (request_id, claimed_at, scopes),
                        )
                        active = [
                            self._decode("workflow", row[0])
                            for row in cursor.fetchall()
                        ]
                else:
                    active = []
                try:
                    workflow = apply_remediation_budget(
                        workflow,
                        active,
                        remediation_budget_claims,
                        now=claimed_at,
                    )
                except RemediationBudgetError as exc:
                    workflow = blocked_by_remediation_budget(
                        workflow,
                        str(exc),
                        now=claimed_at,
                    )
                    budget_error = exc
            if budget_error is not None:
                self._put("workflow", request_id, workflow)
            else:
                new_epoch = (
                    workflow.execution_owner_id != executor_id or not lease_active
                )
                workflow = workflow.model_copy(
                    update={
                        "execution_owner_id": executor_id,
                        "execution_epoch": (
                            workflow.execution_epoch + 1
                            if new_epoch
                            else max(workflow.execution_epoch, 1)
                        ),
                        "execution_lease_expires_at": (claimed_at + lease_duration),
                        **(
                            {"status": WorkflowStatus.RUNNING}
                            if remediation_budget_claims is not None
                            else {}
                        ),
                    }
                )
                self._put("workflow", request_id, workflow)
        if budget_error is not None:
            raise budget_error
        return workflow

    def extend_remediation_budget(
        self,
        request_id: str,
        executor_id: str,
        claims: dict[str, int],
        *,
        now: datetime | None = None,
    ) -> WorkflowRequest:
        scopes = sorted(claims)
        with self._db.transaction():
            # Lock order rule (F-B2): advisory locks first, sorted, then rows.
            with self._db.cursor() as cursor:
                for scope in scopes:
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"remediation_budget/{scope}",),
                    )
            workflow = self._get_for_update("workflow", request_id)
            at = now or datetime.now(timezone.utc)
            if (
                workflow.execution_owner_id != executor_id
                or workflow.execution_lease_expires_at is None
                or workflow.execution_lease_expires_at <= at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            active: list[WorkflowRequest] = []
            if scopes:
                with self._db.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT payload
                        FROM gpu_fault_objects
                        WHERE kind='workflow'
                          AND key<>%s
                          AND payload->>'status'='RUNNING'
                          AND (payload->>'execution_lease_expires_at')
                              ::timestamptz > %s
                          AND coalesce(
                              payload->'remediation_budget_claims',
                              '[]'::jsonb
                          ) ?| %s
                        """,
                        (request_id, at, scopes),
                    )
                    active = [
                        self._decode("workflow", row[0]) for row in cursor.fetchall()
                    ]
            workflow = extend_remediation_budget(workflow, active, claims, now=at)
            self._put("workflow", request_id, workflow)
            return workflow

    def renew_workflow_lease(
        self,
        request_id: str,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
    ) -> WorkflowRequest:
        with self._db.transaction():
            workflow: WorkflowRequest = self._get_for_update("workflow", request_id)
            renewed_at = now or datetime.now(timezone.utc)
            if (
                workflow.execution_owner_id != executor_id
                or workflow.execution_epoch != execution_epoch
                or workflow.execution_lease_expires_at is None
                or workflow.execution_lease_expires_at <= renewed_at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            if not lease_extension_due(
                workflow, renewed_at=renewed_at, lease_duration=lease_duration
            ):
                # More than half the lease left: the read is the point (the
                # executor picks merges up through it); the write is not
                # (store review 2026-09-07, item F1).
                return workflow
            workflow = workflow.model_copy(
                update={"execution_lease_expires_at": (renewed_at + lease_duration)}
            )
            self._put("workflow", request_id, workflow)
            return workflow

    def save_workflow(
        self,
        workflow: WorkflowRequest,
        *,
        expected: WorkflowRequest | None = None,
    ) -> None:
        """See ``WorkflowStore.save_workflow`` (store review 2026-09-07, item B).

        The inherited SQLite version was a blind whole-row upsert under the
        process lock that ``PostgresStore`` neutralizes, so a copy read before
        a merge, a re-lease or a generation change overwrote it. With
        ``expected`` this is the full-payload compare-and-set of ``_put``.
        Without it, two statements and no transaction:

        1. a guarded UPDATE that matches only while the three version fields
           still equal the caller's copy;
        2. if that touched nothing, ``INSERT ... ON CONFLICT DO NOTHING``.

        Race analysis: the UPDATE is atomic on its own; a concurrent writer
        either commits first and moves a version (we match nothing) or waits
        behind our row lock and then sees our row. If it matched nothing the
        row was absent or moved. The INSERT then creates it only if it is still
        absent -- two creators race on the unique key and exactly one wins, the
        other gets no row back and is told the row moved, which is true. A row
        that was present with other versions at step 1 stays untouched and
        step 2 finds it, so nothing is ever overwritten. The one interleaving
        that lands is "present at step 1, deleted before step 2", which
        recreates a row the archive just removed; workflow rows are only
        deleted by control-record retention, which is off, so that is accepted
        rather than paid for with a transaction on every save.
        """

        if expected is not None:
            self._put("workflow", workflow.request_id, workflow, expected=expected)
            return
        payload = workflow.model_dump_json()
        with self._db.cursor() as cursor:
            # COALESCE to the model defaults: rows written before a field
            # existed decode with the default, and must compare as such.
            cursor.execute(
                """
                UPDATE gpu_fault_objects
                SET payload=%s::jsonb
                WHERE kind='workflow' AND key=%s
                  AND coalesce(payload->>'merge_revision', '0')=%s
                  AND coalesce(payload->>'execution_epoch', '0')=%s
                  AND payload->>'fencing_token'=%s
                """,
                (
                    payload,
                    workflow.request_id,
                    str(workflow.merge_revision),
                    str(workflow.execution_epoch),
                    str(workflow.fencing_token),
                ),
            )
            if cursor.rowcount == 1:
                return
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES ('workflow', %s, %s::jsonb)
                ON CONFLICT(kind, key) DO NOTHING
                RETURNING key
                """,
                (workflow.request_id, payload),
            )
            if cursor.fetchone() is not None:
                return
            # Neither statement landed: re-read only to name what moved.
            cursor.execute(
                """
                SELECT payload FROM gpu_fault_objects
                WHERE kind='workflow' AND key=%s
                """,
                (workflow.request_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise StaleWriteError(
                f"workflow/{workflow.request_id} changed since it was read"
            )
        stored: WorkflowRequest = self._decode("workflow", row[0])
        stale = stale_workflow_versions(stored, workflow)
        if stale is None:
            # The guard missed but the versions match now: the row moved and
            # moved back, or another writer landed the same versions in
            # between. Either way the caller's copy predates a write.
            stale = StaleWriteError(
                f"workflow/{workflow.request_id} changed since it was read"
            )
        raise stale

    def save_incident(self, incident: FaultIncident) -> None:
        """Write the incident and its event link atomically.

        The inherited version relies on the process-wide lock, which
        PostgresStore neutralizes because it cannot span replicas. These
        are two statements and the pool runs in autocommit, so without a
        transaction a reader could see the event link before the incident
        it points at.
        """

        with self._db.transaction():
            self._put("incident", incident.incident_id, incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )

    def save_workflow_if_leased(
        self,
        workflow: WorkflowRequest,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._db.transaction():
            current = self._get_for_update("workflow", workflow.request_id)
            checked_at = now or datetime.now(timezone.utc)
            if (
                current.execution_owner_id != executor_id
                or current.execution_epoch != execution_epoch
                or current.execution_lease_expires_at is None
                or current.execution_lease_expires_at <= checked_at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            if current.merge_revision != workflow.merge_revision:
                raise WorkflowMergedError("workflow was merged since it was read")
            self._put("workflow", workflow.request_id, workflow)

    def save_workflow_and_incident_if_leased(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._db.transaction():
            # Lock order rule (TransactionalWorkflowMixin): the incident row
            # first, then the workflow. Workflow-first made this the W->I half
            # of a deadlock pair with every merge on the same incident (store
            # review 2026-09-07, item C). A missing incident is tolerated: the
            # write below creates it.
            try:
                self._get_for_update("incident", incident.incident_id)
            except NotFoundError:
                pass
            current = self._get_for_update("workflow", workflow.request_id)
            checked_at = now or datetime.now(timezone.utc)
            if (
                current.execution_owner_id != executor_id
                or current.execution_epoch != execution_epoch
                or current.execution_lease_expires_at is None
                or current.execution_lease_expires_at <= checked_at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            if current.merge_revision != workflow.merge_revision:
                raise WorkflowMergedError("workflow was merged since it was read")
            self._put("workflow", workflow.request_id, workflow)
            self._put("incident", incident.incident_id, incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
