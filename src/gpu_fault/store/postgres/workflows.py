from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Collection

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
    datetime_json_text,
)
from gpu_fault.store.contracts import ACTIVE_WORKFLOW_INCIDENTS_LIMIT
from gpu_fault.store.postgres.state_table_storage import (
    get_state_payload,
    put_state_fields,
    put_state_record,
)
from gpu_fault.store.shared.errors import (
    NotFoundError,
    RemediationBudgetError,
    StaleFencingTokenError,
    StaleWriteError,
    WorkflowLeaseError,
)
from gpu_fault.store.shared.remediation_budgets import (
    apply_remediation_budget,
    blocked_by_remediation_budget,
    extend_remediation_budget,
)
from gpu_fault.store.shared.time import utc_text as _utc_text
from gpu_fault.store.shared.transactional_workflows import (
    incident_pointer_moved,
    lease_extension_due,
    stale_workflow_versions,
    validate_leased_workflow_save,
)
from gpu_fault.store.shared.workflow_scan import dispatch_order_key

LOGGER = logging.getLogger(__name__)


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

    def list_recent_workflows(
        self,
        open_statuses: set[WorkflowStatus],
        *,
        updated_since: datetime | None,
        limit: int,
    ) -> list[WorkflowRequest]:
        # Two range scans rather than one OR (G-2): the open half walks the
        # executable/BLOCKED partial indexes, the recent half walks
        # ``gpu_fault_workflow_updated_all`` backwards from now to the window
        # edge and stops there, so terminal history older than the window is
        # never read however much of it retention has yet to reclaim. Open
        # rows fill the budget first; the cap only ever cuts terminal rows.
        if limit <= 0:
            return []
        rows: list[WorkflowRequest] = []
        if open_statuses:
            rows.extend(self._scan(set(open_statuses), limit=limit, newest_first=True))
        remaining = limit - len(rows)
        recent_statuses = set(WorkflowStatus) - set(open_statuses)
        if remaining > 0 and recent_statuses:
            rows.extend(
                self._scan(
                    recent_statuses,
                    limit=remaining,
                    newest_first=True,
                    updated_since=updated_since,
                )
            )
        return rows

    def _scan(
        self,
        statuses: set[WorkflowStatus],
        *,
        limit: int,
        newest_first: bool,
        updated_since: datetime | None = None,
    ) -> list[WorkflowRequest]:
        sql, parameters = self.workflow_scan_query(
            statuses,
            limit=limit,
            newest_first=newest_first,
            updated_since=updated_since,
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
        "(predecessor.status IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')"
        " OR (predecessor.status = 'BLOCKED'"
        " AND predecessor.blocked_kind"
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
            clauses.append("(w.not_before IS NULL OR w.not_before <= %s)")
            parameters.append(_utc_text(dispatchable_at))
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM gpu_fault_workflow_records AS predecessor"
                " WHERE predecessor.kind='workflow'"
                " AND predecessor.key=w.predecessor_workflow_id"
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
    _DISPATCH_ORDER_SQL = "w.dispatch_eligible_at"

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
        updated_since: datetime | None = None,
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

        ``updated_since`` is the recency bound of ``list_recent_workflows``:
        text comparison on ``payload->>'updated_at'`` like every payload
        timestamp here, so ``gpu_fault_workflow_updated_all`` serves it as a
        range scan that ends at the window edge.
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
            clauses.append(f"w.status IN ({literals})")
        pushdown, parameters = cls._pushdown_clauses(
            dispatchable_at, exclude_request_ids
        )
        clauses.extend(pushdown)
        if updated_since is not None:
            clauses.append("w.updated_at >= %s")
            parameters.append(datetime_json_text(updated_since))
        direction = "DESC" if newest_first else "ASC"
        if dispatchable_at is not None:
            order_key = cls._DISPATCH_ORDER_SQL
            if after is not None:
                eligible_at, request_id = dispatch_order_key(after)
                comparison = "<" if newest_first else ">"
                clauses.append(f"({order_key}, w.key) {comparison} (%s, %s)")
                parameters.extend((eligible_at, request_id))
        else:
            order_key = "w.updated_at"
        sql = (
            "SELECT w.payload FROM gpu_fault_workflow_records AS w WHERE "
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
            clauses.append(f"w.status IN ({literals})")
        excluded = sorted(set(exclude_request_ids))
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT"
                " count(*) FILTER (WHERE w.not_before > %s),"
                " count(*) FILTER (WHERE EXISTS ("
                "SELECT 1 FROM gpu_fault_workflow_records AS predecessor"
                " WHERE predecessor.kind='workflow'"
                " AND predecessor.key=w.predecessor_workflow_id"
                f" AND {self._OPEN_PREDECESSOR_SQL})),"
                " count(*) FILTER (WHERE w.key = ANY(%s))"
                " FROM gpu_fault_workflow_records AS w WHERE " + " AND ".join(clauses),
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

        table = (
            "gpu_fault_workflow_records" if kind == "workflow" else "gpu_fault_objects"
        )
        expression = "status" if kind == "workflow" else f"payload->>'{field}'"
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT s.value, (
                    SELECT count(*) FROM {table}
                    WHERE kind='{kind}' AND {expression} = s.value
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
                FROM gpu_fault_workflow_records w
                WHERE w.kind='workflow'
                  AND w.status='BLOCKED'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM gpu_fault_objects i
                      JOIN gpu_fault_workflow_records s
                        ON s.kind='workflow'
                       AND s.key=i.payload->>'workflow_request_id'
                      WHERE i.kind='incident'
                        AND i.key=w.incident_id
                        AND i.payload->>'state'='RECOVERED'
                        AND s.key<>w.key
                        AND s.incident_id=w.incident_id
                        AND s.status='SUCCEEDED'
                        AND s.fencing_token=w.fencing_token
                        AND i.payload->>'fencing_token'
                            =w.fencing_token::text
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
                FROM gpu_fault_workflow_records w
                LEFT JOIN gpu_fault_objects i
                  ON i.kind='incident'
                 AND i.key=w.incident_id
                WHERE w.kind='workflow'
                  AND w.status IN ('PENDING', 'SAFETY_PENDING')
                  AND w.created_at < %s
                  AND (
                      i.key IS NULL
                      OR COALESCE(i.payload->>'workflow_request_id', '') <> w.key
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM gpu_fault_workflow_records s
                      WHERE s.kind='workflow'
                        AND s.predecessor_workflow_id=w.key
                  )
                ORDER BY w.created_at, w.key
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
                  AND i.payload->>'workflow_request_id' > ''
                  AND NOT EXISTS (
                      SELECT 1 FROM gpu_fault_workflow_records w
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

    def list_incidents_by_state(
        self,
        cluster_id: str,
        states: Collection[IncidentState],
        *,
        node_ids: set[str] | None = None,
        limit: int = ACTIVE_WORKFLOW_INCIDENTS_LIMIT,
    ) -> list[FaultIncident]:
        wanted = sorted({IncidentState(state).value for state in states})
        if not wanted or (node_ids is not None and not node_ids):
            return []
        clauses = [
            "i.kind='incident'",
            "i.payload->>'cluster_id'=%s",
            "i.payload->>'state' = ANY(%s)",
        ]
        parameters: list[object] = [cluster_id, wanted]
        if node_ids is not None:
            clauses.append("i.payload->'node_ids' ?| %s")
            parameters.append(sorted(node_ids))
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT i.payload FROM gpu_fault_objects i WHERE "
                + " AND ".join(clauses)
                + " ORDER BY i.payload->>'updated_at' DESC, i.key DESC LIMIT %s",
                [*parameters, limit],
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
                FROM gpu_fault_workflow_records
                WHERE kind='workflow'
                  AND status='FAILED' AND failure_unhandled
                ORDER BY updated_at, key
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
            "(w.status IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')"
            " OR (w.status = 'BLOCKED'"
            " AND w.blocked_kind IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')))",
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
                FROM gpu_fault_workflow_records w
                JOIN gpu_fault_objects i
                  ON i.key=w.incident_id
                WHERE
                """
                + " AND ".join(clauses)
                + " ORDER BY w.updated_at DESC, w.key DESC"
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
        include_terminal: bool = False,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        # Fixed-width UTC text avoids per-row detoasting and preserves time order.
        # Legacy fractions may differ within one second; callers filter by rank.
        status_predicate = (
            "TRUE"
            if include_terminal
            else "w.status IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')"
        )
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT i.payload, w.payload
                FROM gpu_fault_workflow_records w
                JOIN gpu_fault_objects i
                  ON i.kind='incident'
                 AND i.key=w.incident_id
                WHERE w.kind='workflow'
                  AND i.payload->>'cluster_id'=%s
                  AND i.payload->>'job_id'=%s
                  AND (
                        (
                            {status_predicate}
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
                ORDER BY w.updated_at DESC,
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
                            FROM gpu_fault_workflow_records
                            WHERE kind='workflow'
                              AND key<>%s
                              AND status='RUNNING'
                              AND execution_lease_expires_at
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
                    # The refusing scope is recorded too, so the per-cluster
                    # saturation gauge never has to parse the message (S1).
                    workflow = blocked_by_remediation_budget(
                        workflow, str(exc), scope=exc.scope, now=claimed_at
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
                        FROM gpu_fault_workflow_records
                        WHERE kind='workflow'
                          AND key<>%s
                          AND status='RUNNING'
                          AND execution_lease_expires_at
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
            put_state_fields(
                self._db,
                "workflow",
                request_id,
                workflow,
                frozenset({"execution_lease_expires_at"}),
            )
            return workflow

    def save_workflow(
        self,
        workflow: WorkflowRequest,
        *,
        expected: WorkflowRequest | None = None,
    ) -> None:
        """Keep full CAS or the merge/epoch/fence guard on the authoritative row.

        The routed upsert applies the same version guard in legacy and dedicated
        modes. A failed guard cannot fall through to an unconditional overwrite.
        """

        if expected is not None:
            self._put("workflow", workflow.request_id, workflow, expected=expected)
            return
        if put_state_record(
            self._db,
            "workflow",
            workflow.request_id,
            workflow,
            guard_versions=True,
        ):
            return
        try:
            payload = get_state_payload(self._db, "workflow", workflow.request_id)
        except NotFoundError:
            raise StaleWriteError(
                f"workflow/{workflow.request_id} changed since it was read"
            ) from None
        stored: WorkflowRequest = self._decode("workflow", payload)
        stale = stale_workflow_versions(stored, workflow)
        if stale is None:
            # The guard missed but the versions match now: the row moved and
            # moved back, or another writer landed the same versions in
            # between. Either way the caller's copy predates a write.
            stale = StaleWriteError(
                f"workflow/{workflow.request_id} changed since it was read"
            )
        raise stale

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
            validate_leased_workflow_save(
                current, workflow, executor_id, execution_epoch, checked_at
            )
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
                current_incident = self._get_for_update(
                    "incident", incident.incident_id
                )
            except NotFoundError:
                current_incident = None
            current = self._get_for_update("workflow", workflow.request_id)
            checked_at = now or datetime.now(timezone.utc)
            validate_leased_workflow_save(
                current, workflow, executor_id, execution_epoch, checked_at
            )
            self._put("workflow", workflow.request_id, workflow)
            if incident_pointer_moved(current_incident, incident):
                # Inside the row lock (C-02): the caller's incident snapshot
                # predates a merge that re-parented the incident. Only the
                # workflow is written; see ``incident_pointer_moved``.
                LOGGER.warning(
                    "incident %s moved its workflow pointer to %s since %s read "
                    "it; keeping the merged incident and writing only the workflow",
                    incident.incident_id,
                    current_incident.workflow_request_id,
                    workflow.request_id,
                )
                return
            self._put("incident", incident.incident_id, incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
