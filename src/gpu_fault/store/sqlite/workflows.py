from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Collection, ContextManager, Sequence

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.contracts import ACTIVE_WORKFLOW_INCIDENTS_LIMIT
from gpu_fault.store.shared.errors import (
    RemediationBudgetError,
    StaleFencingTokenError,
    StaleWriteError,
    WorkflowLeaseError,
    WorkflowMergedError,
)
from gpu_fault.store.shared.record_guards import (
    record_matches_expected,
    stale_incident_versions,
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
    workflow_matches_expected,
)
from gpu_fault.store.shared.workflow_scan import dispatch_order_key, held_reason

LOGGER = logging.getLogger(__name__)


class SqliteWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _get: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _link: Callable[..., Any]
    _list: Callable[..., Any]
    _lock: Any
    _put: Callable[..., Any]
    _state_transaction: Callable[[str], ContextManager[None]]

    def save_incident(
        self,
        incident: FaultIncident,
        *,
        expected: FaultIncident | None = None,
        extra_event_ids: Sequence[str] = (),
    ) -> None:
        """See ``WorkflowStore.save_incident`` (architecture review, item D1).

        Read-compare-write inside one write transaction; the process lock
        serializes every writer of this connection, so that is the CAS here.
        """

        with self._state_transaction(f"incident/{incident.incident_id}"):
            current = self._get_optional("incident", incident.incident_id)
            if expected is not None:
                if not record_matches_expected(current, expected):
                    raise StaleWriteError(
                        f"incident/{incident.incident_id} changed since it was read"
                    )
            elif current is not None:
                stale = stale_incident_versions(current, incident)
                if stale is not None:
                    raise stale
            self._put("incident", incident.incident_id, incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
            for event_id in extra_event_ids:
                if event_id != incident.event_id:
                    self._link("incident_by_event", event_id, incident.incident_id)

    def save_workflow(
        self,
        workflow: WorkflowRequest,
        *,
        expected: WorkflowRequest | None = None,
    ) -> None:
        """See ``WorkflowStore.save_workflow`` (store review 2026-09-07, item B).

        Read-compare-write inside one write transaction; the process lock
        serializes every writer of this connection, so that is the CAS here.
        """

        with self._state_transaction(f"workflow/{workflow.request_id}"):
            current = self._get_optional("workflow", workflow.request_id)
            if expected is not None:
                if not workflow_matches_expected(current, expected):
                    raise StaleWriteError(
                        f"workflow/{workflow.request_id} changed since it was read"
                    )
            elif current is not None:
                stale = stale_workflow_versions(current, workflow)
                if stale is not None:
                    raise stale
            self._put("workflow", workflow.request_id, workflow)

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
        if after is not None and dispatchable_at is None:
            raise ValueError(
                "the scan cursor (after) is only defined with dispatchable_at"
            )
        if dispatchable_at is not None:
            # Dispatch mode orders by when the row became eligible, not by its
            # last merge (F-A2a); the cursor pages that order (F-A2c).
            workflows = sorted(
                self._list("workflow"),
                key=dispatch_order_key,
                reverse=newest_first,
            )
            if after is not None:
                anchor = dispatch_order_key(after)
                workflows = [
                    item for item in workflows if dispatch_order_key(item) > anchor
                ]
        else:
            workflows = sorted(
                self._list("workflow"),
                key=lambda item: (
                    item.updated_at,
                    item.request_id,
                ),
                reverse=newest_first,
            )
        if statuses is not None:
            workflows = [item for item in workflows if item.status in statuses]
        if dispatchable_at is not None or exclude_request_ids:
            workflows = [
                item
                for item in workflows
                if held_reason(
                    item,
                    dispatchable_at=dispatchable_at or item.updated_at,
                    exclude_request_ids=exclude_request_ids,
                    lookup=lambda key: self._get_optional("workflow", key),
                )
                is None
            ]
        return workflows[:limit]

    def count_held_workflows(
        self,
        statuses: set[WorkflowStatus] | None,
        *,
        dispatchable_at: datetime,
        exclude_request_ids: Collection[str] = (),
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self._list("workflow"):
            if statuses is not None and item.status not in statuses:
                continue
            reason = held_reason(
                item,
                dispatchable_at=dispatchable_at,
                exclude_request_ids=exclude_request_ids,
                lookup=lambda key: self._get_optional("workflow", key),
            )
            if reason is not None:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def workflow_status_counts(self) -> dict[WorkflowStatus, int]:
        """Count every persisted workflow by status without decoding any.

        The /metrics workflow gauge must stay exact while the detail scan that
        feeds the duration and step families is bounded, so the count is a
        server-side aggregate rather than a by-product of that scan.
        """

        rows = self._db.execute(
            """
            SELECT json_extract(payload, '$.status') AS status, COUNT(*)
            FROM objects WHERE kind='workflow' GROUP BY status
            """
        ).fetchall()
        counts = {status: 0 for status in WorkflowStatus}
        for status, count in rows:
            counts[WorkflowStatus(status)] = int(count)
        return counts

    def incident_state_counts(self) -> dict[IncidentState, int]:
        """Count every persisted incident by state (server-side aggregate
        for the /metrics incident gauge; ESCALATED is the operator queue)."""

        rows = self._db.execute(
            """
            SELECT json_extract(payload, '$.state') AS state, COUNT(*)
            FROM objects WHERE kind='incident' GROUP BY state
            """
        ).fetchall()
        counts = {state: 0 for state in IncidentState}
        for state, count in rows:
            counts[IncidentState(state)] = int(count)
        return counts

    def blocked_workflows_without_verified_restore(self) -> int:
        """Count the BLOCKED workflows whose GPU node is still held.

        ``workflow_status_counts`` counts every workflow ever persisted, and
        BLOCKED is terminal, so its BLOCKED bucket only falls when an operator
        reconciles a record or the incident is archived out of the table. That
        makes it useless as a "there is a backlog now" signal: a BLOCKED
        workflow whose node has already been restored by a successor keeps
        counting forever.

        This is the same predicate as
        ``workflow_resolution.verified_restore_successor``, negated -- the
        condition the release preflight
        (``deploy/control-plane/regional/probes/workflow_safety.py``) already
        uses to decide whether a BLOCKED record still means a node is out of the
        training pool. It recovers on its own the moment the successor
        workflow restores scheduling, without waiting for
        ``gpu-fault-admin workflow-reconcile``.

        Kept as one server-side aggregate for the same reason as
        ``workflow_status_counts``: /metrics must not decode a growing table.
        The driving scan is over BLOCKED rows via
        ``objects_active_workflow_scope``, and each one costs two primary-key
        lookups.
        """

        row = self._db.execute(
            """
            SELECT COUNT(*)
            FROM objects w
            WHERE w.kind='workflow'
              AND json_extract(w.payload, '$.status')='BLOCKED'
              AND NOT EXISTS (
                  SELECT 1
                  FROM objects i
                  JOIN objects s
                    ON s.kind='workflow'
                   AND s.key=json_extract(i.payload, '$.workflow_request_id')
                  WHERE i.kind='incident'
                    AND i.key=json_extract(w.payload, '$.incident_id')
                    AND json_extract(i.payload, '$.state')='RECOVERED'
                    AND s.key<>w.key
                    AND json_extract(s.payload, '$.incident_id')
                        =json_extract(w.payload, '$.incident_id')
                    AND json_extract(s.payload, '$.status')='SUCCEEDED'
                    AND json_extract(s.payload, '$.fencing_token')
                        =json_extract(w.payload, '$.fencing_token')
                    AND json_extract(i.payload, '$.fencing_token')
                        =json_extract(w.payload, '$.fencing_token')
                    AND EXISTS (
                        SELECT 1
                        FROM json_each(
                            s.payload, '$.completed_operations'
                        ) operation
                        WHERE operation.value='RESTORE_SCHEDULING'
                    )
              )
            """
        ).fetchone()
        return int(row[0])

    def list_orphan_workflows(
        self, *, created_before: datetime, limit: int = 1000
    ) -> list[WorkflowRequest]:
        """See ``ControlPlaneStore.list_orphan_workflows``.

        One server-side predicate rather than a decode of the workflow table:
        the gauge that reads it runs on every scrape. ``created_at`` is
        compared as the ISO-8601 text the models serialize, whose order is
        time order.
        """

        rows = self._db.execute(
            """
            SELECT w.payload
            FROM objects w
            LEFT JOIN objects i
              ON i.kind='incident'
             AND i.key=json_extract(w.payload, '$.incident_id')
            WHERE w.kind='workflow'
              AND json_extract(w.payload, '$.status')
                  IN ('PENDING', 'SAFETY_PENDING')
              AND json_extract(w.payload, '$.created_at') < ?
              AND (
                  i.key IS NULL
                  OR COALESCE(
                      json_extract(i.payload, '$.workflow_request_id'), ''
                  ) <> w.key
              )
              AND NOT EXISTS (
                  SELECT 1 FROM objects s
                  WHERE s.kind='workflow'
                    AND json_extract(s.payload, '$.predecessor_workflow_id')=w.key
              )
            ORDER BY json_extract(w.payload, '$.created_at'), w.key
            LIMIT ?
            """,
            (_utc_text(created_before), limit),
        ).fetchall()
        return [WorkflowRequest.model_validate_json(row[0]) for row in rows]

    def list_incidents_with_missing_workflow(
        self, *, limit: int = 1000
    ) -> list[FaultIncident]:
        rows = self._db.execute(
            """
            SELECT i.payload
            FROM objects i
            WHERE i.kind='incident'
              AND COALESCE(json_extract(i.payload, '$.workflow_request_id'), '') <> ''
              AND NOT EXISTS (
                  SELECT 1 FROM objects w
                  WHERE w.kind='workflow'
                    AND w.key=json_extract(i.payload, '$.workflow_request_id')
              )
            ORDER BY json_extract(i.payload, '$.created_at'), i.key
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [FaultIncident.model_validate_json(row[0]) for row in rows]

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
            "json_extract(i.payload, '$.cluster_id')=?",
            "json_extract(i.payload, '$.state') IN ("
            + ",".join("?" for _ in wanted)
            + ")",
        ]
        parameters: list[object] = [cluster_id, *wanted]
        if node_ids is not None:
            placeholders = ",".join("?" for _ in node_ids)
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM json_each(i.payload, '$.node_ids') n "
                f"WHERE n.value IN ({placeholders})"
                ")"
            )
            parameters.extend(sorted(node_ids))
        rows = self._db.execute(
            "SELECT i.payload FROM objects i WHERE "
            + " AND ".join(clauses)
            + " ORDER BY json_extract(i.payload, '$.updated_at') DESC, i.key DESC"
            " LIMIT ?",
            [*parameters, limit],
        ).fetchall()
        return [FaultIncident.model_validate_json(row[0]) for row in rows]

    def list_unhandled_failed_workflows(
        self, *, limit: int = 1000
    ) -> list[WorkflowRequest]:
        return sorted(
            (
                workflow
                for workflow in self._list("workflow")
                if workflow.status is WorkflowStatus.FAILED
                and workflow.failure_handled_at is None
            ),
            key=lambda item: (
                item.updated_at,
                item.request_id,
            ),
        )[:limit]

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
            "(json_extract(w.payload, '$.status') "
            "IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')"
            " OR (json_extract(w.payload, '$.status') = 'BLOCKED'"
            " AND json_extract(w.payload, '$.blocked_kind')"
            " IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')))",
            "json_extract(i.payload, '$.cluster_id')=?",
        ]
        parameters: list[object] = [cluster_id]
        if job_id is not None:
            clauses.append("json_extract(i.payload, '$.job_id')=?")
            parameters.append(job_id)
        if node_ids is not None:
            if not node_ids:
                return []
            placeholders = ",".join("?" for _ in node_ids)
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM json_each(i.payload, '$.node_ids') n "
                f"WHERE n.value IN ({placeholders})"
                ")"
            )
            parameters.extend(sorted(node_ids))
        rows = self._db.execute(
            """
            SELECT i.payload, w.payload
            FROM objects w
            JOIN objects i
              ON i.key=json_extract(w.payload, '$.incident_id')
            WHERE
            """
            + " AND ".join(clauses)
            + " ORDER BY json_extract(w.payload, '$.updated_at') DESC, "
            "w.key DESC LIMIT ?",
            [*parameters, limit],
        ).fetchall()
        return [
            (
                FaultIncident.model_validate_json(incident),
                WorkflowRequest.model_validate_json(workflow),
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
        rows = self._db.execute(
            """
            SELECT i.payload, w.payload
            FROM objects w
            JOIN objects i
              ON i.kind='incident'
             AND i.key=json_extract(w.payload, '$.incident_id')
            WHERE w.kind='workflow'
              AND json_extract(i.payload, '$.cluster_id')=?
              AND json_extract(i.payload, '$.job_id')=?
              AND (
                    (
                        json_extract(w.payload, '$.status')
                            IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')
                        AND json_extract(i.payload, '$.attempt_id')=?
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM json_each(
                            w.payload, '$.step_executions'
                        ) execution
                        WHERE json_extract(
                                  execution.value, '$.operation'
                              )='RESTART_WORKLOAD'
                          AND json_extract(
                                  execution.value,
                                  '$.details.restart_attempt_id'
                              )=?
                    )
              )
            ORDER BY json_extract(w.payload, '$.updated_at') DESC,
                     w.key DESC
            LIMIT ?
            """,
            (
                cluster_id,
                job_id,
                attempt_id,
                attempt_id,
                limit,
            ),
        ).fetchall()
        return [
            (
                FaultIncident.model_validate_json(incident),
                WorkflowRequest.model_validate_json(workflow),
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
        with self._lock:
            # A budget refusal is a committed write (the workflow goes BLOCKED)
            # followed by a raise, so it leaves the transaction block normally
            # and raises after it.
            budget_error: RemediationBudgetError | None = None
            with self._state_transaction(f"workflow/{request_id}"):
                workflow = self._get("workflow", request_id)
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
                    try:
                        workflow = apply_remediation_budget(
                            workflow,
                            self._list("workflow"),
                            remediation_budget_claims,
                            now=claimed_at,
                        )
                    except RemediationBudgetError as exc:
                        workflow = blocked_by_remediation_budget(
                            workflow,
                            str(exc),
                            scope=exc.scope,
                            now=claimed_at,
                        )
                        self._put("workflow", request_id, workflow)
                        budget_error = exc
                if budget_error is None:
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
        with self._state_transaction(f"workflow/{request_id}"):
            workflow = self._get("workflow", request_id)
            at = now or datetime.now(timezone.utc)
            if (
                workflow.execution_owner_id != executor_id
                or workflow.execution_lease_expires_at is None
                or workflow.execution_lease_expires_at <= at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            workflow = extend_remediation_budget(
                workflow, self._list("workflow"), claims, now=at
            )
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
        with self._state_transaction(f"workflow/{request_id}"):
            workflow: WorkflowRequest = self._get("workflow", request_id)
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
                return workflow  # store review 2026-09-07, item F1
            workflow = workflow.model_copy(
                update={"execution_lease_expires_at": (renewed_at + lease_duration)}
            )
            self._put("workflow", request_id, workflow)
            return workflow

    def save_workflow_if_leased(
        self,
        workflow: WorkflowRequest,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._state_transaction(f"workflow/{workflow.request_id}"):
            current = self._get("workflow", workflow.request_id)
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
            self._put(
                "workflow",
                workflow.request_id,
                workflow,
            )

    def save_workflow_and_incident_if_leased(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._state_transaction(f"workflow/{workflow.request_id}"):
            current_incident = self._get_optional("incident", incident.incident_id)
            current = self._get("workflow", workflow.request_id)
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
            self._put(
                "workflow",
                workflow.request_id,
                workflow,
            )
            if incident_pointer_moved(current_incident, incident):
                # Same rule as PostgreSQL (C-02); the process lock is the row
                # lock here.
                LOGGER.warning(
                    "incident %s moved its workflow pointer to %s since %s read "
                    "it; keeping the merged incident and writing only the workflow",
                    incident.incident_id,
                    current_incident.workflow_request_id,
                    workflow.request_id,
                )
                return
            self._put(
                "incident",
                incident.incident_id,
                incident,
            )
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
