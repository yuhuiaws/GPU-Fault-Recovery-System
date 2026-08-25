from __future__ import annotations

from typing import Any, Callable

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    FaultIncident,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError


class SqliteWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _get: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _link: Callable[..., Any]
    _list: Callable[..., Any]
    _lock: Any
    _put: Callable[..., Any]

    def save_incident(self, incident: FaultIncident) -> None:
        with self._lock:
            self._put("incident", incident.incident_id, incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )

    def get_incident(self, incident_id: str) -> FaultIncident:
        return self._get("incident", incident_id)

    def get_incident_by_event(self, event_id: str) -> FaultIncident | None:
        incident_id = self._get_link("incident_by_event", event_id)
        return self._get_optional("incident", incident_id) if incident_id else None

    def link_event_to_incident(self, event_id: str, incident_id: str) -> None:
        self.get_incident(incident_id)
        with self._lock:
            self._link("incident_by_event", event_id, incident_id)

    def save_workflow(self, workflow: WorkflowRequest) -> None:
        with self._lock:
            self._put("workflow", workflow.request_id, workflow)

    def get_workflow(self, request_id: str) -> WorkflowRequest:
        return self._get("workflow", request_id)

    def list_workflows(
        self,
        statuses: set[WorkflowStatus] | None = None,
        *,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[WorkflowRequest]:
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
        return workflows[:limit]

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
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        clauses = [
            "w.kind='workflow'",
            "i.kind='incident'",
            "json_extract(w.payload, '$.status') "
            "IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')",
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
            "w.key DESC",
            parameters,
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
    ) -> WorkflowRequest:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                workflow = self._get("workflow", request_id)
                if workflow.fencing_token != fencing_token:
                    raise ValueError("stale workflow fencing token")
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
                    }
                )
                self._put("workflow", request_id, workflow)
                self._db.execute("COMMIT")
                return workflow
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def renew_workflow_lease(
        self,
        request_id: str,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
    ) -> WorkflowRequest:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                workflow = self._get("workflow", request_id)
                renewed_at = now or datetime.now(timezone.utc)
                if (
                    workflow.execution_owner_id != executor_id
                    or workflow.execution_epoch != execution_epoch
                    or workflow.execution_lease_expires_at is None
                    or workflow.execution_lease_expires_at <= renewed_at
                ):
                    raise WorkflowLeaseError("workflow execution lease is stale")
                workflow = workflow.model_copy(
                    update={"execution_lease_expires_at": (renewed_at + lease_duration)}
                )
                self._put("workflow", request_id, workflow)
                self._db.execute("COMMIT")
                return workflow
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def save_workflow_if_leased(
        self,
        workflow: WorkflowRequest,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._get("workflow", workflow.request_id)
                checked_at = now or datetime.now(timezone.utc)
                if (
                    current.execution_owner_id != executor_id
                    or current.execution_epoch != execution_epoch
                    or current.execution_lease_expires_at is None
                    or current.execution_lease_expires_at <= checked_at
                ):
                    raise WorkflowLeaseError("workflow execution lease is stale")
                self._put(
                    "workflow",
                    workflow.request_id,
                    workflow,
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def save_workflow_and_incident_if_leased(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._get("workflow", workflow.request_id)
                checked_at = now or datetime.now(timezone.utc)
                if (
                    current.execution_owner_id != executor_id
                    or current.execution_epoch != execution_epoch
                    or current.execution_lease_expires_at is None
                    or current.execution_lease_expires_at <= checked_at
                ):
                    raise WorkflowLeaseError("workflow execution lease is stale")
                self._put(
                    "workflow",
                    workflow.request_id,
                    workflow,
                )
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
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
