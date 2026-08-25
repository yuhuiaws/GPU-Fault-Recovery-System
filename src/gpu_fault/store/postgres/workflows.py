from __future__ import annotations

from typing import Any, Callable

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    FaultIncident,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError


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
    ) -> list[WorkflowRequest]:
        clauses = ["kind='workflow'"]
        parameters: list[object] = []
        if statuses is not None:
            if not statuses:
                return []
            clauses.append("payload->>'status'=ANY(%s)")
            parameters.append(
                [
                    status.value
                    for status in sorted(
                        statuses,
                        key=lambda item: item.value,
                    )
                ]
            )
        direction = "DESC" if newest_first else "ASC"
        parameters.append(limit)
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM gpu_fault_objects WHERE "
                + " AND ".join(clauses)
                + f" ORDER BY payload->>'updated_at' {direction}, "
                + f"key {direction} LIMIT %s",
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("workflow", row[0]) for row in rows]

    def list_unhandled_failed_workflows(
        self, *, limit: int = 1000
    ) -> list[WorkflowRequest]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
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
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [self._decode("workflow", row[0]) for row in rows]

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
            "w.payload->>'status' IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')",
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
                + " ORDER BY w.payload->>'updated_at' DESC, w.key DESC",
                parameters,
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
                ORDER BY (w.payload->>'updated_at')::timestamptz DESC,
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
    ) -> WorkflowRequest:
        with self._db.transaction():
            workflow = self._get_for_update("workflow", request_id)
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
            new_epoch = workflow.execution_owner_id != executor_id or not lease_active
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
            workflow = self._get_for_update("workflow", request_id)
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
            return workflow

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
            current = self._get_for_update("workflow", workflow.request_id)
            checked_at = now or datetime.now(timezone.utc)
            if (
                current.execution_owner_id != executor_id
                or current.execution_epoch != execution_epoch
                or current.execution_lease_expires_at is None
                or current.execution_lease_expires_at <= checked_at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            self._put("workflow", workflow.request_id, workflow)
            self._put("incident", incident.incident_id, incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
