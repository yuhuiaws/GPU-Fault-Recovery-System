from __future__ import annotations

from typing import Any

from datetime import datetime, timedelta, timezone
from typing import Callable

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import (
    NotFoundError,
    WorkflowLeaseError,
)


class MemoryWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _incident_by_event: Any
    _incidents: Any
    _workflows: Any

    _lock: Any
    _replacement_fault_groups: Any
    _sxid_fault_groups: Any

    def save_incident(self, incident: FaultIncident) -> None:
        with self._lock:
            self._incidents[incident.incident_id] = incident
            self._incident_by_event[incident.event_id] = incident.incident_id

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> None:
        if incident.workflow_request_id != workflow.request_id:
            raise ValueError("incident workflow pointer does not match workflow")
        if workflow.incident_id != incident.incident_id:
            raise ValueError("workflow incident pointer does not match incident")
        with self._lock:
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._incident_by_event[incident.event_id] = incident.incident_id

    def get_incident(self, incident_id: str) -> FaultIncident:
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                raise NotFoundError(incident_id)
            return incident

    def get_incident_by_event(self, event_id: str) -> FaultIncident | None:
        with self._lock:
            incident_id = self._incident_by_event.get(event_id)
            return self._incidents.get(incident_id) if incident_id else None

    def link_event_to_incident(self, event_id: str, incident_id: str) -> None:
        with self._lock:
            if incident_id not in self._incidents:
                raise NotFoundError(incident_id)
            self._incident_by_event[event_id] = incident_id

    def create_incident_workflow_if_absent(
        self,
        event_id: str,
        builder: Callable[[], tuple[FaultIncident, WorkflowRequest]],
        *,
        serialization_key: str | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest, bool]:
        """Atomically create an event's incident and workflow."""
        with self._lock:
            existing = self.get_incident_by_event(event_id)
            if existing is not None:
                if not existing.workflow_request_id:
                    raise RuntimeError("event incident has no workflow request")
                return (
                    existing,
                    self.get_workflow(existing.workflow_request_id),
                    False,
                )
            incident, workflow = builder()
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._incident_by_event[event_id] = incident.incident_id
            self._incident_by_event[incident.event_id] = incident.incident_id
            return incident, workflow, True

    def merge_replacement_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        """Atomically merge one node fault into a workload replacement."""
        with self._lock:
            duplicate = self.get_incident_by_event(event_id)
            if duplicate is not None:
                return (
                    duplicate,
                    self.get_workflow(duplicate.workflow_request_id),
                )
            incident_id = self._replacement_fault_groups.get(group_key)
            existing_incident = (
                self._incidents.get(incident_id) if incident_id is not None else None
            )
            existing_workflow = (
                self._workflows.get(existing_incident.workflow_request_id)
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._incident_by_event[incident.event_id] = incident.incident_id
            self._incident_by_event[event_id] = incident.incident_id
            self._replacement_fault_groups[group_key] = incident.incident_id
            return incident, workflow

    def merge_attempt_fault_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        """Atomically merge one fault into an attempt-scoped workflow."""
        with self._lock:
            duplicate = self.get_incident_by_event(event_id)
            if duplicate is not None:
                return (
                    duplicate,
                    self.get_workflow(duplicate.workflow_request_id),
                )
            incident_id = self._sxid_fault_groups.get(group_key)
            existing_incident = (
                self._incidents.get(incident_id) if incident_id is not None else None
            )
            existing_workflow = (
                self._workflows.get(existing_incident.workflow_request_id)
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._incident_by_event[incident.event_id] = incident.incident_id
            self._incident_by_event[event_id] = incident.incident_id
            self._sxid_fault_groups[group_key] = incident.incident_id
            return incident, workflow

    def merge_sxid_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        return self.merge_attempt_fault_workflow(group_key, event_id, builder)

    def save_workflow(self, workflow: WorkflowRequest) -> None:
        with self._lock:
            self._workflows[workflow.request_id] = workflow

    def get_workflow(self, request_id: str) -> WorkflowRequest:
        with self._lock:
            workflow = self._workflows.get(request_id)
            if workflow is None:
                raise NotFoundError(request_id)
            return workflow

    def list_workflows(
        self,
        statuses: set[WorkflowStatus] | None = None,
        *,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[WorkflowRequest]:
        with self._lock:
            workflows = sorted(
                self._workflows.values(),
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
        with self._lock:
            return sorted(
                (
                    workflow
                    for workflow in self._workflows.values()
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
        active_statuses = {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
            WorkflowStatus.SAFETY_PENDING,
        }
        with self._lock:
            matches = []
            for workflow in self._workflows.values():
                if workflow.status not in active_statuses:
                    continue
                incident = self._incidents.get(workflow.incident_id)
                if incident is None or incident.cluster_id != cluster_id:
                    continue
                if job_id is not None and incident.job_id != job_id:
                    continue
                if node_ids is not None and not set(incident.node_ids).intersection(
                    node_ids
                ):
                    continue
                matches.append((incident, workflow))
            return sorted(
                matches,
                key=lambda item: (
                    item[1].updated_at,
                    item[1].request_id,
                ),
                reverse=True,
            )

    def list_job_recovery_workflow_incidents(
        self,
        cluster_id: str,
        job_id: str,
        attempt_id: str,
        *,
        limit: int = 100,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        active_statuses = {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
            WorkflowStatus.SAFETY_PENDING,
        }
        with self._lock:
            matches = []
            for workflow in self._workflows.values():
                incident = self._incidents.get(workflow.incident_id)
                if (
                    incident is None
                    or incident.cluster_id != cluster_id
                    or incident.job_id != job_id
                ):
                    continue
                restarted_attempt = next(
                    (
                        str(execution.details["restart_attempt_id"])
                        for execution in reversed(workflow.step_executions)
                        if (
                            execution.operation is WorkflowOperation.RESTART_WORKLOAD
                            and execution.details.get("restart_attempt_id")
                        )
                    ),
                    None,
                )
                if not (
                    (
                        workflow.status in active_statuses
                        and incident.attempt_id == attempt_id
                    )
                    or restarted_attempt == attempt_id
                ):
                    continue
                matches.append((incident, workflow))
            return sorted(
                matches,
                key=lambda item: (
                    item[1].updated_at,
                    item[1].request_id,
                ),
                reverse=True,
            )[:limit]

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
            workflow = self.get_workflow(request_id)
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
            self._workflows[request_id] = workflow
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
        with self._lock:
            workflow = self.get_workflow(request_id)
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
            self._workflows[request_id] = workflow
            return workflow

    def save_workflow_if_leased(
        self,
        workflow: WorkflowRequest,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None:
        with self._lock:
            current = self.get_workflow(workflow.request_id)
            checked_at = now or datetime.now(timezone.utc)
            if (
                current.execution_owner_id != executor_id
                or current.execution_epoch != execution_epoch
                or current.execution_lease_expires_at is None
                or current.execution_lease_expires_at <= checked_at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            self._workflows[workflow.request_id] = workflow

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
            current = self.get_workflow(workflow.request_id)
            checked_at = now or datetime.now(timezone.utc)
            if (
                current.execution_owner_id != executor_id
                or current.execution_epoch != execution_epoch
                or current.execution_lease_expires_at is None
                or current.execution_lease_expires_at <= checked_at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            self._workflows[workflow.request_id] = workflow
            self._incidents[incident.incident_id] = incident
            self._incident_by_event[incident.event_id] = incident.incident_id
