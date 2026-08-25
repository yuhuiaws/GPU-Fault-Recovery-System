from __future__ import annotations

from typing import Any

from typing import Callable

from gpu_fault.models import FaultIncident, WorkflowRequest


class TransactionalWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _link: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> None:
        if incident.workflow_request_id != workflow.request_id:
            raise ValueError("incident workflow pointer does not match workflow")
        if workflow.incident_id != incident.incident_id:
            raise ValueError("workflow incident pointer does not match incident")
        with self._state_transaction(f"incident_workflow/{incident.incident_id}"):
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )

    def create_incident_workflow_if_absent(
        self,
        event_id: str,
        builder: Callable[[], tuple[FaultIncident, WorkflowRequest]],
        *,
        serialization_key: str | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest, bool]:
        with self._state_transaction(
            "incident_workflow/" + (serialization_key or event_id)
        ):
            existing_id = self._get_link("incident_by_event", event_id)
            if existing_id is not None:
                existing = self._get("incident", existing_id)
                if not existing.workflow_request_id:
                    raise RuntimeError("event incident has no workflow request")
                return (
                    existing,
                    self._get(
                        "workflow",
                        existing.workflow_request_id,
                    ),
                    False,
                )
            incident, workflow = builder()
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._link(
                "incident_by_event",
                event_id,
                incident.incident_id,
            )
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
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
        with self._state_transaction(f"replacement_fault_group/{group_key}"):
            duplicate_id = self._get_link("incident_by_event", event_id)
            if duplicate_id is not None:
                duplicate = self._get("incident", duplicate_id)
                return (
                    duplicate,
                    self._get(
                        "workflow",
                        duplicate.workflow_request_id,
                    ),
                )
            incident_id = self._get_link("replacement_fault_group", group_key)
            existing_incident = (
                self._get_optional("incident", incident_id)
                if incident_id is not None
                else None
            )
            existing_workflow = (
                self._get_optional(
                    "workflow",
                    existing_incident.workflow_request_id,
                )
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
            self._link(
                "incident_by_event",
                event_id,
                incident.incident_id,
            )
            self._link(
                "replacement_fault_group",
                group_key,
                incident.incident_id,
            )
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
        with self._state_transaction(f"sxid_fault_group/{group_key}"):
            duplicate_id = self._get_link("incident_by_event", event_id)
            if duplicate_id is not None:
                duplicate = self._get("incident", duplicate_id)
                return (
                    duplicate,
                    self._get(
                        "workflow",
                        duplicate.workflow_request_id,
                    ),
                )
            incident_id = self._get_link("sxid_fault_group", group_key)
            existing_incident = (
                self._get_optional("incident", incident_id)
                if incident_id is not None
                else None
            )
            existing_workflow = (
                self._get_optional(
                    "workflow",
                    existing_incident.workflow_request_id,
                )
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
            self._link(
                "incident_by_event",
                event_id,
                incident.incident_id,
            )
            self._link(
                "sxid_fault_group",
                group_key,
                incident.incident_id,
            )
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
