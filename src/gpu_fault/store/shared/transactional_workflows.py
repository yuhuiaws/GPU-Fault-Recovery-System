from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from gpu_fault.models import FaultIncident, RecoveryPlan, WorkflowRequest
from gpu_fault.retired_generation import retired_generation_records
from gpu_fault.workflow_resolution import reconciled_restore_records


class TransactionalWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: Callable[..., Any]
    _get_for_update: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
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

    def reconcile_restored_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        expected_workflow_updated_at: datetime,
        reference: str,
        reconciled_at: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident, RecoveryPlan]:
        with self._state_transaction(f"workflow_reconcile/{workflow_request_id}"):
            getter = getattr(self, "_get_for_update", self._get)
            workflow = getter("workflow", workflow_request_id)
            incident = getter("incident", workflow.incident_id)
            successor = getter("workflow", successor_workflow_id)
            if not workflow.source_plan_id:
                raise ValueError("workflow has no source recovery plan")
            source_plan = getter("plan", workflow.source_plan_id)
            updated_workflow, updated_incident, updated_plan = (
                reconciled_restore_records(
                    workflow,
                    incident,
                    successor,
                    source_plan,
                    self._list("remote_command"),
                    expected_fencing_token=expected_fencing_token,
                    expected_workflow_updated_at=expected_workflow_updated_at,
                    reference=reference,
                    reconciled_at=reconciled_at,
                )
            )
            self._put("workflow", workflow_request_id, updated_workflow)
            self._put("incident", incident.incident_id, updated_incident)
            self._put("plan", source_plan.plan_id, updated_plan)
            return updated_workflow, updated_incident, updated_plan

    def reconcile_retired_generation_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        reference: str | None,
        reconciled_at: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        """Terminalize a workflow whose incident has moved to a later generation.

        Unlike ``reconcile_restored_workflow`` this takes no expected
        ``updated_at``. A retired generation is still being dispatched, and each
        dispatch renews its execution lease and stamps the row, so an
        ``updated_at`` precondition would never hold. The compare-and-set is on
        ``fencing_token`` -- the value that decides whether the record is behind
        its incident -- and every other condition is re-derived inside this
        transaction by ``retired_generation_records``.

        The write releases the execution lease rather than requiring it. That is
        the point of a revocation: the lease holder is the dispatcher that keeps
        the retired record alive, and its next ``save_workflow_if_leased`` is
        meant to fail.
        """

        with self._state_transaction(f"retired_generation/{workflow_request_id}"):
            getter = getattr(self, "_get_for_update", self._get)
            workflow = getter("workflow", workflow_request_id)
            incident = getter("incident", workflow.incident_id)
            successor = getter("workflow", successor_workflow_id)
            updated_workflow, updated_incident = retired_generation_records(
                workflow,
                incident,
                successor,
                self._list("remote_command"),
                expected_fencing_token=expected_fencing_token,
                reference=reference,
                reconciled_at=reconciled_at,
            )
            self._put("workflow", workflow_request_id, updated_workflow)
            self._put("incident", incident.incident_id, updated_incident)
            return updated_workflow, updated_incident

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
