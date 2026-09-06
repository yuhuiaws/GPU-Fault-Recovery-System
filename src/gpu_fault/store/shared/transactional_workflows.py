from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping

from gpu_fault.models import FaultIncident, RecoveryPlan, WorkflowRequest
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.retired_generation import retired_generation_records
from gpu_fault.workflow_resolution import reconciled_restore_records
from gpu_fault.store.shared.preemption import preemption_pending_update

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand

LOGGER = logging.getLogger(__name__)


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
    list_remote_commands: Callable[..., list[RemoteActionCommand]]

    # How many ``incident_by_event`` links pointed at a record that was not
    # there when a duplicate event arrived, and were repaired by rebuilding
    # (F-B7). Per store instance; read by the metrics family. A plain class
    # default (not an annotated contract attribute): it is state this mixin
    # owns, not something the composed store has to supply.
    stale_event_link_repairs = 0

    def _duplicate_event_records(
        self, event_id: str
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """The incident and workflow an already-ingested event resolves to.

        ``None`` means "treat this event as new". That covers the plain case
        of no link, and the dirty case: a link whose incident is gone, whose
        incident never got a workflow pointer, or whose workflow row is
        missing. The fast path used an unguarded ``_get`` there, so a dangling
        pointer made every re-post of the event fail inside the transaction --
        and a re-post is exactly the retry path, so the event became a poison
        message (P1-57F, P1-69D). Falling through is the idempotent repair:
        the caller rebuilds from the group state and re-links this
        ``event_id``, and ``_link`` is an upsert, so the dirty link is
        overwritten rather than left for the next retry to trip on.

        The reads take the row lock (``_locked_optional``): a duplicate return
        is read-only, but a fall-through writes, and the group incident it will
        read next may be the same row.
        """

        incident_id = self._get_link("incident_by_event", event_id)
        if incident_id is None:
            return None
        incident = self._locked_optional("incident", incident_id)
        workflow = (
            self._locked_optional("workflow", incident.workflow_request_id)
            if incident is not None and incident.workflow_request_id
            else None
        )
        if incident is not None and workflow is not None:
            return incident, workflow
        self.stale_event_link_repairs += 1
        LOGGER.warning(
            "incident_by_event link for %s points at incident %s whose %s is "
            "missing; rebuilding the event instead of failing the re-post",
            event_id,
            incident_id,
            (
                "record"
                if incident is None
                else "workflow pointer"
                if not incident.workflow_request_id
                else f"workflow {incident.workflow_request_id}"
            ),
        )
        return None

    def _locked_optional(self, kind: str, key: str) -> Any:
        """Read a record for update inside the current transaction, or None.

        The merge paths used ``_get_optional`` -- no row lock -- so an executor
        holding the row could commit between the merge's read and its blind
        write, and one of the two writers lost (P0-78A). Backends without row
        locks (sqlite, memory) serialize on their process-wide lock instead.
        """

        getter = getattr(self, "_get_for_update", self._get)
        try:
            return getter(kind, key)
        except NotFoundError:
            return None

    @staticmethod
    def _merged(
        existing: WorkflowRequest | None, workflow: WorkflowRequest
    ) -> WorkflowRequest:
        """Stamp a merge into an existing row so leased writers can see it (F-B1)."""

        if existing is None or existing.request_id != workflow.request_id:
            return workflow
        return workflow.model_copy(
            update={"merge_revision": existing.merge_revision + 1}
        )

    def amend_workflow(
        self,
        request_id: str,
        updates: Mapping[str, object],
    ) -> WorkflowRequest:
        with self._state_transaction(f"workflow/{request_id}"):
            current = self._locked_optional("workflow", request_id)
            if current is None:
                raise NotFoundError(request_id)
            amended: WorkflowRequest = current.model_copy(
                update={
                    **dict(updates),
                    "merge_revision": current.merge_revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._put("workflow", request_id, amended)
            return amended

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
            existing = self._locked_optional("workflow", workflow.request_id)
            workflow = self._merged(existing, workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
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
        expected_execution_epoch: int,
        expected_workflow_updated_at: datetime | None = None,
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
                    # Only this workflow's commands (F-J5): the full-table load
                    # ran inside the lock-holding transaction (P1-78D).
                    self.list_remote_commands(
                        workflow_request_ids=[workflow_request_id]
                    ),
                    expected_fencing_token=expected_fencing_token,
                    expected_execution_epoch=expected_execution_epoch,
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
                self.list_remote_commands(workflow_request_ids=[workflow_request_id]),
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
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate[0], duplicate[1], False
            incident, workflow = builder()
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
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

    def _stamp_preemption_pending(self, successor: WorkflowRequest) -> None:
        """In the merge transaction, hold the predecessor a successor preempts (F-C1)."""

        if not (successor.preempt_predecessor and successor.predecessor_workflow_id):
            return
        predecessor = self._locked_optional(
            "workflow", successor.predecessor_workflow_id
        )
        stamped = preemption_pending_update(successor, predecessor)
        if stamped is not None:
            self._put("workflow", stamped.request_id, stamped)

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
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate
            incident_id = self._get_link("replacement_fault_group", group_key)
            existing_incident = (
                self._locked_optional("incident", incident_id)
                if incident_id is not None
                else None
            )
            existing_workflow = (
                self._locked_optional(
                    "workflow",
                    existing_incident.workflow_request_id,
                )
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            workflow = self._merged(existing_workflow, workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
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
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate
            incident_id = self._get_link("sxid_fault_group", group_key)
            existing_incident = (
                self._locked_optional("incident", incident_id)
                if incident_id is not None
                else None
            )
            existing_workflow = (
                self._locked_optional(
                    "workflow",
                    existing_incident.workflow_request_id,
                )
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            workflow = self._merged(existing_workflow, workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
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
