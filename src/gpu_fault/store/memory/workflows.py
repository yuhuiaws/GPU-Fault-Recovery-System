from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Sequence

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryPlan,
    WorkflowEvent,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    append_workflow_event,
    record_operator_event,
    workflow_is_open,
)
from gpu_fault.store.contracts import ACTIVE_WORKFLOW_INCIDENTS_LIMIT
from gpu_fault.store.shared.errors import (
    NotFoundError,
    RemediationBudgetError,
    StaleFencingTokenError,
    StaleWriteError,
    WorkflowLeaseError,
    WorkflowMergedError,
)
from gpu_fault.store.shared.preemption import preemption_pending_update
from gpu_fault.store.shared.record_guards import (
    record_matches_expected,
    stale_incident_versions,
)
from gpu_fault.store.shared.remediation_budgets import (
    apply_remediation_budget,
    blocked_by_remediation_budget,
    extend_remediation_budget,
)
from gpu_fault.store.shared.transactional_workflows import (
    incident_pointer_moved,
    lease_extension_due,
    stale_workflow_versions,
    workflow_matches_expected,
)
from gpu_fault.store.shared.workflow_scan import dispatch_order_key, held_reason
from gpu_fault.workflow_resolution import (
    reconciled_restore_records,
    retired_generation_records,
)

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand

LOGGER = logging.getLogger(__name__)


# The two statuses in which an aggregated-but-unnamed workflow waits forever
# (F-B3): not RUNNING, which the executor owns, and not terminal.
ORPHAN_WORKFLOW_STATUSES = frozenset(
    {WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING}
)


class MemoryWorkflowMixin:
    # Attributes supplied by the composed concrete implementation.
    _incident_by_event: Any
    _incidents: Any
    _plans: dict[str, RecoveryPlan]
    _remote_commands: dict[str, RemoteActionCommand]
    _workflows: Any

    get_plan: Callable[[str], RecoveryPlan]
    list_remote_commands: Callable[..., list[RemoteActionCommand]]
    _lock: Any
    _replacement_fault_groups: Any
    _sxid_fault_groups: Any

    # Same counter and meaning as ``TransactionalWorkflowMixin`` (F-B7): read
    # by ``gpu_fault_ingest_stale_event_link_repairs_total``.
    stale_event_link_repairs = 0

    def _duplicate_event_records(
        self, event_id: str
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """The pair an already-ingested event resolves to, or None to rebuild.

        The memory copy of ``TransactionalWorkflowMixin._duplicate_event_records``
        (C-05): ``tests/store/test_merge_duplicate_event_contract.py`` pins the
        three backends to the same answer. The three fast paths here used to
        raise (``RuntimeError`` / ``NotFoundError``) on a dangling pointer, so
        every family test on the in-memory store -- and the ``--dry-run``
        adapter path -- saw a poison message where PostgreSQL rebuilt.
        """

        incident_id = self._incident_by_event.get(event_id)
        if incident_id is None:
            return None
        incident = self._incidents.get(incident_id)
        workflow = (
            self._workflows.get(incident.workflow_request_id)
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

    def _stamped(
        self,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> WorkflowRequest:
        """See ``TransactionalWorkflowMixin._stamped`` (C-01): an adopted row
        is bumped from its *stored* revision and its incident goes through the
        generation guard; the process lock stands in for the row lock."""

        if (
            existing_incident is None
            or existing_incident.incident_id != incident.incident_id
        ):
            stored_incident = self._incidents.get(incident.incident_id)
            if stored_incident is not None:
                stale = stale_incident_versions(stored_incident, incident)
                if stale is not None:
                    raise stale
        stored_workflow = (
            existing_workflow
            if existing_workflow is not None
            and existing_workflow.request_id == workflow.request_id
            else self._workflows.get(workflow.request_id)
        )
        if stored_workflow is None:
            return workflow
        return workflow.model_copy(
            update={"merge_revision": stored_workflow.merge_revision + 1}
        )

    def save_incident(
        self,
        incident: FaultIncident,
        *,
        expected: FaultIncident | None = None,
        extra_event_ids: Sequence[str] = (),
    ) -> None:
        """See ``WorkflowStore.save_incident`` (architecture review, item D1)."""

        with self._lock:
            current = self._incidents.get(incident.incident_id)
            if expected is not None:
                if not record_matches_expected(current, expected):
                    raise StaleWriteError(
                        f"incident/{incident.incident_id} changed since it was read"
                    )
            elif current is not None:
                stale = stale_incident_versions(current, incident)
                if stale is not None:
                    raise stale
            self._incidents[incident.incident_id] = incident
            self._incident_by_event[incident.event_id] = incident.incident_id
            for event_id in extra_event_ids:
                self._incident_by_event[event_id] = incident.incident_id

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        *,
        extra_event_ids: Sequence[str] = (),
    ) -> None:
        if incident.workflow_request_id != workflow.request_id:
            raise ValueError("incident workflow pointer does not match workflow")
        if workflow.incident_id != incident.incident_id:
            raise ValueError("workflow incident pointer does not match incident")
        with self._lock:
            existing = self._workflows.get(workflow.request_id)
            if existing is not None:
                workflow = workflow.model_copy(
                    update={"merge_revision": existing.merge_revision + 1}
                )
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)
            self._incident_by_event[incident.event_id] = incident.incident_id
            for event_id in extra_event_ids:
                self._incident_by_event[event_id] = incident.incident_id

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
        actor: str | None = None,
        approval: Mapping[str, object] | None = None,
    ) -> tuple[WorkflowRequest, FaultIncident, RecoveryPlan]:
        with self._lock:
            workflow = self.get_workflow(workflow_request_id)
            incident = self.get_incident(workflow.incident_id)
            successor = self.get_workflow(successor_workflow_id)
            if not workflow.source_plan_id:
                raise ValueError("workflow has no source recovery plan")
            source_plan = self.get_plan(workflow.source_plan_id)
            updated_workflow, updated_incident, updated_plan = (
                reconciled_restore_records(
                    workflow,
                    incident,
                    successor,
                    source_plan,
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
            # Same event, same write as the shared transactional mixin (I1):
            # a refused reconcile raised above and records nothing.
            updated_workflow = record_operator_event(
                updated_workflow,
                WorkflowEventKind.OPERATOR_RECONCILED,
                actor=actor,
                reference=reference,
                previous_status=workflow.status,
                at=reconciled_at,
                details={
                    "successor_workflow_id": successor_workflow_id,
                    "expected_fencing_token": expected_fencing_token,
                    "expected_execution_epoch": expected_execution_epoch,
                    **dict(approval or {}),
                },
            )
            self._workflows[workflow_request_id] = updated_workflow
            self._incidents[incident.incident_id] = updated_incident
            self._plans[source_plan.plan_id] = updated_plan
            return updated_workflow, updated_incident, updated_plan

    def reconcile_retired_generation_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        reference: str | None,
        reconciled_at: datetime,
        actor: str | None = None,
        approval: Mapping[str, object] | None = None,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        """Terminalize a workflow whose incident moved to a later generation.

        See ``TransactionalWorkflowMixin.reconcile_retired_generation_workflow``
        for why this takes no expected ``updated_at`` and why it releases the
        execution lease instead of requiring it.
        """

        with self._lock:
            workflow = self.get_workflow(workflow_request_id)
            incident = self.get_incident(workflow.incident_id)
            successor = self.get_workflow(successor_workflow_id)
            updated_workflow, updated_incident = retired_generation_records(
                workflow,
                incident,
                successor,
                self.list_remote_commands(workflow_request_ids=[workflow_request_id]),
                expected_fencing_token=expected_fencing_token,
                reference=reference,
                reconciled_at=reconciled_at,
                actor=actor,
                approval=approval,
            )
            self._workflows[workflow_request_id] = updated_workflow
            self._incidents[incident.incident_id] = updated_incident
            return updated_workflow, updated_incident

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
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate[0], duplicate[1], False
            incident, workflow = builder()
            workflow = self._stamped(None, None, incident, workflow)
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)
            self._incident_by_event[event_id] = incident.incident_id
            self._incident_by_event[incident.event_id] = incident.incident_id
            return incident, workflow, True

    def _stamp_preemption_pending(self, successor: WorkflowRequest) -> None:
        if not (successor.preempt_predecessor and successor.predecessor_workflow_id):
            return
        stamped = preemption_pending_update(
            successor, self._workflows.get(successor.predecessor_workflow_id)
        )
        if stamped is not None:
            self._workflows[stamped.request_id] = stamped

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
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate
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
            workflow = self._stamped(
                existing_incident, existing_workflow, incident, workflow
            )
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)
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
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate
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
            workflow = self._stamped(
                existing_incident, existing_workflow, incident, workflow
            )
            self._incidents[incident.incident_id] = incident
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)
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

    def has_workflow_successor(self, predecessor_workflow_id: str) -> bool:
        with self._lock:
            return any(
                workflow.predecessor_workflow_id == predecessor_workflow_id
                for workflow in self._workflows.values()
            )

    def list_orphan_workflows(
        self, *, created_before: datetime, limit: int = 1000
    ) -> list[WorkflowRequest]:
        """See ``ControlPlaneStore.list_orphan_workflows``.

        ``tests/store/test_orphan_workflow_inspection.py`` pins the three
        backends to the same answer, so a change here has to move the SQLite
        and Postgres predicates too.
        """

        with self._lock:
            named_as_predecessor = {
                workflow.predecessor_workflow_id
                for workflow in self._workflows.values()
                if workflow.predecessor_workflow_id is not None
            }
            orphans = [
                workflow
                for workflow in self._workflows.values()
                if workflow.status in ORPHAN_WORKFLOW_STATUSES
                and workflow.created_at < created_before
                and workflow.request_id not in named_as_predecessor
                and self._incident_names_someone_else(workflow)
            ]
        orphans.sort(key=lambda item: (item.created_at, item.request_id))
        return orphans[:limit]

    def _incident_names_someone_else(self, workflow: WorkflowRequest) -> bool:
        incident = self._incidents.get(workflow.incident_id)
        return (
            incident is None
            or (incident.workflow_request_id or "") != workflow.request_id
        )

    def list_incidents_with_missing_workflow(
        self, *, limit: int = 1000
    ) -> list[FaultIncident]:
        with self._lock:
            dangling = [
                incident
                for incident in self._incidents.values()
                if incident.workflow_request_id
                and incident.workflow_request_id not in self._workflows
            ]
        dangling.sort(key=lambda item: (item.created_at, item.incident_id))
        return dangling[:limit]

    def list_incidents_by_state(
        self,
        cluster_id: str,
        states: Collection[IncidentState],
        *,
        node_ids: set[str] | None = None,
        limit: int = ACTIVE_WORKFLOW_INCIDENTS_LIMIT,
    ) -> list[FaultIncident]:
        if node_ids is not None and not node_ids:
            return []
        wanted = {IncidentState(state) for state in states}
        with self._lock:
            matches = [
                incident
                for incident in self._incidents.values()
                if incident.cluster_id == cluster_id
                and incident.state in wanted
                and (node_ids is None or set(incident.node_ids) & node_ids)
            ]
        matches.sort(key=lambda item: (item.updated_at, item.incident_id), reverse=True)
        return matches[:limit]

    def save_workflow(
        self,
        workflow: WorkflowRequest,
        *,
        expected: WorkflowRequest | None = None,
    ) -> None:
        """See ``WorkflowStore.save_workflow`` (store review 2026-09-07, item B)."""

        with self._lock:
            current = self._workflows.get(workflow.request_id)
            if expected is not None:
                if not workflow_matches_expected(current, expected):
                    raise StaleWriteError(
                        f"workflow/{workflow.request_id} changed since it was read"
                    )
            elif current is not None:
                stale = stale_workflow_versions(current, workflow)
                if stale is not None:
                    raise stale
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)

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
        dispatchable_at: datetime | None = None,
        exclude_request_ids: Collection[str] = (),
        after: WorkflowRequest | None = None,
    ) -> list[WorkflowRequest]:
        if after is not None and dispatchable_at is None:
            raise ValueError(
                "the scan cursor (after) is only defined with dispatchable_at"
            )
        with self._lock:
            if dispatchable_at is not None:
                # Dispatch mode orders by when the row became eligible, not by
                # its last merge (F-A2a); the cursor pages that order (F-A2c).
                workflows = sorted(
                    self._workflows.values(),
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
                    self._workflows.values(),
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
                        lookup=self._workflows.get,
                    )
                    is None
                ]
            return workflows[:limit]

    def amend_workflow(
        self,
        request_id: str,
        updates: Mapping[str, object],
        *,
        event: WorkflowEvent | None = None,
    ) -> WorkflowRequest:
        with self._lock:
            current = self._workflows.get(request_id)
            if current is None:
                raise NotFoundError(request_id)
            amended: WorkflowRequest = current.model_copy(
                update={
                    **dict(updates),
                    "merge_revision": current.merge_revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            if event is not None:
                amended = append_workflow_event(amended, event)
            self._workflows[request_id] = amended
            return amended

    def count_held_workflows(
        self,
        statuses: set[WorkflowStatus] | None,
        *,
        dispatchable_at: datetime,
        exclude_request_ids: Collection[str] = (),
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for item in self._workflows.values():
                if statuses is not None and item.status not in statuses:
                    continue
                reason = held_reason(
                    item,
                    dispatchable_at=dispatchable_at,
                    exclude_request_ids=exclude_request_ids,
                    lookup=self._workflows.get,
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

        with self._lock:
            counts = {status: 0 for status in WorkflowStatus}
            for workflow in self._workflows.values():
                counts[workflow.status] += 1
            return counts

    def incident_state_counts(self) -> dict[IncidentState, int]:
        """Count every persisted incident by state (server-side aggregate
        for the /metrics incident gauge; ESCALATED is the operator queue)."""

        with self._lock:
            counts = {state: 0 for state in IncidentState}
            for incident in self._incidents.values():
                counts[incident.state] += 1
            return counts

    def blocked_workflows_without_verified_restore(self) -> int:
        """Count the BLOCKED workflows whose GPU node is still held.

        See the SQLite implementation for why the lifetime BLOCKED count cannot
        answer this. The predicate below is
        ``workflow_resolution.verified_restore_successor`` negated;
        ``tests/store/test_blocked_backlog_gauge.py`` pins the three backends
        and that function to the same answer, so a change to one has to move all
        of them.
        """

        with self._lock:
            blocked = [
                workflow
                for workflow in self._workflows.values()
                if workflow.status is WorkflowStatus.BLOCKED
            ]
            return sum(
                1 for workflow in blocked if not self._restore_is_verified(workflow)
            )

    def _restore_is_verified(self, workflow: WorkflowRequest) -> bool:
        incident = self._incidents.get(workflow.incident_id)
        if incident is None or incident.state is not IncidentState.RECOVERED:
            return False
        successor_id = incident.workflow_request_id
        if not successor_id or successor_id == workflow.request_id:
            return False
        successor = self._workflows.get(successor_id)
        if successor is None:
            return False
        return (
            successor.incident_id == workflow.incident_id
            and successor.status is WorkflowStatus.SUCCEEDED
            and successor.fencing_token == workflow.fencing_token
            and incident.fencing_token == workflow.fencing_token
            and WorkflowOperation.RESTORE_SCHEDULING in successor.completed_operations
        )

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
        limit: int = ACTIVE_WORKFLOW_INCIDENTS_LIMIT,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        with self._lock:
            matches = []
            for workflow in self._workflows.values():
                # Executable rows plus BLOCKED rows that still occupy their
                # node (F-A4); mirrors the SQL backends.
                if not workflow_is_open(workflow.status, workflow.blocked_kind):
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
            )[:limit]

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
        remediation_budget_claims: dict[str, int] | None = None,
    ) -> WorkflowRequest:
        with self._lock:
            workflow = self.get_workflow(request_id)
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
                        list(self._workflows.values()),
                        remediation_budget_claims,
                        now=claimed_at,
                    )
                except RemediationBudgetError as exc:
                    self._workflows[request_id] = blocked_by_remediation_budget(
                        workflow,
                        str(exc),
                        scope=exc.scope,
                        now=claimed_at,
                    )
                    raise
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
                    **(
                        {"status": WorkflowStatus.RUNNING}
                        if remediation_budget_claims is not None
                        else {}
                    ),
                }
            )
            self._workflows[request_id] = workflow
            return workflow

    def extend_remediation_budget(
        self,
        request_id: str,
        executor_id: str,
        claims: dict[str, int],
        *,
        now: datetime | None = None,
    ) -> WorkflowRequest:
        with self._lock:
            workflow = self.get_workflow(request_id)
            at = now or datetime.now(timezone.utc)
            if (
                workflow.execution_owner_id != executor_id
                or workflow.execution_lease_expires_at is None
                or workflow.execution_lease_expires_at <= at
            ):
                raise WorkflowLeaseError("workflow execution lease is stale")
            workflow = extend_remediation_budget(
                workflow, list(self._workflows.values()), claims, now=at
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
            if not lease_extension_due(
                workflow, renewed_at=renewed_at, lease_duration=lease_duration
            ):
                return workflow  # store review 2026-09-07, item F1
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
            if current.merge_revision != workflow.merge_revision:
                raise WorkflowMergedError("workflow was merged since it was read")
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)

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
            if current.merge_revision != workflow.merge_revision:
                raise WorkflowMergedError("workflow was merged since it was read")
            self._workflows[workflow.request_id] = workflow
            self._stamp_preemption_pending(workflow)
            current_incident = self._incidents.get(incident.incident_id)
            if incident_pointer_moved(current_incident, incident):
                # Same rule as PostgreSQL (C-02).
                LOGGER.warning(
                    "incident %s moved its workflow pointer to %s since %s read "
                    "it; keeping the merged incident and writing only the workflow",
                    incident.incident_id,
                    current_incident.workflow_request_id,
                    workflow.request_id,
                )
                return
            self._incidents[incident.incident_id] = incident
            self._incident_by_event[incident.event_id] = incident.incident_id
