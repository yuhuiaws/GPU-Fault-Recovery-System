"""Closing an ESCALATED incident: by the restore that freed its node, or by hand.

An incident goes ESCALATED when its remediation is handed to an operator: the
workflow ran out of lifetime (F-N1), or the support escalation delivered its
ticket. ``NodeConflictService.reopen_if_terminal`` keeps such a pair as the
node's merge target on purpose -- while the node is with an operator, a new
fault is recorded on the incident instead of opening a second remediation.
But nothing ever moved the incident out of ESCALATED: when the node was later
restored through *another* incident (the support escalation's validated
restore), the first one stayed ESCALATED and kept absorbing every later fault
on the node as record-only (DESTR-018, product gap fixed 2026-09-08).

Two exits, one write path:

* ``on_terminal`` -- an executor ``TerminalHook``. A SUCCEEDED workflow that
  restored a node (``restores_node``) closes every other ESCALATED incident of
  the cluster whose nodes it covers and that has no open workflow.
* ``close_incident`` -- the operator API (``POST /v1/incidents/{id}/close``
  and ``gpu-fault-admin workflow-reconcile --close-incident``).

Both write RECOVERED through ``save_incident(expected=)`` (compare-and-set,
never a blind overwrite), append the reason to the incident, record one audit
event on the incident's last workflow (the incident row has no event log of its
own), and retire the incident's markers. A RECOVERED pair is no longer a merge
target, so the next fault on the node opens its own remediation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    bounded_reasons,
    record_workflow_event,
    workflow_is_open,
)
from gpu_fault.operation_registry import (
    CONTAINMENT_ONLY_OPERATIONS,
    DESTRUCTIVE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
)
from gpu_fault.orchestration.workflow_merge import never_executed_operator_block
from gpu_fault.store.shared.errors import NotFoundError, StaleWriteError

if TYPE_CHECKING:
    from gpu_fault.store.contracts import ControlPlaneStore

LOGGER = logging.getLogger(__name__)

# Actor stamped on the audit event of an automatic close; an operator close is
# signed by the operator identity the caller supplied.
AUTO_CLOSE_ACTOR = "incident-closure:restore"
# The only state an operator may close by hand. QUARANTINED means the node is
# still isolated and needs a validated restore workflow, not a signature; the
# planning states have a workflow that will end them on its own.
OPERATOR_CLOSABLE_STATES = frozenset({IncidentState.ESCALATED})
CLOSED_BY_RESTORE = "restore"
CLOSED_BY_OPERATOR = "operator"


class IncidentNotClosable(ValueError):
    """The incident is not in a state an operator may close; ``str()`` says why."""


def restores_node(workflow: WorkflowRequest) -> bool:
    """Whether a SUCCEEDED ``workflow`` put its node back into service.

    Judged on the registry classes the executor's ``_diagnostic_only`` uses
    (destructive -- containment included -- and node-wide operations): a
    workflow that only observed the node (evidence, diagnostics, validation)
    concluded nothing about it. A completed containment must have been
    released by ``RESTORE_SCHEDULING``, the one release operation the
    executor's ``_terminal_incident_state`` also names: a cordon that is still
    in place is not a restored node.
    """

    if workflow.status is not WorkflowStatus.SUCCEEDED:
        return False
    completed = set(workflow.completed_operations)
    if not completed & (DESTRUCTIVE_OPERATIONS | NODE_WIDE_RECOVERY_OPERATIONS):
        return False
    contained = completed & CONTAINMENT_ONLY_OPERATIONS
    if contained and WorkflowOperation.RESTORE_SCHEDULING not in contained:
        return False
    return True


class IncidentClosureService:
    def __init__(self, store: ControlPlaneStore) -> None:
        self.store = store
        # Rendered on /metrics as gpu_fault_incident_operator_closed_total and
        # gpu_fault_incident_auto_closed_by_restore_total.
        self.operator_closed_total = 0
        self.auto_closed_by_restore_total = 0

    # ------------------------------------------------------------ operator

    def preview(self, incident_id: str) -> dict[str, Any]:
        """The verdict ``close_incident`` would reach, without writing (dry run)."""

        try:
            incident = self.store.get_incident(incident_id)
        except NotFoundError:
            return {
                "incident_id": incident_id,
                "state": None,
                "closable": False,
                "refusal": "incident not found",
                "open_workflow_id": None,
            }
        refusal, open_workflow = self._refusal(incident)
        return {
            "incident_id": incident_id,
            "state": incident.state.value,
            "closable": refusal is None
            and incident.state is not IncidentState.RECOVERED,
            "refusal": refusal,
            "open_workflow_id": (
                open_workflow.request_id if open_workflow is not None else None
            ),
        }

    def close_incident(
        self,
        incident_id: str,
        *,
        reason: str,
        operator: str,
        reference: str | None = None,
    ) -> tuple[FaultIncident, bool]:
        """Close ``incident_id`` RECOVERED on an operator's authority.

        Returns the incident and whether this call changed it: an incident
        that is already RECOVERED is returned unchanged (idempotent). Raises
        :class:`NotFoundError` for an unknown id, :class:`IncidentNotClosable`
        when the state is not ESCALATED or a workflow of the incident is still
        open, and :class:`StaleWriteError` when the row moved between the read
        and the compare-and-set (the caller retries).
        """

        incident = self.store.get_incident(incident_id)
        if incident.state is IncidentState.RECOVERED:
            return incident, False
        refusal, _ = self._refusal(incident)
        if refusal is not None:
            raise IncidentNotClosable(refusal)
        closed = self._close(
            incident,
            reason=f"operator closed: {reason} by {operator}",
            actor=operator,
            kind=WorkflowEventKind.OPERATOR_RECONCILED,
            details={
                "closed_by": CLOSED_BY_OPERATOR,
                "operator": operator,
                "reference": reference,
            },
        )
        self.operator_closed_total += 1
        LOGGER.info(
            "incident %s closed by operator %s: %s", incident_id, operator, reason
        )
        return closed, True

    # ------------------------------------------------------- terminal hook

    def on_terminal(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
        steps: list[WorkflowStepSpec],
    ) -> list[str]:
        """Executor ``TerminalHook``: a restored node closes the ESCALATED
        incidents that were waiting on it. Returns the incident ids closed.

        Only a SUCCEEDED workflow whose own incident ended RECOVERED and that
        restored the node (``restores_node``) qualifies; the executor's
        derivation of RECOVERED already says no isolation is left. An
        ESCALATED incident is closed only when every one of its nodes was
        restored and none of its workflows is still open. A row that moved
        under the compare-and-set is skipped: whatever moved it owns it now.
        """

        if (
            incident is None
            or workflow.status is not WorkflowStatus.SUCCEEDED
            or incident.state is not IncidentState.RECOVERED
            or not restores_node(workflow)
        ):
            return []
        restored = {node for step in steps for node in step.node_ids} or set(
            incident.node_ids
        )
        if not restored:
            return []
        closed: list[str] = []
        reason = f"node restored via incident {incident.incident_id}"
        for other in self.store.list_incidents_by_state(
            incident.cluster_id, {IncidentState.ESCALATED}, node_ids=restored
        ):
            if other.incident_id == incident.incident_id:
                continue
            if not other.node_ids or not set(other.node_ids) <= restored:
                continue
            if self._open_workflow(other) is not None:
                continue
            try:
                self._close(
                    other,
                    reason=reason,
                    actor=AUTO_CLOSE_ACTOR,
                    kind=WorkflowEventKind.TERMINAL,
                    details={
                        "closed_by": CLOSED_BY_RESTORE,
                        "restored_by_incident_id": incident.incident_id,
                        "restored_by_workflow_id": workflow.request_id,
                    },
                )
            except StaleWriteError:
                LOGGER.info(
                    "incident %s moved while workflow %s was closing it; left alone",
                    other.incident_id,
                    workflow.request_id,
                )
                continue
            self.auto_closed_by_restore_total += 1
            closed.append(other.incident_id)
            LOGGER.info(
                "incident %s closed: %s (workflow %s)",
                other.incident_id,
                reason,
                workflow.request_id,
            )
        return closed

    # --------------------------------------------------------------- shared

    def _refusal(
        self, incident: FaultIncident
    ) -> tuple[str | None, WorkflowRequest | None]:
        if incident.state is IncidentState.RECOVERED:
            return None, None
        if incident.state not in OPERATOR_CLOSABLE_STATES:
            return (
                f"incident {incident.incident_id} is {incident.state.value}; only an "
                "ESCALATED incident can be closed by an operator (a QUARANTINED "
                "node is released by a validated restore workflow, a planning "
                "state by the workflow that ends it)",
                None,
            )
        open_workflow = self._open_workflow(incident)
        if open_workflow is not None:
            status = open_workflow.status.value
            if open_workflow.blocked_kind is not None:
                status = f"{status}/{open_workflow.blocked_kind.value}"
            return (
                f"incident {incident.incident_id} still has an open workflow "
                f"{open_workflow.request_id} ({status}); wait for it to end or "
                "reconcile it with gpu-fault-admin workflow-reconcile first",
                open_workflow,
            )
        return None, None

    def _open_workflow(self, incident: FaultIncident) -> WorkflowRequest | None:
        """A workflow of ``incident`` that still gates the node, if any.

        Executable rows and BLOCKED rows that occupy their node
        (``workflow_is_open``), except a BLOCKED(NEEDS_OPERATOR) row that never
        ran a step: it holds nothing on the node (C-03) and the next fault
        replaces it in place.
        """

        candidates: list[WorkflowRequest] = []
        if incident.workflow_request_id:
            try:
                candidates.append(self.store.get_workflow(incident.workflow_request_id))
            except NotFoundError:
                pass
        if incident.node_ids:
            for owner, workflow in self.store.list_active_workflow_incidents(
                incident.cluster_id, node_ids=set(incident.node_ids)
            ):
                if owner.incident_id == incident.incident_id:
                    candidates.append(workflow)
        seen: set[str] = set()
        for workflow in candidates:
            if workflow.request_id in seen:
                continue
            seen.add(workflow.request_id)
            if workflow_is_open(
                workflow.status, workflow.blocked_kind
            ) and not never_executed_operator_block(workflow):
                return workflow
        return None

    def _close(
        self,
        incident: FaultIncident,
        *,
        reason: str,
        actor: str,
        kind: WorkflowEventKind,
        details: Mapping[str, Any],
    ) -> FaultIncident:
        """The one write path: CAS the incident, then audit and markers.

        The incident write is the transition; the audit event and the marker
        retirement follow it and are isolated like an ``on_terminal`` hook --
        a failure there is logged and does not unwind a close that landed.
        """

        now = datetime.now(timezone.utc)
        closed = incident.model_copy(
            update={
                "state": IncidentState.RECOVERED,
                "reasons": bounded_reasons([*incident.reasons, reason]),
                "updated_at": now,
            }
        )
        self.store.save_incident(closed, expected=incident)
        try:
            self._record_event(
                incident, reason=reason, actor=actor, kind=kind, details=details, at=now
            )
        except Exception:  # noqa: BLE001 - the close already landed
            LOGGER.exception(
                "recording the close of incident %s on workflow %s",
                incident.incident_id,
                incident.workflow_request_id,
            )
        try:
            retire_markers_for_incident(
                self.store,
                incident.incident_id,
                reason=reason,
                retired_by=actor,
                now=now,
            )
        except Exception:  # noqa: BLE001 - the close already landed
            LOGGER.exception(
                "retiring markers of closed incident %s", incident.incident_id
            )
        return closed

    def _record_event(
        self,
        incident: FaultIncident,
        *,
        reason: str,
        actor: str,
        kind: WorkflowEventKind,
        details: Mapping[str, Any],
        at: datetime,
    ) -> None:
        """Append the close to the incident's last workflow, compare-and-set.

        One re-read on a stale write: a merge may have restamped the row (a
        never-run BLOCKED record is still a merge target) between the read
        and the write. A missing row is not an error -- the incident had no
        workflow left to carry the event.
        """

        if not incident.workflow_request_id:
            return
        for _attempt in range(2):
            try:
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except NotFoundError:
                return
            recorded = record_workflow_event(
                workflow,
                kind,
                code=WorkflowEventCode.INCIDENT_CLOSED.value,
                reason=reason,
                actor=actor,
                status=workflow.status.value,
                details={
                    "incident_id": incident.incident_id,
                    "previous_incident_state": incident.state.value,
                    "incident_state": IncidentState.RECOVERED.value,
                    **{
                        key: value
                        for key, value in details.items()
                        if value is not None
                    },
                },
                at=at,
            )
            try:
                self.store.save_workflow(recorded, expected=workflow)
                return
            except StaleWriteError:
                continue
        LOGGER.warning(
            "workflow %s kept moving; the close of incident %s is on the incident "
            "and its markers only",
            incident.workflow_request_id,
            incident.incident_id,
        )
