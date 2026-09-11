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

* ``on_terminal`` -- an executor ``TerminalHook``. Every SUCCEEDED workflow whose
  incident ended RECOVERED retires that incident's markers (ARCH-I4); one that
  restored a node (``restores_node``) also closes every other ESCALATED incident of
  the cluster whose nodes it covers and that has no open workflow.
* ``close_incident`` -- the operator API (``POST /v1/incidents/{id}/close``
  and ``gpu-fault-admin workflow-reconcile --close-incident``). A QUARANTINED
  incident is closable this way only with ``NodeIsolationEvidence`` for every
  node it names, proving the isolation it owned is gone (``isolation_verdict``):
  the node is schedulable, carries no ``gpu-fault.io/quarantined`` taint with
  this incident's value and no isolation annotation naming this incident. The
  API has no kubeconfig and never supplies evidence; the admin CLI reads it
  through the site's GPU kubeconfig (2026-09-10: 24 incidents sat QUARANTINED
  on nodes whose isolation a later incident or a cleanup had released).

Both write RECOVERED through ``save_incident(expected=)`` (compare-and-set,
never a blind overwrite), append the reason to the incident, record one audit
event on the incident's last workflow (the incident row has no event log of its
own), and retire the incident's markers. A RECOVERED pair is no longer a merge
target, so the next fault on the node opens its own remediation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.compile_blocked import close_settled_incident_blocked_workflow
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

# Re-exported on purpose: the QUARANTINED exit path is built by this module's
# sibling, and ``submit-remediation --disposition restore`` imports it inside
# the CPU Pod. Naming it here keeps it in the control-plane wheel closure
# (component_wheels builds wheels from imports, not from the package tree).
from gpu_fault.orchestration.validated_restore import (
    build_validated_restore_workflow as build_validated_restore_workflow,
)

# Same reason: ``gpu-fault-admin collector-outbox`` imports the outbox
# maintenance builder inside the CPU Pod.
from gpu_fault.orchestration.collector_outbox_maintenance import (
    build_collector_outbox_workflow as build_collector_outbox_workflow,
)
from gpu_fault.store.shared.errors import NotFoundError, StaleWriteError

if TYPE_CHECKING:
    from gpu_fault.store.contracts import ControlPlaneStore

LOGGER = logging.getLogger(__name__)

# Actor stamped on the audit event of an automatic close; an operator close is
# signed by the operator identity the caller supplied.
AUTO_CLOSE_ACTOR = "incident-closure:restore"
# Names the terminal hook as the path that retired a RECOVERED incident's markers.
RECOVERED_RETIREMENT_ACTOR = "incident-closure:recovered"
# The only state an operator may close on a signature alone. QUARANTINED means
# the node is still isolated and needs a validated restore workflow -- unless
# node evidence shows the isolation is already gone (``EVIDENCE_CLOSABLE_STATES``);
# the planning states have a workflow that will end them on its own.
OPERATOR_CLOSABLE_STATES = frozenset({IncidentState.ESCALATED})
EVIDENCE_CLOSABLE_STATES = frozenset({IncidentState.QUARANTINED})
CLOSED_BY_RESTORE = "restore"
CLOSED_BY_OPERATOR = "operator"
# The node annotations the kubernetes adapter writes on MARK_UNSCHEDULABLE /
# QUARANTINE and clears on RESTORE_SCHEDULING (``_node_isolation_patch``).
ISOLATION_ANNOTATION_KEYS = (
    ANNOTATION_INCIDENT,
    ANNOTATION_FENCING,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
)


class IncidentNotClosable(ValueError):
    """The incident is not in a state an operator may close; ``str()`` says why."""


@dataclass(frozen=True)
class NodeIsolationEvidence:
    """What one node of a QUARANTINED incident carries right now.

    Read by the caller from the cluster (``kubectl get node``): the cordon
    flag, the value of the ``gpu-fault.io/quarantined`` taint if any, and the
    gpu-fault isolation annotations present (``ISOLATION_ANNOTATION_KEYS``,
    key -> value). ``exists`` is False for a node the cluster no longer has.
    """

    node_id: str
    unschedulable: bool = False
    quarantine_taint_value: str | None = None
    isolation_annotations: Mapping[str, str] = field(default_factory=dict)
    exists: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NodeIsolationEvidence":
        annotations = value.get("isolation_annotations") or {}
        taint = value.get("quarantine_taint_value")
        return cls(
            node_id=str(value["node_id"]),
            unschedulable=bool(value.get("unschedulable", False)),
            quarantine_taint_value=None if taint in (None, "") else str(taint),
            isolation_annotations={
                str(key): str(item)
                for key, item in dict(annotations).items()
                if key in ISOLATION_ANNOTATION_KEYS and item not in (None, "")
            },
            exists=bool(value.get("exists", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "exists": self.exists,
            "unschedulable": self.unschedulable,
            "quarantine_taint_value": self.quarantine_taint_value,
            "isolation_annotations": dict(self.isolation_annotations),
        }


def owned_quarantine_taint_values(incident_id: str) -> frozenset[str]:
    """The taint values the executor treats as this incident's own
    (``node_operations._node_isolation_patch``: the digest form it writes and
    the raw id an older release wrote)."""

    return frozenset({incident_id, quarantine_taint_value(incident_id)})


def isolation_verdict(
    incident: FaultIncident,
    evidence: Sequence[NodeIsolationEvidence],
) -> tuple[str | None, list[str]]:
    """Whether ``evidence`` proves the isolation ``incident`` owned is gone.

    Returns ``(refusal, reasons)``: ``refusal`` names the first node that still
    blocks the close, else ``reasons`` carries one line per node,
    ``isolation no longer present on node <node>``, extended with the
    isolation another incident holds on it -- that other incident owns it and
    is not this one's business, but the record should say so. Every node the
    incident names needs evidence; a node the cluster no longer has is not
    proof either way and is refused.
    """

    by_node = {item.node_id: item for item in evidence}
    owned_taints = owned_quarantine_taint_values(incident.incident_id)
    reasons: list[str] = []
    for node_id in incident.node_ids:
        node = by_node.get(node_id)
        if node is None:
            return f"no isolation evidence was supplied for node {node_id}", []
        if not node.exists:
            return (
                f"node {node_id} is not in the cluster; its isolation state "
                "cannot be verified",
                [],
            )
        if node.unschedulable:
            return f"node {node_id} is still cordoned (unschedulable)", []
        if node.quarantine_taint_value in owned_taints:
            return (
                f"node {node_id} still carries the {QUARANTINE_TAINT} taint of "
                f"incident {incident.incident_id}",
                [],
            )
        annotations = dict(node.isolation_annotations)
        owner = annotations.get(ANNOTATION_INCIDENT)
        if owner == incident.incident_id:
            return (
                f"node {node_id} still carries the gpu-fault isolation annotations "
                f"of incident {incident.incident_id}",
                [],
            )
        line = f"isolation no longer present on node {node_id}"
        others: list[str] = []
        if node.quarantine_taint_value is not None:
            others.append(
                f"{QUARANTINE_TAINT} taint {node.quarantine_taint_value} is owned by "
                + (f"incident {owner}" if owner else "another incident")
            )
        if annotations:
            others.append(
                "isolation annotations "
                + ", ".join(sorted(annotations))
                + (f" are owned by incident {owner}" if owner else " name no owner")
            )
        if others:
            line += " (" + "; ".join(others) + ")"
        reasons.append(line)
    return None, reasons


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

    def preview(
        self,
        incident_id: str,
        *,
        evidence: Sequence[NodeIsolationEvidence] | None = None,
    ) -> dict[str, Any]:
        """The verdict ``close_incident`` would reach, without writing (dry run).

        ``evidence_required`` is True for a QUARANTINED incident judged without
        node evidence: the caller that can read the nodes (the admin CLI)
        gathers ``NodeIsolationEvidence`` for ``node_ids`` on ``cluster_id``
        and asks again; ``isolation_reasons`` are the per-node lines the close
        would append.
        """

        try:
            incident = self.store.get_incident(incident_id)
        except NotFoundError:
            return {
                "incident_id": incident_id,
                "state": None,
                "closable": False,
                "refusal": "incident not found",
                "open_workflow_id": None,
                "cluster_id": None,
                "node_ids": [],
                "evidence_required": False,
                "isolation_reasons": [],
            }
        refusal, open_workflow, isolation_reasons = self._refusal(incident, evidence)
        return {
            "incident_id": incident_id,
            "state": incident.state.value,
            "closable": refusal is None
            and incident.state is not IncidentState.RECOVERED,
            "refusal": refusal,
            "open_workflow_id": (
                open_workflow.request_id if open_workflow is not None else None
            ),
            "cluster_id": incident.cluster_id,
            "node_ids": list(incident.node_ids),
            "evidence_required": (
                incident.state in EVIDENCE_CLOSABLE_STATES and evidence is None
            ),
            "isolation_reasons": isolation_reasons,
        }

    def close_incident(
        self,
        incident_id: str,
        *,
        reason: str,
        operator: str,
        reference: str | None = None,
        evidence: Sequence[NodeIsolationEvidence] | None = None,
    ) -> tuple[FaultIncident, bool]:
        """Close ``incident_id`` RECOVERED on an operator's authority.

        Returns the incident and whether this call changed it: an incident
        that is already RECOVERED is returned unchanged (idempotent). Raises
        :class:`NotFoundError` for an unknown id, :class:`IncidentNotClosable`
        when the state is not ESCALATED (or QUARANTINED with ``evidence`` that
        clears every node, see ``isolation_verdict``) or a workflow of the
        incident is still open, and :class:`StaleWriteError` when the row
        moved between the read and the compare-and-set (the caller retries).
        """

        incident = self.store.get_incident(incident_id)
        if incident.state is IncidentState.RECOVERED:
            return incident, False
        refusal, _, isolation_reasons = self._refusal(incident, evidence)
        if refusal is not None:
            raise IncidentNotClosable(refusal)
        details: dict[str, Any] = {
            "closed_by": CLOSED_BY_OPERATOR,
            "operator": operator,
            "reference": reference,
        }
        if isolation_reasons:
            # A QUARANTINED close: the node evidence it rested on goes on the
            # audit event, so the record shows what was seen.
            named = set(incident.node_ids)
            details["isolation_evidence"] = [
                item.as_dict() for item in (evidence or ()) if item.node_id in named
            ]
        closed = self._close(
            incident,
            reason=f"operator closed: {reason} by {operator}",
            actor=operator,
            kind=WorkflowEventKind.OPERATOR_RECONCILED,
            details=details,
            extra_reasons=isolation_reasons,
        )
        self.operator_closed_total += 1
        LOGGER.info(
            "incident %s closed by operator %s: %s", incident_id, operator, reason
        )
        self._settle_blocked_workflow(closed, operator=operator, reference=reference)
        return closed, True

    def _settle_blocked_workflow(
        self, incident: FaultIncident, *, operator: str, reference: str | None
    ) -> None:
        """End the BLOCKED workflow a closed incident leaves behind.

        ``_open_workflow`` lets the close through past a BLOCKED row that holds
        nothing on the node, so that row would outlive its incident as
        paperwork -- and the release preflight counts BLOCKED destructive work.
        The close must not fail on it: a refused or stale write is logged and
        the dispatcher's sweep ends the record on its next tick.
        """

        if not incident.workflow_request_id:
            return
        try:
            workflow = self.store.get_workflow(incident.workflow_request_id)
            if workflow.status is not WorkflowStatus.BLOCKED:
                return
            close_settled_incident_blocked_workflow(
                self.store,
                workflow,
                now=datetime.now(timezone.utc),
                actor=operator,
                reference=reference,
            )
        except Exception:  # noqa: BLE001 - the incident close already landed
            LOGGER.exception(
                "BLOCKED workflow %s of closed incident %s left for the sweep",
                incident.workflow_request_id,
                incident.incident_id,
            )

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
        ):
            return []
        # ARCH-I4: a RECOVERED incident leaves no live marker. The isolating
        # shape retires through RESTORE_SCHEDULING and a failed diagnostic
        # through its inconclusive close, but a workflow that only froze
        # evidence, diagnosed and validated -- the node-health RUN_DIAGNOSTICS
        # shape -- ended RECOVERED with its marker active until the TTL,
        # holding the GPU "under remediation" an hour after the verdict said
        # it was fine (COLLECT-020). Idempotent: an earlier retirement keeps
        # its own reason.
        retire_markers_for_incident(
            self.store,
            incident.incident_id,
            reason=f"workflow {workflow.request_id} SUCCEEDED: incident RECOVERED",
            retired_by=RECOVERED_RETIREMENT_ACTOR,
        )
        if not restores_node(workflow):
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
        self,
        incident: FaultIncident,
        evidence: Sequence[NodeIsolationEvidence] | None = None,
    ) -> tuple[str | None, WorkflowRequest | None, list[str]]:
        """``(refusal, open_workflow, isolation_reasons)`` for an operator close.

        Without ``evidence`` the rule is the historical one: ESCALATED only.
        With it a QUARANTINED incident is judged by ``isolation_verdict`` --
        the evidence must clear every node -- and then, like ESCALATED, by the
        absence of an open workflow.
        """

        if incident.state is IncidentState.RECOVERED:
            return None, None, []
        isolation_reasons: list[str] = []
        if incident.state in EVIDENCE_CLOSABLE_STATES and evidence is not None:
            refusal, isolation_reasons = isolation_verdict(incident, evidence)
            if refusal is not None:
                return (
                    f"incident {incident.incident_id} is {incident.state.value} and "
                    f"the node evidence does not clear it: {refusal}",
                    None,
                    [],
                )
        elif incident.state not in OPERATOR_CLOSABLE_STATES:
            return (
                f"incident {incident.incident_id} is {incident.state.value}; only an "
                "ESCALATED incident can be closed by an operator (a QUARANTINED "
                "node is released by a validated restore workflow, or closed by "
                "gpu-fault-admin workflow-reconcile --close-quarantined with node "
                "evidence that the isolation is gone; a planning state by the "
                "workflow that ends it)",
                None,
                [],
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
                [],
            )
        return None, None, isolation_reasons

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
        extra_reasons: Sequence[str] = (),
    ) -> FaultIncident:
        """The one write path: CAS the incident, then audit and markers.

        The incident write is the transition; the audit event and the marker
        retirement follow it and are isolated like an ``on_terminal`` hook --
        a failure there is logged and does not unwind a close that landed.
        ``extra_reasons`` (the per-node isolation lines of a QUARANTINED
        close) follow ``reason`` on the incident.
        """

        now = datetime.now(timezone.utc)
        closed = incident.model_copy(
            update={
                "state": IncidentState.RECOVERED,
                "reasons": bounded_reasons([*incident.reasons, reason, *extra_reasons]),
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
