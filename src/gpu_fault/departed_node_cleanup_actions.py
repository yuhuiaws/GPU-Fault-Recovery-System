"""What the departed-node cleanup writes for one proven departure, and why not.

Three kinds of record, each ended through the path ``workflow-reconcile`` ends
it with, so the automation adds no write of its own:

* the node's fleet Agent -- ``departed_nodes.retire_departed_agent``, the
  registry's ``drain_agent``/``revoke_agent`` under one transition with the
  audit line the CLI writes (``retirement_audit``);
* the never-changed workflows of the node's incidents -- a ``PENDING`` record
  never dispatched through ``workflow_reconcile.close_never_dispatched_workflow``
  (the ``apply-never-dispatched`` write), a ``BLOCKED`` record the deployed
  planner judges ``never-changed`` through ``apply_workflow_reconcile_plan``
  (its own digest-checked apply);
* the node's ``ESCALATED`` incidents once no workflow of theirs is open --
  ``IncidentClosureService.close_incident``, the operator close.

The actor on every write is ``ACTOR`` and the reference is the departure's
generated id, so the audit reads exactly as the CLI's would with those two
substituted. Everything else is reported, never written: a workflow that
changed a node, a remote command on it, a live execution, a QUARANTINED
incident, an incident that also names a node not proven gone. Each refusal is a
``Blocker`` with a bounded ``kind`` (the metric label) and, when there is one,
the exact administrator command that is the manual path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping

from gpu_fault.departed_nodes import retire_departed_agent, retirement_audit
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.orchestration.incident_closure import IncidentNotClosable
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.workflow_reconcile import (
    NEVER_CHANGED,
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
    close_never_dispatched_workflow,
    never_dispatched_plan_items,
)
from gpu_fault.workflow_resolution import DEPARTED_NODE_PROOF_REASON

#: Who the cleanup's writes are attributed to.
ACTOR = "departed-node-cleanup"

# Blocker kinds (bounded: they label the refusal counter). The first group
# resolves on its own -- the next cycle retries; the rest is the operator's.
KUBERNETES_UNPROVEN = "kubernetes_unproven"
AGENT_ALIVE = "agent_alive"
PROVIDER_STILL_LISTS = "provider_still_lists"
PROVIDER_UNKNOWN = "provider_unknown"
RECORD_MOVED = "record_moved"
WORKFLOW_LIVE = "workflow_live"
INCIDENT_NODES_PENDING = "incident_nodes_pending"
CLUSTER_NOT_ACTIVE = "cluster_not_active"
DECLARED_SPARE = "declared_spare"
WORKFLOW_CHANGED_NODE = "workflow_changed_node"
REMOTE_COMMAND = "remote_command"
QUARANTINED_INCIDENT = "quarantined_incident"
INCIDENT_NAMES_OTHER_NODES = "incident_names_other_nodes"
INCIDENT_NOT_CLOSABLE = "incident_not_closable"
OTHER = "other"
WAITING_KINDS = frozenset(
    {
        KUBERNETES_UNPROVEN,
        AGENT_ALIVE,
        PROVIDER_STILL_LISTS,
        PROVIDER_UNKNOWN,
        RECORD_MOVED,
        WORKFLOW_LIVE,
        INCIDENT_NODES_PENDING,
    }
)
BLOCKER_KINDS = WAITING_KINDS | {
    CLUSTER_NOT_ACTIVE,
    DECLARED_SPARE,
    WORKFLOW_CHANGED_NODE,
    REMOTE_COMMAND,
    QUARANTINED_INCIDENT,
    INCIDENT_NAMES_OTHER_NODES,
    INCIDENT_NOT_CLOSABLE,
    OTHER,
}
# When one record has several reasons, the one reported is the most specific:
# what the operator must look at first, then what will resolve on its own.
_KIND_PRECEDENCE = (
    WORKFLOW_CHANGED_NODE,
    REMOTE_COMMAND,
    INCIDENT_NOT_CLOSABLE,
    OTHER,
    WORKFLOW_LIVE,
    RECORD_MOVED,
)
# Action labels (bounded: they label the action counter).
AGENT_RETIRED = "agent_retired"
WORKFLOW_CLOSED = "workflow_closed"
INCIDENT_CLOSED = "incident_closed"
ACTION_KINDS = (AGENT_RETIRED, WORKFLOW_CLOSED, INCIDENT_CLOSED)

#: The incident states that still need ending; RECOVERED is done.
OPEN_INCIDENT_STATES = frozenset(set(IncidentState) - {IncidentState.RECOVERED})
_TERMINAL_WORKFLOW_STATUSES = frozenset(
    {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.SUPERSEDED}
)
_STATE_DIR = "--state-dir <state-dir>"


def retire_command(node_id: str) -> str:
    return (
        f"gpu-fault-admin workflow-reconcile {_STATE_DIR} --retire-departed-agents "
        f"--node {node_id} --dry-run, then without --dry-run and with --reference "
        "<change>"
    )


def workflow_command(request_id: str) -> str:
    return (
        f"gpu-fault-admin workflow-reconcile {_STATE_DIR} --workflow-id "
        f"{request_id} --dry-run, then with --reference <change>"
    )


def incident_command(incident_id: str) -> str:
    return (
        f"gpu-fault-admin workflow-reconcile {_STATE_DIR} --close-incident "
        f"{incident_id} --reason <why> --dry-run, then with --reference <change>"
    )


def spare_command(node_id: str, cluster_id: str) -> str:
    return (
        f"gpu-fault-admin config spare {_STATE_DIR} --node {node_id} --cluster-id "
        f"{cluster_id} --release --reference <change> --confirm "
        "RELEASE_WARM_SPARE_UNCORDON"
    )


@dataclass(frozen=True)
class Blocker:
    """Why one record of a departure was not ended, and the manual path."""

    kind: str
    text: str
    command: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in BLOCKER_KINDS:
            raise ValueError(f"unknown departed-node blocker kind: {self.kind}")


@dataclass
class CleanupOutcome:
    """What one cycle ended for a node and what it left, with why."""

    actions: list[str] = field(default_factory=list)
    action_kinds: list[str] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)

    def acted(self, kind: str, label: str) -> None:
        self.actions.append(label)
        self.action_kinds.append(kind)


def _workflow_reason_kind(reason: str) -> str:
    """The blocker kind of one planner reason (``workflow_resolution`` wording)."""

    if "remote command" in reason:
        return REMOTE_COMMAND
    if any(
        marker in reason
        for marker in (" changed", "not PENDING", "not BLOCKED", "does not exist")
    ):
        return RECORD_MOVED
    if any(
        marker in reason
        for marker in (
            "execution owner",
            "lease has not expired",
            "dispatch guard",
            "still waiting on a workflow",
            "dispatcher sweep owns",
        )
    ):
        return WORKFLOW_LIVE
    if any(
        marker in reason
        for marker in (
            "node-mutating",
            "verified restore successor",
            "step executions",
            "completed or superseded a step",
        )
    ):
        return WORKFLOW_CHANGED_NODE
    return OTHER


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass
class DepartureWrites:
    """The writes of one cycle; ``proven`` is every node proven gone this cycle."""

    store: Any
    registry: Any
    incident_closure: Any
    now: datetime
    proven: frozenset[str]
    pending: frozenset[str]
    evidence: Mapping[str, Mapping[str, Any]]
    waiting_ttl: timedelta | None = None

    def clean(
        self,
        cluster_id: str,
        node_id: str,
        *,
        agent: AgentRecord | None,
        reference: str,
    ) -> CleanupOutcome:
        outcome = CleanupOutcome()
        if (
            agent is not None
            and agent.lifecycle_state is not AgentLifecycleState.REVOKED
        ):
            self._retire(cluster_id, node_id, agent, reference, outcome)
        incidents = self.store.list_incidents_by_state(
            cluster_id, OPEN_INCIDENT_STATES, node_ids={node_id}
        )
        for incident in sorted(incidents, key=lambda item: item.incident_id):
            self._incident(incident, node_id, reference, outcome)
        return outcome

    # ----------------------------------------------------------------- Agent

    def _retire(
        self,
        cluster_id: str,
        node_id: str,
        agent: AgentRecord,
        reference: str,
        outcome: CleanupOutcome,
    ) -> None:
        evidence = self.evidence[node_id]
        try:
            retire_departed_agent(
                self.registry,
                self.store,
                cluster_id=cluster_id,
                node_id=node_id,
                expected_generation=agent.generation,
                expected_lifecycle=agent.lifecycle_state.value,
                expected_last_seen_at=agent.last_seen_at,
                evidence=evidence,
                transition_id=f"{ACTOR}/{reference}",
                reason=retirement_audit(
                    reference=reference,
                    node_id=node_id,
                    evidence=evidence,
                    recorded_instance=str(agent.node_instance_id or ""),
                    actor=ACTOR,
                    plan_sha256=_digest(evidence),
                ),
                now=self.now,
            )
        except (ValueError, NotFoundError) as exc:
            # The record moved between the read and the write (a heartbeat,
            # an operator): the next cycle judges it again.
            outcome.blockers.append(
                Blocker(
                    RECORD_MOVED,
                    f"Agent {node_id} was not retired: {exc}",
                    retire_command(node_id),
                )
            )
            return
        outcome.acted(AGENT_RETIRED, f"agent-retired:{node_id}")

    # ------------------------------------------------------------- incidents

    def _incident(
        self,
        incident: FaultIncident,
        node_id: str,
        reference: str,
        outcome: CleanupOutcome,
    ) -> None:
        others = sorted(set(incident.node_ids) - self.proven)
        if others:
            waiting = set(others) <= self.pending
            outcome.blockers.append(
                Blocker(
                    INCIDENT_NODES_PENDING if waiting else INCIDENT_NAMES_OTHER_NODES,
                    f"incident {incident.incident_id} also names "
                    f"{', '.join(others)}, not proven departed"
                    + ("" if waiting else "; its records are the operator's"),
                    None if waiting else incident_command(incident.incident_id),
                )
            )
            return
        left_open = False
        for workflow in self._open_workflows(incident):
            closed = self._workflow(workflow, incident, reference, outcome)
            left_open = left_open or not closed
        if not left_open:
            # A workflow left open already says why; the close would only
            # repeat it as "still has an open workflow".
            self._close_incident(incident.incident_id, node_id, reference, outcome)

    def _open_workflows(self, incident: FaultIncident) -> list[WorkflowRequest]:
        """The incident's workflows that are not terminal, oldest id first."""

        candidates: dict[str, WorkflowRequest] = {}
        if incident.workflow_request_id:
            try:
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except (KeyError, NotFoundError):
                workflow = None
            if workflow is not None:
                candidates[workflow.request_id] = workflow
        for owner, workflow in self.store.list_active_workflow_incidents(
            incident.cluster_id, node_ids=set(incident.node_ids)
        ):
            if owner.incident_id == incident.incident_id:
                candidates.setdefault(workflow.request_id, workflow)
        return [
            candidates[key]
            for key in sorted(candidates)
            if candidates[key].status not in _TERMINAL_WORKFLOW_STATUSES
        ]

    def _close_incident(
        self,
        incident_id: str,
        node_id: str,
        reference: str,
        outcome: CleanupOutcome,
    ) -> None:
        incident = self.store.get_incident(incident_id)
        if incident.state is IncidentState.RECOVERED:
            return
        if incident.state is IncidentState.QUARANTINED:
            # No command either: the administrator's close needs isolation
            # evidence too, and refuses a node the cluster no longer has.
            outcome.blockers.append(
                Blocker(
                    QUARANTINED_INCIDENT,
                    f"incident {incident_id} is QUARANTINED; its isolation cannot "
                    "be verified on a node that is gone, so it is never closed "
                    "automatically, and --close-incident refuses it for the same "
                    "reason: no command ends it yet",
                )
            )
            return
        if incident.state is not IncidentState.ESCALATED:
            outcome.blockers.append(
                Blocker(
                    WORKFLOW_LIVE,
                    f"incident {incident_id} is {incident.state.value}; it is "
                    "closed once its workflow ends",
                )
            )
            return
        reason = departure_reason(
            [self.evidence[item] for item in sorted(incident.node_ids)], reference
        )
        try:
            _closed, changed = self.incident_closure.close_incident(
                incident_id, reason=reason, operator=ACTOR, reference=reference
            )
        except IncidentNotClosable as exc:
            text = str(exc)
            outcome.blockers.append(
                Blocker(
                    WORKFLOW_LIVE if "open workflow" in text else INCIDENT_NOT_CLOSABLE,
                    text,
                    incident_command(incident_id),
                )
            )
            return
        except StaleWriteError as exc:
            outcome.blockers.append(
                Blocker(RECORD_MOVED, f"incident {incident_id} moved: {exc}")
            )
            return
        if changed:
            outcome.acted(INCIDENT_CLOSED, f"incident-closed:{incident_id}")

    # ------------------------------------------------------------- workflows

    def _workflow(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        reference: str,
        outcome: CleanupOutcome,
    ) -> bool:
        """Close one workflow of the departure; ``False`` when it stays open."""

        request_id = workflow.request_id
        if workflow.status is WorkflowStatus.PENDING:
            reasons = self._close_never_dispatched(request_id, incident, reference)
        elif workflow.status is WorkflowStatus.BLOCKED:
            reasons = self._close_never_changed(request_id, incident, reference)
        else:
            outcome.blockers.append(
                Blocker(
                    WORKFLOW_LIVE,
                    f"workflow {request_id} is {workflow.status.value}; the "
                    "executor ends it",
                )
            )
            return False
        if not reasons:
            outcome.acted(WORKFLOW_CLOSED, f"workflow-closed:{request_id}")
            return True
        kinds = {_workflow_reason_kind(reason) for reason in reasons}
        kind = next(item for item in _KIND_PRECEDENCE if item in kinds)
        outcome.blockers.append(
            Blocker(
                kind,
                f"workflow {request_id} ({workflow.status.value}) is not closed: "
                + "; ".join(reasons),
                None if kind in WAITING_KINDS else workflow_command(request_id),
            )
        )
        return False

    def _node_evidence(self, incident: FaultIncident) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node,
                "instance_id": self.evidence[node].get("instance_id"),
                "provider": self.evidence[node].get("provider"),
                "absent_from_kubernetes": True,
                "absent_from_provider": True,
                "sources": self.evidence[node].get("sources"),
                "read_at": dict(self.evidence[node].get("read_at") or {}),
            }
            for node in sorted(incident.node_ids)
        ]

    def _close_never_dispatched(
        self, request_id: str, incident: FaultIncident, reference: str
    ) -> list[str]:
        items, _report = never_dispatched_plan_items(
            self.store, [request_id], now=self.now
        )
        item = items[0]
        reasons = [
            str(reason)
            for reason in item.get("reasons") or []
            if reason != DEPARTED_NODE_PROOF_REASON
        ]
        if reasons:
            return reasons
        evidence = self._node_evidence(incident)
        try:
            close_never_dispatched_workflow(
                self.store,
                {
                    "request_id": request_id,
                    "fencing_token": int(item["fencing_token"]),
                    "execution_epoch": int(item["execution_epoch"]),
                    "workflow_created_at": str(item["workflow_created_at"]),
                    "departed_node_evidence": evidence,
                },
                reference=reference,
                actor=ACTOR,
                admin_plan_sha256=_digest(evidence),
                now=self.now,
            )
        except (ValueError, NotFoundError) as exc:
            return [str(exc)]
        return []

    def _close_never_changed(
        self, request_id: str, incident: FaultIncident, reference: str
    ) -> list[str]:
        plan = build_workflow_reconcile_plan(self.store, [request_id], now=self.now)
        item = plan["items"][0]
        reasons = [str(reason) for reason in item.get("reasons") or []]
        if item.get("terminalization") != NEVER_CHANGED or not item.get(
            "never_changed_a_node"
        ):
            # A record that completed a node-mutating operation -- restored by
            # a later workflow or not -- is the administrator's reconcile.
            return ["workflow completed a node-mutating operation", *reasons]
        if not item.get("eligible"):
            return reasons
        try:
            result = apply_workflow_reconcile_plan(
                self.store,
                workflow_ids=[request_id],
                expected_plan_sha256=plan["plan_sha256"],
                reference=reference,
                now=self.now,
                waiting_ttl=self.waiting_ttl,
                actor=ACTOR,
                admin_plan_sha256=_digest(self._node_evidence(incident)),
            )
        except ValueError as exc:
            return [f"workflow record changed before the close: {exc}"]
        failures = result.get("failures") or {}
        return [str(failures[key]) for key in sorted(failures)]


def departure_reason(evidence: list[Mapping[str, Any]], reference: str) -> str:
    """The reason an incident is closed with: the departure evidence, briefly."""

    parts = []
    for node in evidence:
        sources = node.get("sources") or {}
        label = "EC2" if "ec2" in sources else "HyperPod"
        read_at = node.get("read_at") or {}
        parts.append(
            f"{node.get('node_id')} absent from Kubernetes since "
            f"{node.get('kubernetes_absent_since')}, {label} no longer has "
            f"instance {node.get('instance_id')} (read "
            f"{read_at.get('ec2') or read_at.get('hyperpod')})"
        )
    return "node departed: " + "; ".join(parts) + f"; departed-node cleanup {reference}"
