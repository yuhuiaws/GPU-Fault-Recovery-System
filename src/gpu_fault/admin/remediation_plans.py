"""The plans ``submit-remediation`` derives for a parked or quarantined record.

Two dispositions have no CHECK_MECHANICALS step to answer; their plans are
built here from the control plane's inspection and the live nodes, and
``gpu_fault.admin.submit_remediation`` submits them.

``restore`` (``RestorePlan``)
    the validated restore of a QUARANTINED incident. Judged *per node*: a node
    still isolated by this incident (its ``gpu-fault.io/quarantined`` taint, or
    its isolation annotation -- the ownership the restore's kubernetes patch
    reads) is restored; a node already clean, or another incident's business,
    is skipped and the plan says why. A two-node job DAG whose workflow
    restored one node itself and parked on the other (GF-REGIONAL-DESTR-014)
    used to be refused on the clean node; now the isolated one is restored.
    When no node is isolated any more the advice stays: close the incident.

``confirm-node-action`` (``ConfirmNodeActionPlan``)
    the operator's confirmation of a node action whose outcome the executor
    never observed. The plan picks the incident's BLOCKED workflow that left
    the named node unresolved, reads the node through the GPU kubeconfig and
    the fleet record from the inspection, and computes the same verdict the
    Pod recomputes before it writes (``execution.node_action_confirmation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, cast

from gpu_fault.adapters.common import ANNOTATION_INCIDENT, QUARANTINE_TAINT
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    confirm_node_action,
    confirmed_node_actions_on,
    unresolved_node_actions_on,
)
from gpu_fault.models import FaultIncident, WorkflowRequest
from gpu_fault.orchestration.incident_closure import owned_quarantine_taint_values
from gpu_fault.orchestration.validated_restore import (
    build_validated_restore_workflow,
    is_validated_restore_workflow,
    restore_reason,
)

DISPOSITION_RESTORE = "restore"
DISPOSITION_CONFIRM_NODE_ACTION = "confirm-node-action"
# A workflow in one of these still owns its node; ``restore`` waits for it.
OPEN_WORKFLOW_STATUSES = frozenset({"PENDING", "RUNNING", "SAFETY_PENDING"})
QUARANTINED_STATE = "QUARANTINED"
BLOCKED_STATE = "BLOCKED"
DECISION_RESTORE = "restore"
DECISION_SKIP = "skip"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# restore


@dataclass(frozen=True)
class RestorePlan:
    """The validated restore ``--disposition restore`` will create, derived
    from the QUARANTINED incident record and the live nodes.

    ``steps`` are the exact steps the product builder yields (operation,
    owner, nodes, GPU scope) for ``restore_node_ids`` -- the nodes still
    isolated by this incident; ``node_decisions`` says, per incident node,
    whether it is restored or skipped and why. The request id is minted in the
    Pod at submit time. ``existing_restore_workflow_id`` is set when such a
    workflow is already PENDING/RUNNING under the incident: the rerun is a
    no-op.
    """

    incident_id: str
    cluster_id: str
    node_ids: tuple[str, ...]
    gpu_uuids: tuple[str, ...]
    fencing_token: int
    workflow_request_id: str
    state: str
    reason: str
    steps: tuple[dict[str, Any], ...]
    existing_restore_workflow_id: str | None = None
    warnings: tuple[str, ...] = ()
    disposition: str = DISPOSITION_RESTORE
    restore_node_ids: tuple[str, ...] = ()
    node_decisions: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "cluster_id": self.cluster_id,
            "disposition": self.disposition,
            "node_ids": list(self.node_ids),
            "restore_node_ids": list(self.restore_node_ids),
            "node_decisions": [dict(item) for item in self.node_decisions],
            "gpu_uuids": list(self.gpu_uuids),
            "fencing_token": self.fencing_token,
            "workflow_request_id": self.workflow_request_id,
            "state": self.state,
            "reason": self.reason,
            "steps": [dict(item) for item in self.steps],
            "existing_restore_workflow_id": self.existing_restore_workflow_id,
            "next_operations": [str(item["operation"]) for item in self.steps],
            "warnings": list(self.warnings),
        }


def _quarantine_taint(node: Mapping[str, Any]) -> str | None:
    spec = node.get("spec")
    if not isinstance(spec, dict):
        return None
    for item in spec.get("taints") or []:
        if isinstance(item, dict) and item.get("key") == QUARANTINE_TAINT:
            return str(item.get("value") or "")
    return None


def _node_annotations(node: Mapping[str, Any]) -> dict[str, Any]:
    metadata = cast(dict[str, Any], node.get("metadata") or {})
    return dict(metadata.get("annotations") or {})


def node_restore_decision(
    node_id: str, node: Mapping[str, Any], *, incident_id: str
) -> dict[str, Any]:
    """Restore or skip ``node_id`` for ``incident_id``, with the reason.

    Restored when the node carries the incident's quarantine taint or its
    isolation annotation (``_node_restore_patch`` keys ownership on the
    annotation, so a node whose taint and cordon an operator lifted by hand
    is still the product's to release). Skipped when another incident owns
    the isolation (its restore releases it) or nothing of this incident is on
    the node. A node whose annotation names this incident but whose taint
    belongs to another is contradictory and skipped rather than guessed.
    """

    owned = owned_quarantine_taint_values(incident_id)
    annotations = _node_annotations(node)
    owner = annotations.get(ANNOTATION_INCIDENT)
    owner = str(owner) if owner else None
    taint = _quarantine_taint(node)
    spec = node.get("spec")
    decision: dict[str, Any] = {
        "node_id": node_id,
        "quarantine_taint_value": taint,
        "incident_annotation": owner,
        "unschedulable": bool(
            isinstance(spec, dict) and spec.get("unschedulable", False)
        ),
    }
    if owner and owner != incident_id:
        reason = (
            f"node {node_id} is isolated by incident {owner}, not {incident_id}; "
            "its restore releases it"
        )
        return {**decision, "decision": DECISION_SKIP, "reason": reason}
    if taint is not None and taint in owned:
        reason = f"node {node_id} carries the {QUARANTINE_TAINT} taint of {incident_id}"
        return {**decision, "decision": DECISION_RESTORE, "reason": reason}
    if owner == incident_id and taint is not None:
        # No annotation named another owner, so the taint value alone says
        # who placed it -- and it was not this incident.
        reason = (
            f"node {node_id} carries the isolation annotation of {incident_id} but a "
            f"{QUARANTINE_TAINT} taint of another incident ({taint}); the restore "
            "would strip that taint, so the node needs a human first"
        )
        return {**decision, "decision": DECISION_SKIP, "reason": reason}
    if owner == incident_id:
        reason = (
            f"node {node_id} carries the isolation annotation of {incident_id} "
            f"without its {QUARANTINE_TAINT} taint; the restore releases the annotation"
        )
        return {**decision, "decision": DECISION_RESTORE, "reason": reason}
    if taint is not None:
        reason = (
            f"node {node_id} is quarantined by another incident (taint {taint}), "
            f"not {incident_id}"
        )
        return {**decision, "decision": DECISION_SKIP, "reason": reason}
    reason = (
        f"node {node_id} carries no {QUARANTINE_TAINT} taint and no isolation "
        f"annotation of {incident_id}"
    )
    return {**decision, "decision": DECISION_SKIP, "reason": reason}


def _validate_restore_identity(
    inspection: Mapping[str, Any], *, incident_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    incident = cast(dict[str, Any] | None, inspection.get("incident"))
    workflow = cast(dict[str, Any] | None, inspection.get("workflow"))
    if incident is None:
        raise BootstrapError(f"incident {incident_id} was not returned")
    if str(incident.get("incident_id")) != incident_id:
        raise BootstrapError("the control plane returned a different incident")
    state = str(incident.get("state"))
    if state != QUARANTINED_STATE:
        raise BootstrapError(
            f"incident {incident_id} is {state}, not {QUARANTINED_STATE}; "
            f"{DISPOSITION_RESTORE} applies to a quarantined node only"
        )
    if workflow is None:
        raise BootstrapError(
            f"incident {incident_id} has no workflow; the quarantine it records "
            "was not placed by this product"
        )
    if workflow.get("request_id") != incident.get("workflow_request_id"):
        raise BootstrapError(
            f"incident {incident_id} points at workflow "
            f"{incident.get('workflow_request_id')} but "
            f"{workflow.get('request_id')} was returned"
        )
    if int(workflow.get("fencing_token") or 0) != int(
        incident.get("fencing_token") or 0
    ):
        raise BootstrapError(
            f"stale generation: incident fencing token {incident.get('fencing_token')} "
            f"differs from workflow {workflow.get('fencing_token')}"
        )
    if not incident.get("node_ids"):
        raise BootstrapError(f"incident {incident_id} names no nodes")
    return incident, workflow


def _validate_restore_record(
    inspection: Mapping[str, Any],
    *,
    incident_id: str,
    live_nodes: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str | None, list[dict[str, Any]]]:
    """The restore preconditions, all before any write.

    Returns the incident, its last workflow, the id of a validated restore
    workflow already open under it (the idempotent rerun; no node is judged
    then) and the per-node decisions, or raises the first refusal: not
    QUARANTINED; another workflow still open; a node missing from the cluster
    (its isolation cannot be judged); incident and workflow fencing tokens
    differing (a stale generation the node's annotation would reject); no
    node still isolated by this incident (close it instead).
    """

    incident, workflow = _validate_restore_identity(inspection, incident_id=incident_id)
    node_ids = [str(item) for item in incident.get("node_ids") or []]
    open_rows = (
        [dict(workflow)] if workflow.get("status") in OPEN_WORKFLOW_STATUSES else []
    )
    for item in inspection.get("open_workflows") or []:
        if isinstance(item, dict) and item.get("request_id") != workflow.get(
            "request_id"
        ):
            open_rows.append(item)
    existing = [
        str(item.get("request_id"))
        for item in open_rows
        if is_validated_restore_workflow(str(item.get("request_id") or ""))
    ]
    if existing:
        return incident, workflow, existing[0], []
    if open_rows:
        raise BootstrapError(
            f"incident {incident_id} still has an open workflow "
            + ", ".join(
                f"{item.get('request_id')} ({item.get('status')})" for item in open_rows
            )
            + "; wait for it to end or reconcile it first"
        )
    missing = sorted(set(node_ids) - set(live_nodes))
    if missing:
        raise BootstrapError(
            f"incident nodes are missing from cluster {incident['cluster_id']}: "
            + ", ".join(missing)
        )
    decisions = [
        node_restore_decision(node_id, live_nodes[node_id], incident_id=incident_id)
        for node_id in node_ids
    ]
    if not any(item["decision"] == DECISION_RESTORE for item in decisions):
        raise BootstrapError(
            f"no node of incident {incident_id} is still isolated by it: "
            + "; ".join(str(item["reason"]) for item in decisions)
            + " -- close the incident with gpu-fault-admin workflow-reconcile "
            f"--close-quarantined (or --close-incident {incident_id}) instead"
        )
    return incident, workflow, None, decisions


def build_restore_plan(
    inspection: Mapping[str, Any],
    *,
    incident_id: str,
    operator: str,
    reference: str | None,
    live_nodes: Mapping[str, Mapping[str, Any]],
    now: datetime | None = None,
) -> RestorePlan:
    """Turn the QUARANTINED record into the restore it will get, or refuse."""

    stamp = now or _utc_now()
    incident, workflow, existing, decisions = _validate_restore_record(
        inspection, incident_id=incident_id, live_nodes=live_nodes
    )
    record = FaultIncident.model_validate(incident)
    restore_nodes = tuple(
        str(item["node_id"])
        for item in decisions
        if item["decision"] == DECISION_RESTORE
    )
    inventory = inspection.get("node_gpu_uuids")
    _, preview = build_validated_restore_workflow(
        record,
        operator=operator,
        reference=reference,
        now=stamp,
        node_ids=list(restore_nodes) or None,
        node_gpu_uuids=inventory if isinstance(inventory, Mapping) else None,
    )
    warnings = tuple(
        f"node {item['node_id']} is cordoned without gpu-fault isolation metadata of "
        f"{incident_id}; the restore cannot release a cordon it does not own and "
        "--close-quarantined refuses while it stands"
        for item in decisions
        if item["decision"] == DECISION_SKIP
        and item["unschedulable"]
        and item["quarantine_taint_value"] is None
        and item["incident_annotation"] is None
    )
    return RestorePlan(
        incident_id=incident_id,
        cluster_id=str(incident["cluster_id"]),
        node_ids=tuple(str(item) for item in incident["node_ids"]),
        gpu_uuids=tuple(str(item) for item in incident.get("gpu_uuids") or []),
        fencing_token=int(incident["fencing_token"]),
        workflow_request_id=str(workflow["request_id"]),
        state=str(incident["state"]),
        reason=restore_reason(operator, reference),
        steps=tuple(
            {
                "operation": step.operation.value,
                "execution_owner": step.execution_owner,
                "node_ids": list(step.node_ids),
                "gpu_uuids": list(step.gpu_uuids),
            }
            for step in preview.official_steps
        ),
        existing_restore_workflow_id=existing,
        warnings=warnings,
        restore_node_ids=restore_nodes,
        node_decisions=tuple(decisions),
    )


# --------------------------------------------------------------------------
# confirm-node-action


@dataclass(frozen=True)
class ConfirmNodeActionPlan:
    """What ``--disposition confirm-node-action`` will write, and on what.

    ``confirmations`` are the records the Pod will put on the step executions
    (one per unresolved node action on the node); ``already_confirmed`` the
    ones already there (a rerun is a no-op). ``node_evidence`` is what the
    admin side read from Kubernetes and hands to the Pod verbatim;
    ``agent_evidence`` what the Pod reported from the fleet registry. The
    record identity the Pod compares before writing is ``fencing_token``,
    ``execution_epoch``, ``merge_revision`` and ``workflow_status``.
    """

    incident_id: str
    cluster_id: str
    node_id: str
    node_ids: tuple[str, ...]
    workflow_request_id: str
    fencing_token: int
    execution_epoch: int
    merge_revision: int
    workflow_status: str
    blocked_kind: str | None
    confirmations: tuple[dict[str, Any], ...]
    already_confirmed: tuple[dict[str, Any], ...]
    node_evidence: dict[str, Any]
    agent_evidence: dict[str, Any]
    # The step records the confirmations answer, exactly as inspected (phase,
    # index, operation, status, adapter operation, details): the Pod binds its
    # write to these, so a record that changed in between is refused.
    executions: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    disposition: str = DISPOSITION_CONFIRM_NODE_ACTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "cluster_id": self.cluster_id,
            "disposition": self.disposition,
            "node_id": self.node_id,
            "node_ids": list(self.node_ids),
            "workflow_request_id": self.workflow_request_id,
            "fencing_token": self.fencing_token,
            "execution_epoch": self.execution_epoch,
            "merge_revision": self.merge_revision,
            "workflow_status": self.workflow_status,
            "blocked_kind": self.blocked_kind,
            "steps": [
                {
                    "step_index": item["step_index"],
                    "operation": item["operation"],
                    "phase": item["phase"],
                    "remote_command_id": item["remote_command_id"],
                    "previous_boot_id": item["previous_boot_id"],
                    "previous_boot_id_source": item["previous_boot_id_source"],
                    "observed_boot_id": item["observed_boot_id"],
                }
                for item in self.confirmations
            ],
            "confirmations": [dict(item) for item in self.confirmations],
            "already_confirmed": [dict(item) for item in self.already_confirmed],
            "executions": [dict(item) for item in self.executions],
            "node_evidence": dict(self.node_evidence),
            "agent_evidence": dict(self.agent_evidence),
            "next_operations": [],
            "warnings": list(self.warnings),
        }


def _blocked_candidates(inspection: Mapping[str, Any]) -> list[WorkflowRequest]:
    rows: dict[str, dict[str, Any]] = {}
    pointer = inspection.get("workflow")
    if isinstance(pointer, dict) and pointer.get("status") == BLOCKED_STATE:
        rows[str(pointer.get("request_id"))] = pointer
    for item in inspection.get("blocked_workflows") or []:
        if isinstance(item, dict) and item.get("status") == BLOCKED_STATE:
            rows[str(item.get("request_id"))] = item
    return [WorkflowRequest.model_validate(rows[key]) for key in sorted(rows)]


def _parked_workflow(
    inspection: Mapping[str, Any], *, incident_id: str, node_id: str
) -> WorkflowRequest:
    """The one BLOCKED workflow of the incident with a node action on ``node_id``."""

    candidates = _blocked_candidates(inspection)
    relevant: list[WorkflowRequest] = []
    elsewhere: list[str] = []
    for workflow in candidates:
        on_node, other = unresolved_node_actions_on(workflow, node_id)
        if on_node or confirmed_node_actions_on(workflow, node_id):
            relevant.append(workflow)
        elsewhere.extend(f"{workflow.request_id}: {item}" for item in other)
    if not relevant:
        raise BootstrapError(
            f"no BLOCKED workflow of incident {incident_id} has an unresolved node "
            f"action on node {node_id}"
            + ("; unresolved elsewhere: " + "; ".join(elsewhere) if elsewhere else "")
            + (
                ""
                if candidates
                else "; the incident has no BLOCKED workflow on its nodes"
            )
        )
    if len(relevant) > 1:
        raise BootstrapError(
            f"more than one BLOCKED workflow of incident {incident_id} has a node "
            f"action on node {node_id}: "
            + ", ".join(item.request_id for item in relevant)
            + "; the CLI does not guess which record the confirmation answers"
        )
    return relevant[0]


def build_confirm_node_action_plan(
    inspection: Mapping[str, Any],
    *,
    incident_id: str,
    node_id: str,
    operator: str,
    reference: str | None,
    live_nodes: Mapping[str, Mapping[str, Any]],
    now: datetime | None = None,
) -> ConfirmNodeActionPlan:
    """Derive the confirmation from the record, the fleet and the node, or refuse.

    The verdict is ``execution.node_action_confirmation.confirm_node_action``
    over the inspection's records; every refusal it names is raised verbatim.
    """

    stamp = now or _utc_now()
    raw_incident = cast(dict[str, Any] | None, inspection.get("incident"))
    if raw_incident is None:
        raise BootstrapError(f"incident {incident_id} was not returned")
    if str(raw_incident.get("incident_id")) != incident_id:
        raise BootstrapError("the control plane returned a different incident")
    if "agents" not in inspection or "remote_commands" not in inspection:
        raise BootstrapError(
            "the control plane image predates confirm-node-action: the inspection "
            "carries no fleet agent or remote command evidence; deploy the release "
            "that ships it before confirming"
        )
    incident = FaultIncident.model_validate(raw_incident)
    if node_id not in incident.node_ids:
        raise BootstrapError(
            f"node {node_id} is not named by incident {incident_id} "
            f"(nodes: {', '.join(sorted(incident.node_ids)) or 'none'})"
        )
    workflow = _parked_workflow(inspection, incident_id=incident_id, node_id=node_id)
    node = KubernetesNodeEvidence.from_node(node_id, live_nodes.get(node_id))
    raw_agent = cast(dict[str, Any], inspection.get("agents") or {}).get(node_id)
    agent = (
        AgentEvidence.from_mapping(raw_agent)
        if isinstance(raw_agent, Mapping)
        else AgentEvidence(node_id=node_id)
    )
    commands = cast(dict[str, Any], inspection.get("remote_commands") or {}).get(
        workflow.request_id
    )
    # The incident's current workflow, when it is not the parked record, is
    # the candidate successor: a later validated restore that validated the
    # node is evidence for a node action other than a reboot.
    pointer = inspection.get("workflow")
    successor = (
        WorkflowRequest.model_validate(pointer)
        if isinstance(pointer, dict)
        and str(pointer.get("request_id")) != workflow.request_id
        else None
    )
    verdict = confirm_node_action(
        workflow=workflow,
        incident=incident,
        node_id=node_id,
        node=node,
        agent=agent,
        remote_commands=[item for item in commands or [] if isinstance(item, Mapping)],
        actor=operator,
        reference=reference or "(no reference: plan only)",
        now=stamp,
        successor=successor,
    )
    if verdict.refusals:
        raise BootstrapError(
            f"{DISPOSITION_CONFIRM_NODE_ACTION} refused: " + "; ".join(verdict.refusals)
        )
    return ConfirmNodeActionPlan(
        incident_id=incident_id,
        cluster_id=incident.cluster_id,
        node_id=node_id,
        node_ids=tuple(incident.node_ids),
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
        execution_epoch=workflow.execution_epoch,
        merge_revision=workflow.merge_revision,
        workflow_status=workflow.status.value,
        blocked_kind=(
            workflow.blocked_kind.value if workflow.blocked_kind is not None else None
        ),
        confirmations=verdict.confirmations,
        already_confirmed=verdict.already_confirmed,
        node_evidence=node.as_dict(),
        agent_evidence=agent.as_dict(),
        executions=tuple(
            {
                "phase": execution.phase,
                "step_index": execution.step_index,
                "operation": execution.operation.value,
                "status": execution.status.value,
                "adapter_operation_id": execution.adapter_operation_id,
                "details": execution.model_dump(mode="json")["details"],
            }
            for execution in verdict.executions
        ),
        warnings=verdict.warnings,
    )
