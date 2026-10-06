"""``workflow-reconcile`` for a parked record whose node action an operator confirmed.

Live (gpu-eks-cluster, XID 46): a RESET_GPU parked BLOCKED / NEEDS_OPERATOR
with ``outcome_unknown``; the node was rebooted, the GPU validated healthy by a
later restore, and ``submit-remediation --disposition confirm-node-action``
recorded the operator's confirmation on the step. The record still could not
end: the deployed planner knows two BLOCKED shapes (a verified restore
successor, or never having changed a node) and this is neither; the deployed
incident close refuses an ESCALATED incident while the record is open; and
``gpu-fault-admin config`` re-rolls the deployed release from its own source
snapshot, so its workflow safety gate -- the deployed probe -- counted the
record as an active destructive workflow. The release carrying the fixes was
the one the record blocked.

This module is the admin half of closing such a record. The rule it closes on
is ``gpu_fault.workflow_reconcile.operator_confirmed_reconciliation_reasons``
(BLOCKED, nothing live, a complete confirmation on a node-mutating step, no
node action still unresolved, the incident settled and naming the confirmed
nodes); the proof it adds is the node evidence every BLOCKED reconcile reads
through the site's GPU kubeconfig: the node carries no cordon, no isolation
taint and no isolation annotation -- or, like a ``never-changed`` record, it
left Kubernetes and the provider. The write goes through
``apply-operator-confirmed``, a Pod script branch with primitives every
deployed image has; ``SCRIPT_TAIL`` also plans the shape itself when the
deployed ``gpu_fault.workflow_reconcile`` predates it, and the fallback copy
is pinned to the module by ``tests/admin``.

The record ends SUPERSEDED with the audit on its ``OPERATOR_RECONCILED`` event
and the incident's reasons; the incident keeps its state. An ESCALATED one is
then closed on the operator's authority with ``--close-incident`` -- a close
the deployed service accepts once no workflow of the incident is open -- and
every release gate, the deployed one included, sees a terminal record.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.workflow_reconcile import (
    NODE_RESTORE_PROOF_REASON as NODE_RESTORE_PROOF_REASON,
)
from gpu_fault.workflow_reconcile import OPERATOR_CONFIRMED as OPERATOR_CONFIRMED

BRIDGE_MODE = "apply-operator-confirmed"

# Appended to ``workflow_reconcile.RECONCILE_SCRIPT``'s mode chain after the
# never-dispatched tail; ``result`` is the deployed planner's plan in ``plan``
# mode and ``None`` for a mode no earlier branch knew. Everything below runs
# against whatever ``gpu_fault`` is deployed in the Pod.
SCRIPT_TAIL = """
import hashlib
from datetime import datetime, timezone

try:
    from gpu_fault.workflow_reconcile import operator_confirmed_plan_items
except ImportError:  # the deployed planner predates the shape
    operator_confirmed_plan_items = None
try:
    from gpu_fault.workflow_reconcile import (
        operator_confirmed_node_actions,
        operator_confirmed_reconciliation_reasons,
    )
except ImportError:
    operator_confirmed_node_actions = None
    operator_confirmed_reconciliation_reasons = None
try:
    from gpu_fault.retired_generation import plan_digest_items
except ImportError:

    def plan_digest_items(items):
        return [
            {k: v for k, v in item.items() if k != "workflow_updated_at"}
            for item in items
        ]


OPERATOR_CONFIRMED = "operator-confirmed"
NODE_RESTORE_PROOF_REASON = (
    "node restoration is unproven: the administrator reads the GPU node's "
    "scheduling state"
)
CONFIRMATION_KEYS = ("actor", "reference", "confirmed_at", "node_id", "operation")
UNCERTAINTY_FLAGS = (
    "outcome_unknown",
    "node_action_interrupted",
    "node_action_response_unknown",
    "ownership_permit_delivery_unknown",
)


def _fallback_confirmation(details):
    # ``node_action_uncertainty.operator_confirmation`` for an image without it.
    value = details.get("operator_confirmed") if isinstance(details, dict) else None
    if not isinstance(value, dict):
        return None
    for key in CONFIRMATION_KEYS:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            return None
    return dict(value)


def _fallback_latest_node_actions(workflow):
    from gpu_fault.operation_registry import NODE_MUTATING_OPERATIONS

    seen = set()
    latest = []
    for execution in reversed(workflow.step_executions):
        identity = (
            getattr(execution, "phase", None),
            execution.step_index,
            execution.operation,
        )
        if identity in seen or execution.operation not in NODE_MUTATING_OPERATIONS:
            continue
        seen.add(identity)
        latest.append(execution)
    latest.reverse()
    return latest


def _fallback_unresolved(execution):
    # ``node_action_uncertainty.unresolved_node_action`` for an image without
    # the attestation; tests pin the two.
    from gpu_fault.models import WorkflowOperation, WorkflowStepStatus

    details = execution.details
    if execution.status is WorkflowStepStatus.SUCCEEDED or (
        execution.operation is WorkflowOperation.RESTORE_GPU_SERVICES
        and details.get("restore_gpu_services_withheld") is True
    ):
        return False
    if _fallback_confirmation(details) is not None:
        return False
    if any(details.get(name) is True for name in UNCERTAINTY_FLAGS):
        return True
    if (
        details.get("node_action_state") == "PENDING"
        and isinstance(details.get("node_action_command_id"), str)
        and details["node_action_command_id"]
    ):
        return True
    return (
        details.get("manual_confirmation_required") is True
        and details.get("node_action_not_started") is not True
        and details.get("node_action_outcome_known") is not True
    )


def _fallback_operator_confirmed_node_actions(workflow):
    confirmed = []
    for execution in _fallback_latest_node_actions(workflow):
        confirmation = _fallback_confirmation(execution.details)
        if confirmation is None:
            continue
        settled_by = confirmation.get("settled_by")
        confirmed.append(
            {
                "step_index": execution.step_index,
                "operation": execution.operation.value,
                "phase": getattr(execution, "phase", None),
                "node_id": confirmation["node_id"],
                "actor": confirmation["actor"],
                "reference": confirmation["reference"],
                "confirmed_at": confirmation["confirmed_at"],
                "settled_by": settled_by if isinstance(settled_by, str) else None,
            }
        )
    return confirmed


def _fallback_operator_confirmed_reasons(workflow, incident, commands, *, evaluated_at):
    # The copy of ``workflow_reconcile.operator_confirmed_reconciliation_reasons``
    # for an image without it; tests pin the two.
    from gpu_fault.models import IncidentState, WorkflowStatus, WorkflowStepStatus
    from gpu_fault.remote_command_models import RemoteCommandStatus

    open_statuses = {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.WAITING,
        RemoteCommandStatus.LEASED,
    }
    reasons = []
    if workflow.status is not WorkflowStatus.BLOCKED:
        reasons.append(f"workflow status is {workflow.status.value}, not BLOCKED")
    if workflow.execution_owner_id is not None:
        reasons.append("workflow still has an execution owner")
    if (
        workflow.execution_lease_expires_at is not None
        and workflow.execution_lease_expires_at > evaluated_at
    ):
        reasons.append("workflow execution lease has not expired")
    if any(
        item.status is WorkflowStepStatus.WAITING for item in workflow.step_executions
    ):
        reasons.append("workflow has an unknown provider action")
    if any(
        item.workflow_request_id == workflow.request_id and item.status in open_statuses
        for item in commands
    ):
        reasons.append("workflow has open remote commands")
    confirmations = _fallback_operator_confirmed_node_actions(workflow)
    if not confirmations:
        reasons.append("no node action of the record carries an operator confirmation")
    if any(_fallback_unresolved(item) for item in _fallback_latest_node_actions(workflow)):
        reasons.append("workflow has an unresolved node action no operator confirmed")
    if incident is None:
        reasons.append("incident is missing")
    else:
        if incident.state not in (IncidentState.RECOVERED, IncidentState.ESCALATED):
            reasons.append(
                f"incident is {incident.state.value}, still waiting on a workflow"
            )
        outside = sorted(
            {str(item["node_id"]) for item in confirmations} - set(incident.node_ids)
        )
        if outside:
            reasons.append(
                "confirmation names a node outside the incident: " + ", ".join(outside)
            )
    reasons.append(NODE_RESTORE_PROOF_REASON)
    return reasons


def _fallback_operator_confirmed_items(store, workflow_ids, *, now=None):
    # The copy of ``workflow_reconcile.operator_confirmed_plan_items`` for an
    # image without it; tests pin the two.
    from gpu_fault.remote_command_models import RemoteCommandStatus
    from gpu_fault.store import NotFoundError

    evaluated_at = now or datetime.now(timezone.utc)
    request_ids = sorted(
        {str(item).strip() for item in workflow_ids or () if str(item).strip()}
    )
    commands = (
        store.list_remote_commands(workflow_request_ids=request_ids)
        if request_ids
        else []
    )
    items = []
    for request_id in request_ids:
        try:
            workflow = store.get_workflow(request_id)
        except (KeyError, NotFoundError):
            continue
        confirmations = _fallback_operator_confirmed_node_actions(workflow)
        if not confirmations:
            continue
        try:
            incident = store.get_incident(workflow.incident_id)
        except (KeyError, NotFoundError):
            incident = None
        source_plan = None
        if workflow.source_plan_id:
            try:
                source_plan = store.get_plan(workflow.source_plan_id)
            except (KeyError, NotFoundError):
                source_plan = None
        items.append(
            {
                "request_id": request_id,
                "incident_id": workflow.incident_id,
                "cluster_id": incident.cluster_id if incident is not None else None,
                "node_ids": sorted(incident.node_ids) if incident is not None else [],
                "incident_state": (
                    incident.state.value if incident is not None else None
                ),
                "blocked_kind": (
                    workflow.blocked_kind.value
                    if workflow.blocked_kind is not None
                    else None
                ),
                "fencing_token": workflow.fencing_token,
                "execution_epoch": workflow.execution_epoch,
                "workflow_updated_at": workflow.updated_at.isoformat(),
                "source_plan_id": workflow.source_plan_id,
                "source_plan_status": (
                    source_plan.status.value if source_plan is not None else None
                ),
                "open_remote_commands": sorted(
                    item.command_id
                    for item in commands
                    if item.workflow_request_id == request_id
                    and item.status
                    in {
                        RemoteCommandStatus.PENDING,
                        RemoteCommandStatus.WAITING,
                        RemoteCommandStatus.LEASED,
                    }
                ),
                "confirmations": confirmations,
                "terminalization": OPERATOR_CONFIRMED,
                "eligible": False,
                "reasons": _fallback_operator_confirmed_reasons(
                    workflow, incident, commands, evaluated_at=evaluated_at
                ),
            }
        )
    return items


if payload["mode"] == "plan":
    # The deployed planner's items -- discovered or named -- are judged again
    # for this shape: a record it refused that carries an operator
    # confirmation takes this verdict; an eligible one, a record without a
    # confirmation and a record that does not exist keep the deployed verdict.
    # No discovery of its own: the BLOCKED backlog the deployed planner scans
    # already holds every such record, under the operator's batch selectors.
    planner = operator_confirmed_plan_items or _fallback_operator_confirmed_items
    judged = planner(
        store,
        [str(item.get("request_id")) for item in result["items"]],
        now=datetime.fromisoformat(result["evaluated_at"]),
    )
    by_id = {str(item["request_id"]): item for item in judged}
    merged = []
    for item in result["items"]:
        candidate = by_id.get(str(item.get("request_id")))
        if candidate is not None and not item.get("eligible"):
            item = candidate
        merged.append(item)
    result["items"] = merged
    result["plan_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "mode": result["mode"],
                "items": plan_digest_items(merged),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
elif payload["mode"] == "apply-operator-confirmed":
    # A parked record whose node action the operator confirmed, whose node the
    # administrator proved back in service (or gone). Re-derived here with the
    # rule above minus the proof the evidence answers, bound to the keys the
    # approval covered -- fencing token, execution epoch and the very
    # confirmations the operator saw -- and to the incident's node set, then
    # written with ``amend_workflow`` carrying the audit event, the source
    # plan's reconciliation reference when the record names a plan, and a
    # compare-and-set ``save_incident`` appending the same audit line. The
    # incident keeps its state: an ESCALATED one is the operator's to close.
    from gpu_fault.models import (
        WorkflowEventKind,
        WorkflowStatus,
        bounded_reasons,
        build_operator_event,
    )
    from gpu_fault.store import NotFoundError

    try:
        from gpu_fault.execution.restart_budget_preflight import (
            release_unattempted_restart_reservations,
        )
    except ImportError:  # an image that predates the reservation release
        release_unattempted_restart_reservations = None

    reasons_for = (
        operator_confirmed_reconciliation_reasons or _fallback_operator_confirmed_reasons
    )
    confirmations_of = (
        operator_confirmed_node_actions or _fallback_operator_confirmed_node_actions
    )
    now = datetime.now(timezone.utc)
    applied = []
    failures = {}
    warnings = []
    incidents = set()
    resolved_plans = []
    for item in payload["items"]:
        request_id = item["request_id"]
        try:
            workflow = store.get_workflow(request_id)
            incident = store.get_incident(workflow.incident_id)
            if workflow.fencing_token != item["fencing_token"]:
                raise ValueError(
                    f"workflow fencing token changed: expected {item['fencing_token']}, "
                    f"found {workflow.fencing_token}"
                )
            if workflow.execution_epoch != item["execution_epoch"]:
                raise ValueError(
                    "workflow execution epoch changed: expected "
                    f"{item['execution_epoch']}, found {workflow.execution_epoch}"
                )
            reasons = [
                reason
                for reason in reasons_for(
                    workflow,
                    incident,
                    store.list_remote_commands(workflow_request_ids=[request_id]),
                    evaluated_at=now,
                )
                if reason != NODE_RESTORE_PROOF_REASON
            ]
            if reasons:
                raise ValueError("; ".join(reasons))
            confirmations = confirmations_of(workflow)
            if confirmations != list(item["confirmations"]):
                raise ValueError(
                    "operator confirmations changed since the plan; plan again"
                )
            evidence = list(item["node_evidence"])
            restored = sorted(
                str(node["node_id"]) for node in evidence if node.get("restored") is True
            )
            if not restored or restored != sorted(
                str(node["node_id"]) for node in evidence
            ):
                raise ValueError("node evidence does not show every node restored")
            if restored != sorted(incident.node_ids):
                raise ValueError(
                    "incident node set changed: evidence covers "
                    f"{restored}, incident names {sorted(incident.node_ids)}"
                )
            source_plan = None
            if workflow.source_plan_id:
                try:
                    source_plan = store.get_plan(workflow.source_plan_id)
                except (KeyError, NotFoundError):
                    raise ValueError(
                        f"source recovery plan {workflow.source_plan_id} is missing"
                    ) from None
            operations = ", ".join(
                f"step {entry['step_index']} {entry['operation']} "
                f"({entry['settled_by'] or 'confirmed'})"
                for entry in confirmations
            )
            audit = (
                f"operator reconciliation {payload['reference']}: closed "
                f"{request_id}, whose node action the operator confirmed settled "
                f"({operations}) on {', '.join(restored)}, with its incident "
                f"{incident.state.value}"
            )
            superseded = workflow.model_copy(update={"status": WorkflowStatus.SUPERSEDED})
            amended = store.amend_workflow(
                request_id,
                {
                    "status": WorkflowStatus.SUPERSEDED,
                    "preempted_by_workflow_id": None,
                    "preemption_reason": audit,
                    "superseded_at": now,
                    "remediation_budget_claims": [],
                },
                event=build_operator_event(
                    superseded,
                    WorkflowEventKind.OPERATOR_RECONCILED,
                    actor=payload.get("actor"),
                    reference=payload["reference"],
                    previous_status=workflow.status,
                    at=now,
                    details={
                        "terminalization": OPERATOR_CONFIRMED,
                        "expected_fencing_token": item["fencing_token"],
                        "expected_execution_epoch": item["execution_epoch"],
                        "admin_plan_sha256": payload.get("admin_plan_sha256"),
                        # Bounded two levels deep: one flat record per
                        # confirmation and per node.
                        "operator_confirmations": confirmations,
                        "node_evidence": [
                            {
                                "node_id": node["node_id"],
                                "exists": node.get("exists"),
                                "restored": True,
                                "unschedulable": node.get("unschedulable"),
                                "quarantine_taint": node.get("quarantine_taint"),
                                "isolation_annotation_count": node.get(
                                    "isolation_annotation_count"
                                ),
                                "provider": node.get("provider"),
                                "absent_from_provider": node.get("absent_from_provider"),
                            }
                            for node in evidence
                        ],
                    },
                ),
            )
            if source_plan is not None:
                store.save_plan(
                    source_plan.model_copy(
                        update={
                            "reconciliation_reference": payload["reference"],
                            "reconciled_at": now,
                        }
                    )
                )
                resolved_plans.append(source_plan.plan_id)
            store.save_incident(
                incident.model_copy(
                    update={
                        "reasons": bounded_reasons([*incident.reasons, audit]),
                        "updated_at": now,
                    }
                ),
                expected=incident,
            )
            applied.append(request_id)
            incidents.add(incident.incident_id)
            if release_unattempted_restart_reservations is not None:
                try:
                    release_unattempted_restart_reservations(store, amended)
                except Exception as exc:  # the row is terminal; a warning
                    warnings.append(
                        f"{request_id}: closed, but releasing its restart "
                        f"reservations failed ({type(exc).__name__}: {exc})"
                    )
        except Exception as exc:  # per-item isolation, reported like the apply
            failures[request_id] = f"{type(exc).__name__}: {exc}"
    result = {
        "mode": "workflow-reconcile-apply",
        "applied_workflow_ids": applied,
        "failed_workflow_ids": sorted(failures),
        "failures": dict(sorted(failures.items())),
        "restart_reservation_warnings": sorted(warnings),
        "resolved_plan_ids": resolved_plans,
        "archive_eligible_incident_ids": sorted(incidents),
        "records_deleted": 0,
    }
"""


def promote_operator_confirmed(item: dict[str, Any]) -> None:
    """Mark an operator-confirmed item eligible, in place, on node evidence.

    Only the exact shape: the Pod judged the record a confirmed parked record
    and left ``NODE_RESTORE_PROOF_REASON`` as its one reason, and the node
    evidence shows every node of the incident restored -- present without a
    cordon, an isolation taint or an isolation annotation, or gone from
    Kubernetes and the provider. Any other reason, any node still isolated, or
    an item without node evidence keeps the Pod's verdict.
    """

    if item.get("terminalization") != OPERATOR_CONFIRMED or item.get("eligible"):
        return
    if [str(reason) for reason in item.get("reasons") or []] != [
        NODE_RESTORE_PROOF_REASON
    ]:
        return
    evidence = item.get("scheduling_evidence")
    if not isinstance(evidence, dict) or not evidence.get("restored"):
        return
    nodes = evidence.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return
    if any(node.get("restored") is not True for node in nodes):
        return
    item["eligible"] = True
    item["reasons"] = []


def _node_evidence_record(node: dict[str, Any]) -> dict[str, Any]:
    """One node of the item's scheduling evidence, flattened for the Pod.

    A present node carries the scheduling facts ``node_scheduling_evidence``
    read; a departed one carries the provider verdict instead. Only scalars:
    the Pod puts the record on the workflow's audit event, which is bounded
    two levels deep.
    """

    annotations = node.get("isolation_annotations")
    return {
        "node_id": str(node["node_id"]),
        "exists": node.get("exists") is True,
        "restored": node.get("restored") is True,
        "unschedulable": node.get("unschedulable"),
        "quarantine_taint": node.get("quarantine_taint"),
        "isolation_annotation_count": (
            len(annotations) if isinstance(annotations, list) else None
        ),
        "provider": node.get("provider"),
        "absent_from_provider": node.get("absent_from_provider"),
    }


def bridge_entry(item: dict[str, Any]) -> dict[str, Any]:
    """The ``apply-operator-confirmed`` entry: the keys the approval bound.

    Fencing token and execution epoch identify the record; ``confirmations``
    are the attestations the operator approved closing on, compared verbatim
    in the Pod; ``node_evidence`` is the per-node proof, checked there against
    the incident's node set and recorded on the audit event.
    """

    evidence = item["scheduling_evidence"]
    return {
        "request_id": str(item["request_id"]),
        "fencing_token": int(item["fencing_token"]),
        "execution_epoch": int(item["execution_epoch"]),
        "confirmations": [dict(entry) for entry in item["confirmations"]],
        "node_evidence": [_node_evidence_record(node) for node in evidence["nodes"]],
    }
