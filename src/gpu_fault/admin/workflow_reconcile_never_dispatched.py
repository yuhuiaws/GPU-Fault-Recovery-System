"""``workflow-reconcile`` for a PENDING record whose node left the cluster.

On 2026-10-01 HyperPod reclaimed a spot GPU node while the control plane was
ingesting its last telemetry. Two workflows were created for that node's
incidents and stayed PENDING for good: no step execution, no remote command,
no event -- the node's Agent was gone, so the fleet preflight never dispatched
them -- and the release preflight (``workflow_safety``) refused every deploy on
them, ``uninstall`` counted them as live, and ``workflow-reconcile`` scanned
BLOCKED records only.

This module is the admin half of closing such a record. The rule it closes on
is ``gpu_fault.workflow_resolution.never_dispatched_reconciliation_reasons``
(PENDING, never dispatched, no remote command, older than the guard age, the
incident still names it); the proof it adds is the departed-node evidence of
``gpu_fault.admin.workflow_reconcile``: the node is absent from the cluster's
Kubernetes Nodes *and* HyperPod no longer lists its instance. A node that is
NotReady, still in Kubernetes, still listed by HyperPod, on a cluster the site
does not describe as HyperPod-managed, or behind a failed lookup keeps the
record ineligible, naming why. The plan lists, per record, which sources were
consulted, their verdicts (in the digest) and when they were read (outside
it, so a re-plan a moment later is not drift).

The write goes through ``apply-never-dispatched``, a Pod script branch with
primitives every deployed image has -- the deployed planner may predate the
shape, and the image it blocks from being replaced is exactly the one that has
to apply it. ``SCRIPT_TAIL`` also plans the shape itself when the deployed
``gpu_fault.workflow_reconcile`` has no ``never_dispatched_plan_items``; the
fallback copy is pinned to the module by ``tests/admin``.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.workflow_resolution import DEPARTED_NODE_PROOF_REASON

NEVER_DISPATCHED = "never-dispatched"
BRIDGE_MODE = "apply-never-dispatched"
# The reason the deployed planner gives an explicit PENDING id; the
# never-dispatched item is the verdict for that record instead.
NOT_BLOCKED_REASON = "workflow status is PENDING, not BLOCKED"
# Read times are reported beside the evidence and left out of the digest: the
# verdicts (present/listed) bind the approval, the clock does not.
DIGEST_EXCLUDED_ITEM_FIELDS = frozenset({"evidence_read_at"})
KUBERNETES_PRESENT_BLOCKER = (
    "node is still in Kubernetes; a never-dispatched record is closed only "
    "for a node that left Kubernetes and HyperPod"
)

# Appended to ``workflow_reconcile.RECONCILE_SCRIPT``'s mode chain, which ends
# with ``result = None`` for a mode it does not know. Everything below runs
# against whatever ``gpu_fault`` is deployed in the Pod.
SCRIPT_TAIL = """
import hashlib
from datetime import datetime, timedelta, timezone

try:
    from gpu_fault.workflow_reconcile import never_dispatched_plan_items
except ImportError:  # the deployed planner predates the shape
    never_dispatched_plan_items = None
try:
    from gpu_fault.workflow_resolution import never_dispatched_reconciliation_reasons
except ImportError:
    never_dispatched_reconciliation_reasons = None
try:
    from gpu_fault.retired_generation import plan_digest_items
except ImportError:

    def plan_digest_items(items):
        return [
            {k: v for k, v in item.items() if k != "workflow_updated_at"}
            for item in items
        ]


NEVER_DISPATCHED = "never-dispatched"
NEVER_DISPATCHED_GUARD_AGE = timedelta(minutes=10)
DEPARTED_NODE_PROOF_REASON = (
    "node departure is unproven: the administrator reads Kubernetes and HyperPod"
)
NOT_BLOCKED_REASON = "workflow status is PENDING, not BLOCKED"


def _fallback_never_dispatched_reasons(workflow, incident, commands, *, evaluated_at):
    # The copy of ``workflow_resolution.never_dispatched_reconciliation_reasons``
    # for an image without it; tests pin the two.
    from gpu_fault.models import WorkflowStatus

    reasons = []
    if workflow.status is not WorkflowStatus.PENDING:
        reasons.append(f"workflow status is {workflow.status.value}, not PENDING")
    if workflow.step_executions:
        reasons.append("workflow has step executions")
    if (
        workflow.completed_step_indexes
        or workflow.superseded_step_indexes
        or workflow.completed_operations
    ):
        reasons.append("workflow completed or superseded a step")
    if workflow.execution_owner_id is not None:
        reasons.append("workflow still has an execution owner")
    if (
        workflow.execution_lease_expires_at is not None
        and workflow.execution_lease_expires_at > evaluated_at
    ):
        reasons.append("workflow execution lease has not expired")
    if workflow.remediation_budget_claims:
        reasons.append("workflow holds remediation budget claims")
    if any(item.workflow_request_id == workflow.request_id for item in commands):
        reasons.append("workflow has remote commands")
    if workflow.created_at > evaluated_at - NEVER_DISPATCHED_GUARD_AGE:
        reasons.append("workflow is younger than the 10-minute dispatch guard")
    if incident is None:
        reasons.append("incident is missing")
    elif incident.workflow_request_id != workflow.request_id:
        reasons.append(
            "incident names another workflow; the dispatcher sweep owns a "
            "retired generation"
        )
    reasons.append(DEPARTED_NODE_PROOF_REASON)
    return reasons


def _fallback_never_dispatched_items(
    store, workflow_ids, *, incident_ids=None, max_items=None, now=None
):
    # The copy of ``workflow_reconcile.never_dispatched_plan_items`` for an
    # image without it; tests pin the two.
    from gpu_fault.models import WorkflowStatus
    from gpu_fault.store import NotFoundError

    evaluated_at = now or datetime.now(timezone.utc)
    report = None
    requested = sorted({str(item).strip() for item in workflow_ids or () if str(item).strip()})
    if requested:
        request_ids = requested
    else:
        scan_limit = 10_000
        workflows = store.list_workflows(
            statuses={WorkflowStatus.PENDING}, limit=scan_limit + 1
        )
        truncated = len(workflows) > scan_limit
        workflows = workflows[:scan_limit]
        wanted = {str(item).strip() for item in incident_ids or () if item}
        oldest_allowed = evaluated_at - NEVER_DISPATCHED_GUARD_AGE
        candidates = [
            item.request_id
            for item in workflows
            if item.status is WorkflowStatus.PENDING
            and not item.step_executions
            and not item.completed_step_indexes
            and not item.superseded_step_indexes
            and not item.completed_operations
            and item.execution_owner_id is None
            and item.created_at <= oldest_allowed
            and (not wanted or item.incident_id in wanted)
        ]
        selected = candidates if max_items is None else candidates[:max_items]
        request_ids = sorted(selected)
        report = {
            "scanned": len(workflows),
            "selected": len(selected),
            "remaining": max(0, len(candidates) - len(selected)),
            "scan_truncated": truncated,
        }
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
            items.append(
                {
                    "request_id": request_id,
                    "eligible": False,
                    "reasons": ["workflow does not exist"],
                }
            )
            continue
        try:
            incident = store.get_incident(workflow.incident_id)
        except (KeyError, NotFoundError):
            incident = None
        items.append(
            {
                "request_id": request_id,
                "incident_id": workflow.incident_id,
                "cluster_id": incident.cluster_id if incident is not None else None,
                "node_ids": sorted(incident.node_ids) if incident is not None else [],
                "incident_state": (
                    incident.state.value if incident is not None else None
                ),
                "fencing_token": workflow.fencing_token,
                "execution_epoch": workflow.execution_epoch,
                "workflow_created_at": workflow.created_at.isoformat(),
                "workflow_updated_at": workflow.updated_at.isoformat(),
                "step_execution_count": len(workflow.step_executions),
                "remote_command_count": sum(
                    1 for item in commands if item.workflow_request_id == request_id
                ),
                "terminalization": NEVER_DISPATCHED,
                "eligible": False,
                "reasons": _fallback_never_dispatched_reasons(
                    workflow, incident, commands, evaluated_at=evaluated_at
                ),
            }
        )
    return items, report


if payload["mode"] == "plan":
    # The deployed planner covers BLOCKED; the never-dispatched PENDING shape
    # is planned here with the checkout's rule and merged in: an explicit id
    # the deployed planner refused only for being PENDING takes the
    # never-dispatched verdict, discovered records are appended, and the
    # runtime digest is rebuilt over the merged items.
    planner = never_dispatched_plan_items or _fallback_never_dispatched_items
    options = {
        key: payload[key]
        for key in ("incident_ids", "max_items")
        if payload.get(key) is not None
    }
    extra, never_dispatched_report = planner(
        store,
        payload.get("workflow_ids"),
        now=datetime.fromisoformat(result["evaluated_at"]),
        **options,
    )
    by_id = {str(item["request_id"]): item for item in extra}
    merged = []
    for item in result["items"]:
        candidate = by_id.pop(str(item.get("request_id")), None)
        if candidate is not None and NOT_BLOCKED_REASON in (item.get("reasons") or []):
            item = candidate
        merged.append(item)
    merged.extend(by_id[key] for key in sorted(by_id))
    result["items"] = merged
    result["never_dispatched_discovery"] = never_dispatched_report
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
elif payload["mode"] == "apply-never-dispatched":
    # A PENDING record never dispatched, whose node the administrator proved
    # gone from Kubernetes and HyperPod. Re-derived here with the rule above
    # (minus the proof reason the evidence in the payload answers), bound to
    # the keys the approval covered and to the incident's node set, then
    # written with ``amend_workflow`` carrying the audit event and a
    # compare-and-set ``save_incident`` that hands the incident to the
    # operator (ESCALATED) with the same audit line.
    from gpu_fault.models import (
        IncidentState,
        WorkflowEventKind,
        WorkflowStatus,
        bounded_reasons,
        build_operator_event,
    )

    try:
        from gpu_fault.execution.restart_budget_preflight import (
            release_unattempted_restart_reservations,
        )
    except ImportError:  # an image that predates the reservation release
        release_unattempted_restart_reservations = None

    reasons_for = (
        never_dispatched_reconciliation_reasons or _fallback_never_dispatched_reasons
    )
    now = datetime.now(timezone.utc)
    applied = []
    failures = {}
    warnings = []
    incidents = set()
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
            if workflow.created_at.isoformat() != item["workflow_created_at"]:
                raise ValueError(
                    "workflow created_at changed: expected "
                    f"{item['workflow_created_at']}, found "
                    f"{workflow.created_at.isoformat()}"
                )
            reasons = [
                reason
                for reason in reasons_for(
                    workflow,
                    incident,
                    store.list_remote_commands(workflow_request_ids=[request_id]),
                    evaluated_at=now,
                )
                if reason != DEPARTED_NODE_PROOF_REASON
            ]
            if reasons:
                raise ValueError("; ".join(reasons))
            evidence = list(item["departed_node_evidence"])
            proven = sorted(
                str(node["node_id"])
                for node in evidence
                if node.get("absent_from_kubernetes") is True
                and node.get("absent_from_provider") is True
            )
            if not proven or proven != sorted(
                str(node["node_id"]) for node in evidence
            ):
                raise ValueError(
                    "departed-node evidence does not prove every node gone"
                )
            if proven != sorted(incident.node_ids):
                raise ValueError(
                    "incident node set changed: evidence covers "
                    f"{proven}, incident names {sorted(incident.node_ids)}"
                )
            audit = (
                f"operator reconciliation {payload['reference']}: closed "
                f"{request_id}, never dispatched, after its node left "
                f"Kubernetes and HyperPod ({', '.join(proven)})"
            )
            # Event details are bounded two levels deep; one flat record per
            # node keeps every source, verdict and read time legible.
            recorded = [
                {
                    "node_id": node["node_id"],
                    "instance_id": node.get("instance_id"),
                    "provider": node.get("provider"),
                    "absent_from_kubernetes": True,
                    "absent_from_provider": True,
                    "hyperpod_cluster_name": (
                        ((node.get("sources") or {}).get("hyperpod") or {}).get(
                            "cluster_name"
                        )
                    ),
                    "region": (
                        ((node.get("sources") or {}).get("hyperpod") or {}).get("region")
                    ),
                    "kubernetes_read_at": (node.get("read_at") or {}).get("kubernetes"),
                    "hyperpod_read_at": (node.get("read_at") or {}).get("hyperpod"),
                }
                for node in evidence
            ]
            superseded = workflow.model_copy(update={"status": WorkflowStatus.SUPERSEDED})
            amended = store.amend_workflow(
                request_id,
                {
                    "status": WorkflowStatus.SUPERSEDED,
                    "preempted_by_workflow_id": None,
                    "preemption_reason": audit,
                    "superseded_at": now,
                },
                event=build_operator_event(
                    superseded,
                    WorkflowEventKind.OPERATOR_RECONCILED,
                    actor=payload.get("actor"),
                    reference=payload["reference"],
                    previous_status=workflow.status,
                    at=now,
                    details={
                        "terminalization": NEVER_DISPATCHED,
                        "expected_fencing_token": item["fencing_token"],
                        "expected_execution_epoch": item["execution_epoch"],
                        "admin_plan_sha256": payload.get("admin_plan_sha256"),
                        "departed_node_evidence": recorded,
                    },
                ),
            )
            updates = {
                "reasons": bounded_reasons([*incident.reasons, audit]),
                "updated_at": now,
            }
            if incident.state in (
                IncidentState.DETECTED,
                IncidentState.ACTION_PENDING,
                IncidentState.SAFETY_PENDING,
            ):
                # Nobody is going to act on this incident's behalf any more;
                # the operator closing the record is who it is handed to.
                updates["state"] = IncidentState.ESCALATED
            store.save_incident(incident.model_copy(update=updates), expected=incident)
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
        "resolved_plan_ids": [],
        "archive_eligible_incident_ids": sorted(incidents),
        "records_deleted": 0,
    }
elif result is None:
    raise ValueError("unsupported workflow reconcile mode")
print(json.dumps(result, sort_keys=True))
"""


def node_evidence(
    node_id: str,
    *,
    departed: dict[str, Any] | None,
    target: dict[str, Any],
    aws_region: object,
) -> dict[str, Any]:
    """The per-node evidence of a never-dispatched item.

    ``departed`` is ``workflow_reconcile._departed_node_evidence`` for a node
    Kubernetes no longer has, ``None`` while it still has it. Unlike a
    ``never-changed`` item, a node that is present is a blocker here whatever
    its scheduling state: the record is closed for a node that is *gone*, not
    for one that is clean. ``sources`` names what was consulted (``target`` is
    the site's cluster entry, for the HyperPod cluster name and region) and
    what each said; a source that could not be read says so instead of
    answering.
    """

    if departed is None:
        return {
            "node_id": node_id,
            "exists": True,
            "absent_from_kubernetes": False,
            "absent_from_provider": False,
            "restored": False,
            "blockers": [KUBERNETES_PRESENT_BLOCKER],
            "sources": {"kubernetes": {"consulted": True, "present": True}},
        }
    provider: dict[str, Any] = {
        "consulted": "absent_from_provider" in departed,
        "cluster_name": str(target.get("hyperpod_cluster_name") or "").strip() or None,
        "region": str(target.get("region") or aws_region or "").strip() or None,
    }
    if "absent_from_provider" in departed:
        provider["listed"] = not bool(departed["absent_from_provider"])
    return {
        **departed,
        "absent_from_kubernetes": True,
        "absent_from_provider": bool(departed.get("absent_from_provider", False)),
        "sources": {
            "kubernetes": {"consulted": True, "present": False},
            "hyperpod": provider,
        },
    }


def promote_never_dispatched(item: dict[str, Any]) -> None:
    """Mark a never-dispatched item eligible, in place, on departed-node proof.

    Only the exact shape: the Pod judged the record never dispatched and left
    ``DEPARTED_NODE_PROOF_REASON`` as its one reason, and the evidence shows
    every node absent from Kubernetes and from HyperPod. Any other reason, any
    node still present somewhere, or an item without node evidence keeps the
    Pod's verdict.
    """

    if item.get("terminalization") != NEVER_DISPATCHED or item.get("eligible"):
        return
    if [str(reason) for reason in item.get("reasons") or []] != [
        DEPARTED_NODE_PROOF_REASON
    ]:
        return
    evidence = item.get("scheduling_evidence")
    if not isinstance(evidence, dict) or not evidence.get("restored"):
        return
    nodes = evidence.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return
    if any(
        node.get("absent_from_kubernetes") is not True
        or node.get("absent_from_provider") is not True
        for node in nodes
    ):
        return
    item["eligible"] = True
    item["reasons"] = []


def bridge_entry(item: dict[str, Any]) -> dict[str, Any]:
    """The ``apply-never-dispatched`` entry: the keys the approval bound.

    Fencing token, execution epoch and ``created_at`` identify the record; the
    departed-node evidence goes on the workflow's audit event (flattened to one
    record per node there) and is checked against the incident's node set in
    the Pod.
    """

    evidence = item["scheduling_evidence"]
    return {
        "request_id": str(item["request_id"]),
        "fencing_token": int(item["fencing_token"]),
        "execution_epoch": int(item["execution_epoch"]),
        "workflow_created_at": str(item["workflow_created_at"]),
        "departed_node_evidence": [
            {
                "node_id": str(node["node_id"]),
                "instance_id": node.get("instance_id"),
                "provider": node.get("provider"),
                "absent_from_kubernetes": node.get("absent_from_kubernetes") is True,
                "absent_from_provider": node.get("absent_from_provider") is True,
                "sources": node.get("sources"),
                "read_at": dict(item.get("evidence_read_at") or {}),
            }
            for node in evidence["nodes"]
        ],
    }
