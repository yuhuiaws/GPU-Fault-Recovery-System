"""``gpu-fault-admin workflow-reconcile``: close BLOCKED records a later workflow restored.

One shape, one command. A workflow that is BLOCKED and never changed a node --
or whose incident a *later* workflow already recovered and whose node a
``RESTORE_SCHEDULING`` already put back -- is paperwork the runtime cannot
close on its own, because the proof needs node evidence the control plane does
not hold: the GPU node carries no cordon, no quarantine taint and no gpu-fault
isolation annotation. This module reads that evidence through the site's own
GPU kubeconfig and hands the verdict to ``gpu_fault.workflow_reconcile`` in the
CPU ingress Pod.

A node that is gone carries none of that. When a ``never-changed`` record's
node is missing from Kubernetes and the GPU cluster is HyperPod-managed, the
node name (``hyperpod-<instance-id>``) is looked up in the HyperPod node list:
an instance HyperPod no longer lists has nothing left to restore and the item
records that as its evidence (``absent_from_provider``); an instance still
listed, a cluster that is not HyperPod-managed or a failed lookup keep failing
closed, naming why. ``verified-restore`` items keep the Kubernetes rule.

The other three shapes the command used to take modes for are closed by the
dispatcher's periodic sweep now (``WorkflowDispatcher.sweep_stuck_records``):
retired generations, compile-time BLOCKED no-ops (``gpu_fault.compile_blocked``)
and remote commands a terminal workflow left open
(``gpu_fault.orphaned_commands``). Every live use of those modes had been to
unblock a release preflight, and every condition they checked was a Store
predicate.

One invocation plans and applies. The plan is built in the Pod, extended here
with the site identity and the node evidence, re-built immediately before the
apply and compared field by field (``_plan_drift``); the apply carries the
runtime digest, the admin digest and the operator's STS identity, and the Pod
puts all three on the workflow's ``OPERATOR_RECONCILED`` event. ``--dry-run``
prints the plan and writes nothing, on the Pod or on disk. Without it the plan
and the result are archived under ``workflow-reconcile/history/<digest>/``.
Nothing is ever deleted: ``records_deleted`` is always 0.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from gpu_fault import retired_generation
from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import run_command
from gpu_fault.admin.site import RenderedSite

REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
HISTORY_PATH = Path("workflow-reconcile/history")
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
# The bridge for a record the *deployed* planner refuses for lacking a source
# recovery plan while naming a verified restore successor. A job DAG's node
# branch is never plan-driven; when it parks BLOCKED and a later validated
# restore RECOVERS its incident, the successor proves the restore and the
# checkout's resolver accepts the record (``workflow_resolution``). The image
# whose ``apply_workflow_reconcile_plan`` still applies the plan gate is the one
# such records block from being replaced (DESTR-014 and HA-004 held deploy #15
# in the release preflight), so the admin side promotes the item and applies it
# through ``apply-verified-restore`` with primitives every image has.
VERIFIED_RESTORE_NON_PLAN = "verified-restore-non-plan"
# The sibling bridge for a ``never-changed`` record without a source plan: its
# plan was replaced in place by a merge (``PLAN_REWRITE`` leaves
# ``source_plan_id`` empty) or it was never plan-driven, it completed no
# node-mutating operation and its incident is settled. The checkout's resolver
# accepts it; the deployed image's may still refuse it for the plan it never
# had, so the admin side promotes it and applies it through
# ``apply-never-changed`` with primitives every image has.
NEVER_CHANGED = "never-changed"
NEVER_CHANGED_NON_PLAN = "never-changed-non-plan"
NON_PLAN_REASON = "workflow has no source recovery plan"
# Which bridge mode applies a promoted item; anything else goes through the
# deployed ``apply_workflow_reconcile_plan``.
BRIDGE_MODES = {
    VERIFIED_RESTORE_NON_PLAN: "apply-verified-restore",
    NEVER_CHANGED_NON_PLAN: "apply-never-changed",
}
# Items whose missing node may be explained by the provider instead of failing
# on "node is missing": there is no restore successor to name, and a node
# HyperPod no longer lists carries no cordon, taint or annotation to restore.
DEPARTED_NODE_TERMINALIZATIONS = frozenset({NEVER_CHANGED, NEVER_CHANGED_NON_PLAN})
HYPERPOD_PROVIDER = "hyperpod"
# HyperPod names an EKS node after its EC2 instance.
HYPERPOD_NODE_NAME = re.compile(r"^hyperpod-(i-[0-9a-f]{8,17})$")
# Pages of ``list-cluster-nodes`` read before the lookup is declared broken.
HYPERPOD_INVENTORY_PAGE_LIMIT = 100
ISOLATION_ANNOTATIONS = (
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
)
RECONCILE_SCRIPT = """
import inspect
import json
import sys
from datetime import timedelta

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
store = context.store
if payload["mode"] == "plan":
    # Batch options are passed only when given, so the script still runs
    # against a deployed image whose planner predates them.
    options = {
        key: payload[key]
        for key in ("incident_ids", "max_items")
        if payload.get(key) is not None
    }
    result = build_workflow_reconcile_plan(
        store,
        payload.get("workflow_ids"),
        **options,
    )
elif payload["mode"] == "apply":
    options = {}
    parameters = inspect.signature(apply_workflow_reconcile_plan).parameters
    # The executor's RESTART_WORKLOAD waiting cap decides which WAITING
    # restart reservations the terminalization releases; the module has no
    # executor config of its own, and an older deployed image's apply does
    # not take it.
    if "waiting_ttl" in parameters:
        options["waiting_ttl"] = timedelta(
            seconds=context.production_executor_config.step_waiting_limit(
                WorkflowOperation.RESTART_WORKLOAD
            )
        )
    # The operator's identity and the admin-side plan digest go on the
    # workflow's audit event; an image whose apply predates them cannot take
    # them, and the write must still happen.
    if "actor" in parameters:
        options["actor"] = payload.get("actor")
    if "admin_plan_sha256" in parameters:
        options["admin_plan_sha256"] = payload.get("admin_plan_sha256")
    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=payload["workflow_ids"],
        expected_plan_sha256=payload["plan_sha256"],
        reference=payload["reference"],
        **options,
    )
elif payload["mode"] == "apply-verified-restore":
    # A record that was never plan-driven, superseded by its verified restore
    # successor. The deployed ``apply_workflow_reconcile_plan`` may still apply
    # the source-plan gate to it, so the write is made here with primitives
    # every image has: the resolver's own eligibility minus that one reason,
    # ``amend_workflow`` with the audit event in the same write (the path the
    # never-changed close uses), and a compare-and-set ``save_incident``.
    from datetime import datetime, timezone

    from gpu_fault.models import (
        WorkflowEventKind,
        WorkflowStatus,
        bounded_reasons,
        build_operator_event,
    )
    from gpu_fault.workflow_resolution import (
        restore_reconciliation_reasons,
        verified_restore_successor,
    )

    NON_PLAN_REASON = "workflow has no source recovery plan"
    now = datetime.now(timezone.utc)
    applied = []
    failures = {}
    incidents = set()
    for item in payload["items"]:
        request_id = item["request_id"]
        try:
            workflow = store.get_workflow(request_id)
            incident = store.get_incident(workflow.incident_id)
            successor = verified_restore_successor(store, workflow)
            if successor is None or successor.request_id != item["successor_workflow_id"]:
                raise ValueError(
                    "workflow has no verified restore successor "
                    + str(item["successor_workflow_id"])
                )
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
            if workflow.source_plan_id:
                raise ValueError(
                    "workflow has a source recovery plan; the restore reconcile owns it"
                )
            reasons = [
                reason
                for reason in restore_reconciliation_reasons(
                    workflow,
                    incident,
                    successor,
                    None,
                    store.list_remote_commands(workflow_request_ids=[request_id]),
                    evaluated_at=now,
                )
                if reason != NON_PLAN_REASON
            ]
            if reasons:
                raise ValueError("; ".join(reasons))
            audit = (
                f"operator reconciliation {payload['reference']}: superseded "
                f"{request_id} after verified restore {successor.request_id}"
            )
            superseded = workflow.model_copy(update={"status": WorkflowStatus.SUPERSEDED})
            store.amend_workflow(
                request_id,
                {
                    "status": WorkflowStatus.SUPERSEDED,
                    "preempted_by_workflow_id": successor.request_id,
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
                        "terminalization": "verified-restore-non-plan",
                        "successor_workflow_id": successor.request_id,
                        "expected_fencing_token": item["fencing_token"],
                        "expected_execution_epoch": item["execution_epoch"],
                        "admin_plan_sha256": payload.get("admin_plan_sha256"),
                    },
                ),
            )
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
        except Exception as exc:  # per-item isolation, reported like the apply
            failures[request_id] = f"{type(exc).__name__}: {exc}"
    result = {
        "mode": "workflow-reconcile-apply",
        "applied_workflow_ids": applied,
        "failed_workflow_ids": sorted(failures),
        "failures": dict(sorted(failures.items())),
        "restart_reservation_warnings": [],
        "resolved_plan_ids": [],
        "archive_eligible_incident_ids": sorted(incidents),
        "records_deleted": 0,
    }
elif payload["mode"] == "apply-never-changed":
    # A record that completed no node-mutating operation and names no source
    # plan (replaced in place by a merge, or never plan-driven). The deployed
    # ``apply_workflow_reconcile_plan`` may still refuse it for the plan it
    # never had, so the write is made here the way the checkout's
    # ``_close_never_changed`` makes it: the resolver's own eligibility minus
    # that one reason, ``amend_workflow`` with the audit event in the same
    # write, no plan and no incident write.
    from datetime import datetime, timezone

    from gpu_fault.models import (
        WorkflowEventKind,
        WorkflowStatus,
        build_operator_event,
    )
    from gpu_fault.workflow_resolution import (
        restore_reconciliation_reasons,
        verified_restore_successor,
        workflow_never_changed_a_node,
    )

    try:
        from gpu_fault.execution.restart_budget_preflight import (
            release_unattempted_restart_reservations,
        )
    except ImportError:  # an image that predates the reservation release
        release_unattempted_restart_reservations = None

    NON_PLAN_REASON = "workflow has no source recovery plan"
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
            if workflow.source_plan_id:
                raise ValueError(
                    "workflow has a source recovery plan; the restore reconcile owns it"
                )
            if verified_restore_successor(store, workflow) is not None:
                raise ValueError(
                    "workflow has a verified restore successor; "
                    "the restore reconcile owns it"
                )
            if not workflow_never_changed_a_node(workflow):
                raise ValueError("workflow completed a node-mutating operation")
            reasons = [
                reason
                for reason in restore_reconciliation_reasons(
                    workflow,
                    incident,
                    None,
                    None,
                    store.list_remote_commands(workflow_request_ids=[request_id]),
                    evaluated_at=now,
                )
                if reason != NON_PLAN_REASON
            ]
            if reasons:
                raise ValueError("; ".join(reasons))
            audit = (
                f"operator reconciliation {payload['reference']}: closed "
                f"{request_id}, which completed no node-mutating operation, "
                f"with its incident {incident.state.value}"
            )
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
                        "terminalization": "never-changed-non-plan",
                        "expected_fencing_token": item["fencing_token"],
                        "expected_execution_epoch": item["execution_epoch"],
                        "admin_plan_sha256": payload.get("admin_plan_sha256"),
                    },
                ),
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
        "resolved_plan_ids": [],
        "archive_eligible_incident_ids": sorted(incidents),
        "records_deleted": 0,
    }
else:
    raise ValueError("unsupported workflow reconcile mode")
print(json.dumps(result, sort_keys=True))
"""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _apply_payload(
    *,
    workflow_ids: list[str],
    runtime_plan_sha256: object,
    admin_plan_sha256: str,
    reference: str,
    actor: str,
) -> dict[str, Any]:
    """The apply request sent to the Pod.

    Besides the ids, the runtime digest and the reference, it names the operator
    (the STS caller ARN, else ``user@host`` -- resolving it never blocks the
    apply) and the admin-side plan digest, so the Pod can put both on the
    workflow's audit event (I1). The archived ``applied.json`` carries the same
    identity.
    """

    return {
        "mode": "apply",
        "workflow_ids": workflow_ids,
        "plan_sha256": runtime_plan_sha256,
        "reference": reference,
        "actor": actor,
        "admin_plan_sha256": admin_plan_sha256,
    }


def promote_non_plan_successor(item: dict[str, Any]) -> None:
    """Mark a ``VERIFIED_RESTORE_NON_PLAN`` item eligible, in place.

    Only the exact shape: the deployed planner refused the record for
    ``NON_PLAN_REASON`` alone, named a verified restore successor and the record
    carries no source plan. Anything else stays as the planner judged it.
    """

    if (
        not item.get("eligible")
        and [str(reason) for reason in item.get("reasons") or []] == [NON_PLAN_REASON]
        and item.get("terminalization") == "verified-restore"
        and item.get("successor_workflow_id")
        and not item.get("source_plan_id")
    ):
        item["eligible"] = True
        item["reasons"] = []
        item["terminalization"] = VERIFIED_RESTORE_NON_PLAN


def promote_non_plan_never_changed(item: dict[str, Any]) -> None:
    """Mark a ``NEVER_CHANGED_NON_PLAN`` item eligible, in place.

    Only the exact shape: the deployed planner refused the record for
    ``NON_PLAN_REASON`` alone -- so it already judged the record never changed
    a node and its incident settled -- names no successor and the record
    carries no source plan. Anything else stays as the planner judged it.
    """

    if (
        not item.get("eligible")
        and [str(reason) for reason in item.get("reasons") or []] == [NON_PLAN_REASON]
        and item.get("terminalization") == NEVER_CHANGED
        and not item.get("successor_workflow_id")
        and not item.get("source_plan_id")
    ):
        item["eligible"] = True
        item["reasons"] = []
        item["terminalization"] = NEVER_CHANGED_NON_PLAN


def _bridge_payload(
    *,
    mode: str,
    items: list[dict[str, Any]],
    admin_plan_sha256: str,
    reference: str,
    actor: str,
) -> dict[str, Any]:
    """A bridge request (``BRIDGE_MODES``): the keys the approval bound.

    The verified-restore bridge names the successor it supersedes the record
    with; the never-changed bridge has none to name.
    """

    bound: list[dict[str, Any]] = []
    for item in items:
        entry: dict[str, Any] = {"request_id": str(item["request_id"])}
        if mode == BRIDGE_MODES[VERIFIED_RESTORE_NON_PLAN]:
            entry["successor_workflow_id"] = str(item["successor_workflow_id"])
        entry["fencing_token"] = int(item["fencing_token"])
        entry["execution_epoch"] = int(item["execution_epoch"])
        bound.append(entry)
    return {
        "mode": mode,
        "items": bound,
        "reference": reference,
        "actor": actor,
        "admin_plan_sha256": admin_plan_sha256,
    }


def _merge_apply_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One apply result out of the runtime apply and the bridge apply."""

    merged: dict[str, Any] = {
        "mode": "workflow-reconcile-apply",
        "applied_workflow_ids": [],
        "failed_workflow_ids": [],
        "failures": {},
        "restart_reservation_warnings": [],
        "resolved_plan_ids": [],
        "archive_eligible_incident_ids": [],
        "records_deleted": 0,
    }
    for result in results:
        for key in (
            "applied_workflow_ids",
            "failed_workflow_ids",
            "restart_reservation_warnings",
            "resolved_plan_ids",
            "archive_eligible_incident_ids",
        ):
            merged[key] = [*merged[key], *(result.get(key) or [])]
        merged["failures"] = {**merged["failures"], **(result.get("failures") or {})}
    merged["failed_workflow_ids"] = sorted(set(merged["failed_workflow_ids"]))
    merged["archive_eligible_incident_ids"] = sorted(
        set(merged["archive_eligible_incident_ids"])
    )
    return merged


def run_control_plane_script(
    site: RenderedSite,
    payload: dict[str, Any],
    *,
    script: str = RECONCILE_SCRIPT,
) -> dict[str, Any]:
    """Exec ``script`` in a Running CPU ingress Pod with ``payload`` on stdin.

    The exec channel is how every administrator command reaches the Store
    without holding a control-plane token on the deploy host; ``submit-
    remediation`` reuses it with its own script.
    """

    kubectl = [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
    ]
    pod_result = run_command(
        [
            *kubectl,
            "get",
            "pod",
            "-l",
            "app=gpu-fault-api-ha",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        timeout_seconds=120,
    )
    pod = pod_result.stdout.strip()
    if pod_result.returncode or not pod:
        raise BootstrapError("workflow reconcile found no Running CPU ingress Pod")
    completed = run_command(
        [*kubectl, "exec", "-i", pod, "--", "python", "-c", script],
        input_text=json.dumps(payload, separators=(",", ":")),
        timeout_seconds=900,
    )
    if completed.returncode:
        detail = diagnostic_text(completed.stderr, sensitive=True)
        raise BootstrapError(
            "workflow reconcile execution failed" + (f": {detail}" if detail else "")
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError("workflow reconcile returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapError("workflow reconcile returned a non-object result")
    return value


# The historical name; the reconcile paths and their tests still bind it.
_run_reconcile = run_control_plane_script


def _site_identity(site: RenderedSite) -> dict[str, Any]:
    return {
        "site_name": site.release_config["site_name"],
        "aws_region": site.release_config["aws_region"],
        "cpu_eks_arn": site.release_config["cpu_eks_arn"],
        "site_sha256": site.source_sha256,
        "clusters": sorted(
            (
                {
                    "cluster_id": str(item["cluster_id"]),
                    "context": str(item["context"]),
                }
                for item in site.release_config["clusters"]
            ),
            key=lambda item: item["cluster_id"],
        ),
    }


def gpu_kubectl_command(site: RenderedSite, target: dict[str, Any]) -> list[str]:
    """``kubectl`` bound to the site's GPU kubeconfig, or a refusal.

    The evidence this reads decides whether a record is closed, so it has to
    come from the cluster the site manages. The shell's ``KUBECONFIG`` and
    ``~/.kube/config`` are whatever the operator last pointed at -- possibly a
    different site -- so a site without a rendered GPU kubeconfig is refused
    instead of silently read through the default.
    """

    kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if not kubeconfig:
        raise BootstrapError(
            "workflow reconcile has no GPU kubeconfig for cluster "
            f"{target['cluster_id']}: the managed site renders none, and the "
            "default kubeconfig is never used for node evidence"
        )
    return [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        str(target["context"]),
    ]


_gpu_kubectl = gpu_kubectl_command


def cluster_target(site: RenderedSite, cluster_id: str) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {
        str(item["cluster_id"]): dict(item) for item in site.release_config["clusters"]
    }
    target = targets.get(cluster_id)
    if target is None:
        raise BootstrapError(
            f"workflow reconcile cluster is not in the managed site: {cluster_id}"
        )
    return target


def _orphaned_isolation_metadata(
    node: dict[str, Any], incident_id: str
) -> dict[str, Any]:
    metadata = node.get("metadata")
    spec = node.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise BootstrapError("cannot strip isolation annotations: node is incomplete")
    if any(
        not isinstance(metadata.get(key), str) or not metadata[key].strip()
        for key in ("name", "uid", "resourceVersion")
    ):
        raise BootstrapError(
            "cannot strip isolation annotations: node identity or resourceVersion is missing"
        )
    annotations = metadata.get("annotations")
    if (
        not incident_id
        or not isinstance(annotations, dict)
        or annotations.get(ISOLATION_ANNOTATIONS[0]) != incident_id
        or not isinstance(annotations.get(ISOLATION_ANNOTATIONS[1]), str)
        or not annotations[ISOLATION_ANNOTATIONS[1]].strip()
    ):
        raise BootstrapError(
            "cannot strip isolation annotations: ownership is unproven"
        )
    taints = spec.get("taints", [])
    if (
        metadata.get("deletionTimestamp") is not None
        or spec.get("unschedulable", False) is not False
        or not isinstance(taints, list)
        or any(
            not isinstance(taint, dict)
            or not isinstance(taint.get("key"), str)
            or taint["key"] == QUARANTINE_TAINT
            for taint in taints
        )
    ):
        raise BootstrapError(
            "cannot strip isolation annotations: node is still isolated"
        )
    return metadata


def strip_node_isolation_annotations(
    site: RenderedSite,
    cluster_id: str,
    node: dict[str, Any],
    *,
    incident_id: str,
    expected_node: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove proven orphaned annotations, bound to the original node and owner."""

    previous = _orphaned_isolation_metadata(expected_node, incident_id)
    metadata = _orphaned_isolation_metadata(node, incident_id)
    node_id = str(metadata["name"])
    if any(metadata[key] != previous[key] for key in ("name", "uid")) or any(
        metadata["annotations"].get(key) != previous["annotations"].get(key)
        for key in ISOLATION_ANNOTATIONS
    ):
        raise BootstrapError(
            f"cannot strip isolation annotations of {node_id}: node or ownership changed"
        )
    resource_version = str(metadata["resourceVersion"])
    patch = {
        "metadata": {
            "uid": metadata["uid"],
            "resourceVersion": resource_version,
            "annotations": {key: None for key in ISOLATION_ANNOTATIONS},
        }
    }
    if not dry_run:
        completed = run_command(
            [
                *_gpu_kubectl(site, cluster_target(site, cluster_id)),
                "patch",
                "node",
                node_id,
                "--type",
                "merge",
                "-p",
                json.dumps(patch, sort_keys=True),
            ],
            timeout_seconds=120,
        )
        if completed.returncode:
            raise BootstrapError(
                f"cannot strip the orphaned isolation annotations of {node_id} on "
                f"{cluster_id}: {diagnostic_text(completed.stderr, sensitive=True)}"
            )
    return {
        "node_id": node_id,
        "node_uid": metadata["uid"],
        "resource_version": resource_version,
        "annotations": list(ISOLATION_ANNOTATIONS),
    }


def cluster_nodes(
    site: RenderedSite,
    cluster_id: str,
) -> dict[str, dict[str, Any]]:
    target = cluster_target(site, cluster_id)
    completed = run_command(
        [*_gpu_kubectl(site, target), "get", "nodes", "-o", "json"],
        timeout_seconds=120,
    )
    if completed.returncode:
        raise BootstrapError(
            f"workflow reconcile cannot read GPU nodes for {cluster_id}: "
            f"{diagnostic_text(completed.stderr, sensitive=True)}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            f"workflow reconcile received invalid node JSON for {cluster_id}"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise BootstrapError(
            f"workflow reconcile received invalid node inventory for {cluster_id}"
        )
    nodes: dict[str, dict[str, Any]] = {}
    for raw in value["items"]:
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            continue
        name = str(metadata.get("name") or "")
        if name:
            nodes[name] = raw
    return nodes


def _node_scheduling_evidence(
    node_id: str,
    node: dict[str, Any] | None,
) -> dict[str, Any]:
    """What ``node`` (a ``kubectl get node -o json`` item) carries.

    ``quarantine_taint_value`` and ``isolation_annotation_values`` (key ->
    value) are the ownership facts ``IncidentClosureService`` needs to judge a
    QUARANTINED close: whether the taint or the annotations belong to the
    incident being closed or to another one. The boolean ``quarantine_taint``
    and the key list ``isolation_annotations`` are the older shape the BLOCKED
    reconcile plan reads and hashes.
    """

    if node is None:
        return {
            "node_id": node_id,
            "exists": False,
            "restored": False,
            "blockers": ["node is missing"],
        }
    metadata = node.get("metadata")
    spec = node.get("spec")
    annotations: dict[str, Any] = {}
    if isinstance(metadata, dict):
        raw_annotations = metadata.get("annotations")
        if isinstance(raw_annotations, dict):
            annotations = raw_annotations
    taints: list[Any] = []
    if isinstance(spec, dict):
        raw_taints = spec.get("taints")
        if isinstance(raw_taints, list):
            taints = raw_taints
    unschedulable = bool(
        spec.get("unschedulable", False) if isinstance(spec, dict) else False
    )
    taint_value: str | None = None
    for item in taints:
        if isinstance(item, dict) and item.get("key") == QUARANTINE_TAINT:
            taint_value = str(item.get("value") or "")
            break
    quarantine = taint_value is not None
    isolation_annotation_values = {
        key: str(annotations[key])
        for key in ISOLATION_ANNOTATIONS
        if annotations.get(key) not in (None, "")
    }
    isolation_annotations = sorted(isolation_annotation_values)
    blockers = []
    if unschedulable:
        blockers.append("node remains unschedulable")
    if quarantine:
        blockers.append("node retains the gpu-fault quarantine taint")
    if isolation_annotations:
        blockers.append("node retains gpu-fault isolation annotations")
    return {
        "node_id": node_id,
        "exists": True,
        "unschedulable": unschedulable,
        "quarantine_taint": quarantine,
        "quarantine_taint_value": taint_value,
        "isolation_annotations": isolation_annotations,
        "isolation_annotation_values": isolation_annotation_values,
        "restored": not blockers,
        "blockers": blockers,
    }


def node_isolation_evidence(
    site: RenderedSite,
    cluster_id: str,
    node_ids: Sequence[str],
    *,
    inventory: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """``NodeIsolationEvidence`` mappings for ``node_ids`` on ``cluster_id``.

    Read through the site's GPU kubeconfig (``cluster_nodes``; a site without
    one is refused there, never the shell's default kubeconfig). The shape is
    what ``IncidentClosureService`` consumes
    (``NodeIsolationEvidence.from_mapping``): existence, the cordon flag, the
    quarantine taint's value and the isolation annotations with their values.
    ``inventory`` lets a caller reuse one ``kubectl get nodes`` per cluster.
    """

    nodes = inventory if inventory is not None else cluster_nodes(site, cluster_id)
    evidence: list[dict[str, Any]] = []
    for node_id in sorted({str(item) for item in node_ids}):
        raw = _node_scheduling_evidence(node_id, nodes.get(node_id))
        evidence.append(
            {
                "node_id": node_id,
                "exists": bool(raw["exists"]),
                "unschedulable": bool(raw.get("unschedulable", False)),
                "quarantine_taint_value": raw.get("quarantine_taint_value"),
                "isolation_annotations": dict(
                    raw.get("isolation_annotation_values") or {}
                ),
            }
        )
    return evidence


def hyperpod_instance_ids(site: RenderedSite, cluster_id: str) -> frozenset[str]:
    """The EC2 instance ids HyperPod currently lists for ``cluster_id``.

    Read with ``aws sagemaker list-cluster-nodes`` in the cluster's region,
    every page. A cluster the site does not describe as HyperPod-managed, a
    failed call and an unreadable answer all raise ``BootstrapError``: the
    caller turns that into a blocker, never into "absent".
    """

    target = cluster_target(site, cluster_id)
    name = str(target.get("hyperpod_cluster_name") or "").strip()
    if not name:
        raise BootstrapError(
            f"GPU cluster {cluster_id} is not HyperPod-managed in the site"
        )
    region = str(
        target.get("region") or site.release_config.get("aws_region") or ""
    ).strip()
    if not region:
        raise BootstrapError(f"GPU cluster {cluster_id} has no AWS region in the site")
    instance_ids: set[str] = set()
    token: str | None = None
    for _page in range(HYPERPOD_INVENTORY_PAGE_LIMIT):
        command = [
            "aws",
            "sagemaker",
            "list-cluster-nodes",
            "--cluster-name",
            name,
            "--region",
            region,
            "--output",
            "json",
        ]
        if token:
            command.extend(["--next-token", token])
        completed = run_command(command, timeout_seconds=120)
        if completed.returncode:
            raise BootstrapError(
                f"cannot list HyperPod nodes of {cluster_id}: "
                f"{diagnostic_text(completed.stderr, sensitive=True)}"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BootstrapError(
                f"HyperPod node inventory of {cluster_id} is invalid JSON"
            ) from exc
        summaries = (
            value.get("ClusterNodeSummaries") if isinstance(value, dict) else None
        )
        if not isinstance(summaries, list):
            raise BootstrapError(f"HyperPod node inventory of {cluster_id} is invalid")
        for row in summaries:
            if isinstance(row, dict):
                instance_id = row.get("InstanceId")
                if isinstance(instance_id, str) and instance_id.strip():
                    instance_ids.add(instance_id.strip())
        next_token = value.get("NextToken")
        if not next_token:
            return frozenset(instance_ids)
        if not isinstance(next_token, str) or next_token == token:
            raise BootstrapError(
                f"HyperPod node inventory of {cluster_id} repeats its page token"
            )
        token = next_token
    raise BootstrapError(
        f"HyperPod node inventory of {cluster_id} exceeds "
        f"{HYPERPOD_INVENTORY_PAGE_LIMIT} pages"
    )


def _departed_node_evidence(
    site: RenderedSite,
    cluster_id: str,
    node_id: str,
    provider_inventories: dict[str, frozenset[str] | str],
) -> dict[str, Any]:
    """Evidence for a ``never-changed`` item's node that Kubernetes no longer has.

    Restored only when the node name maps to an instance HyperPod no longer
    lists: nothing exists to carry a cordon, a taint or an isolation
    annotation. Every other answer keeps the Kubernetes blocker and adds why
    the provider could not clear it. ``provider_inventories`` caches one
    provider read (or its failure) per cluster for the plan being built.
    """

    missing = _node_scheduling_evidence(node_id, None)
    match = HYPERPOD_NODE_NAME.fullmatch(node_id)
    if match is None:
        return {
            **missing,
            "blockers": [
                *missing["blockers"],
                "provider membership unknown: node name carries no HyperPod "
                "instance id",
            ],
        }
    instance_id = match.group(1)
    inventory = provider_inventories.get(cluster_id)
    if inventory is None:
        try:
            inventory = hyperpod_instance_ids(site, cluster_id)
        except BootstrapError as exc:
            inventory = str(exc)
        provider_inventories[cluster_id] = inventory
    if isinstance(inventory, str):
        return {
            **missing,
            "instance_id": instance_id,
            "blockers": [
                *missing["blockers"],
                f"provider membership unknown: {inventory}",
            ],
        }
    if instance_id in inventory:
        return {
            "node_id": node_id,
            "exists": False,
            "provider": HYPERPOD_PROVIDER,
            "instance_id": instance_id,
            "absent_from_provider": False,
            "restored": False,
            "blockers": [
                "node is missing from Kubernetes but HyperPod still lists "
                f"instance {instance_id}"
            ],
        }
    return {
        "node_id": node_id,
        "exists": False,
        "provider": HYPERPOD_PROVIDER,
        "instance_id": instance_id,
        "absent_from_provider": True,
        "restored": True,
        "blockers": [],
    }


def _scheduling_evidence(
    site: RenderedSite,
    items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    provider_inventories: dict[str, frozenset[str] | str] = {}
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        request_id = str(item.get("request_id") or "")
        cluster_id = str(item.get("cluster_id") or "")
        node_ids = sorted({str(value) for value in item.get("node_ids") or []})
        if not cluster_id or not node_ids:
            result[request_id] = {
                "cluster_id": cluster_id or None,
                "nodes": [],
                "restored": False,
                "blockers": ["cluster identity or node set is missing"],
            }
            continue
        inventory = inventories.get(cluster_id)
        if inventory is None:
            inventory = cluster_nodes(site, cluster_id)
            inventories[cluster_id] = inventory
        # Only a never-changed item may have its missing node explained by the
        # provider; a verified-restore item is judged on Kubernetes alone.
        departed_allowed = item.get("terminalization") in DEPARTED_NODE_TERMINALIZATIONS
        nodes = [
            _departed_node_evidence(site, cluster_id, node_id, provider_inventories)
            if departed_allowed and inventory.get(node_id) is None
            else _node_scheduling_evidence(node_id, inventory.get(node_id))
            for node_id in node_ids
        ]
        blockers = [
            f"{node['node_id']}: {reason}"
            for node in nodes
            for reason in node["blockers"]
        ]
        result[request_id] = {
            "cluster_id": cluster_id,
            "nodes": nodes,
            "restored": not blockers,
            "blockers": blockers,
        }
    return result


def plan_digest_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The items reduced to the fields the digest binds.

    The admin layer re-hashes the runtime items with the site identity and the
    node evidence, so it needs the same trimming the runtime uses: hashing
    ``workflow_updated_at`` left the apply unwinnable no matter what the runtime
    digest did (P0-72A). See ``retired_generation.DIGEST_EXCLUDED_ITEM_FIELDS``.
    """

    return retired_generation.plan_digest_items(items)


def _plan_drift(
    saved_items: list[dict[str, Any]],
    current_items: list[dict[str, Any]],
) -> list[str]:
    """Which item fields moved between the plan and the re-plan before apply.

    "plan changed before apply" on its own told the operator nothing: the record
    is live, so the plan changing is the expected case, and the useful question
    is *what* changed -- a dispatch tick, a re-plan, or a node whose state
    drifted. Compared over the digest fields, so a field the digest ignores is
    never named as the cause.
    """

    saved = {
        str(item.get("request_id")): item for item in plan_digest_items(saved_items)
    }
    current = {
        str(item.get("request_id")): item for item in plan_digest_items(current_items)
    }
    drift: list[str] = []
    for request_id in sorted(set(saved) - set(current)):
        drift.append(f"{request_id}: no longer in the plan")
    for request_id in sorted(set(current) - set(saved)):
        drift.append(f"{request_id}: newly in the plan")
    for request_id in sorted(set(saved) & set(current)):
        before, after = saved[request_id], current[request_id]
        for field in sorted(set(before) | set(after)):
            if before.get(field) != after.get(field):
                drift.append(
                    f"{request_id}: {field} {before.get(field)!r} -> "
                    f"{after.get(field)!r}"
                )
    return drift


def _finalize_plan(
    site: RenderedSite,
    runtime_plan: dict[str, Any],
) -> dict[str, Any]:
    raw_items = runtime_plan.get("items")
    if not isinstance(raw_items, list) or not all(
        isinstance(item, dict) for item in raw_items
    ):
        raise BootstrapError("workflow reconcile runtime plan is invalid")
    items = [dict(item) for item in raw_items]
    for item in items:
        promote_non_plan_successor(item)
        promote_non_plan_never_changed(item)
    scheduling = _scheduling_evidence(site, items)
    for item in items:
        evidence = scheduling[str(item.get("request_id") or "")]
        item["scheduling_evidence"] = evidence
        if not evidence["restored"]:
            item["eligible"] = False
            item["reasons"] = [
                *list(item.get("reasons") or []),
                "GPU node scheduling state has not been restored",
            ]
    plan = {
        "schema_version": 2,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": runtime_plan.get("evaluated_at"),
        "site_identity": _site_identity(site),
        "runtime_plan_sha256": runtime_plan.get("plan_sha256"),
        "discovery": runtime_plan.get("discovery"),
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            "site_identity": plan["site_identity"],
            "runtime_plan_sha256": plan["runtime_plan_sha256"],
            "items": plan_digest_items(items),
        }
    )
    return plan


def _plan(
    site: RenderedSite,
    *,
    workflow_ids: Sequence[str],
    incident_ids: Sequence[str] = (),
    max_items: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": "plan", "workflow_ids": list(workflow_ids)}
    if incident_ids:
        payload["incident_ids"] = list(incident_ids)
    if max_items is not None:
        payload["max_items"] = int(max_items)
    return _finalize_plan(site, _run_reconcile(site, payload))


def _ineligible(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(item.get("request_id")): [
            str(reason) for reason in item.get("reasons") or []
        ]
        for item in items
        if not item.get("eligible")
    }


def _validate_request(
    *,
    workflow_ids: Sequence[str],
    incident_ids: Sequence[str],
    max_items: int | None,
    reference: str | None,
    dry_run: bool,
) -> str | None:
    """Refuse a request whose flags would otherwise be silently ignored.

    Explicit ``--workflow-id``s bypass discovery, so a selector given with them
    (``--incident-id``, ``--max-items``) would do nothing; the reference is what
    the audit trail keys on, so an apply without one is refused before any read.
    Returns the normalised reference, or ``None`` on a dry run without one.
    """

    if workflow_ids and (incident_ids or max_items is not None):
        raise BootstrapError(
            "--incident-id and --max-items select BLOCKED records for discovery; "
            "do not combine them with --workflow-id"
        )
    if max_items is not None and max_items < 1:
        raise BootstrapError("--max-items must be at least 1")
    normalized = (reference or "").strip()
    if not normalized:
        if dry_run:
            return None
        raise BootstrapError(
            "workflow-reconcile requires --reference (an approved change or "
            "maintenance-window reference) unless --dry-run is given"
        )
    if not REFERENCE_PATTERN.fullmatch(normalized):
        raise BootstrapError("workflow reconcile reference is invalid")
    return normalized


def _apply(
    site: RenderedSite,
    current: dict[str, Any],
    *,
    eligible_ids: list[str],
    reference: str,
    actor: str,
) -> dict[str, Any]:
    """Apply the eligible items: the deployed apply for the records it
    accepts, a bridge (``BRIDGE_MODES``) for each promoted shape.

    The deployed apply rebuilds its plan over exactly the ids it is handed and
    compares the digest, so when bridged items are set apart the remaining ids
    are re-planned on their own and checked for drift against ``current``
    before that digest is sent. Every apply is per-record isolated in the
    Pod; the results merge into one.
    """

    bridged = [
        item for item in current["items"] if item.get("terminalization") in BRIDGE_MODES
    ]
    bridged_ids = {str(item["request_id"]) for item in bridged}
    runtime_ids = [item for item in eligible_ids if item not in bridged_ids]
    results: list[dict[str, Any]] = []
    if runtime_ids:
        runtime_plan = current
        if bridged:
            runtime_plan = _plan(site, workflow_ids=runtime_ids)
            drift = _plan_drift(
                [
                    item
                    for item in current["items"]
                    if str(item.get("request_id")) in runtime_ids
                ],
                list(runtime_plan["items"]),
            )
            if drift:
                raise BootstrapError(
                    "workflow reconcile plan changed before apply: " + " | ".join(drift)
                )
        results.append(
            _run_reconcile(
                site,
                _apply_payload(
                    workflow_ids=runtime_ids,
                    runtime_plan_sha256=runtime_plan["runtime_plan_sha256"],
                    admin_plan_sha256=current["plan_sha256"],
                    reference=reference,
                    actor=actor,
                ),
            )
        )
    for terminalization, mode in BRIDGE_MODES.items():
        group = [
            item for item in bridged if item.get("terminalization") == terminalization
        ]
        if group:
            results.append(
                _run_reconcile(
                    site,
                    _bridge_payload(
                        mode=mode,
                        items=group,
                        admin_plan_sha256=current["plan_sha256"],
                        reference=reference,
                        actor=actor,
                    ),
                )
            )
    return _merge_apply_results(results)


def run_workflow_reconcile(
    site: RenderedSite,
    state_dir: Path,
    *,
    workflow_ids: Sequence[str] = (),
    incident_ids: Sequence[str] = (),
    max_items: int | None = None,
    reference: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Plan, and unless ``dry_run``, apply, in one invocation.

    Discovery (no ``--workflow-id``) may name records that are not eligible;
    those are reported under ``ineligible`` and skipped. A record the operator
    named explicitly that is not eligible refuses the whole apply before any
    write, with its reasons: the operator asked for something the evidence does
    not support, and ``--dry-run`` shows why. Eligible records are re-planned
    immediately before the apply and any field that moved refuses the apply by
    name. Each record is written on its own inside the Pod; ``failed_workflow_ids``
    and ``failures`` say which did not land, and the CLI exits 1 on any.
    """

    if any(not str(item).strip() for item in (*workflow_ids, *incident_ids)):
        raise BootstrapError("workflow reconcile selectors must not be blank")
    requested = [str(item).strip() for item in workflow_ids]
    normalized_reference = _validate_request(
        workflow_ids=requested,
        incident_ids=incident_ids,
        max_items=max_items,
        reference=reference,
        dry_run=dry_run,
    )
    plan = _plan(
        site,
        workflow_ids=requested,
        incident_ids=incident_ids,
        max_items=max_items,
    )
    if dry_run:
        return {**plan, "dry_run": True}
    assert normalized_reference is not None
    ineligible = _ineligible(plan["items"])
    if requested and ineligible:
        raise BootstrapError(
            "workflow reconcile refuses ineligible records: "
            + " | ".join(
                f"{request_id}: " + "; ".join(reasons)
                for request_id, reasons in sorted(ineligible.items())
            )
        )
    eligible_ids = [
        str(item["request_id"]) for item in plan["items"] if item.get("eligible")
    ]
    result: dict[str, Any]
    if not eligible_ids:
        result = {
            "mode": "workflow-reconcile-apply",
            "applied_workflow_ids": [],
            "failed_workflow_ids": [],
            "failures": {},
            "records_deleted": 0,
        }
    else:
        # Re-planned for exactly the records about to be written: the runtime
        # digest binds the item set, and a discovery plan may hold ineligible
        # items beside them. Compared field by field with the plan just made.
        current = _plan(site, workflow_ids=eligible_ids)
        drift = _plan_drift(
            [item for item in plan["items"] if item["request_id"] in eligible_ids],
            list(current["items"]),
        )
        if drift:
            raise BootstrapError(
                "workflow reconcile plan changed before apply: " + " | ".join(drift)
            )
        still_ineligible = _ineligible(current["items"])
        if still_ineligible:
            raise BootstrapError(
                "workflow reconcile plan contains ineligible records: "
                + " | ".join(
                    f"{request_id}: " + "; ".join(reasons)
                    for request_id, reasons in sorted(still_ineligible.items())
                )
            )
        actor = operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
        result = _apply(
            site,
            current,
            eligible_ids=eligible_ids,
            reference=normalized_reference,
            actor=actor,
        )
        result["actor"] = actor
        result["admin_plan_sha256"] = current["plan_sha256"]
        plan = current
    result.setdefault("records_deleted", 0)
    result["reference"] = normalized_reference
    result["plan_sha256"] = plan["plan_sha256"]
    result["ineligible"] = ineligible
    result["dry_run"] = False
    archive = state_dir / HISTORY_PATH / str(plan["plan_sha256"])
    write_json_atomic(archive / "plan.json", plan)
    write_json_atomic(archive / "applied.json", result)
    return result
