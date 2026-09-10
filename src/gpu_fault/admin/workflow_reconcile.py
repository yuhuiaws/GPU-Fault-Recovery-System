"""``gpu-fault-admin workflow-reconcile``: close BLOCKED records a later workflow restored.

One shape, one command. A workflow that is BLOCKED and never changed a node --
or whose incident a *later* workflow already recovered and whose node a
``RESTORE_SCHEDULING`` already put back -- is paperwork the runtime cannot
close on its own, because the proof needs node evidence the control plane does
not hold: the GPU node carries no cordon, no quarantine taint and no gpu-fault
isolation annotation. This module reads that evidence through the site's own
GPU kubeconfig and hands the verdict to ``gpu_fault.workflow_reconcile`` in the
CPU ingress Pod.

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
import subprocess
from pathlib import Path
from typing import Any, Sequence

from gpu_fault import retired_generation
from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite

REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
HISTORY_PATH = Path("workflow-reconcile/history")
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
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
    pod_result = subprocess.run(
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
        check=False,
        capture_output=True,
        text=True,
    )
    pod = pod_result.stdout.strip()
    if pod_result.returncode or not pod:
        raise BootstrapError("workflow reconcile found no Running CPU ingress Pod")
    completed = subprocess.run(
        [*kubectl, "exec", "-i", pod, "--", "python", "-c", script],
        input=json.dumps(payload, separators=(",", ":")),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise BootstrapError(
            completed.stderr.strip() or "workflow reconcile execution failed"
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


def cluster_nodes(
    site: RenderedSite,
    cluster_id: str,
) -> dict[str, dict[str, Any]]:
    targets = {
        str(item["cluster_id"]): item for item in site.release_config["clusters"]
    }
    target = targets.get(cluster_id)
    if target is None:
        raise BootstrapError(
            f"workflow reconcile cluster is not in the managed site: {cluster_id}"
        )
    completed = subprocess.run(
        [*_gpu_kubectl(site, target), "get", "nodes", "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise BootstrapError(
            f"workflow reconcile cannot read GPU nodes for {cluster_id}: "
            f"{completed.stderr.strip()}"
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


def _scheduling_evidence(
    site: RenderedSite,
    items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
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
        nodes = [
            _node_scheduling_evidence(node_id, inventory.get(node_id))
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

    requested = [str(item).strip() for item in workflow_ids if str(item).strip()]
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
        result = _run_reconcile(
            site,
            _apply_payload(
                workflow_ids=eligible_ids,
                runtime_plan_sha256=current["runtime_plan_sha256"],
                admin_plan_sha256=current["plan_sha256"],
                reference=normalized_reference,
                actor=actor,
            ),
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
