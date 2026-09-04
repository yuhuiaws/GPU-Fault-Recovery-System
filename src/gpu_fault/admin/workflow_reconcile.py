from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from gpu_fault import retired_generation
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.digests import SHA256_PATTERN

REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
PLAN_PATH = Path("workflow-reconcile/plan.json")
HISTORY_PATH = Path("workflow-reconcile/history")
RETIRED_GENERATION_PLAN_PATH = Path("workflow-reconcile/retired-generation/plan.json")
RETIRED_GENERATION_HISTORY_PATH = Path("workflow-reconcile/retired-generation/history")
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
ISOLATION_ANNOTATIONS = (
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
)
RECONCILE_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)

payload = json.load(sys.stdin)
store = ApplicationContext.from_environment().store
if payload["mode"] == "plan":
    result = build_workflow_reconcile_plan(
        store,
        payload.get("workflow_ids"),
    )
elif payload["mode"] == "apply":
    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=payload["workflow_ids"],
        expected_plan_sha256=payload["plan_sha256"],
        reference=payload["reference"],
    )
else:
    raise ValueError("unsupported workflow reconcile mode")
print(json.dumps(result, sort_keys=True))
"""
RETIRED_GENERATION_DRIVER = """

import json as _json
import sys as _sys

from gpu_fault.app import ApplicationContext

_payload = _json.load(_sys.stdin)
_store = ApplicationContext.from_environment().store
if _payload["mode"] == "plan":
    _result = build_retired_generation_plan(_store, _payload.get("workflow_ids"))
elif _payload["mode"] == "apply":
    _result = apply_retired_generation_plan(
        _store,
        workflow_ids=_payload["workflow_ids"],
        expected_plan_sha256=_payload["plan_sha256"],
        reference=_payload["reference"],
    )
else:
    raise ValueError("unsupported retired generation reconcile mode")
print(_json.dumps(_result, sort_keys=True))
"""


def retired_generation_script() -> str:
    """``gpu_fault.retired_generation``'s own source, plus a stdin/stdout driver.

    The other mode above imports its logic from the Pod's ``gpu_fault``. This one
    cannot. A retired generation is precisely the record the release preflight
    refuses to roll past, so the operator has to close it *before* the release
    that carries the code for closing it -- against an image whose
    ``gpu_fault.retired_generation`` does not exist yet. The engine has the same
    problem with its control-plane probes and solves it the same way: ship the
    decision as source and run it against the deployed runtime.

    Shipping the module's own file, rather than a hand-copied excerpt of it, is
    what keeps the two from drifting: there is no second copy to update.
    ``tests/admin/test_retired_generation_reconcile.py`` pins that this script
    still compiles, still calls the two entry points the driver names, and still
    imports nothing the previously deployed image may lack.
    """

    source = Path(retired_generation.__file__).read_text(encoding="utf-8")
    return source + RETIRED_GENERATION_DRIVER


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run_reconcile(
    site: RenderedSite,
    payload: dict[str, Any],
    *,
    script: str = RECONCILE_SCRIPT,
) -> dict[str, Any]:
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


def _gpu_kubectl(site: RenderedSite, target: dict[str, Any]) -> list[str]:
    command = ["kubectl"]
    kubeconfig = (
        site.release_config.get("gpu_kubeconfig")
        or site.environment.get("KUBECONFIG")
        or os.environ.get("KUBECONFIG")
        or str(Path.home() / ".kube/config")
    )
    command.extend(
        [
            "--kubeconfig",
            str(kubeconfig),
            "--context",
            str(target["context"]),
        ]
    )
    return command


def _cluster_nodes(
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
    quarantine = any(
        isinstance(item, dict) and item.get("key") == QUARANTINE_TAINT
        for item in taints
    )
    isolation_annotations = sorted(
        key for key in ISOLATION_ANNOTATIONS if annotations.get(key) not in (None, "")
    )
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
        "isolation_annotations": isolation_annotations,
        "restored": not blockers,
        "blockers": blockers,
    }


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
            inventory = _cluster_nodes(site, cluster_id)
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
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            "site_identity": plan["site_identity"],
            "runtime_plan_sha256": plan["runtime_plan_sha256"],
            "items": plan["items"],
        }
    )
    return plan


def plan_workflow_reconcile(
    site: RenderedSite,
    state_dir: Path,
    *,
    workflow_ids: Sequence[str],
) -> dict[str, Any]:
    runtime_plan = _run_reconcile(
        site,
        {"mode": "plan", "workflow_ids": list(workflow_ids)},
    )
    plan = _finalize_plan(site, runtime_plan)
    write_json_atomic(state_dir / PLAN_PATH, plan)
    return plan


def _finalize_retired_generation_plan(
    site: RenderedSite,
    runtime_plan: dict[str, Any],
) -> dict[str, Any]:
    raw_items = runtime_plan.get("items")
    if not isinstance(raw_items, list) or not all(
        isinstance(item, dict) for item in raw_items
    ):
        raise BootstrapError("retired generation reconcile runtime plan is invalid")
    items: list[dict[str, Any]] = [dict(item) for item in raw_items]
    plan = {
        "schema_version": 1,
        "mode": "retired-generation-plan",
        "evaluated_at": runtime_plan.get("evaluated_at"),
        "site_identity": _site_identity(site),
        "runtime_plan_sha256": runtime_plan.get("plan_sha256"),
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            "site_identity": plan["site_identity"],
            "runtime_plan_sha256": plan["runtime_plan_sha256"],
            # Same exclusion as the runtime digest, and for the same reason: a
            # record still being dispatched restamps ``updated_at`` every tick,
            # and hashing it here would leave the apply unwinnable no matter what
            # the runtime digest did. See DIGEST_EXCLUDED_ITEM_FIELDS.
            "items": retired_generation.plan_digest_items(items),
        }
    )
    return plan


def plan_retired_generation_reconcile(
    site: RenderedSite,
    state_dir: Path,
    *,
    workflow_ids: Sequence[str],
) -> dict[str, Any]:
    runtime_plan = _run_reconcile(
        site,
        {"mode": "plan", "workflow_ids": list(workflow_ids)},
        script=retired_generation_script(),
    )
    plan = _finalize_retired_generation_plan(site, runtime_plan)
    write_json_atomic(state_dir / RETIRED_GENERATION_PLAN_PATH, plan)
    return plan


def apply_retired_generation_reconcile(
    site: RenderedSite,
    state_dir: Path,
    *,
    expected_plan_sha256: str,
    reference: str,
) -> dict[str, Any]:
    digest = expected_plan_sha256.strip()
    normalized_reference = reference.strip()
    if not SHA256_PATTERN.fullmatch(digest):
        raise BootstrapError("retired generation reconcile plan SHA-256 is invalid")
    if not REFERENCE_PATTERN.fullmatch(normalized_reference):
        raise BootstrapError("retired generation reconcile reference is invalid")
    plan_path = state_dir / RETIRED_GENERATION_PLAN_PATH
    if not plan_path.is_file():
        raise BootstrapError("retired generation reconcile has no saved plan")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(
            "retired generation reconcile saved plan is invalid"
        ) from exc
    if not isinstance(plan, dict) or plan.get("plan_sha256") != digest:
        raise BootstrapError(
            "retired generation reconcile saved plan SHA-256 does not match"
        )
    if plan.get("mode") != "retired-generation-plan":
        raise BootstrapError("saved plan is not a retired generation plan")
    if plan.get("site_identity") != _site_identity(site):
        raise BootstrapError(
            "retired generation reconcile managed site identity changed"
        )
    items = plan.get("items")
    if not isinstance(items, list) or not items:
        raise BootstrapError("retired generation reconcile saved plan has no workflows")
    workflow_ids = [str(item["request_id"]) for item in items]
    script = retired_generation_script()
    runtime_plan = _run_reconcile(
        site,
        {"mode": "plan", "workflow_ids": workflow_ids},
        script=script,
    )
    current = _finalize_retired_generation_plan(site, runtime_plan)
    if current["plan_sha256"] != digest:
        raise BootstrapError("retired generation reconcile plan changed before apply")
    # Refused here as well as in the runtime, so the operator reads the reasons
    # from the plan they approved instead of from a Pod's stderr. An item whose
    # only blocker is an open remote command is *not* refused: cancelling those
    # is the first half of the apply.
    blocked = [
        f"{item.get('request_id')}: " + "; ".join(item.get("reasons") or [])
        for item in current["items"]
        if not item.get("eligible") and not item.get("cancellable")
    ]
    if blocked:
        raise BootstrapError(
            "retired generation reconcile plan contains ineligible records: "
            + " | ".join(blocked)
        )
    result = _run_reconcile(
        site,
        {
            "mode": "apply",
            "workflow_ids": workflow_ids,
            "plan_sha256": current["runtime_plan_sha256"],
            "reference": normalized_reference,
        },
        script=script,
    )
    result["admin_plan_sha256"] = digest
    archive = state_dir / RETIRED_GENERATION_HISTORY_PATH / digest
    write_json_atomic(archive / "plan.json", plan)
    write_json_atomic(archive / "applied.json", result)
    plan_path.unlink(missing_ok=True)
    return result


def apply_workflow_reconcile(
    site: RenderedSite,
    state_dir: Path,
    *,
    expected_plan_sha256: str,
    reference: str,
) -> dict[str, Any]:
    digest = expected_plan_sha256.strip()
    normalized_reference = reference.strip()
    if not SHA256_PATTERN.fullmatch(digest):
        raise BootstrapError("workflow reconcile plan SHA-256 is invalid")
    if not REFERENCE_PATTERN.fullmatch(normalized_reference):
        raise BootstrapError("workflow reconcile reference is invalid")
    plan_path = state_dir / PLAN_PATH
    if not plan_path.is_file():
        raise BootstrapError("workflow reconcile has no saved plan")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError("workflow reconcile saved plan is invalid") from exc
    if not isinstance(plan, dict) or plan.get("plan_sha256") != digest:
        raise BootstrapError("workflow reconcile saved plan SHA-256 does not match")
    if plan.get("site_identity") != _site_identity(site):
        raise BootstrapError("workflow reconcile managed site identity changed")
    items = plan.get("items")
    if not isinstance(items, list) or not items:
        raise BootstrapError("workflow reconcile saved plan has no workflows")
    workflow_ids = [str(item["request_id"]) for item in items]
    runtime_plan = _run_reconcile(
        site,
        {"mode": "plan", "workflow_ids": workflow_ids},
    )
    current = _finalize_plan(site, runtime_plan)
    if current["plan_sha256"] != digest:
        raise BootstrapError("workflow reconcile plan changed before apply")
    rejected = [item for item in current["items"] if not item.get("eligible")]
    if rejected:
        raise BootstrapError("workflow reconcile plan contains ineligible records")
    result = _run_reconcile(
        site,
        {
            "mode": "apply",
            "workflow_ids": workflow_ids,
            "plan_sha256": current["runtime_plan_sha256"],
            "reference": normalized_reference,
        },
    )
    result["admin_plan_sha256"] = digest
    archive = state_dir / HISTORY_PATH / digest
    write_json_atomic(archive / "plan.json", plan)
    write_json_atomic(archive / "applied.json", result)
    plan_path.unlink(missing_ok=True)
    return result
