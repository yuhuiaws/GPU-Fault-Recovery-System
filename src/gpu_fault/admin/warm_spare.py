"""``gpu-fault-admin config spare``: declare or release one warm spare GPU node.

``REPLACE_NODE`` fails over only to a node the operator declared as a warm
spare: labeled ``gpu-fault.io/spare=true`` *and* cordoned, monitored by an
ACTIVE Node Agent, unreserved and untainted. The declaration used to be a hand
``kubectl label`` plus ``kubectl cordon`` -- the manual step that leaves nothing
to restore from -- and then an acceptance-only script bound to a site profile.
This module is the sanctioned lever: it rides the ``config`` verb, keeps its
record under the state directory, and its console output is teed into the
admin command log like every other mutating verb.

It deliberately performs the smallest mutation a declaration can be: the spare
label and the cordon. It never writes ``gpu-fault.io/spare-pool-state``; that
annotation belongs to the control plane, and the failover path accepts it
absent. Writing ``AVAILABLE`` by hand would fabricate the state the pool's own
health check exists to observe.

A node that is labeled but schedulable is accepted for re-declaration: a
validated restore that released the spare's quarantine also uncordoned it
(DESTR-003 cleanup, 2026-09-08), and the pool refuses an "unreserved spare
[that] is schedulable". Re-declaring completes the cordon. Release restores
exactly the recorded baseline, so a node cordoned before the declaration stays
cordoned afterwards.

Cluster access follows the other admin verbs: ``kubectl`` bound to the site's
GPU kubeconfig and the cluster's context. The access object exposes the three
``CoreV1Api`` method names the flow uses, so a test -- or a Kubernetes client
-- can stand in for it; the Node Agent check reaches the Store through the same
exec channel ``workflow-reconcile`` uses and is a named refusal, never a silent
pass, when that channel is unavailable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import REFERENCE_PATTERN
from gpu_fault.admin.workflow_reconcile import (
    _run_reconcile as run_control_plane_script,
)

SPARE_LABEL = "gpu-fault.io/spare"
SPARE_RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
SPARE_RESERVED_AT_ANNOTATION = "gpu-fault.io/spare-reserved-at"
SPARE_POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
HYPERPOD_HEALTH_LABEL = "sagemaker.amazonaws.com/node-health-status"
INSTANCE_GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"
INSTANCE_TYPE_LABELS = (
    "node.kubernetes.io/instance-type",
    "beta.kubernetes.io/instance-type",
)
OWNERSHIP_ANNOTATIONS = (
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
)
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
GPU_RESOURCE = "nvidia.com/gpu"
FINISHED_POD_PHASES = frozenset({"Succeeded", "Failed"})

DECLARE_CONFIRMATION = "DECLARE_WARM_SPARE_CORDON"
RELEASE_CONFIRMATION = "RELEASE_WARM_SPARE_UNCORDON"
RECORDS_PATH = Path("warm-spares")
MODES = ("check", "declare", "release")

# Runs inside a CPU ingress Pod: the only place the Store is reachable from an
# administrator command. Prints one JSON object; a missing Agent is a result,
# not an error, so the caller can tell "no Agent" from "no Store".
AGENT_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

payload = json.load(sys.stdin)
store = ApplicationContext.from_environment().store
try:
    agent = store.get_agent(payload["cluster_id"], payload["node_id"])
except NotFoundError:
    print(json.dumps({"found": False}))
else:
    print(json.dumps({"found": True, "lifecycle_state": str(agent.lifecycle_state)}))
"""


class WarmSpareError(BootstrapError):
    """A refusal or failure of the warm-spare flow; reported on one line."""


class NodeApi(Protocol):
    """The ``CoreV1Api`` surface the flow uses; JSON objects or client models."""

    def read_node(self, name: str) -> Any: ...

    def patch_node(self, name: str, body: Mapping[str, Any]) -> Any: ...

    def list_pod_for_all_namespaces(self) -> Any: ...

    def list_node(self, label_selector: str) -> Any: ...


@dataclass(frozen=True)
class AgentState:
    """What the Store says about the node's Agent, or why it could not say."""

    lifecycle_state: str | None
    error: str | None = None


AgentLookup = Callable[[str, str], AgentState]


@dataclass(frozen=True)
class WarmSpareRequest:
    state_dir: Path
    node: str
    fault_node: str = ""
    cluster_id: str | None = None
    reference: str | None = None
    mode: str = "check"
    confirmation: str = ""


class KubectlNodeApi:
    """``NodeApi`` over ``kubectl``, the client every admin verb already uses."""

    def __init__(
        self, kubectl: Sequence[str], *, run: Callable[..., Any] = subprocess.run
    ) -> None:
        self.kubectl = list(kubectl)
        self.run = run

    def _json(self, *arguments: str) -> Any:
        completed = self.run(
            [*self.kubectl, *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise WarmSpareError(
                f"kubectl {' '.join(arguments[:3])} failed: "
                f"{(completed.stderr or '').strip() or 'no error output'}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise WarmSpareError(
                f"kubectl {' '.join(arguments[:3])} returned invalid JSON"
            ) from exc

    def read_node(self, name: str) -> Any:
        return self._json("get", "node", name, "-o", "json")

    def patch_node(self, name: str, body: Mapping[str, Any]) -> Any:
        return self._json(
            "patch",
            "node",
            name,
            "--type=merge",
            "-o",
            "json",
            "-p",
            json.dumps(body, sort_keys=True),
        )

    def list_pod_for_all_namespaces(self) -> Any:
        return self._json("get", "pod", "--all-namespaces", "-o", "json")

    def list_node(self, label_selector: str) -> Any:
        return self._json("get", "node", "-l", label_selector, "-o", "json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def plain(value: Any) -> dict[str, Any]:
    """``value`` as the JSON object Kubernetes serves.

    ``kubectl`` already returns that; a ``kubernetes`` client model is
    serialised through the client's own attribute map so the same snapshot
    code reads both.
    """

    if isinstance(value, dict):
        return value
    from kubernetes.client import ApiClient

    result = ApiClient().sanitize_for_serialization(value)
    return dict(result) if isinstance(result, dict) else {}


def listed(value: Any) -> list[dict[str, Any]]:
    """The items of a list response, as JSON objects."""

    return [plain(item) for item in plain(value).get("items", []) or []]


def node_snapshot(raw: Any) -> dict[str, Any]:
    """The fields the refusals read, and nothing that names the site."""

    value = plain(raw)
    metadata = value.get("metadata", {})
    spec = value.get("spec", {})
    status = value.get("status", {})
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    try:
        gpu_allocatable = int(status.get("allocatable", {}).get(GPU_RESOURCE, 0))
    except (TypeError, ValueError):
        gpu_allocatable = 0
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "resource_version": metadata.get("resourceVersion"),
        "provider_id": spec.get("providerID"),
        "ready": next(
            (
                item.get("status")
                for item in status.get("conditions", []) or []
                if item.get("type") == "Ready"
            ),
            None,
        ),
        "gpu_allocatable": gpu_allocatable,
        "unschedulable": bool(spec.get("unschedulable", False)),
        "taints": list(spec.get("taints") or []),
        "labels": {
            key: labels.get(key)
            for key in (
                SPARE_LABEL,
                HYPERPOD_HEALTH_LABEL,
                INSTANCE_GROUP_LABEL,
                *INSTANCE_TYPE_LABELS,
            )
        },
        "annotations": {
            key: annotations.get(key)
            for key in (
                SPARE_RESERVATION_ANNOTATION,
                SPARE_RESERVED_AT_ANNOTATION,
                SPARE_POOL_STATE_ANNOTATION,
                *OWNERSHIP_ANNOTATIONS,
            )
        },
    }


def instance_type(snapshot: Mapping[str, Any]) -> str | None:
    return next(
        (
            snapshot["labels"].get(key)
            for key in INSTANCE_TYPE_LABELS
            if snapshot["labels"].get(key)
        ),
        None,
    )


def topology(snapshot: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "instance_group": snapshot["labels"].get(INSTANCE_GROUP_LABEL),
        "instance_type": instance_type(snapshot),
    }


def pod_gpu_count(pod: Mapping[str, Any]) -> int:
    """The largest ``nvidia.com/gpu`` request or limit across containers."""

    gpu_count = 0
    spec = pod.get("spec", {}) or {}
    for container in [
        *(spec.get("containers") or []),
        *(spec.get("initContainers") or []),
    ]:
        resources = container.get("resources", {}) or {}
        for values in (resources.get("requests") or {}, resources.get("limits") or {}):
            try:
                gpu_count = max(gpu_count, int(values.get(GPU_RESOURCE, 0)))
            except (TypeError, ValueError):
                continue
    return gpu_count


def gpu_workloads(pods: Sequence[Any], node: str) -> list[dict[str, Any]]:
    """Live pods on ``node`` that hold GPUs; cordoning does not evict them."""

    result = []
    for raw in pods:
        pod = plain(raw)
        if (pod.get("spec", {}) or {}).get("nodeName") != node:
            continue
        phase = (pod.get("status", {}) or {}).get("phase")
        if phase in FINISHED_POD_PHASES:
            continue
        gpu_count = pod_gpu_count(pod)
        if gpu_count <= 0:
            continue
        metadata = pod.get("metadata", {}) or {}
        result.append(
            {
                "namespace": metadata.get("namespace"),
                "name": metadata.get("name"),
                "node": node,
                "phase": phase,
                "gpu_count": gpu_count,
            }
        )
    return sorted(result, key=lambda item: (str(item["namespace"]), str(item["name"])))


def declared_spares(api: NodeApi) -> list[str]:
    return sorted(
        str(plain(item).get("metadata", {}).get("name"))
        for item in listed(api.list_node(f"{SPARE_LABEL}=true"))
    )


def has_quarantine_ownership(snapshot: Mapping[str, Any]) -> bool:
    return any(
        item.get("key") == QUARANTINE_TAINT for item in snapshot["taints"]
    ) or any(snapshot["annotations"].get(key) for key in OWNERSHIP_ANNOTATIONS)


def declare_refusals(
    node: str,
    *,
    spare: Mapping[str, Any],
    fault: Mapping[str, Any] | None,
    fault_node: str,
    declared: Sequence[str],
    workloads: Sequence[Mapping[str, Any]],
    agent: AgentState,
) -> list[str]:
    """Why ``node`` cannot honestly be a spare; empty means it can."""

    refusals = []
    if spare["ready"] != "True":
        refusals.append("node is not Ready")
    if spare["labels"].get(SPARE_LABEL) == "true" and spare["unschedulable"]:
        # Already labeled *and* cordoned: nothing to declare. Labeled but
        # schedulable is a half-declared spare and re-declaring completes it.
        refusals.append("node is already declared as a spare and cordoned")
    if spare["labels"].get(HYPERPOD_HEALTH_LABEL) != "Schedulable":
        refusals.append("HyperPod health label is not Schedulable")
    if spare["annotations"].get(SPARE_RESERVATION_ANNOTATION):
        refusals.append("node already carries a spare reservation")
    if spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION) not in {None, "AVAILABLE"}:
        refusals.append("spare pool state is not AVAILABLE")
    if has_quarantine_ownership(spare):
        refusals.append("node carries quarantine ownership from an earlier incident")
    if [item for item in workloads if item.get("node") == node]:
        refusals.append("node still has an active GPU workload")
    if agent.error:
        refusals.append(
            "Node Agent state could not be read from the control-plane store: "
            + agent.error
        )
    elif agent.lifecycle_state != "ACTIVE":
        refusals.append("node does not have exactly one ACTIVE Agent")
    if node in declared and spare["unschedulable"]:
        refusals.append("node is already in the declared spare set")
    if fault is not None:
        if node == fault_node:
            refusals.append("spare and fault node are identical")
        if topology(fault) != topology(spare):
            refusals.append(
                "topology does not match the fault node: "
                f"{topology(fault)} vs {topology(spare)}"
            )
    return refusals


def release_refusals(
    spare: Mapping[str, Any], record: Mapping[str, Any] | None
) -> list[str]:
    refusals = []
    if spare["annotations"].get(SPARE_RESERVATION_ANNOTATION):
        # The pool allocated this node to an incident; uncordoning it now hands
        # a reserved spare back to the scheduler under a running workflow.
        refusals.append(
            "node is reserved by an incident; release it through the incident"
        )
    if spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION) == "ALLOCATED":
        # The pool state outlives the reservation on some paths (a failover
        # whose restart is still running); the node is carrying the job.
        refusals.append(
            "node is ALLOCATED by the spare pool; release it through the incident"
        )
    if has_quarantine_ownership(spare):
        refusals.append(
            "node is quarantined; restore the quarantine before releasing the spare"
        )
    if record is None or record.get("released_at"):
        refusals.append(
            "node was not declared by gpu-fault-admin (no unreleased record)"
        )
    return refusals


def record_path(state_dir: Path, node: str) -> Path:
    """``<state-dir>/warm-spares/<node>.json``; one record per node."""

    if not node or "/" in node or node in {".", ".."} or node != node.strip():
        raise WarmSpareError(f"invalid node name for a warm-spare record: {node!r}")
    return state_dir.expanduser().resolve() / RECORDS_PATH / f"{node}.json"


def read_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WarmSpareError(f"warm-spare record {path} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WarmSpareError(f"warm-spare record {path} is not a JSON object")
    return value


def survey(
    api: NodeApi,
    *,
    node: str,
    fault_node: str,
    cluster_id: str,
    agent_lookup: AgentLookup,
) -> dict[str, Any]:
    """Everything the refusals read, gathered once; also the run's report."""

    spare = node_snapshot(api.read_node(node))
    agent = agent_lookup(cluster_id, node)
    result: dict[str, Any] = {
        "observed_at": now(),
        "cluster_id": cluster_id,
        "node": spare,
        "topology": topology(spare),
        "declared_spares": declared_spares(api),
        "gpu_workloads": gpu_workloads(listed(api.list_pod_for_all_namespaces()), node),
        "agent": {"lifecycle_state": agent.lifecycle_state, "error": agent.error},
    }
    if fault_node:
        fault = node_snapshot(api.read_node(fault_node))
        result["fault_node"] = fault
        result["fault_topology"] = topology(fault)
    return result


def without_survey(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record minus the embedded surveys, so a report never contains itself."""

    return {key: value for key, value in record.items() if not key.endswith("_survey")}


def _patch(
    api: NodeApi, node: str, *, spare_label: str | None, unschedulable: bool
) -> dict[str, Any]:
    api.patch_node(
        node,
        {
            "metadata": {"labels": {SPARE_LABEL: spare_label}},
            "spec": {"unschedulable": unschedulable},
        },
    )
    return node_snapshot(api.read_node(node))


def declare(
    api: NodeApi,
    *,
    node: str,
    record: Path,
    reference: str,
    actor: str,
    survey: Mapping[str, Any],
) -> dict[str, Any]:
    """Label and cordon ``node``; the baseline is on disk before the patch."""

    existing = read_record(record)
    if existing is not None and not existing.get("released_at"):
        raise WarmSpareError(
            f"{record} still records an unreleased declaration; "
            "release it before declaring again"
        )
    before = survey["node"]
    document: dict[str, Any] = {
        "node": node,
        "cluster_id": survey.get("cluster_id"),
        "declared_at": now(),
        "confirmation": DECLARE_CONFIRMATION,
        "reference": reference,
        "actor": actor,
        "baseline": {
            "labels": {SPARE_LABEL: before["labels"].get(SPARE_LABEL)},
            "unschedulable": bool(before["unschedulable"]),
        },
        "pre_declaration_survey": dict(survey),
    }
    # Written before the mutation, so an interrupted declaration still leaves
    # an exact record of what to put back.
    write_json_atomic(record, document)
    after = _patch(api, node, spare_label="true", unschedulable=True)
    if after["labels"].get(SPARE_LABEL) != "true" or not after["unschedulable"]:
        raise WarmSpareError(
            "declaration did not take effect: the node is not labeled and cordoned"
        )
    document["declared_state"] = after
    document["declared_spares"] = declared_spares(api)
    write_json_atomic(record, document)
    return document


def release(
    api: NodeApi,
    *,
    node: str,
    record: Path,
    reference: str,
    actor: str,
    survey: Mapping[str, Any],
) -> dict[str, Any]:
    """Put back exactly what the declaration recorded."""

    document = read_record(record)
    if document is None:
        raise WarmSpareError(f"no warm-spare record at {record}")
    if document.get("released_at"):
        raise WarmSpareError(
            f"{record} was already released at {document['released_at']}"
        )
    if document.get("node") != node:
        raise WarmSpareError(
            f"{record} records node {document.get('node')}, not {node}"
        )
    baseline = document.get("baseline") or {}
    # Restore against the recorded baseline rather than the node as it is now,
    # so a node that was already cordoned before the declaration stays cordoned.
    restored = _patch(
        api,
        node,
        spare_label=(baseline.get("labels") or {}).get(SPARE_LABEL),
        unschedulable=bool(baseline.get("unschedulable", False)),
    )
    document["released_at"] = now()
    document["release_reference"] = reference
    document["release_actor"] = actor
    document["release_survey"] = dict(survey)
    document["restored_state"] = restored
    write_json_atomic(record, document)
    return document


def control_plane_agent_lookup(site: RenderedSite) -> AgentLookup:
    """``get_agent`` through the CPU ingress Pod; unavailability is a state."""

    def lookup(cluster_id: str, node: str) -> AgentState:
        try:
            result = run_control_plane_script(
                site, {"cluster_id": cluster_id, "node_id": node}, script=AGENT_SCRIPT
            )
        except BootstrapError as exc:
            return AgentState(lifecycle_state=None, error=str(exc))
        if not result.get("found"):
            return AgentState(lifecycle_state=None)
        return AgentState(lifecycle_state=str(result.get("lifecycle_state")))

    return lookup


def site_cluster(site: RenderedSite, cluster_id: str | None) -> dict[str, Any]:
    clusters = {
        str(item["cluster_id"]): dict(item) for item in site.release_config["clusters"]
    }
    if cluster_id:
        if cluster_id not in clusters:
            raise WarmSpareError(
                f"cluster {cluster_id} is not in the managed site "
                f"(managed: {', '.join(sorted(clusters)) or 'none'})"
            )
        return clusters[cluster_id]
    if len(clusters) == 1:
        return next(iter(clusters.values()))
    raise WarmSpareError(
        "the site manages "
        f"{len(clusters)} GPU clusters; name the node's cluster with --cluster-id "
        f"(one of: {', '.join(sorted(clusters)) or 'none'})"
    )


def gpu_kubectl_command(site: RenderedSite, target: Mapping[str, Any]) -> list[str]:
    """``kubectl`` bound to the site's GPU kubeconfig, or a refusal.

    A cordon must land on the cluster the site manages. The shell's
    ``KUBECONFIG`` and ``~/.kube/config`` are whatever the operator last
    pointed at, so a site without a rendered GPU kubeconfig is refused rather
    than read -- or written -- through the default.
    """

    kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if not kubeconfig:
        raise WarmSpareError(
            "config spare has no GPU kubeconfig for cluster "
            f"{target['cluster_id']}: the managed site renders none, and the "
            "default kubeconfig is never used to cordon a node"
        )
    return [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        str(target["context"]),
    ]


def node_api_for_site(site: RenderedSite, target: Mapping[str, Any]) -> KubectlNodeApi:
    return KubectlNodeApi(gpu_kubectl_command(site, target))


def _approved_reference(request: WarmSpareRequest) -> str:
    normalized = (request.reference or "").strip()
    if not normalized:
        raise WarmSpareError(
            f"config spare --{request.mode} requires --reference "
            "(an approved change or maintenance-window reference)"
        )
    if not REFERENCE_PATTERN.fullmatch(normalized):
        raise WarmSpareError("config spare reference is invalid")
    return normalized


def _confirmed(request: WarmSpareRequest) -> None:
    expected = {"declare": DECLARE_CONFIRMATION, "release": RELEASE_CONFIRMATION}
    token = expected.get(request.mode)
    if token is None:
        raise WarmSpareError(f"unknown warm-spare mode {request.mode!r}")
    if request.confirmation != token:
        raise WarmSpareError(
            f"config spare --{request.mode} requires --confirm {token}"
        )


def perform_warm_spare(
    api: NodeApi,
    request: WarmSpareRequest,
    *,
    cluster_id: str,
    record: Path,
    agent_lookup: AgentLookup,
    actor: str | None = None,
) -> dict[str, Any]:
    """Check, declare or release against ``api``; the report is the result.

    The site-free core: the CLI binds it to the managed site's cluster and
    record directory (:func:`run_warm_spare`); the acceptance wrapper binds it
    to a site profile and its ``--baseline`` file. ``actor`` defaults to the
    operator's STS identity and is resolved only when something is written.
    """

    if request.mode not in MODES:
        raise WarmSpareError(f"unknown warm-spare mode {request.mode!r}")
    report = survey(
        api,
        node=request.node,
        fault_node=request.fault_node,
        cluster_id=cluster_id,
        agent_lookup=agent_lookup,
    )
    report["mode"] = request.mode
    report["record"] = str(record)
    if request.mode == "release":
        refusals = release_refusals(report["node"], read_record(record))
    else:
        refusals = declare_refusals(
            request.node,
            spare=report["node"],
            fault=report.get("fault_node"),
            fault_node=request.fault_node,
            declared=report["declared_spares"],
            workloads=report["gpu_workloads"],
            agent=AgentState(
                lifecycle_state=report["agent"]["lifecycle_state"],
                error=report["agent"]["error"],
            ),
        )
    report["refusals"] = refusals
    report["ready"] = not refusals
    if request.mode == "check":
        return report
    _confirmed(request)
    reference = _approved_reference(request)
    if refusals:
        verb = "declared a warm spare" if request.mode == "declare" else "released"
        raise WarmSpareError(f"node cannot be {verb}: " + "; ".join(refusals))
    if actor is None:
        actor = operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
    report["reference"] = reference
    report["actor"] = actor
    action = declare if request.mode == "declare" else release
    document = action(
        api,
        node=request.node,
        record=record,
        reference=reference,
        actor=actor,
        survey=report,
    )
    report["declaration" if request.mode == "declare" else "release"] = without_survey(
        document
    )
    return report


def run_warm_spare(
    site: RenderedSite,
    request: WarmSpareRequest,
    *,
    api: NodeApi | None = None,
    agent_lookup: AgentLookup | None = None,
) -> dict[str, Any]:
    """The managed-site binding: the cluster, kubectl, Store channel and record
    directory all come from the site and the state directory."""

    target = site_cluster(site, request.cluster_id)
    return perform_warm_spare(
        api if api is not None else node_api_for_site(site, target),
        request,
        cluster_id=str(target["cluster_id"]),
        record=record_path(request.state_dir, request.node),
        agent_lookup=(
            agent_lookup
            if agent_lookup is not None
            else control_plane_agent_lookup(site)
        ),
    )


def add_config_spare_command(config: argparse.ArgumentParser) -> None:
    """``config spare``: the one sub-action the ``config`` verb carries."""

    actions = config.add_subparsers(dest="config_action", metavar="{spare}")
    spare = actions.add_parser(
        "spare",
        usage=(
            "gpu-fault-admin config spare --state-dir STATE_DIR --node NODE "
            "[--fault-node NODE] [--cluster-id ID] [--reference REFERENCE] "
            f"[--declare --confirm {DECLARE_CONFIRMATION} | "
            f"--release --confirm {RELEASE_CONFIRMATION}]"
        ),
        help=(
            "declare one GPU node a warm spare (label + cordon) or release it; "
            "read-only readiness check without --declare/--release"
        ),
    )
    spare.add_argument("--state-dir", required=True, type=Path, metavar="STATE_DIR")
    spare.add_argument(
        "--node", required=True, metavar="NODE", help="the GPU node to declare"
    )
    spare.add_argument(
        "--fault-node",
        default="",
        metavar="NODE",
        help="optional: verify the spare matches this node's topology",
    )
    spare.add_argument(
        "--cluster-id",
        default=None,
        metavar="CLUSTER_ID",
        help="the node's GPU cluster; needed only when the site manages several",
    )
    spare.add_argument(
        "--reference",
        metavar="REFERENCE",
        help="approved change or maintenance-window reference; required to mutate",
    )
    mode = spare.add_mutually_exclusive_group()
    mode.add_argument("--declare", action="store_true", help="label and cordon")
    mode.add_argument(
        "--release", action="store_true", help="restore the recorded baseline"
    )
    spare.add_argument(
        "--confirm",
        default="",
        metavar="TOKEN",
        help=(
            f"--declare requires exactly {DECLARE_CONFIRMATION}; "
            f"--release requires exactly {RELEASE_CONFIRMATION}"
        ),
    )


def spare_request(arguments: argparse.Namespace) -> WarmSpareRequest:
    if arguments.declare:
        mode = "declare"
    elif arguments.release:
        mode = "release"
    else:
        mode = "check"
    return WarmSpareRequest(
        state_dir=arguments.state_dir,
        node=str(arguments.node).strip(),
        fault_node=str(arguments.fault_node or "").strip(),
        cluster_id=arguments.cluster_id or None,
        reference=arguments.reference,
        mode=mode,
        confirmation=str(arguments.confirm or ""),
    )


def run_config_spare_command(
    arguments: argparse.Namespace,
    *,
    site: RenderedSite,
    api: NodeApi | None = None,
    agent_lookup: AgentLookup | None = None,
) -> int:
    """Print the report; exit 1 when a read-only check finds refusals."""

    request = spare_request(arguments)
    if request.mode == "check":
        report = run_warm_spare(site, request, api=api, agent_lookup=agent_lookup)
    else:
        # The same lock ``config`` and ``deploy`` hold: a cordon during a
        # rollout is exactly the kind of concurrent site change it exists for.
        with administrator_operation_lock(request.state_dir):
            report = run_warm_spare(site, request, api=api, agent_lookup=agent_lookup)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["ready"] else 1
