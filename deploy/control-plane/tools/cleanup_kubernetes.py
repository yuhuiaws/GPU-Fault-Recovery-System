"""Bounded Kubernetes cleanup using transaction-owned object and node identities."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from cleanup_state import (
    PHASE_INDEX,
    CleanupStateError,
    atomic_write,
    attach_provider_nodes,
    completed_phases,
    read_state,
    required_phases,
    validate_request,
    verify_provider_nodes,
)

from gpu_fault.admin.cluster_removal_rbac import (
    delete_recorded_workload_namespace_rbac,
)
from gpu_fault.admin.execution import (
    cleanup_deadline,
    deployment_deadline,
    remaining_timeout,
    run_command,
)


class CleanupClient:
    def __init__(self, config: dict[str, Any], context: str) -> None:
        self.prefix = ["kubectl", "--request-timeout=30s"]
        if context == "cpu":
            self.prefix += ["--kubeconfig", config["cpu_kubeconfig"]]
        else:
            if not context.startswith("gpu:") or not context.removeprefix("gpu:"):
                raise CleanupStateError("GPU cleanup context lacks an explicit plane")
            kubeconfig = config.get("gpu_kubeconfig") or os.environ.get("KUBECONFIG")
            if kubeconfig:
                self.prefix += ["--kubeconfig", kubeconfig]
            self.prefix += ["--context", context.removeprefix("gpu:")]

    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
        result = run_command(
            [*self.prefix, *arguments],
            input_text=json.dumps(payload) if payload is not None else None,
            timeout_seconds=remaining_timeout(45),
        )
        if result.returncode:
            raise CleanupStateError(
                f"Kubernetes cleanup command failed (exit {result.returncode})"
            )
        return result.stdout.strip()

    def get(self, kind: str, name: str, namespace: str = "") -> dict[str, Any] | None:
        arguments = ["-n", namespace] if namespace else []
        text = self.run(
            *arguments, "get", kind, name, "--ignore-not-found", "-o", "json"
        )
        if not text:
            return None
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
            raise CleanupStateError(
                "Kubernetes cleanup returned invalid object metadata"
            )
        identity = object_identity(value)
        if identity[1:3] != (kind, name) or value["metadata"].get("namespace") != (
            namespace or None
        ):
            raise CleanupStateError(
                "Kubernetes cleanup returned a different object identity"
            )
        return value

    def items(self, *arguments: str) -> list[dict[str, Any]]:
        value = json.loads(self.run(*arguments, "-o", "json"))
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise CleanupStateError(
                "Kubernetes cleanup returned an invalid resource list"
            )
        items = value["items"]
        if any(not isinstance(item, dict) for item in items):
            raise CleanupStateError("Kubernetes cleanup returned an invalid list item")
        return cast(list[dict[str, Any]], items)


def object_identity(item: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = item.get("metadata") or {}
    values = (
        item.get("apiVersion"),
        str(item.get("kind", "")).lower(),
        metadata.get("name"),
        metadata.get("uid"),
    )
    if not all(isinstance(value, str) and value for value in values):
        raise CleanupStateError("cleanup object identity is incomplete")
    return cast(tuple[str, str, str, str], values)


def verify_cluster(
    client: CleanupClient, document: dict[str, Any], context: str
) -> str:
    anchor = client.get("namespace", "kube-system")
    if anchor is None:
        raise CleanupStateError("Kubernetes cluster identity is unavailable")
    uid = object_identity(anchor)[3]
    expected = document.get("cluster_uids", {}).get(context)
    if expected is not None and expected != uid:
        raise CleanupStateError("Kubernetes context points to a different cluster")
    if any(
        other != context and value == uid
        for other, value in document.get("cluster_uids", {}).items()
    ):
        raise CleanupStateError(
            "CPU/GPU cleanup targets alias the same Kubernetes cluster"
        )
    return uid


def verify_namespace(
    client: CleanupClient,
    document: dict[str, Any],
    context: str,
    *,
    allow_absent: bool = False,
) -> dict[str, Any] | None:
    snapshot = document.get("namespace_snapshots", {}).get(context)
    if not isinstance(snapshot, dict) or not snapshot.get("uid"):
        raise CleanupStateError("namespace cleanup lacks an ownership checkpoint")
    live = client.get("namespace", document["targets"]["namespace"])
    if live is None:
        if allow_absent:
            return None
        raise CleanupStateError("solution namespace is absent before cleanup")
    if object_identity(live)[3] != snapshot["uid"]:
        raise CleanupStateError("solution namespace was replaced")
    if not allow_absent and live["metadata"].get("deletionTimestamp"):
        raise CleanupStateError("solution namespace is terminating")
    return live


def verify_quiesced(
    client: CleanupClient, document: dict[str, Any], context: str
) -> None:
    completed = set(completed_phases(document))
    namespace = document["targets"]["namespace"]
    if "NAMESPACES_DELETED" in completed:
        if client.get("namespace", namespace) is not None:
            raise CleanupStateError("completed cleanup namespace has reappeared")
        return
    verify_namespace(
        client,
        document,
        context,
        allow_absent="APPLICATION_OBJECTS_DELETED" in completed,
    )
    plane = "cpu" if context == "cpu" else "gpu"
    if plane == "cpu" and document["scope"] != "all":
        return
    section = document["inventory_snapshot"][plane]
    phases = {
        "ingress": "INGRESS_STOPPED",
        "consumer": "CONTROL_CONSUMERS_STOPPED",
        "auxiliary": "CPU_AUXILIARIES_STOPPED",
        "producer": "GPU_DATA_PLANE_SOURCES_STOPPED",
        "executor": "GPU_EXECUTORS_STOPPED",
    }
    for resource in section.get("by_context", {}).get(
        context.removeprefix("gpu:"), section
    )["resources"]:
        if (
            resource["kind"] not in {"deployment", "daemonset", "cronjob"}
            or phases.get(resource.get("phase")) not in completed
        ):
            continue
        live = client.get(
            resource["kind"], resource["name"], resource.get("namespace") or namespace
        )
        if resource["kind"] == "deployment" and client.items(
            "-n",
            resource.get("namespace") or namespace,
            "get",
            "pods",
            "-l",
            f"app={resource['name']}",
        ):
            raise CleanupStateError("completed cleanup workload still has Pod objects")
        if live is None:
            continue
        spec = live.get("spec") or {}
        if resource["kind"] == "deployment":
            stopped = spec.get("replicas") == 0 and not any(
                (live.get("status") or {}).get(key, 0)
                for key in ("replicas", "readyReplicas", "availableReplicas")
            )
        else:
            stopped = resource["kind"] == "cronjob" and spec.get("suspend") is True
        if not stopped or "APPLICATION_OBJECTS_DELETED" in completed:
            raise CleanupStateError(
                "completed cleanup workload has reappeared or restarted"
            )


# Aggregated APIs that project live measurements as list results: their items
# carry no uid, cannot be owned and cannot be deleted, so they are not part of
# a namespace inventory (live 2026-09-20: metrics-server's PodMetrics stopped an
# uninstall at PREFLIGHT with "cleanup object identity is incomplete").
PROJECTION_API_GROUPS = frozenset({"metrics.k8s.io", "metrics.eks.amazonaws.com"})


def is_projection_resource(name: str) -> bool:
    group = name.partition(".")[2]
    return group in PROJECTION_API_GROUPS or group.endswith(".metrics.k8s.io")


def namespace_items(client: CleanupClient, namespace: str) -> list[dict[str, Any]]:
    discovered = sorted(
        set(
            client.run(
                "api-resources", "--verbs=list", "--namespaced=true", "-o", "name"
            ).splitlines()
        )
    )
    if not discovered or len(discovered) > 512:
        raise CleanupStateError("namespace resource discovery is incomplete")
    kinds = [kind for kind in discovered if not is_projection_resource(kind)]
    if not kinds:
        raise CleanupStateError("namespace resource discovery is incomplete")
    items: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for start in range(0, len(kinds), 32):
        for item in client.items(
            "-n", namespace, "get", ",".join(kinds[start : start + 32])
        ):
            if (item.get("metadata") or {}).get("namespace") != namespace:
                raise CleanupStateError(
                    "namespace resource discovery crossed its scope"
                )
            items[object_identity(item)] = item
    return list(items.values())


def implicit_namespace_object(item: dict[str, Any]) -> bool:
    _api, kind, name, _uid = object_identity(item)
    return (
        kind in {"configmap", "secret"}
        and name.startswith("gpu-fault-")
        or kind == "configmap"
        and name == "kube-root-ca.crt"
        or kind == "serviceaccount"
        and name == "default"
        or kind == "event"
    )


def service_derived_object(
    item: dict[str, Any], approved: set[tuple[str, str, str, str]]
) -> bool:
    """A controller-written mirror of an approved Service in this namespace.

    Kubernetes keeps a ``v1 Endpoints`` object for every selector Service and
    the AWS Load Balancer Controller keeps one TargetGroupBinding per load
    balancer Service. Neither carries an ownerReference; both disappear with
    the Service, so they belong to the Service that is already approved.
    """

    api, kind, name, _uid = object_identity(item)

    def approved_service(service_name: object) -> bool:
        return isinstance(service_name, str) and any(
            identity[:3] == ("v1", "service", service_name) for identity in approved
        )

    if (api, kind) == ("v1", "endpoints"):
        return approved_service(name)
    if kind == "targetgroupbinding" and api.startswith("elbv2.k8s.aws/"):
        metadata = item.get("metadata") or {}
        labels = metadata.get("labels") or {}
        service_reference = (item.get("spec") or {}).get("serviceRef") or {}
        service_name = service_reference.get("name")
        return (
            labels.get("service.k8s.aws/stack-name") == service_name
            and labels.get("service.k8s.aws/stack-namespace")
            == metadata.get("namespace")
            and approved_service(service_name)
        )
    return False


def installer_job_of_reconciler(
    item: dict[str, Any], approved: set[tuple[str, str, str, str]]
) -> bool:
    """A node-installer Job created by the approved reconciler Deployment.

    The reconciler labels each per-node Job ``gpu-fault.io/node-installer=true``
    instead of owner-referencing itself and leaves a TTL to retire it; within
    that hour an uninstall or cluster removal finds the finished Jobs.
    """

    api, kind, _name, _uid = object_identity(item)
    if kind != "job" or not api.startswith("batch/"):
        return False
    labels = (item.get("metadata") or {}).get("labels") or {}
    return labels.get("gpu-fault.io/node-installer") == "true" and any(
        identity[:3] == ("apps/v1", "deployment", "gpu-fault-node-installer-reconciler")
        for identity in approved
    )


def capture_namespace(
    client: CleanupClient, document: dict[str, Any], context: str
) -> None:
    namespace = document["targets"]["namespace"]
    if namespace in {"default", "kube-system", "kube-public", "kube-node-lease"}:
        raise CleanupStateError("refusing cleanup of a shared Kubernetes namespace")
    live_namespace = client.get("namespace", namespace)
    if live_namespace is None:
        raise CleanupStateError("solution namespace is absent before cleanup")
    namespace_uid = object_identity(live_namespace)[3]
    previous = document.get("namespace_snapshots", {}).get(context)
    if previous is not None and previous.get("uid") != namespace_uid:
        raise CleanupStateError("solution namespace was replaced during preflight")
    if document["mode"] != "reset":
        document.setdefault("namespace_snapshots", {})[context] = {
            "uid": namespace_uid,
            "objects": [],
        }
        return
    section = document["inventory_snapshot"]["cpu" if context == "cpu" else "gpu"]
    resources = section.get("by_context", {}).get(
        context.removeprefix("gpu:"), section
    )["resources"]
    registered = {
        (item["kind"], item["name"])
        for item in resources
        if item["scope"] == "namespaced"
        and (item.get("namespace") or namespace) == namespace
    }
    objects = namespace_items(client, namespace)
    approved = {
        object_identity(item)
        for item in objects
        if (str(item["kind"]).lower(), item["metadata"]["name"]) in registered
        or implicit_namespace_object(item)
    }
    while True:
        parent_uids = {identity[3] for identity in approved}
        children = {
            object_identity(item)
            for item in objects
            if len(owners := item["metadata"].get("ownerReferences") or []) == 1
            and owners[0].get("uid") in parent_uids
            and any(
                identity[:3]
                == (
                    owners[0].get("apiVersion"),
                    str(owners[0].get("kind", "")).lower(),
                    owners[0].get("name"),
                )
                and identity[3] == owners[0]["uid"]
                for identity in approved
            )
        }
        derived = {
            object_identity(item)
            for item in objects
            if service_derived_object(item, approved)
            or installer_job_of_reconciler(item, approved)
        }
        if children.issubset(approved) and derived.issubset(approved):
            break
        approved.update(children)
        approved.update(derived)
    if any(object_identity(item) not in approved for item in objects):
        raise CleanupStateError("solution namespace contains unowned resources")
    document.setdefault("namespace_snapshots", {})[context] = {
        "uid": namespace_uid,
        "objects": [list(identity) for identity in sorted(approved)],
    }


def delete_owned(
    client: CleanupClient,
    *,
    kind: str,
    name: str,
    uid: str,
    namespace: str = "",
) -> None:
    live = client.get(kind, name, namespace)
    if live is None:
        return
    if object_identity(live)[3] != uid:
        raise CleanupStateError("cleanup target UID changed; refusing deletion")
    path = (
        f"/api/v1/namespaces/{name}"
        if kind == "namespace"
        else f"/apis/apps/v1/namespaces/{namespace}/daemonsets/{name}"
    )
    client.run(
        "delete",
        "--raw",
        path,
        "-f",
        "-",
        payload={
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": uid},
            "propagationPolicy": "Foreground",
        },
    )
    while True:
        live = client.get(kind, name, namespace)
        remaining_timeout(1)
        if live is None:
            return
        if object_identity(live)[3] != uid:
            raise CleanupStateError("cleanup target was replaced while deleting")
        time.sleep(min(2, remaining_timeout(2)))


def delete_namespace(
    client: CleanupClient, document: dict[str, Any], context: str
) -> None:
    namespace = document["targets"]["namespace"]
    snapshot = document.get("namespace_snapshots", {}).get(context)
    if not isinstance(snapshot, dict):
        raise CleanupStateError("namespace cleanup lacks an ownership checkpoint")
    live = verify_namespace(client, document, context, allow_absent=True)
    if live is None:
        return
    approved = {tuple(identity) for identity in snapshot["objects"]}
    if any(
        object_identity(item) not in approved and not implicit_namespace_object(item)
        for item in namespace_items(client, namespace)
    ):
        raise CleanupStateError("namespace gained unowned resources during cleanup")
    delete_owned(client, kind="namespace", name=namespace, uid=snapshot["uid"])


def workload_rbac_proof(
    document: dict[str, Any], context: str
) -> dict[str, Any] | None:
    section = document["inventory_snapshot"]["gpu"]
    snapshot = section.get("by_context", {}).get(context.removeprefix("gpu:"))
    if snapshot is None:
        if any(row.get("guarded_delete") for row in section["resources"]):
            raise CleanupStateError("guarded RBAC lacks a context-specific snapshot")
        return None
    rows = snapshot["resources"]
    guarded = []
    for row in rows:
        guard = row.get("guarded_delete", "")
        if guard not in {"", "workload-rbac"}:
            raise CleanupStateError("unknown guarded cleanup action")
        if guard:
            if (
                row.get("kind") not in {"role", "rolebinding"}
                or row.get("scope") != "namespaced"
                or row.get("phase") != "support"
                or row.get("clean") != "delete"
            ):
                raise CleanupStateError("guarded workload RBAC row has invalid scope")
            guarded.append(
                f"{row.get('namespace')}/"
                f"{'RoleBinding' if row['kind'] == 'rolebinding' else 'Role'}/"
                f"{row.get('name')}"
            )
    proof = snapshot.get("workload_rbac")
    if proof is None and not guarded:
        return None
    if (
        not isinstance(proof, dict)
        or not isinstance(proof.get("resources"), dict)
        or len(set(guarded)) != len(guarded)
        or set(guarded) != set(proof["resources"])
    ):
        raise CleanupStateError("guarded workload RBAC rows differ from recorded proof")
    return proof


def delete_workload_rbac(
    config: dict[str, Any],
    document: dict[str, Any],
    state_path: Path,
    *,
    context: str,
    cluster_id: str,
    skip_cpu: bool = False,
) -> list[str]:
    if document["mode"] not in {"clean", "reset"}:
        raise CleanupStateError("workload RBAC deletion requires clean or reset mode")
    raw_context = context.removeprefix("gpu:")
    selected = [
        target
        for target in document["targets"]["clusters"]
        if target["cluster_id"] == cluster_id and target["context"] == raw_context
    ]
    members = [
        target for target in config["clusters"] if target["context"] == raw_context
    ]
    if (
        not context.startswith("gpu:")
        or not cluster_id
        or len(selected) != 1
        or len(members) != 1
        or members[0]["cluster_id"] != cluster_id
    ):
        raise CleanupStateError(
            "workload RBAC cleanup target is outside the bound scope"
        )
    completed = set(completed_phases(document))
    if skip_cpu and "CLEANUP_COMPLETED" not in completed:
        raise CleanupStateError("skip-cpu requires completed read-only cleanup replay")
    prerequisites = {
        phase
        for phase in required_phases(document)
        if PHASE_INDEX[phase] < PHASE_INDEX["APPLICATION_OBJECTS_DELETED"]
    }
    if not prerequisites.issubset(completed):
        raise CleanupStateError(
            "workload RBAC cleanup requires completed drain and stop phases"
        )
    if document["targets"]["namespace"] != config.get("namespace", "gpu-fault-system"):
        raise CleanupStateError(
            "workload RBAC cleanup namespace differs from source config"
        )
    namespace = document.get("namespace_snapshots", {}).get(context, {})
    namespace_uid = namespace.get("uid")
    if (
        not isinstance(namespace_uid, str)
        or not namespace_uid
        or not isinstance(document.get("cluster_uids", {}).get(context), str)
        or not document["cluster_uids"][context]
    ):
        raise CleanupStateError(
            "workload RBAC cleanup lacks bound cluster/namespace identities"
        )
    captured = workload_rbac_proof(document, context)
    if captured is None:
        return []
    progress = document.get("workload_rbac_removed", {})
    if not isinstance(progress, dict):
        raise CleanupStateError("workload RBAC cleanup progress is invalid")
    proof = copy.deepcopy(captured)
    proof["removed"] = copy.deepcopy(progress.get(context, captured.get("removed")))
    if "APPLICATION_OBJECTS_DELETED" in completed and (
        not isinstance(proof["removed"], list)
        or set(proof["removed"]) != set(proof["resources"])
    ):
        raise CleanupStateError(
            "completed application cleanup lacks RBAC deletion checkpoints"
        )
    client = CleanupClient(config, context)
    verify_cluster(client, document, context)
    verify_quiesced(client, document, context)
    if document["scope"] == "all" and not skip_cpu:
        cpu = CleanupClient(config, "cpu")
        verify_cluster(cpu, document, "cpu")
        verify_quiesced(cpu, document, "cpu")
    target = {**members[0], "expected_namespace_uid": namespace_uid}

    def checkpoint() -> None:
        if any(
            proof.get(key) != captured.get(key)
            for key in ("binding", "resources", "anchors", "documents")
        ):
            raise CleanupStateError(
                "workload RBAC checkpoint changed its captured authority"
            )
        previous = document.get("workload_rbac_removed", {}).get(context, [])
        if not set(previous).issubset(proof["removed"]):
            raise CleanupStateError("workload RBAC deletion progress moved backwards")
        document.setdefault("workload_rbac_removed", {})[context] = list(
            proof["removed"]
        )
        atomic_write(state_path, document)

    removed = delete_recorded_workload_namespace_rbac(
        config,
        target,
        proof,
        run=run_command,
        kubectl=client.prefix,
        checkpoint=checkpoint,
    )
    if set(proof["removed"]) != set(proof["resources"]):
        raise CleanupStateError(
            "workload RBAC deletion did not confirm every recorded resource"
        )
    return removed


# HyperPod names every EKS node ``hyperpod-<instance-id>``; the fleet agent
# record carries the same instance id (``gpu_fault.hyperpod`` aliases).
HYPERPOD_NODE_PREFIX = "hyperpod-"
# Node annotations the node-installer-reconciler writes while (or after) it
# tries to install a node; ``gpu-fault.io/node-key-*`` ownership annotations
# are deliberately not in this family.
INSTALLER_ANNOTATION_PREFIX = "gpu-fault.io/installer-"
INSTALLER_STATE_ANNOTATION = "gpu-fault.io/installer-state"
# Installer states that prove the node never received a runtime from us. A
# spot replacement without a node key sits in one of these with no fleet agent
# (live 2026-09-30: ``Failed``, attempts 2); ``Succeeded`` means a runtime was
# installed and stays a blocker without a fleet agent.
ORPHANED_INSTALLER_STATES = frozenset(
    {"Failed", "Installing", "Retrying", "WaitingForKey"}
)


def hyperpod_target(
    config: dict[str, Any] | None, context: str, cluster_id: str
) -> tuple[str, str]:
    """The HyperPod cluster name and Region bound to this cleanup cluster."""

    if config is None:
        raise CleanupStateError("cluster is not HyperPod-managed")
    members = [
        item
        for item in config.get("clusters") or []
        if isinstance(item, dict) and item.get("cluster_id") == cluster_id
    ]
    if len(members) != 1 or members[0].get("context") != context.removeprefix("gpu:"):
        raise CleanupStateError("cleanup cluster is outside the bound config scope")
    name = members[0].get("hyperpod_cluster_name")
    region = members[0].get("region") or config.get("aws_region")
    if not isinstance(name, str) or not name:
        raise CleanupStateError("cluster is not HyperPod-managed")
    if not isinstance(region, str) or not region:
        raise CleanupStateError("HyperPod cluster has no bound Region")
    return name, region


def hyperpod_instances(
    config: dict[str, Any] | None,
    document: dict[str, Any],
    context: str,
    cluster_id: str,
) -> set[str]:
    """Instance ids HyperPod currently lists for the cluster, captured once."""

    cached = (document.get("provider_nodes") or {}).get(cluster_id)
    if cached is not None:
        return set(verify_provider_nodes(cached))
    name, region = hyperpod_target(config, context, cluster_id)
    try:
        result = run_command(
            [
                "aws",
                "sagemaker",
                "list-cluster-nodes",
                "--cluster-name",
                name,
                "--region",
                region,
                "--output",
                "json",
            ],
            timeout_seconds=remaining_timeout(45),
        )
    except OSError as exc:
        raise CleanupStateError(
            f"HyperPod node listing failed ({type(exc).__name__})"
        ) from exc
    if result.returncode:
        raise CleanupStateError(
            f"HyperPod node listing failed (exit {result.returncode})"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CleanupStateError("HyperPod node listing is not JSON") from exc
    summaries = value.get("ClusterNodeSummaries") if isinstance(value, dict) else None
    if not isinstance(summaries, list) or value.get("NextToken"):
        raise CleanupStateError("HyperPod node listing is invalid or truncated")
    instance_ids = [
        item.get("InstanceId") if isinstance(item, dict) else None for item in summaries
    ]
    if not all(isinstance(item, str) and item for item in instance_ids):
        raise CleanupStateError("HyperPod node listing lacks instance identities")
    record = attach_provider_nodes(
        document,
        cluster_id,
        hyperpod_cluster_name=name,
        region=region,
        instance_ids=cast(list[str], instance_ids),
    )
    return set(verify_provider_nodes(record))


def departed_fleet_node(
    config: dict[str, Any] | None,
    document: dict[str, Any],
    context: str,
    agent: dict[str, Any],
) -> dict[str, Any]:
    """Prove an ACTIVE agent's node left the cluster with its spot instance.

    The node is already absent from Kubernetes. It is DEPARTED only when its
    name identifies a HyperPod instance that the provider no longer lists;
    every other outcome (not HyperPod-managed, listing failed, instance still
    listed, unrecognized name) keeps the original refusal, with the reason.
    """

    name = str(agent["node_id"])
    cluster_id = str(agent["cluster_id"])

    def refuse(reason: str) -> CleanupStateError:
        return CleanupStateError(f"an active fleet node is missing: {name}; {reason}")

    instance_id = name.removeprefix(HYPERPOD_NODE_PREFIX)
    if not name.startswith(HYPERPOD_NODE_PREFIX) or not instance_id:
        raise refuse("node name does not identify a HyperPod instance")
    recorded_instance = agent.get("node_instance_id")
    if recorded_instance not in (None, "", instance_id):
        raise refuse("agent instance identity differs from the node name")
    try:
        members = hyperpod_instances(config, document, context, cluster_id)
    except CleanupStateError as exc:
        raise refuse(str(exc)) from exc
    if instance_id in members:
        raise refuse(f"HyperPod still lists instance {instance_id}")
    return {
        "cluster_id": cluster_id,
        "node_id": name,
        "node_instance_id": instance_id,
    }


def orphaned_installer_node(
    node: dict[str, Any], agents: object, cluster_id: str
) -> bool:
    """A node carrying only unfinished installer annotations and no agent.

    Anything else of ours on the node (labels, quarantine taint, ownership or
    other annotations, ``installer-state=Succeeded``) or any fleet agent record
    for the node keeps it a blocker.
    """

    metadata = node["metadata"]
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    taints = (node.get("spec") or {}).get("taints") or []
    if any(key.startswith("gpu-fault.io/") for key in labels) or any(
        str(taint.get("key", "")).startswith("gpu-fault.io/") for taint in taints
    ):
        return False
    owned = [key for key in annotations if key.startswith("gpu-fault.io/")]
    if not owned or any(
        not key.startswith(INSTALLER_ANNOTATION_PREFIX) for key in owned
    ):
        return False
    if annotations.get(INSTALLER_STATE_ANNOTATION) not in ORPHANED_INSTALLER_STATES:
        return False
    if not isinstance(agents, list):
        return False
    name = metadata["name"]
    return not any(
        agent.get("cluster_id") == cluster_id and agent.get("node_id") == name
        for agent in agents
    )


def node_targets(
    client: CleanupClient,
    document: dict[str, Any],
    context: str,
    cluster_id: str,
    *,
    nodes: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    if nodes is None:
        nodes = client.items("get", "nodes")
    by_name = {object_identity(node)[2]: node for node in nodes}
    if len(by_name) != len(nodes):
        raise CleanupStateError("duplicate Kubernetes node identity")
    agents = document.get("fleet_snapshot")
    recorded = document.get("node_targets", {}).get(context)
    if recorded is None:
        if not isinstance(agents, list):
            raise CleanupStateError("node cleanup requires captured fleet inventory")
        recorded = {}
        departed: list[dict[str, Any]] = []
        for agent in agents:
            if agent.get("cluster_id") != cluster_id:
                continue
            name = agent["node_id"]
            if name not in by_name:
                if agent.get("lifecycle_state") == "ACTIVE":
                    departed.append(
                        departed_fleet_node(config, document, context, agent)
                    )
                continue
            if not (agent.get("installed_unit_inventory") or {}).get("units"):
                raise CleanupStateError("node cleanup lacks installed unit inventory")
            recorded[name] = object_identity(by_name[name])[3]
        if not recorded:
            raise CleanupStateError("node cleanup has no proven fleet targets")
        document.setdefault("node_targets", {})[context] = recorded
        journal = document.setdefault("departed_fleet_nodes", [])
        journal.extend(entry for entry in departed if entry not in journal)
    for name, uid in recorded.items():
        if name not in by_name or object_identity(by_name[name])[3] != uid:
            raise CleanupStateError("node cleanup target disappeared or changed UID")
    orphaned: dict[str, str] = {}
    for name, node in by_name.items():
        metadata = node["metadata"]
        keys = set(metadata.get("labels") or {}) | set(
            metadata.get("annotations") or {}
        )
        if name in recorded or not any(key.startswith("gpu-fault.io/") for key in keys):
            continue
        if orphaned_installer_node(node, agents, cluster_id):
            orphaned[name] = object_identity(node)[3]
            continue
        raise CleanupStateError(
            "unregistered node runtime remains outside cleanup targets"
        )
    document.setdefault("orphaned_installer_nodes", {})[context] = orphaned
    return dict(recorded)


def installer_annotation_patch(node: dict[str, Any], uid: str) -> list[dict[str, Any]]:
    """Remove every installer annotation from an orphaned node, UID/RV bound."""

    metadata = node["metadata"]
    operations: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": metadata["resourceVersion"],
        },
    ]
    for key in sorted(metadata.get("annotations") or {}):
        if key.startswith(INSTALLER_ANNOTATION_PREFIX):
            escaped = key.replace("~", "~0").replace("/", "~1")
            operations.append(
                {"op": "remove", "path": f"/metadata/annotations/{escaped}"}
            )
    return operations


def clear_node_metadata(
    client: CleanupClient,
    document: dict[str, Any],
    context: str,
    cluster_id: str,
    *,
    config: dict[str, Any] | None = None,
) -> None:
    verify_cluster(client, document, context)
    verify_namespace(client, document, context, allow_absent=True)
    nodes = client.items("get", "nodes")
    targets = node_targets(
        client, document, context, cluster_id, nodes=nodes, config=config
    )
    orphaned = dict(document.get("orphaned_installer_nodes", {}).get(context, {}))
    inventory = document["inventory_snapshot"]["gpu"]
    patches: list[tuple[str, list[dict[str, Any]], bool]] = []
    for node in nodes:
        metadata = node["metadata"]
        name = metadata["name"]
        if name in orphaned:
            if metadata.get("uid") != orphaned[name]:
                raise CleanupStateError("orphaned installer node changed UID")
            patches.append(
                (name, installer_annotation_patch(node, orphaned[name]), False)
            )
            continue
        if name not in targets:
            continue
        if metadata.get("uid") != targets[name]:
            raise CleanupStateError("node metadata cleanup target changed UID")
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        restore_scheduling = False
        if labels.get("gpu-fault.io/spare") == "true":
            previous = annotations.get("gpu-fault.io/previous-unschedulable")
            if not isinstance(previous, str) or previous not in {"true", "false"}:
                raise CleanupStateError(
                    "declared warm spare has no scheduling baseline on the node; "
                    "release it with gpu-fault-admin (config spare --release; "
                    "uninstall does it) before node cleanup"
                )
            if labels.get("gpu-fault.io/quarantined") not in {None, "false"}:
                raise CleanupStateError("quarantined spare cannot be released")
            unschedulable = (node.get("spec") or {}).get("unschedulable", False)
            if type(unschedulable) is not bool:
                raise CleanupStateError("spare cleanup has unknown scheduling state")
            restore_scheduling = previous == "false" and unschedulable
        operations: list[dict[str, Any]] = [
            {"op": "test", "path": "/metadata/uid", "value": targets[name]},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": metadata["resourceVersion"],
            },
        ]
        if restore_scheduling:
            operations.extend(
                [
                    {"op": "test", "path": "/spec/unschedulable", "value": True},
                    {"op": "replace", "path": "/spec/unschedulable", "value": False},
                ]
            )
        for field, keys in (
            ("annotations", inventory["node_annotations"]),
            ("labels", inventory["node_labels"]),
        ):
            for key in keys:
                if key in (metadata.get(field) or {}):
                    escaped = key.replace("~", "~0").replace("/", "~1")
                    operations.append(
                        {"op": "remove", "path": f"/metadata/{field}/{escaped}"}
                    )
        if len(operations) > 2:
            patches.append((name, operations, restore_scheduling))
    for name, operations, restored in patches:
        client.run("patch", "node", name, "--type=json", "-p", json.dumps(operations))
        current = client.get("node", name)
        expected_uid = orphaned[name] if name in orphaned else targets[name]
        if current is None or object_identity(current)[3] != expected_uid:
            raise CleanupStateError("node changed during metadata cleanup")
        metadata = current["metadata"]
        if name in orphaned:
            if any(
                key.startswith("gpu-fault.io/")
                for key in metadata.get("annotations") or {}
            ):
                raise CleanupStateError("node metadata cleanup did not converge")
            continue
        if any(
            key in (metadata.get(field) or {})
            for field, keys in (
                ("annotations", inventory["node_annotations"]),
                ("labels", inventory["node_labels"]),
            )
            for key in keys
        ) or (
            restored and (current.get("spec") or {}).get("unschedulable") is not False
        ):
            raise CleanupStateError("node metadata cleanup did not converge")


def node_manifest(
    *,
    name: str,
    namespace: str,
    run_id: str,
    nodes: list[str],
    image: str,
    mode: str,
    host_script_b64: str,
) -> dict[str, Any]:
    if mode not in {"stop", "uninstall"} or not nodes:
        raise CleanupStateError("invalid node cleanup request")
    if not base64.b64decode(host_script_b64, validate=True).strip():
        raise CleanupStateError("node cleanup host script is empty")
    labels = {"app": name, "gpu-fault.io/cleanup-run": run_id}
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "hostPID": True,
                    "automountServiceAccountToken": False,
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                # Terms are ORed; a ``metadata.name`` field
                                # selector accepts exactly one value per term.
                                "nodeSelectorTerms": [
                                    {
                                        "matchFields": [
                                            {
                                                "key": "metadata.name",
                                                "operator": "In",
                                                "values": [node],
                                            }
                                        ]
                                    }
                                    for node in sorted(nodes)
                                ],
                            },
                        }
                    },
                    "tolerations": [{"operator": "Exists"}],
                    "containers": [
                        {
                            "name": "cleanup",
                            "image": image,
                            "securityContext": {"privileged": True},
                            "command": ["/bin/sh", "-ec"],
                            "args": [
                                f"printf '%s' '{host_script_b64}' | base64 -d | "
                                f"chroot /host /bin/sh -s -- '{mode}'\n"
                                "touch /tmp/cleanup-complete\nexec sleep 86400"
                            ],
                            "readinessProbe": {
                                "exec": {
                                    "command": [
                                        "/bin/sh",
                                        "-c",
                                        "test -f /tmp/cleanup-complete",
                                    ]
                                },
                                "periodSeconds": 2,
                            },
                            "volumeMounts": [
                                {
                                    "name": "host",
                                    "mountPath": "/host",
                                    "mountPropagation": "HostToContainer",
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "host", "hostPath": {"path": "/", "type": "Directory"}}
                    ],
                },
            },
        },
    }


def verify_node_manifest(live: dict[str, Any], expected: dict[str, Any]) -> None:
    metadata = live.get("metadata") or {}
    if (
        live.get("apiVersion") != "apps/v1"
        or live.get("kind") != "DaemonSet"
        or any(
            metadata.get(key) != expected["metadata"][key]
            for key in ("name", "namespace")
        )
        or any(
            (metadata.get("labels") or {}).get(key) != value
            for key, value in expected["metadata"]["labels"].items()
        )
    ):
        raise CleanupStateError("temporary node cleanup object identity differs")
    spec = copy.deepcopy(live.get("spec") or {})
    defaults = {
        "revisionHistoryLimit": 10,
        "updateStrategy": {
            "type": "RollingUpdate",
            "rollingUpdate": {"maxUnavailable": 1, "maxSurge": 0},
        },
    }
    for key, value in defaults.items():
        if spec.get(key) == value:
            spec.pop(key)
    template = spec.get("template") or {}
    if (template.get("metadata") or {}).get("creationTimestamp", "absent") is None:
        template["metadata"].pop("creationTimestamp")
    pod = template.get("spec") or {}
    for key, value in {
        "dnsPolicy": "ClusterFirst",
        "restartPolicy": "Always",
        "schedulerName": "default-scheduler",
        "securityContext": {},
        "terminationGracePeriodSeconds": 30,
    }.items():
        if pod.get(key) == value:
            pod.pop(key)
    for container in pod.get("containers") or []:
        image = str(container.get("image") or "").rsplit("/", 1)[-1]
        pull_policy = (
            "Always"
            if image.endswith(":latest") or ":" not in image and "@" not in image
            else "IfNotPresent"
        )
        for key, value in {
            "imagePullPolicy": pull_policy,
            "resources": {},
            "terminationMessagePath": "/dev/termination-log",
            "terminationMessagePolicy": "File",
        }.items():
            if container.get(key) == value:
                container.pop(key)
        probe = container.get("readinessProbe") or {}
        for key, value in {
            "failureThreshold": 3,
            "successThreshold": 1,
            "timeoutSeconds": 1,
        }.items():
            if probe.get(key) == value:
                probe.pop(key)
    if spec != expected["spec"]:
        raise CleanupStateError("temporary node cleanup spec differs from its journal")


def cleanup_pending(
    config: dict[str, Any], document: dict[str, Any], state_path: Path
) -> None:
    contexts = {"gpu:" + item["context"] for item in document["targets"]["clusters"]}
    for context, record in document.get("node_cleanup", {}).items():
        if context == "cpu" or context not in contexts:
            raise CleanupStateError(
                "temporary cleanup context is outside the selected scope"
            )
        if record.get("status") == "REMOVED":
            continue
        client = CleanupClient(config, context)
        if context not in document.get("cluster_uids", {}):
            raise CleanupStateError("temporary node cleanup lacks cluster identity")
        verify_cluster(client, document, context)
        verify_namespace(client, document, context, allow_absent=True)
        live = client.get("daemonset", record["name"], document["targets"]["namespace"])
        if live is not None:
            verify_node_manifest(live, record["manifest"])
            metadata = live["metadata"]
            if (metadata.get("labels") or {}).get(
                "gpu-fault.io/cleanup-run"
            ) != document["run_id"] or record.get("uid") not in (
                None,
                metadata.get("uid"),
            ):
                raise CleanupStateError(
                    "temporary node cleanup object ownership changed"
                )
            delete_owned(
                client,
                kind="daemonset",
                name=record["name"],
                uid=object_identity(live)[3],
                namespace=document["targets"]["namespace"],
            )
        record["status"] = "REMOVED"
        atomic_write(state_path, document)


def run_node_cleanup(
    config: dict[str, Any],
    document: dict[str, Any],
    state_path: Path,
    *,
    context: str,
    cluster_id: str,
    image: str,
    mode: str,
    host_script_b64: str,
) -> None:
    client = CleanupClient(config, context)
    if context not in document.get("cluster_uids", {}):
        raise CleanupStateError("node cleanup lacks a cluster identity checkpoint")
    verify_cluster(client, document, context)
    verify_namespace(client, document, context)
    nodes = node_targets(client, document, context, cluster_id, config=config)
    namespace = document["targets"]["namespace"]
    suffix = hashlib.sha256(f"{document['run_id']}:{context}".encode()).hexdigest()[:16]
    name = f"gpu-fault-node-cleanup-{suffix}"
    manifest = node_manifest(
        name=name,
        namespace=namespace,
        run_id=document["run_id"],
        nodes=list(nodes),
        image=image,
        mode=mode,
        host_script_b64=host_script_b64,
    )
    cleanup_pending(config, document, state_path)
    if client.get("daemonset", name, namespace) is not None:
        raise CleanupStateError("temporary node cleanup name is already occupied")
    preview = json.loads(
        client.run(
            "create", "--dry-run=server", "-f", "-", "-o", "json", payload=manifest
        )
    )
    verify_node_manifest(preview, manifest)
    record: dict[str, Any] = {
        "name": name,
        "uid": None,
        "status": "PLANNED",
        "manifest": manifest,
    }
    document.setdefault("node_cleanup", {})[context] = record
    atomic_write(state_path, document)
    failure: BaseException | None = None
    try:
        verify_namespace(client, document, context)
        client.run("create", "-f", "-", payload=manifest)
        live = client.get("daemonset", name, namespace)
        if live is None:
            raise CleanupStateError("node cleanup creation was not observed")
        verify_node_manifest(live, manifest)
        record["uid"] = object_identity(live)[3]
        record["status"] = "RUNNING"
        atomic_write(state_path, document)
        while True:
            live = client.get("daemonset", name, namespace)
            if live is None or object_identity(live)[3] != record["uid"]:
                raise CleanupStateError("node cleanup DaemonSet was replaced")
            verify_node_manifest(live, manifest)
            status = live.get("status") or {}
            remaining_timeout(1)
            if (
                status.get("observedGeneration", 0)
                >= live["metadata"].get("generation", 1)
                and status.get("desiredNumberScheduled") == len(nodes)
                and status.get("updatedNumberScheduled") == len(nodes)
                and status.get("numberReady") == len(nodes)
                and status.get("numberMisscheduled", 0) == 0
            ):
                node_targets(client, document, context, cluster_id)
                break
            time.sleep(min(2, remaining_timeout(2)))
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            with cleanup_deadline("temporary node cleanup", 90):
                cleanup_pending(config, document, state_path)
        except BaseException as cleanup_error:
            if failure is None:
                raise
            failure.add_note(
                "temporary node cleanup remains unverified: "
                + type(cleanup_error).__name__
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "capture",
            "delete-namespace",
            "node-cleanup",
            "cleanup-owned",
            "clear-node-metadata",
            "delete-workload-rbac",
            "verify-targets",
            "drain-registry",
            "command",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--context", default="cpu")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--image", default="")
    parser.add_argument(
        "--node-mode", choices=("stop", "uninstall"), default="uninstall"
    )
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    with deployment_deadline(
        "Kubernetes cleanup", arguments.timeout_seconds, recovery_seconds=120
    ):
        client = CleanupClient(config, arguments.context)
        if arguments.action == "command":
            command = arguments.arguments
            if command[:1] == ["--"]:
                command = command[1:]
            result = run_command(
                [*client.prefix, *command], timeout_seconds=arguments.timeout_seconds
            )
            if result.returncode:
                raise CleanupStateError(
                    f"Kubernetes cleanup command failed (exit {result.returncode})"
                )
            print(result.stdout, end="")
            return 0
        if arguments.state_file is None:
            raise CleanupStateError("cleanup requires a state file")
        document = read_state(arguments.state_file)
        validate_request(
            document,
            config_path=arguments.config,
            scope=document["scope"],
            mode=document["mode"],
            node_mode=document["node_mode"],
            cluster_ids=(
                [item["cluster_id"] for item in document["targets"]["clusters"]]
                if document["scope"] == "gpu"
                else []
            ),
            state_path=arguments.state_file,
        )
        if arguments.action == "drain-registry":
            completed = completed_phases(document)
            resumable = (
                "CLUSTERS_DRAINING" in completed
                and "CONTROL_CONSUMERS_STOPPED" not in completed
                and PHASE_INDEX[document["phase"]]
                <= PHASE_INDEX["CONTROL_CONSUMERS_STOPPED"]
            )
            if (
                document["scope"] != "all"
                or document["phase"] != "CLUSTERS_DRAINING"
                and not resumable
                or "PREFLIGHT" not in completed
            ):
                raise CleanupStateError("registry drain requires completed preflight")
            cluster_arguments = [
                argument
                for target in document["targets"]["clusters"]
                for argument in ("--cluster-id", target["cluster_id"])
            ]
            if cluster_arguments:
                result = run_command(
                    [
                        str(
                            Path(__file__).resolve().parents[1]
                            / "regional"
                            / "rollout-regional-release.sh"
                        ),
                        "drain-cluster",
                        "--config",
                        str(arguments.config),
                        *cluster_arguments,
                    ],
                    timeout_seconds=arguments.timeout_seconds,
                )
                if result.returncode:
                    raise CleanupStateError("regional registry drain did not converge")
        elif arguments.action == "capture":
            document.setdefault("cluster_uids", {})[arguments.context] = verify_cluster(
                client, document, arguments.context
            )
            capture_namespace(client, document, arguments.context)
            if arguments.context != "cpu" and document["node_mode"] != "skip":
                node_targets(
                    client,
                    document,
                    arguments.context,
                    arguments.cluster_id,
                    config=config,
                )
            atomic_write(arguments.state_file, document)
        elif arguments.action == "delete-namespace":
            delete_namespace(client, document, arguments.context)
        elif arguments.action == "cleanup-owned":
            cleanup_pending(config, document, arguments.state_file)
        elif arguments.action == "clear-node-metadata":
            clear_node_metadata(
                client,
                document,
                arguments.context,
                arguments.cluster_id,
                config=config,
            )
        elif arguments.action == "delete-workload-rbac":
            delete_workload_rbac(
                config,
                document,
                arguments.state_file,
                context=arguments.context,
                cluster_id=arguments.cluster_id,
                skip_cpu=arguments.skip_cpu,
            )
        elif arguments.action == "verify-targets":
            contexts = [
                "gpu:" + item["context"] for item in document["targets"]["clusters"]
            ]
            if not arguments.skip_cpu:
                contexts.insert(0, "cpu")
            for context in contexts:
                if context not in document.get("cluster_uids", {}):
                    raise CleanupStateError("cleanup lacks a bound cluster identity")
                verify_cluster(CleanupClient(config, context), document, context)
                verify_quiesced(CleanupClient(config, context), document, context)
            if "APPLICATION_OBJECTS_DELETED" in completed_phases(document):
                for target in document["targets"]["clusters"]:
                    delete_workload_rbac(
                        config,
                        document,
                        arguments.state_file,
                        context="gpu:" + target["context"],
                        cluster_id=target["cluster_id"],
                        skip_cpu=arguments.skip_cpu,
                    )
        else:
            run_node_cleanup(
                config,
                document,
                arguments.state_file,
                context=arguments.context,
                cluster_id=arguments.cluster_id,
                image=arguments.image,
                mode=arguments.node_mode,
                host_script_b64=os.environ["GPU_FAULT_CLEANUP_HOST_SCRIPT_B64"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
