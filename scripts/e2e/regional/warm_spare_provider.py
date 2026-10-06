"""Provider-side facts the warm-spare cases read, for HyperPod and plain EKS.

``GF-REGIONAL-DESTR-003/008/022`` were written against HyperPod: the cluster
fact was ``sagemaker describe-cluster`` (``InService``, ``NodeRecovery=None``),
the node inventory ``sagemaker list-cluster-nodes``, the spare's health gate
the health-monitoring-agent label and its topology the instance group. A
plain-EKS (``eks-managed``) cluster has none of those; the equivalent facts
are the EKS cluster status (``ACTIVE``), the GPU nodes as Kubernetes reports
them (name, EC2 instance id from ``spec.providerID``, managed node group,
instance type, Ready), no health label at all, and the node group as the
failure-domain group. Each function here takes the provider and returns the
same shape the runners already judge, so the case logic reads one way for
both.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    PROVIDER_NODE_REBOOT_OWNERS,
    cluster_provider,
)
from scripts.e2e.regional.guardrail_audit_evidence import (
    ProviderCallInventory,
    provider_call_inventory,
)
from scripts.e2e.regional.regional_workload_inventory import node_provider

HYPERPOD = "hyperpod"
EKS_MANAGED = "eks-managed"
HYPERPOD_HEALTH_LABEL = "sagemaker.amazonaws.com/node-health-status"
INSTANCE_GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"
INSTANCE_TYPE_LABELS = (
    "node.kubernetes.io/instance-type",
    "beta.kubernetes.io/instance-type",
)
TOPOLOGY_GROUP_LABELS = {
    HYPERPOD: INSTANCE_GROUP_LABEL,
    EKS_MANAGED: EKS_NODEGROUP_LABEL,
}
#: What a healthy cluster reads per provider: HyperPod must be InService with the
#: product (not the provider) owning node recovery; a plain EKS cluster must be
#: ACTIVE and has no NodeRecovery to speak of.
HEALTHY_CLUSTER = {
    HYPERPOD: {"status": "InService", "node_recovery": "None"},
    EKS_MANAGED: {"status": "ACTIVE", "node_recovery": "not-applicable"},
}
#: The sagemaker mutations a HyperPod warm-spare case must never see.
PROVIDER_REPLACE_EVENTS = frozenset(
    {
        "BatchDeleteClusterNodes",
        "BatchReplaceClusterNodes",
        "DeleteClusterNodes",
        "ReplaceClusterNodes",
    }
)


def normalized(provider: str | None) -> str:
    return cluster_provider(provider).value


def gpu_node_provider(nodes_document: Mapping[str, Any]) -> str:
    """The provider the GPU nodes' labels show; HyperPod when nothing says EKS."""

    for item in nodes_document.get("items", []) or []:
        allocatable = (item.get("status") or {}).get("allocatable") or {}
        if not allocatable.get("nvidia.com/gpu"):
            continue
        found = node_provider((item.get("metadata") or {}).get("labels") or {})
        if found in TOPOLOGY_GROUP_LABELS:
            return found
    return HYPERPOD


def instance_type(snapshot: Mapping[str, Any]) -> str | None:
    labels = snapshot.get("labels") or {}
    return next(
        (labels.get(key) for key in INSTANCE_TYPE_LABELS if labels.get(key)), None
    )


def topology(snapshot: Mapping[str, Any], provider: str | None) -> dict[str, Any]:
    """The group and instance type a spare must share with its fault node."""

    label = TOPOLOGY_GROUP_LABELS[normalized(provider)]
    return {
        "instance_group": (snapshot.get("labels") or {}).get(label),
        "instance_type": instance_type(snapshot),
    }


def spare_health_errors(spare: Mapping[str, Any], provider: str | None) -> list[str]:
    """The provider health gate a declared spare must pass; none on plain EKS."""

    if normalized(provider) == HYPERPOD:
        if (spare.get("labels") or {}).get(HYPERPOD_HEALTH_LABEL) != "Schedulable":
            return ["spare HyperPod health label is not Schedulable"]
        return []
    if not (spare.get("labels") or {}).get(EKS_NODEGROUP_LABEL):
        return ["spare eks-managed node carries no managed node group label"]
    return []


def cluster_recovery_errors(
    cluster: Mapping[str, Any], provider: str | None
) -> list[str]:
    expected = HEALTHY_CLUSTER[normalized(provider)]
    if (
        cluster.get("status") != expected["status"]
        or cluster.get("node_recovery") != expected["node_recovery"]
    ):
        if normalized(provider) == HYPERPOD:
            return ["HyperPod cluster is not InService with NodeRecovery=None"]
        return ["EKS cluster is not ACTIVE (plain EKS has no managed node recovery)"]
    return []


def lifecycle_owner(provider: str | None) -> str:
    """The executor adapter that OWNs nodeReboot and nodeReplace on ``provider``."""

    return PROVIDER_NODE_REBOOT_OWNERS[normalized(provider)]


def eks_cluster_recovery(
    describe: Mapping[str, Any], cluster_name: str
) -> dict[str, Any]:
    """``aws eks describe-cluster`` reduced to the HyperPod-shaped cluster fact."""

    cluster = describe.get("cluster") or {}
    return {
        "cluster_name": cluster.get("name") or cluster_name,
        "status": cluster.get("status"),
        "node_recovery": "not-applicable",
        "provider": EKS_MANAGED,
    }


def eks_provider_inventory(nodes_document: Mapping[str, Any]) -> dict[str, Any]:
    """The GPU nodes as Kubernetes reports them, in the HyperPod inventory shape.

    Stable across a case that replaces nothing: a Spot replacement shows up as a
    new instance id under a new name and changes the digest, which is exactly
    the drift the postflight compares against.
    """

    rows = []
    for item in nodes_document.get("items", []) or []:
        status = item.get("status") or {}
        if not (status.get("allocatable") or {}).get("nvidia.com/gpu"):
            continue
        metadata = item.get("metadata") or {}
        labels = metadata.get("labels") or {}
        provider_id = str((item.get("spec") or {}).get("providerID") or "")
        rows.append(
            {
                "node_logical_id": metadata.get("name"),
                "instance_id": provider_id.rsplit("/", 1)[-1] or None,
                "instance_group": labels.get(EKS_NODEGROUP_LABEL),
                "instance_type": instance_type({"labels": labels}),
                "status": next(
                    (
                        "Running" if condition.get("status") == "True" else "NotReady"
                        for condition in status.get("conditions", []) or []
                        if condition.get("type") == "Ready"
                    ),
                    "Unknown",
                ),
            }
        )
    rows.sort(key=lambda row: (str(row["node_logical_id"]), str(row["instance_id"])))
    encoded = json.dumps(rows, sort_keys=True).encode()
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "nodes": rows,
    }


def provider_calls(
    started_at: datetime,
    ended_at: datetime,
    *,
    provider: str | None,
    region: str,
    command: Callable[[list[str]], str],
) -> ProviderCallInventory:
    """Provider replacement calls in the window (sagemaker's, plus EC2's on
    EKS): the accepted mutations, and the calls AWS refused kept apart."""

    return provider_call_inventory(
        started_at,
        ended_at,
        provider=normalized(provider),
        region=region,
        command=command,
        hyperpod_mutations=PROVIDER_REPLACE_EVENTS,
    )


def replace_events(
    started_at: datetime,
    ended_at: datetime,
    *,
    provider: str | None,
    region: str,
    command: Callable[[list[str]], str],
) -> list[dict[str, Any]]:
    """The accepted provider replacement mutations in the window."""

    return provider_calls(
        started_at, ended_at, provider=provider, region=region, command=command
    )["mutations"]
