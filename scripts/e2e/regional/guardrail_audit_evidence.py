"""Identity and read-only provider recording checks for deployed guard audits."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, TypedDict

from gpu_fault.cluster_scope import PROVIDER_NODE_REBOOT_OWNERS
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records


def complete_pod_population(
    deployment: dict[str, Any], inventory: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata, spec, status = (
        deployment["metadata"],
        deployment["spec"],
        deployment.get("status", {}),
    )
    desired, generation = spec.get("replicas"), metadata.get("generation")
    pods = ready_pod_records(inventory)
    if (
        type(desired) is not int
        or desired < 1
        or type(generation) is not int
        or generation < 1
        or not isinstance(metadata.get("uid"), str)
        or not metadata["uid"]
        or metadata.get("deletionTimestamp")
        or type(status.get("observedGeneration")) is not int
        or status["observedGeneration"] < generation
        or any(
            type(status.get(key)) is not int or status[key] != desired
            for key in (
                "replicas",
                "readyReplicas",
                "updatedReplicas",
                "availableReplicas",
            )
        )
        or len(pods) != desired
        or len(inventory["items"]) != desired
    ):
        raise RuntimeError("guard audit requires the complete stable Ready population")
    return pods


def validate_provider_record(
    snapshot: dict[str, Any], expected_cluster: str, *, now: datetime | None = None
) -> None:
    try:
        payload = snapshot["payloads"]
        recorded = datetime.fromisoformat(
            snapshot["recorded_at"].replace("Z", "+00:00")
        )
        if recorded.tzinfo is None:
            raise ValueError("naive timestamp")
        age = ((now or datetime.now(timezone.utc)) - recorded).total_seconds()
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        if not 0 <= age <= 300 or digest != snapshot["payload_digest"]:
            raise ValueError("stale or changed provider record")
        if (
            payload["cluster_name"] != expected_cluster
            or payload["describe_cluster"]["ClusterName"] != expected_cluster
        ):
            raise ValueError("recorded cluster differs")
        pages = payload["list_cluster_nodes"]
        if not isinstance(pages, list) or not pages:
            raise ValueError("missing provider pages")
        tokens, nodes = set(), set()
        for index, page in enumerate(pages):
            token = page.get("NextToken")
            if index < len(pages) - 1:
                if not isinstance(token, str) or not token or token in tokens:
                    raise ValueError("invalid page token")
                tokens.add(token)
            elif token:
                raise ValueError("incomplete page chain")
            for node in page["ClusterNodeSummaries"]:
                logical_id = node["NodeLogicalId"]
                if (
                    not isinstance(logical_id, str)
                    or not logical_id
                    or logical_id in nodes
                ):
                    raise ValueError("invalid or repeated node identity")
                nodes.add(logical_id)
                detail = payload["describe_cluster_node"][logical_id]
                if any(
                    detail.get(key) != node.get(key)
                    for key in ("NodeLogicalId", "InstanceId", "InstanceGroupName")
                ):
                    raise ValueError("provider node identity differs between reads")
        if not nodes or set(payload["describe_cluster_node"]) != nodes:
            raise ValueError("provider node details are incomplete")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise RuntimeError(
            "provider recording has invalid identity, freshness or completeness"
        ) from None


REGISTRY_PROBE = r"""
import json, os
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.pool import StoreCredentials
path = os.getenv("GPU_FAULT_STORE_URL_FILE")
credentials = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL", ""), path=path)
url = credentials.conninfo()
if not url or (path and credentials.source != "file"):
    raise RuntimeError("current store credential is unavailable")
store = PostgresStore(url, initialize_schema=False)
try:
    head = store.get_regional_registry_head()
    revision = store.get_regional_registry_revision(head.generation)
    if revision.content_sha256 != head.content_sha256:
        raise RuntimeError("registry head changed")
    rows = [{"cluster_id": row.cluster_id, "region": row.region,
             "hyperpod_cluster_name": row.hyperpod_cluster_name,
             "eks_cluster_arn": row.eks_cluster_arn,
             "provider": getattr(row, "provider", "") or "hyperpod"}
            for row in revision.registrations]
    if store.get_regional_registry_head() != head:
        raise RuntimeError("registry changed during audit")
    print(json.dumps({"generation": head.generation, "registrations": rows}))
finally:
    store.close()
"""


def _row_identity_complete(row: dict[str, Any]) -> bool:
    """Every registration names its cluster, Region and EKS cluster; a HyperPod
    registration also its HyperPod cluster. A plain-EKS registration
    (``provider: eks-managed``) has no HyperPod and leaves the name empty --
    the mixed site's second cluster (live 2026-10-04: DESTR-005/006/007 on the
    HyperPod cluster refused because of it)."""

    for key in ("cluster_id", "region", "eks_cluster_arn"):
        if not isinstance(row.get(key), str) or not row[key]:
            return False
    provider = str(row.get("provider") or "hyperpod")
    name = row.get("hyperpod_cluster_name")
    if provider == "eks-managed":
        return isinstance(name, str)
    return isinstance(name, str) and bool(name)


def registration_identity(
    registry: dict[str, Any], managed_cluster: str, negative_cluster: str, region: str
) -> dict[str, str]:
    rows = registry.get("registrations")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("durable registry inventory is missing")
    if any(
        not isinstance(row, dict) or not _row_identity_complete(row) for row in rows
    ):
        raise RuntimeError("durable registry identities are incomplete")
    if negative_cluster and any(
        row["hyperpod_cluster_name"] == negative_cluster for row in rows
    ):
        raise RuntimeError(
            "Automatic negative cluster is registered in the managed fleet"
        )
    managed = [row for row in rows if _names_managed_cluster(row, managed_cluster)]
    if len(managed) != 1 or managed[0]["region"] != region:
        raise RuntimeError("managed cluster does not match the durable registry")
    return dict(managed[0])


def _names_managed_cluster(row: dict[str, Any], managed_cluster: str) -> bool:
    """A HyperPod registration is named by its HyperPod cluster; a plain-EKS one
    (no HyperPod) by its ``cluster_id`` or EKS cluster name -- the value the
    provider cluster scope (``--hyperpod-cluster``) carries for it."""

    if row.get("hyperpod_cluster_name") == managed_cluster:
        return True
    if str(row.get("provider") or "hyperpod") != "eks-managed":
        return False
    eks_name = str(row.get("eks_cluster_arn") or "").rsplit("/", 1)[-1]
    return managed_cluster in {row.get("cluster_id"), eks_name}


#: The EC2 mutations that replace, stop or launch a plain-EKS GPU node; the
#: provider-side counterpart of HyperPod's node mutations for ``eks-managed``.
EC2_REPLACEMENT_MUTATIONS = frozenset(
    {"TerminateInstances", "StopInstances", "RunInstances", "StartInstances"}
)


def provider_event_sources(
    provider: str, hyperpod_mutations: frozenset[str]
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Which CloudTrail event sources (and event names) count as provider
    mutations for a managed cluster of ``provider``."""

    sources: tuple[tuple[str, frozenset[str]], ...] = (
        ("sagemaker.amazonaws.com", hyperpod_mutations),
    )
    if provider == "eks-managed":
        sources += (("ec2.amazonaws.com", EC2_REPLACEMENT_MUTATIONS),)
    return sources


class ProviderCallInventory(TypedDict):
    """A CloudTrail window's provider calls: the ones AWS accepted (mutations)
    and the ones it refused (``errorCode`` set, nothing ran -- on a plain EKS
    cluster Karpenter's ``RunInstances`` dry-runs are the common case)."""

    mutations: list[dict[str, Any]]
    refused_calls: list[dict[str, Any]]


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _refusal(item: dict[str, Any]) -> dict[str, Any] | None:
    """The error code and caller of a refused call; None when the record has no
    readable ``errorCode`` (it then stays a mutation, fail closed)."""

    try:
        detail = _object(json.loads(item["CloudTrailEvent"]))
    except (KeyError, TypeError, ValueError):
        return None
    code = detail.get("errorCode")
    if not isinstance(code, str) or not code:
        return None
    issuer = _object(
        _object(_object(detail.get("userIdentity")).get("sessionContext")).get(
            "sessionIssuer"
        )
    )
    arn = issuer.get("arn")
    return {
        "api_error_code": code,
        "username": item.get("Username"),
        "session_issuer_role_name": (
            arn.rsplit("/", 1)[-1] if isinstance(arn, str) and ":role/" in arn else None
        ),
    }


def provider_call_inventory(
    started_at: datetime,
    ended_at: datetime,
    *,
    provider: str,
    region: str,
    command: Any,
    hyperpod_mutations: frozenset[str],
) -> ProviderCallInventory:
    """Provider calls in the window, read from CloudTrail per event source and
    split into accepted mutations and refused calls (``_refusal``)."""

    inventory: ProviderCallInventory = {"mutations": [], "refused_calls": []}
    for source, names in provider_event_sources(provider, hyperpod_mutations):
        value = json.loads(
            command(
                [
                    "aws",
                    "cloudtrail",
                    "lookup-events",
                    "--region",
                    region,
                    "--start-time",
                    started_at.isoformat(),
                    "--end-time",
                    ended_at.isoformat(),
                    "--lookup-attributes",
                    f"AttributeKey=EventSource,AttributeValue={source}",
                    "--output",
                    "json",
                ]
            )
        )
        if not isinstance(value, dict) or not isinstance(value.get("Events"), list):
            raise RuntimeError("CloudTrail event inventory is unreadable")
        for item in value["Events"]:
            if not isinstance(item, dict):
                raise RuntimeError("CloudTrail event inventory is unreadable")
            if item.get("EventName") not in names:
                continue
            record = {
                "event_name": item.get("EventName"),
                "event_time": str(item.get("EventTime")),
                "event_source": source,
            }
            refusal = _refusal(item)
            if refusal is None:
                inventory["mutations"].append(record)
            else:
                inventory["refused_calls"].append({**record, **refusal})
    return inventory


def provider_replace_events(
    started_at: datetime,
    ended_at: datetime,
    *,
    provider: str,
    region: str,
    command: Any,
    hyperpod_mutations: frozenset[str],
) -> list[dict[str, Any]]:
    """The accepted provider mutations in the window (``provider_call_inventory``)."""

    return provider_call_inventory(
        started_at,
        ended_at,
        provider=provider,
        region=region,
        command=command,
        hyperpod_mutations=hyperpod_mutations,
    )["mutations"]


def managed_recovery_invariant_errors(
    gpu: dict[str, Any], *, provider: str
) -> list[str]:
    """The provider-side recovery invariant of the managed cluster.

    HyperPod: ``NodeRecovery`` must be ``None`` (the product, not the provider,
    owns node recovery). Plain EKS has no HyperPod NodeRecovery; its invariant is
    that no provider-side replacement is enabled or observed: the cluster's
    reboot owner is the EKS adapter, nodeReplace is the warm-spare handover only
    (no EC2 instance is replaced), and the CloudTrail window (judged separately)
    carries no EC2 replacement mutation.
    """

    if provider == "eks-managed":
        if gpu.get("node_recovery") != "not-applicable":
            return ["plain-EKS managed cluster reports a HyperPod NodeRecovery"]
        return []
    if gpu.get("node_recovery") != "None":
        return ["managed GPU cluster NodeRecovery is not None"]
    return []


def eks_managed_invariant() -> dict[str, Any]:
    return {
        "provider": "eks-managed",
        "node_recovery": "not-applicable",
        "node_reboot_owner": PROVIDER_NODE_REBOOT_OWNERS["eks-managed"],
        "node_replace": "warm-spare-only",
        "provider_mutation_sources": [
            source
            for source, _names in provider_event_sources("eks-managed", frozenset())
        ],
    }
