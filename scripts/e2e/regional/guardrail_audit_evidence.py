"""Identity and read-only provider recording checks for deployed guard audits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

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
             "eks_cluster_arn": row.eks_cluster_arn}
            for row in revision.registrations]
    if store.get_regional_registry_head() != head:
        raise RuntimeError("registry changed during audit")
    print(json.dumps({"generation": head.generation, "registrations": rows}))
finally:
    store.close()
"""


def registration_identity(
    registry: dict[str, Any], managed_cluster: str, negative_cluster: str, region: str
) -> dict[str, str]:
    rows = registry.get("registrations")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("durable registry inventory is missing")
    if any(
        not isinstance(row, dict)
        or not all(
            isinstance(row.get(key), str) and row[key]
            for key in (
                "cluster_id",
                "region",
                "hyperpod_cluster_name",
                "eks_cluster_arn",
            )
        )
        for row in rows
    ):
        raise RuntimeError("durable registry identities are incomplete")
    if negative_cluster and any(
        row["hyperpod_cluster_name"] == negative_cluster for row in rows
    ):
        raise RuntimeError(
            "Automatic negative cluster is registered in the managed fleet"
        )
    managed = [row for row in rows if row["hyperpod_cluster_name"] == managed_cluster]
    if len(managed) != 1 or managed[0]["region"] != region:
        raise RuntimeError("managed cluster does not match the durable registry")
    return dict(managed[0])
