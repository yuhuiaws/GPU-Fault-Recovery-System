"""Validate stable release, Pod, physical Node and authenticated Agent identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.fleet_endpoint import parse_endpoint_networks, validate_agent_endpoint
from scripts.e2e.regional.auth015_protocol import (
    CHALLENGE_OPERATION,
    PROOF_TIMEOUT_SECONDS,
    Auth015ProofError,
)
from scripts.e2e.regional.auth015_release import SHA256, VerifiedAuth015Release
from scripts.e2e.regional.identity_acceptance_common import ClusterTarget


@dataclass(frozen=True, repr=False)
class BoundSnapshot:
    binding: dict[str, Any]
    agents: dict[str, AgentRecord]


def text(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise Auth015ProofError("AUTH015 identity contains an empty or invalid field")
    return value


def objects(document: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    items = document.get("items")
    if (
        document.get("apiVersion") != "v1"
        or document.get("kind") not in {kind + "List", "List"}
        or (document.get("metadata") or {}).get("continue")
        or not isinstance(items, list)
        or not items
        or any(not isinstance(item, dict) for item in items)
    ):
        raise Auth015ProofError("AUTH015 Kubernetes inventory is incomplete")
    return items


def pod_binding(pod: dict[str, Any], image: str) -> dict[str, str]:
    metadata, spec, status = pod["metadata"], pod["spec"], pod["status"]
    containers = [item for item in spec["containers"] if item["name"] == "api"]
    running = [item for item in status["containerStatuses"] if item["name"] == "api"]
    if (
        pod.get("apiVersion") != "v1"
        or pod.get("kind") != "Pod"
        or metadata.get("deletionTimestamp")
        or metadata.get("labels", {}).get("app") != "gpu-fault-api-ha"
        or status.get("phase") != "Running"
        or not any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in status.get("conditions", [])
        )
        or len(containers) != 1
        or containers[0].get("image") != image
        or len(running) != 1
        or running[0].get("ready") is not True
        or not running[0].get("state", {}).get("running")
        or text(running[0].get("imageID")).rsplit("sha256:", 1)[-1]
        != image.rsplit("sha256:", 1)[-1]
    ):
        raise Auth015ProofError("AUTH015 CPU Pod is not running the signed image")
    return {
        "name": text(metadata.get("name")),
        "namespace": text(metadata.get("namespace")),
        "uid": text(metadata.get("uid")),
        "container_id": text(running[0].get("containerID")),
        "image": image,
    }


def node_binding(
    node: dict[str, Any],
    record: AgentRecord,
    target: ClusterTarget,
    release: VerifiedAuth015Release,
    state: dict[str, Any],
    cidrs: list[str],
    now: datetime,
) -> dict[str, Any]:
    metadata, status = node["metadata"], node["status"]
    provider_id = text(node["spec"].get("providerID"))
    addresses = [
        text(item["address"])
        for item in status.get("addresses", [])
        if item.get("type") == "InternalIP"
    ]
    if (
        node.get("apiVersion") != "v1"
        or node.get("kind") != "Node"
        or metadata.get("deletionTimestamp")
        or metadata.get("name") != record.node_id
        or metadata.get("labels", {}).get("sagemaker.amazonaws.com/cluster-name")
        != target.hyperpod_cluster_name
        or not any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in status.get("conditions", [])
        )
        or not provider_id.startswith("aws:///")
        or provider_id.rsplit("/", 1)[-1] != record.node_instance_id
        or status["nodeInfo"].get("bootID") != record.boot_id
        or not record.boot_id
        or not record.agent_incarnation_id
        or record.cluster_id != target.cluster_id
        or record.lifecycle_state is not AgentLifecycleState.ACTIVE
        or record.node_action_key_version != 2
        or record.agent_protocol_version != release.agent_protocol_version
        or record.artifact_sha256 != release.node_wheel_sha256
        or record.compatibility_digest != release.node_digest
        or record.installer_bundle_sha256 != release.bundle_sha256
        or record.installer_template_sha256 != state["node_template_sha256"]
        or record.config_digest != state["agent_config_digest"]
        or record.runtime_profile_version != state["runtime_profile_version"]
        or CHALLENGE_OPERATION in record.allowed_operations
        or record.last_seen_at.tzinfo is None
        or not now - timedelta(seconds=60)
        <= record.last_seen_at
        <= now + timedelta(seconds=5)
        or record.lease_expires_at is None
        or record.lease_expires_at.tzinfo is None
        or record.lease_expires_at <= now + timedelta(seconds=PROOF_TIMEOUT_SECONDS)
        or not record.endpoint.startswith("https://")
        or urlsplit(record.endpoint).hostname not in addresses
        or not record.tls_certificate_pem
    ):
        raise Auth015ProofError("AUTH015 Node and active Agent identities do not agree")
    validate_agent_endpoint(
        record.node_id,
        record.endpoint,
        allowed_networks=parse_endpoint_networks(",".join(cidrs)),
    )
    return {
        "node_id": record.node_id,
        "node_uid": text(metadata.get("uid")),
        "provider_id": provider_id,
        "boot_id": record.boot_id,
        "agent_incarnation_id": record.agent_incarnation_id,
        "agent_generation": record.generation,
        "endpoint": record.endpoint,
        "certificate_sha256": hashlib.sha256(
            record.tls_certificate_pem.encode()
        ).hexdigest(),
        "artifact_sha256": record.artifact_sha256,
        "compatibility_digest": record.compatibility_digest,
        "installer_bundle_sha256": record.installer_bundle_sha256,
        "installer_template_sha256": record.installer_template_sha256,
        "config_digest": record.config_digest,
        "runtime_profile_version": record.runtime_profile_version,
        "allowed_operations": sorted(item.value for item in record.allowed_operations),
    }


def bind_snapshot(
    raw: dict[str, Any],
    *,
    release: VerifiedAuth015Release,
    target: ClusterTarget,
    nodes: tuple[str, str],
) -> BoundSnapshot:
    try:
        state = raw["release_state"]
        snapshot = raw["agent_snapshot"]
        version = snapshot["version"]
        registration = raw["registration"]
        release_metadata = raw["release_metadata"]
        cidrs = registration["agent_endpoint_allowed_cidrs"]
        if (
            len(set(nodes)) != 2
            or raw["pod"]["metadata"]["namespace"] != raw["namespace"]
            or release_metadata["namespace"] != raw["namespace"]
            or release_metadata["name"] != "gpu-fault-regional-release-state"
            or release_metadata.get("deletionTimestamp")
            or state["release_id"] != release.release_id
            or state["phase"] != "complete"
            or state["transaction_committed"] is not True
            or state["release_delivery_sha256"] != release.delivery_sha256
            or state["bundle_sha256"] != release.bundle_sha256
            or state["runtime_image"] != release.runtime_image
            or snapshot["release_id"] != release.release_id
            or version["module_digest"] != release.control_plane_digest
            or version["deployment_mode"] != "regional"
            or version["required_agent_artifact_sha256"] != release.node_wheel_sha256
            or version["required_agent_compatibility_digest"] != release.node_digest
            or type(version["required_agent_protocol_version"]) is not int
            or version["required_agent_protocol_version"]
            != release.agent_protocol_version
            or version["required_agent_config_digest"] != state["agent_config_digest"]
            or version["required_runtime_profile_version"]
            != state["runtime_profile_version"]
            or type(version["required_node_action_key_version"]) is not int
            or version["required_node_action_key_version"] != 2
            or registration["cluster_id"] != target.cluster_id
            or registration["enabled"] is not True
            or registration["lifecycle_state"] != "ACTIVE"
            or not isinstance(cidrs, list)
            or not cidrs
            or any(not isinstance(value, str) or not value for value in cidrs)
            or set(snapshot["agents"]) != set(nodes)
        ):
            raise ValueError
        for field in ("node_template_sha256", "agent_config_digest"):
            if SHA256.fullmatch(text(state[field])) is None:
                raise ValueError
        physical = objects(raw["nodes"], "Node")
        if len(physical) != 2 or {item["metadata"]["name"] for item in physical} != set(
            nodes
        ):
            raise ValueError
        agents = {
            name: AgentRecord.model_validate_json(
                json.dumps(snapshot["agents"][name]), strict=True
            )
            for name in nodes
        }
        now = datetime.now(timezone.utc)
        bindings = {
            item["metadata"]["name"]: node_binding(
                item,
                agents[item["metadata"]["name"]],
                target,
                release,
                state,
                cidrs,
                now,
            )
            for item in physical
        }
        for field in (
            "node_uid",
            "provider_id",
            "boot_id",
            "agent_incarnation_id",
            "endpoint",
        ):
            if len({item[field] for item in bindings.values()}) != 2:
                raise ValueError
        return BoundSnapshot(
            binding={
                "release_id": release.release_id,
                "release_inputs": release.input_identity,
                "release_state_uid": text(release_metadata.get("uid")),
                "cluster_id": target.cluster_id,
                "eks_cluster_arn": target.eks_cluster_arn,
                "cpu_pod": pod_binding(raw["pod"], release.runtime_image),
                "endpoint_cidrs": sorted(cidrs),
                "nodes": bindings,
            },
            agents=agents,
        )
    except Auth015ProofError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise Auth015ProofError(
            "AUTH015 live identity snapshot is incomplete or differs"
        ) from None
