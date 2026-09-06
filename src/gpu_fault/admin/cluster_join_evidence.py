from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    hydrate_previous_snapshot,
)

REGIONAL_RELEASE_STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
CPU_INGRESS_APP = "gpu-fault-control-plane-ingress"
VERIFICATION_MAX_AGE_SECONDS = 900
REGISTRY_STATUS_CLIENT = r"""
import json
import os
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/regional/registry/status",
    headers={
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""
RELEASE_IDENTITY_FIELDS = (
    "release_id",
    "wheel_sha256",
    "executor_wheel_sha256",
    "node_wheel_sha256",
    "bundle_sha256",
    "database_schema_version",
    "agent_protocol_version",
    "executor_protocol_version",
    "component_digests",
    "runtime_profile_version",
    "runtime_profile_policy_sha256",
    "agent_config_digest",
    "release_delivery_sha256",
    "runtime_image",
    "node_installer_image",
    "dcgm_image",
    "adot_image",
)


def join_activation_is_irreversible(state: dict[str, Any]) -> bool:
    completed = set(state.get("completed_steps") or [])
    return bool(completed.intersection({"ACTIVATION_STARTED", "ACTIVATED"}))


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BootstrapError(f"{description} is not a mapping")
    return {str(key): item for key, item in value.items()}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site_non_membership_sha256(path: Path) -> str:
    try:
        document = _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")),
            "site document",
        )
    except (OSError, yaml.YAMLError) as exc:
        raise BootstrapError(f"cannot read site identity from {path}") from exc
    normalized = copy.deepcopy(document)
    spec = _mapping(normalized.get("spec"), "site spec")
    spec.pop("clusters", None)
    spec.pop("gpuKubeconfig", None)
    normalized["spec"] = spec
    return _canonical_sha256(normalized)


def membership_runtime_snapshot(site: RenderedSite) -> dict[str, Any]:
    cpu = [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
    ]
    namespace = str(site.release_config["namespace"])
    state_result = subprocess.run(
        [
            *cpu,
            "-n",
            namespace,
            "get",
            "configmap",
            REGIONAL_RELEASE_STATE_CONFIG_MAP,
            "-o",
            "json",
        ],
        text=True,
        capture_output=True,
    )
    if state_result.returncode:
        raise BootstrapError(
            "cannot read live regional release state: " + state_result.stderr.strip()
        )
    state_document = _mapping(
        json.loads(state_result.stdout or "{}"),
        "live regional release state ConfigMap",
    )
    state_data = _mapping(state_document.get("data") or {}, "release state data")
    raw_state = str(state_data.get("state.json") or "")
    if not raw_state:
        raise BootstrapError("live regional release state is empty")
    parsed_state = _mapping(json.loads(raw_state), "live regional release state")

    def read_snapshot_config_map(name: str) -> dict[str, Any]:
        result = subprocess.run(
            [
                *cpu,
                "-n",
                namespace,
                "get",
                "configmap",
                name,
                "-o",
                "json",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise BootstrapError(
                f"cannot read live release previous snapshot {name}: "
                + result.stderr.strip()
            )
        return _mapping(
            json.loads(result.stdout or "{}"),
            f"live release previous snapshot {name}",
        )

    try:
        live_state = hydrate_previous_snapshot(
            parsed_state,
            read_snapshot_config_map,
        )
    except ReleaseStateSnapshotError as exc:
        raise BootstrapError(
            "live regional release previous snapshot is invalid"
        ) from exc

    pod_result = subprocess.run(
        [
            *cpu,
            "-n",
            namespace,
            "get",
            "pod",
            "-l",
            f"app={CPU_INGRESS_APP}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        text=True,
        capture_output=True,
    )
    pod = (pod_result.stdout or "").strip()
    if pod_result.returncode or not pod:
        raise BootstrapError("no Running CPU ingress Pod for membership evidence")
    registry_result = subprocess.run(
        [
            *cpu,
            "-n",
            namespace,
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            REGISTRY_STATUS_CLIENT,
        ],
        text=True,
        capture_output=True,
    )
    if registry_result.returncode:
        raise BootstrapError(
            "cannot read regional registry generation: "
            + registry_result.stderr.strip()
        )
    registry = _mapping(
        json.loads(registry_result.stdout or "{}"),
        "regional registry status",
    )
    cluster_states = {
        str(cluster_id): str(lifecycle)
        for cluster_id, lifecycle in _mapping(
            registry.get("cluster_states") or {},
            "regional registry cluster states",
        ).items()
    }
    release_identity = {
        field: live_state.get(field) for field in RELEASE_IDENTITY_FIELDS
    }
    return {
        "live_release_state_sha256": hashlib.sha256(raw_state.encode()).hexdigest(),
        "live_release_identity_sha256": _canonical_sha256(release_identity),
        "registry_generation": int(registry["generation"]),
        "registry_content_sha256": str(registry["content_sha256"]),
        "registry_cluster_states": cluster_states,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def build_verified_membership_evidence(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    candidate_site_sha256: str,
    source_site_sha256: str,
    source_site_non_membership_sha256: str,
    candidate_cluster_ids: list[str],
    cluster_id: str,
    batch_id: str | None = None,
    verified_at: datetime | None = None,
) -> dict[str, Any]:
    compared = (
        "live_release_state_sha256",
        "live_release_identity_sha256",
        "registry_generation",
        "registry_content_sha256",
        "registry_cluster_states",
    )
    drifted = [field for field in compared if before.get(field) != after.get(field)]
    if drifted:
        raise BootstrapError(
            "membership identity drifted during candidate verification: "
            + ", ".join(drifted)
        )
    cluster_states = _mapping(
        after.get("registry_cluster_states") or {},
        "verified registry cluster states",
    )
    if cluster_states.get(cluster_id) != "PENDING":
        raise BootstrapError(
            f"joined cluster {cluster_id} is not PENDING during candidate verification"
        )
    observed = verified_at or datetime.now(timezone.utc)
    return {
        "candidate_site_sha256": candidate_site_sha256,
        "source_site_sha256": source_site_sha256,
        "source_site_non_membership_sha256": (source_site_non_membership_sha256),
        "candidate_cluster_ids": sorted(set(candidate_cluster_ids)),
        "cluster_id": cluster_id,
        "batch_id": batch_id,
        "verified_at": observed.isoformat(),
        **{field: after[field] for field in compared},
    }


def validate_verified_membership(
    *,
    evidence: dict[str, Any],
    state: dict[str, Any],
    current_site: RenderedSite,
    candidate_site: RenderedSite,
    cluster_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    if evidence.get("cluster_id") != cluster_id:
        raise BootstrapError("join verification evidence cluster identity drifted")
    candidate_sha256 = file_sha256(candidate_site.source)
    if (
        evidence.get("candidate_site_sha256") != candidate_sha256
        or candidate_site.source_sha256 != candidate_sha256
    ):
        raise BootstrapError("join candidate site changed after verification")
    if evidence.get("source_site_sha256") != state.get("source_site_sha256"):
        raise BootstrapError("join source site identity differs from transaction state")
    expected_non_membership = str(state.get("source_site_non_membership_sha256") or "")
    if (
        not expected_non_membership
        or evidence.get("source_site_non_membership_sha256") != expected_non_membership
        or site_non_membership_sha256(current_site.source) != expected_non_membership
    ):
        raise BootstrapError("join source site non-membership fields drifted")

    try:
        verified_at = datetime.fromisoformat(str(evidence["verified_at"]))
    except (KeyError, ValueError) as exc:
        raise BootstrapError("join verification timestamp is invalid") from exc
    if verified_at.tzinfo is None:
        raise BootstrapError("join verification timestamp has no timezone")
    observed = now or datetime.now(timezone.utc)
    age = (observed - verified_at).total_seconds()
    if age < -60 or age > VERIFICATION_MAX_AGE_SECONDS:
        raise BootstrapError("join verification evidence expired")

    current = membership_runtime_snapshot(current_site)
    if current["live_release_identity_sha256"] != evidence.get(
        "live_release_identity_sha256"
    ):
        raise BootstrapError("live release identity drifted after join verification")
    baseline_states = {
        str(key): str(value)
        for key, value in _mapping(
            evidence.get("registry_cluster_states") or {},
            "verified registry cluster states",
        ).items()
    }
    current_states = {
        str(key): str(value)
        for key, value in _mapping(
            current.get("registry_cluster_states") or {},
            "current registry cluster states",
        ).items()
    }
    if set(current_states) != set(baseline_states):
        raise BootstrapError("regional registry membership drifted after verification")
    if current_states.get(cluster_id) != "PENDING":
        raise BootstrapError(
            f"joined cluster {cluster_id} crossed the activation boundary unexpectedly"
        )
    candidate_ids = {
        str(value) for value in evidence.get("candidate_cluster_ids") or []
    }
    actual_candidate_ids = {
        str(item["cluster_id"]) for item in candidate_site.release_config["clusters"]
    }
    if candidate_ids != actual_candidate_ids or cluster_id not in candidate_ids:
        raise BootstrapError("join candidate cluster set drifted after verification")
    transitions = {
        item
        for item, lifecycle in current_states.items()
        if lifecycle != baseline_states[item]
    }
    invalid = sorted(
        item
        for item in transitions
        if not (
            item in candidate_ids
            and baseline_states[item] == "PENDING"
            and current_states[item] == "ACTIVE"
        )
    )
    if invalid:
        raise BootstrapError(
            "regional registry lifecycle drifted after verification: "
            + ", ".join(invalid)
        )
    baseline_generation = int(evidence["registry_generation"])
    current_generation = int(current["registry_generation"])
    if current_generation < baseline_generation:
        raise BootstrapError("regional registry generation moved backwards")
    if current_generation == baseline_generation and current[
        "registry_content_sha256"
    ] != evidence.get("registry_content_sha256"):
        raise BootstrapError("regional registry content drifted without a generation")
    transaction_progressed = any(
        step in set(state.get("completed_steps") or [])
        for step in ("SITE_UPDATED", "RELEASE_STATE_UPDATED", "REGISTRY_UPDATED")
    )
    if (
        not transitions
        and not transaction_progressed
        and (
            current_generation != baseline_generation
            or current["live_release_state_sha256"]
            != evidence.get("live_release_state_sha256")
        )
    ):
        raise BootstrapError("membership state drifted after join verification")
    return current


def final_membership_identity(
    site: RenderedSite,
    *,
    candidate_site_sha256: str,
    cluster_id: str,
    verified_at: str,
) -> dict[str, Any]:
    snapshot = membership_runtime_snapshot(site)
    cluster_states = _mapping(
        snapshot.get("registry_cluster_states") or {},
        "final registry cluster states",
    )
    lifecycle = str(cluster_states.get(cluster_id) or "")
    if lifecycle != "ACTIVE":
        raise BootstrapError(
            f"joined cluster {cluster_id} is not ACTIVE in the runtime registry"
        )
    return {
        "candidate_site_sha256": candidate_site_sha256,
        "live_release_state_sha256": snapshot["live_release_state_sha256"],
        "live_release_identity_sha256": snapshot["live_release_identity_sha256"],
        "registry_generation": snapshot["registry_generation"],
        "registry_content_sha256": snapshot["registry_content_sha256"],
        "cluster_id": cluster_id,
        "registry_lifecycle": lifecycle,
        "verified_at": verified_at,
        "committed_at": datetime.now(timezone.utc).isoformat(),
    }
