from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]
from pydantic import ValidationError

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import run_command
from gpu_fault.admin.site import (
    IDENTIFIER_PATTERN,
    RenderedSite,
    effective_environment,
)
from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    hydrate_previous_snapshot,
)
from gpu_fault.regional import RegionalRegistryStatus
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT
from gpu_fault_release.regional_release_state import _live_truth_state

REGIONAL_RELEASE_STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
CPU_INGRESS_APP = CPU_INGRESS_DEPLOYMENT
VERIFICATION_MAX_AGE_SECONDS = 900
RUNTIME_MEMBERSHIP_FIELDS = (
    "live_release_identity_sha256",
    "registry_generation",
    "registry_content_sha256",
    "registry_cluster_states",
)


class JoinVerificationExpired(BootstrapError):
    """The ``VERIFIED`` evidence is older than the window.

    The data plane the join rolled out is still healthy; the right answer is to
    verify it again, not to roll it back. Callers clear the step and re-verify.
    """


def verification_is_stale(
    evidence: dict[str, Any],
    now: datetime | None = None,
) -> bool:
    try:
        verified_at = datetime.fromisoformat(str(evidence["verified_at"]))
    except (KeyError, ValueError):
        return True
    if verified_at.tzinfo is None:
        return True
    age = ((now or datetime.now(timezone.utc)) - verified_at).total_seconds()
    return age < -60 or age > VERIFICATION_MAX_AGE_SECONDS


def clear_verified_step(state_path: Path, state: dict[str, Any]) -> None:
    """Forget a stale ``VERIFIED`` step so the next verify records fresh evidence."""

    state["completed_steps"] = [
        step for step in state.get("completed_steps") or [] if step != "VERIFIED"
    ]
    evidence = state.get("evidence")
    if isinstance(evidence, dict):
        evidence.pop("VERIFIED", None)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)


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
    "executor_image",
    "node_installer_image",
    "node_dependencies",
    "release_manifest_schema_version",
    "node_template_sha256",
    "dcgm_image",
    "adot_image",
)
# The rendered-manifest digest also covers membership, so sync-state legitimately
# changes it. Its complete state bytes are still checked across candidate verify.


def _require_cluster_id(value: str) -> str:
    """Bound a cluster id to the site schema's shape before it selects a target.

    The joined cluster id is compared against registry state read from the live
    control plane and decides which cluster crosses the PENDING/ACTIVE boundary.
    It should always arrive already validated from the site document, so an id
    that does not match the anchored identifier shape means the caller is
    trusting an out-of-band value -- reject it rather than key the registry
    comparison on it.
    """
    if not IDENTIFIER_PATTERN.fullmatch(str(value)):
        raise BootstrapError(f"join cluster identity is malformed: {value!r}")
    return str(value)


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
    state_result = run_command(
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
        environment=effective_environment(site),
        timeout_seconds=60,
    )
    if state_result.returncode:
        raise BootstrapError(
            "cannot read live regional release state: "
            + diagnostic_text(state_result.stderr.strip())
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
        result = run_command(
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
            environment=effective_environment(site),
            timeout_seconds=60,
        )
        if result.returncode:
            raise BootstrapError(
                f"cannot read live release previous snapshot {name}: "
                + diagnostic_text(result.stderr.strip())
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

    pod_result = run_command(
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
        environment=effective_environment(site),
        timeout_seconds=30,
    )
    pod = (pod_result.stdout or "").strip()
    if pod_result.returncode or not pod:
        raise BootstrapError("no Running CPU ingress Pod for membership evidence")
    registry_result = run_command(
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
        environment=effective_environment(site),
        timeout_seconds=60,
    )
    if registry_result.returncode:
        raise BootstrapError(
            "cannot read regional registry generation: "
            + diagnostic_text(registry_result.stderr.strip())
        )
    try:
        registry = RegionalRegistryStatus.model_validate_json(
            registry_result.stdout or "{}"
        )
    except ValidationError:
        raise BootstrapError("regional registry status is malformed") from None
    if not registry.converged or registry.missing_member_ids:
        raise BootstrapError("regional registry has not converged")
    cluster_states = {
        cluster_id: lifecycle.value
        for cluster_id, lifecycle in registry.cluster_states.items()
    }
    live_identity = _live_truth_state(live_state)
    release_identity = {
        field: live_identity.get(field) for field in RELEASE_IDENTITY_FIELDS
    }
    return {
        "live_release_state_sha256": hashlib.sha256(raw_state.encode()).hexdigest(),
        "live_release_identity_sha256": _canonical_sha256(release_identity),
        "registry_generation": registry.generation,
        "registry_content_sha256": registry.content_sha256,
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
    cluster_id = _require_cluster_id(cluster_id)
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
    check_runtime: bool = True,
    allow_expired: bool = False,
) -> None:
    """Bind verification to its files and the live registry before activation."""

    cluster_id = _require_cluster_id(cluster_id)
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
    candidate_ids = {
        str(value) for value in evidence.get("candidate_cluster_ids") or []
    }
    actual_candidate_ids = {
        str(item["cluster_id"]) for item in candidate_site.release_config["clusters"]
    }
    if candidate_ids != actual_candidate_ids or cluster_id not in candidate_ids:
        raise BootstrapError("join candidate cluster set drifted after verification")
    if not allow_expired and verification_is_stale(evidence, now):
        raise JoinVerificationExpired("join verification evidence expired")
    if check_runtime:
        observed = membership_runtime_snapshot(current_site)
        expected = evidence.get("post_verification_runtime") or evidence
        drifted = [
            field
            for field in RUNTIME_MEMBERSHIP_FIELDS
            if observed.get(field) != expected.get(field)
        ]
        if drifted:
            raise BootstrapError(
                "membership identity drifted before activation: " + ", ".join(drifted)
            )
        if not allow_expired and verification_is_stale(evidence, now):
            raise JoinVerificationExpired("join verification evidence expired")


def advance_batch_verification(
    evidence: dict[str, Any],
    *,
    committed_evidence: dict[str, Any],
    final_identity: dict[str, Any],
    cluster_id: str,
) -> dict[str, Any]:
    """Account only for a completed sibling's activation, without renewing age."""
    expected = evidence.get("post_verification_runtime") or evidence
    committed = (
        committed_evidence.get("post_verification_runtime") or committed_evidence
    )
    if any(
        expected.get(field) != committed.get(field)
        for field in RUNTIME_MEMBERSHIP_FIELDS
    ):
        raise BootstrapError("batch join verification baselines diverged")
    expected_states = _mapping(
        expected.get("registry_cluster_states"), "verified batch cluster states"
    )
    if expected_states.get(cluster_id) != "PENDING":
        raise BootstrapError("batch activation did not start from PENDING")
    expected_states[cluster_id] = "ACTIVE"
    if (
        final_identity.get("registry_cluster_states") != expected_states
        or final_identity.get("live_release_identity_sha256")
        != expected.get("live_release_identity_sha256")
        or int(final_identity.get("registry_generation") or 0)
        <= int(expected.get("registry_generation") or 0)
        or not final_identity.get("registry_content_sha256")
    ):
        raise BootstrapError("batch membership drifted during sibling activation")
    return {
        **evidence,
        "post_verification_runtime": {
            field: final_identity[field] for field in RUNTIME_MEMBERSHIP_FIELDS
        },
        "verified_batch_activations": [
            *(evidence.get("verified_batch_activations") or []),
            cluster_id,
        ],
    }


def final_membership_identity(
    site: RenderedSite,
    *,
    candidate_site_sha256: str,
    cluster_id: str,
    verified_at: str,
    verification_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cluster_id = _require_cluster_id(cluster_id)
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
    if verification_evidence is not None and snapshot[
        "live_release_identity_sha256"
    ] != verification_evidence.get("live_release_identity_sha256"):
        raise BootstrapError("live release identity drifted after join verification")
    return {
        **snapshot,
        "candidate_site_sha256": candidate_site_sha256,
        "live_release_state_sha256": snapshot["live_release_state_sha256"],
        "live_release_identity_sha256": snapshot["live_release_identity_sha256"],
        "registry_generation": snapshot["registry_generation"],
        "cluster_id": cluster_id,
        "registry_lifecycle": lifecycle,
        "verified_at": verified_at,
        "committed_at": datetime.now(timezone.utc).isoformat(),
    }
