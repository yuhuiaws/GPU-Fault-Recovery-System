"""Identity and evidence contracts for isolated deployed busy-worker takeover."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

CASE_ID = "GF-REGIONAL-HA-011"
CONFIRMATION = "HA011_ISOLATED_BUSY_CPU_TAKEOVER"
BOUNDARY = "isolated-deployed-production-image"
POD_NAME = "busy-cpu-proof"
PYTHON = "/opt/gpu-fault/control-plane/bin/python"
LABEL = "gpu-fault.nvidia.com/ha011-run"
LEASE_SECONDS = 8
BACKLOG = 3
MIN_CPU_SECONDS = 0.05
ROLES = ("processor", "spool")
SPOOL_PATH = "/v1/collector-events/host-telemetry"
QUEUE_PATH = "/v1/collector-events/nvidia-kernel"
DEPLOYMENT = "gpu-fault-control-worker"
TOKEN = re.compile(r"[a-f0-9]{32}")
IMAGE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9./_:-]*@sha256:[a-f0-9]{64}")
IMAGE_ID = re.compile(
    r"(?:(?:docker-pullable|docker|containerd)://)?"
    r"(?:[A-Za-z0-9._:/-]+@)?(?P<digest>sha256:[a-f0-9]{64})"
)
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
FAILURE_STAGES = frozenset(
    {"identity", "arm", "database", "processor", "spool", "evidence"}
)
FAILURE_TYPES = frozenset(
    {
        "ProofError",
        "OperationalError",
        "InterfaceError",
        "RuntimeError",
        "OSError",
        "PermissionError",
        "FileNotFoundError",
        "ValueError",
        "KeyError",
        "TypeError",
        "ImportError",
        "ModuleNotFoundError",
        "TimeoutError",
        "StaleFencingTokenError",
    }
)


class ProofError(RuntimeError):
    pass


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def image_digest(value: object) -> str:
    match = IMAGE_ID.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ProofError("running container image digest is missing or malformed")
    return match.group("digest")


@dataclass(frozen=True)
class Settings:
    cpu_kubeconfig: Path
    cpu_context: str
    namespace: str
    cluster_id: str
    isolation_id: str
    postgres_image: str
    predecessor_case: str
    predecessor_path: Path
    region: str

    def __post_init__(self) -> None:
        if not self.cpu_kubeconfig.is_file():
            raise ProofError("an explicit CPU kubeconfig file is required")
        if (
            not self.cpu_context.strip()
            or not self.cluster_id.strip()
            or not self.region.strip()
        ):
            raise ProofError(
                "explicit Region, CPU context and cluster identity are required"
            )
        if not DNS_LABEL.fullmatch(self.namespace):
            raise ProofError("control-plane namespace is malformed")
        if not TOKEN.fullmatch(self.isolation_id):
            raise ProofError(
                "isolation ID must be a fresh 32-character lowercase hex ID"
            )
        if not IMAGE.fullmatch(self.postgres_image) or not re.fullmatch(
            r"(?:docker.io/library/)?postgres:16(?:[.][0-9]+)?-bookworm@sha256:[a-f0-9]{64}",
            self.postgres_image,
        ):
            raise ProofError(
                "an explicitly digest-pinned PostgreSQL 16 bookworm image is required"
            )
        if self.namespace == self.isolated_namespace:
            raise ProofError(
                "the isolated namespace must not be the business namespace"
            )
        if not self.predecessor_case:
            raise ProofError("a registered formal predecessor is required")

    @property
    def isolated_namespace(self) -> str:
        return f"gf-regional-ha011-{self.isolation_id}"

    @property
    def priority_class_name(self) -> str:
        return self.isolated_namespace

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "CPU_EKS_CONTEXT": self.cpu_context,
            "AWS_REGION": self.region,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "GPU_FAULT_CLUSTER_ID": self.cluster_id,
            "GPU_FAULT_HA011_ISOLATION_ID": self.isolation_id,
            "GPU_FAULT_HA011_POSTGRES_IMAGE": self.postgres_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def require_owned(
    value: dict[str, Any], isolation_id: str, uid: str | None = None
) -> str:
    metadata = value.get("metadata", {})
    actual_uid = metadata.get("uid")
    if (
        metadata.get("labels", {}).get(LABEL) != isolation_id
        or not isinstance(actual_uid, str)
        or not actual_uid
        or (uid is not None and actual_uid != uid)
        or metadata.get("deletionTimestamp")
    ):
        raise ProofError("isolated resource ownership, UID, or lifecycle changed")
    return actual_uid


def require_subset(actual: object, expected: object) -> None:
    """Admission defaults may add mapping keys, but never containers or mounts."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ProofError("isolated resource shape changed")
        for key, value in expected.items():
            if key not in actual:
                raise ProofError("isolated resource field is missing")
            require_subset(actual[key], value)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ProofError("isolated resource list changed")
        for observed, planned in zip(actual, expected, strict=True):
            require_subset(observed, planned)
    elif actual != expected:
        raise ProofError("isolated resource value changed")


def evidence_errors(
    result: dict[str, Any], *, isolation_id: str, pod_uid: str, intent_sha256: str
) -> list[str]:
    errors = []
    expected = {
        "case_id": CASE_ID,
        "validation_scope": BOUNDARY,
        "isolation_id": isolation_id,
        "pod_uid": pod_uid,
        "arm_intent_sha256": intent_sha256,
        "postgres_major": 16,
        "business_worker_targeted": False,
        "cpu_saturation_tested": False,
    }
    if set(result) != {*expected, "roles"}:
        errors.append("probe contains missing or unrecognized evidence fields")
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result.get(key) != value:
            errors.append(f"probe identity or boundary mismatch: {key}")
    roles = result.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        return [*errors, "processor and spool evidence are both required"]
    for role in ROLES:
        proof = roles[role]
        if not isinstance(proof, dict):
            errors.append(f"{role}: malformed evidence")
            continue
        checks = {
            "same_durable_work": True,
            "fence_changed": True,
            "early_claim_count": 0,
            "old_exitcode": -9,
            "late_completion_refused": True,
            "replacement_live_before_late": True,
            "replacement_live_after_late": True,
            "backlog_at_crash": BACKLOG + 1,
            "completed_count": BACKLOG + 1,
            "final_depth": 0,
            "owned_processes_stopped": True,
        }
        if set(proof) != {
            *checks,
            "work_sha256",
            "owners",
            "old_cpu_seconds",
            "replacement_cpu_seconds",
        }:
            errors.append(f"{role}: unrecognized evidence fields")
        if not isinstance(proof.get("work_sha256"), str) or not re.fullmatch(
            r"[a-f0-9]{64}", proof["work_sha256"]
        ):
            errors.append(f"{role}: durable work digest is missing")
        for key, value in checks.items():
            if type(proof.get(key)) is not type(value) or proof.get(key) != value:
                errors.append(f"{role}: {key} was not proven")
        for key in ("old_cpu_seconds", "replacement_cpu_seconds"):
            value = proof.get(key)
            if type(value) not in (int, float) or not (
                MIN_CPU_SECONDS <= cast(float, value) <= 60
            ):
                errors.append(f"{role}: bounded busy CPU execution was not observed")
        owners = proof.get("owners")
        if (
            not isinstance(owners, list)
            or len(owners) != 2
            or any(
                not isinstance(owner, str)
                or not re.fullmatch(re.escape(pod_uid) + r":[1-9][0-9]*", owner)
                for owner in owners
            )
            or owners[0] == owners[1]
        ):
            errors.append(f"{role}: distinct owned process identities are missing")
    return errors
