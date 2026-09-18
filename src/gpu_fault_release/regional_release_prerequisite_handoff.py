"""A cleaned, unadopted bootstrap may retire its original Aurora repair."""

from __future__ import annotations

import copy
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpu_fault_release.regional_release_aurora_refresh import (
    validate_aurora_refresh_snapshot,
)
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    AURORA_PREREQUISITE_REPAIR_KEY,
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_images import require_digest_pinned_image
from gpu_fault_release.regional_release_prerequisite_baseline import (
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_state import _write_state

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

HANDOFF_AUDIT_KEY = "bootstrap_prerequisite_handoff"
_CLEANUP_STEPS = {
    "installer-jobs-cancelled",
    "gpu-scaled-down",
    "cpu-scaled-down",
}
_ADOPTION_FIELDS = {
    "bootstrap_store_safety",
    "bootstrap_database_origin",
    "retained_database_origin",
    "component_progress",
    "schema_change_acceptance",
}
_BINDING_FIELDS = {
    "release_id",
    "manifest_sha256",
    "cpu_eks_arn",
    "namespace",
    "runtime_image",
    "wheel_config_map",
    "commit_live",
    "database",
    "execution_plan",
}


def validate_bootstrap_repair_handoff(
    release: RegionalRelease,
    state: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate the old candidate against its original, fully hashed baseline."""
    record = state.get(AURORA_PREREQUISITE_REPAIR_KEY)
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int:
        raise ReleaseError("bootstrap prerequisite handoff record is invalid")
    binding = record.get("binding")
    if (
        record["schema_version"] != 1
        or not isinstance(binding, dict)
        or set(binding) not in (_BINDING_FIELDS, _BINDING_FIELDS | {"bootstrap"})
        or binding.get("bootstrap", True) is not True
        or binding.get("commit_live") is not False
        or canonical_sha256(binding) != record.get("binding_sha256")
        or not isinstance(binding.get("release_id"), str)
        or not binding["release_id"]
        or binding["release_id"] == release.release_id
        or re.fullmatch(r"[a-f0-9]{32}", str(record.get("attempt_id") or "")) is None
        or not isinstance(record.get("status"), str)
        or record.get("status") not in {"PREPARED", "APPLYING", "READY", "FAILED"}
        or binding.get("execution_plan")
        != ReleaseExecutionPlan(
            (ReleaseComponent.AURORA_REFRESH, ReleaseComponent.VERIFY)
        ).as_dict()
    ):
        raise ReleaseError("bootstrap prerequisite handoff binding is invalid")
    baseline = validated_prerequisite_baseline(state, record)
    if (
        state.get("phase") != "bootstrap-cleaned"
        or baseline.get("phase") not in {"bootstrap-started", "bootstrap-cleaned"}
        or state.get("bootstrap_cleanup_completed_steps") != sorted(_CLEANUP_STEPS)
        or state.get("bootstrap_cleanup_failure") is not None
        or state.get("resume_phase") != "bootstrap-started"
        or release.config.retained_database_handoff is not None
    ):
        raise ReleaseError("bootstrap prerequisite handoff requires completed cleanup")
    for value in (state, baseline):
        if (
            "previous" not in value
            or value["previous"] is not None
            or value.get("transaction_committed") is not False
            or value.get("completed_cluster_ids") not in (None, [])
            or value.get("completed_phases") not in (None, [])
            or (
                value.get("registry_staged") is not None
                and value["registry_staged"] is not False
            )
            or value.get("release_lifecycle") is not None
            or _ADOPTION_FIELDS.intersection(value)
        ):
            raise ReleaseError(
                "bootstrap prerequisite repair was adopted or progressed"
            )
    expected_members = sorted(target.cluster_id for target in release.config.clusters)
    if (
        not expected_members
        or len(set(expected_members)) != len(expected_members)
        or baseline.get("cluster_ids") != expected_members
        or state.get("cluster_ids") != expected_members
        or binding.get("cpu_eks_arn") != release.config.cpu_eks_arn
        or binding.get("namespace") != release.config.namespace
        or binding.get("database") != identity
    ):
        raise ReleaseError("bootstrap prerequisite handoff site or database drifted")
    for bound, recorded in (
        ("release_id", "release_id"),
        ("manifest_sha256", "rendered_manifest_sha256"),
        ("manifest_sha256", "approved_manifest_sha256"),
        ("runtime_image", "runtime_image"),
        ("wheel_config_map", "wheel_config_map"),
    ):
        if binding.get(bound) != baseline.get(recorded):
            raise ReleaseError("bootstrap prerequisite predecessor identity differs")
    wheel_sha = str(baseline.get("wheel_sha256") or "")
    wheel_name = binding["wheel_config_map"]
    if (
        re.fullmatch(r"[a-f0-9]{64}", wheel_sha) is None
        or not isinstance(wheel_name, str)
        or len(wheel_name) > 253
        or re.fullmatch(
            r"gpu-fault-control-plane-wheel-[a-z0-9]+(?:[.-][a-z0-9]+)*-"
            + re.escape(wheel_sha[:12]),
            wheel_name,
        )
        is None
        or re.fullmatch(r"[a-f0-9]{64}", str(binding["manifest_sha256"])) is None
        or canonical_sha256(record.get("previous_refresher"))
        != record.get("snapshot_sha256")
    ):
        raise ReleaseError("bootstrap prerequisite predecessor snapshot differs")
    require_digest_pinned_image(
        "previous bootstrap refresher", binding["runtime_image"]
    )
    validate_aurora_refresh_snapshot(release, record.get("previous_refresher"))
    jobs = record.get("jobs")
    if (
        not isinstance(jobs, dict)
        or set(jobs) - {"credential", "store"}
        or any(
            not isinstance(job, dict)
            or not isinstance(job.get("status"), str)
            or job.get("status") not in {"PLANNED", "RUNNING", "REMOVED"}
            or not isinstance(job.get("owner_uid"), str)
            or not job["owner_uid"]
            for job in jobs.values()
        )
    ):
        raise ReleaseError("bootstrap prerequisite handoff Job journal is invalid")
    return record


def candidate_bootstrap_baseline(
    release: RegionalRelease, state: dict[str, Any]
) -> dict[str, Any]:
    """Project the existing state writer without a live checkpoint or narration."""
    if "previous" not in state or state["previous"] is not None:
        raise ReleaseError("bootstrap prerequisite handoff has a previous application")
    projected = copy.copy(release)
    projected.runner = copy.copy(release.runner)
    projected.runner.dry_run = True
    projected.state = copy.deepcopy(state)
    projected.state.pop(AURORA_PREREQUISITE_REPAIR_KEY, None)
    # Reuse the actual identity serializer, not a second list of release fields.
    # previous=None keeps this dry-run projection independent of snapshot I/O.
    _write_state(
        projected,
        "bootstrap-started",
        previous=None,
        transaction_committed=False,
        completed_cluster_ids=[],
        bootstrap_cleanup_completed_steps=[],
        bootstrap_cleanup_failure=None,
        resume_phase="bootstrap-started",
    )
    return projected.state


def bootstrap_handoff_audit(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": record["attempt_id"],
        "release_id": record["binding"]["release_id"],
        "binding_sha256": record["binding_sha256"],
        "baseline_sha256": record["baseline_sha256"],
        "snapshot_sha256": record["snapshot_sha256"],
        "record_sha256": canonical_sha256(record),
        "restored_at": datetime.now(UTC).isoformat(),
    }
