"""An unadopted Aurora repair may be retired by the candidate that follows it.

Two shapes exist. A cleaned bootstrap hands its repair to another candidate
(``validate_bootstrap_repair_handoff``). An upgrade candidate that died after
its repair reached READY but before its application transaction began leaves
an orphan over a committed release; that candidate never returns, because a
release id is a digest of the tree, so the next candidate retires the orphan
(``validate_upgrade_repair_handoff``). Both restore the original refresher
from the journal's snapshot before anything of the new candidate is applied.

A bootstrap that adopted its repair leaves no journal to hand off, only the
state of its failed, interrupted or cleaned attempt. The next candidate takes
that state over when it records its own journal (``foreign_bootstrap_baseline``):
the journal hashes its baseline and every later bootstrap checkpoint stamps
the new candidate's identity, so a baseline still naming the old candidate
would drift before adoption, and the old candidate's approved digest is never
the new candidate's plan (M-23).
"""

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
    _valid_bootstrap_progress,
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_state import _write_state

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

HANDOFF_AUDIT_KEY = "bootstrap_prerequisite_handoff"
PREDECESSOR_AUDIT_KEY = "bootstrap_prerequisite_predecessor"
# Phases whose cleanup the bootstrap entry still owes (a cleanup in progress,
# or a failure that auto-rollback cleans): a takeover keeps them and their
# steps so that cleanup still runs. Every other phase is projected to the
# ``bootstrap-started`` checkpoint the new candidate would write first.
_OWED_CLEANUP_PHASES = frozenset(
    {
        "bootstrap-failed",
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-failed",
        "bootstrap-cleaned",
    }
)
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


def _project_bootstrap_state(
    release: RegionalRelease, state: dict[str, Any], phase: str, **progress: Any
) -> dict[str, Any]:
    """Project the existing state writer without a live checkpoint or narration."""
    if "previous" not in state or state["previous"] is not None:
        raise ReleaseError(
            "bootstrap prerequisite predecessor has a previous application"
        )
    projected = copy.copy(release)
    projected.runner = copy.copy(release.runner)
    projected.runner.dry_run = True
    projected.state = copy.deepcopy(state)
    projected.state.pop(AURORA_PREREQUISITE_REPAIR_KEY, None)
    # Reuse the actual identity serializer, not a second list of release fields.
    # previous=None keeps this dry-run projection independent of snapshot I/O.
    _write_state(
        projected,
        phase,
        previous=None,
        transaction_committed=False,
        completed_cluster_ids=[],
        resume_phase="bootstrap-started",
        **progress,
    )
    return projected.state


def candidate_bootstrap_baseline(
    release: RegionalRelease, state: dict[str, Any]
) -> dict[str, Any]:
    """The handoff successor's baseline: a fresh start under this candidate."""
    return _project_bootstrap_state(
        release,
        state,
        "bootstrap-started",
        bootstrap_cleanup_completed_steps=[],
        bootstrap_cleanup_failure=None,
    )


def foreign_bootstrap_baseline(
    release: RegionalRelease, state: dict[str, Any]
) -> dict[str, Any]:
    """Take over another candidate's journal-less bootstrap state.

    The result is what this candidate's own first checkpoint would write: its
    identity and approved digest, no completed clusters and a fresh resume
    phase. Cleanup the old attempt still owes keeps its phase and steps, and
    the database origin and safety proof stay, so the empty-database proof the
    first bootstrap recorded still authorizes this one.
    """
    owed = state.get("phase") in _OWED_CLEANUP_PHASES
    projected = _project_bootstrap_state(
        release,
        state,
        str(state["phase"]) if owed else "bootstrap-started",
        bootstrap_cleanup_completed_steps=(
            copy.deepcopy(state.get("bootstrap_cleanup_completed_steps", []))
            if owed
            else []
        ),
        bootstrap_cleanup_failure=(
            state.get("bootstrap_cleanup_failure") if owed else None
        ),
    )
    if not _valid_bootstrap_progress(projected):
        raise ReleaseError("foreign bootstrap predecessor progress is invalid")
    return projected


def predecessor_audit(state: dict[str, Any]) -> dict[str, Any]:
    """What a takeover replaced; the journal and then the adopted state keep it."""
    return {
        "release_id": state.get("release_id"),
        "phase": state.get("phase"),
        "resume_phase": state.get("resume_phase"),
        "approved_manifest_sha256": state.get("approved_manifest_sha256"),
        "rendered_manifest_sha256": state.get("rendered_manifest_sha256"),
        "state_sha256": canonical_sha256(state),
        "taken_over_at": datetime.now(UTC).isoformat(),
    }


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


def validate_upgrade_repair_handoff(
    release: RegionalRelease,
    state: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate an upgrade repair whose candidate died before its application.

    The journal is trusted only against its own fully hashed baseline: the live
    state minus the journal must still be the committed, quiescent release the
    repair was prepared over (``complete``, committed, cleanup done, and not the
    abandoned candidate), so nothing that candidate did outlived it except the
    refresher objects the journal's snapshot restores. Bootstrap, commit-live
    and in-flight application transactions are refused here; they keep their
    own paths.
    """
    record = state.get(AURORA_PREREQUISITE_REPAIR_KEY)
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int:
        raise ReleaseError("upgrade prerequisite handoff record is invalid")
    binding = record.get("binding")
    if (
        record["schema_version"] != 1
        or not isinstance(binding, dict)
        or set(binding) not in (_BINDING_FIELDS, _BINDING_FIELDS | {"bootstrap"})
        or binding.get("bootstrap", False) is not False
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
        raise ReleaseError("upgrade prerequisite handoff binding is invalid")
    baseline = validated_prerequisite_baseline(state, record)
    live_release_id = state.get("release_id")
    for value in (state, baseline):
        if (
            value.get("phase") != "complete"
            or value.get("transaction_committed") is not True
            or value.get("release_lifecycle") != "COMMITTED"
            or value.get("commit_cleanup_completed") is not True
            or not isinstance(live_release_id, str)
            or not live_release_id
            or value.get("release_id") != live_release_id
            or live_release_id == binding["release_id"]
        ):
            raise ReleaseError(
                "upgrade prerequisite handoff requires a committed release that "
                "never adopted the abandoned candidate"
            )
    if (
        binding.get("cpu_eks_arn") != release.config.cpu_eks_arn
        or binding.get("namespace") != release.config.namespace
        or binding.get("database") != identity
    ):
        raise ReleaseError("upgrade prerequisite handoff site or database drifted")
    if (
        re.fullmatch(r"[a-f0-9]{64}", str(binding["manifest_sha256"])) is None
        or not isinstance(binding["wheel_config_map"], str)
        or not binding["wheel_config_map"]
        or canonical_sha256(record.get("previous_refresher"))
        != record.get("snapshot_sha256")
    ):
        raise ReleaseError("upgrade prerequisite predecessor snapshot differs")
    require_digest_pinned_image("previous upgrade refresher", binding["runtime_image"])
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
        raise ReleaseError("upgrade prerequisite handoff Job journal is invalid")
    return record


def upgrade_handoff_audit(
    record: dict[str, Any], *, state_sha256: str
) -> dict[str, Any]:
    """The predecessor audit plus the state digest the deploy diff was pinned to.

    The deploy driver pins the release-state digest it planned against and the
    apply step compares it again; a repair's own transition is bridged by the
    record's ``baseline_sha256``. After a handoff the state the driver saw held
    the predecessor journal, so the successor carries that digest as well.
    """
    return {**bootstrap_handoff_audit(record), "state_sha256": state_sha256}
