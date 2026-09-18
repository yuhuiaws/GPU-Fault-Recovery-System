"""Evidence required before compensating a failed administrator config release."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigApply,
    AdminConfigError,
    canonical_sha256,
    complete_admin_config_apply,
    config_command,
)
from gpu_fault_release.regional_release_diff import (
    ADMIN_CONFIG_CHANGE_FIELDS,
    ReleaseChangeKind,
    ReleaseDiff,
    build_execution_plan,
)
from gpu_fault_release.regional_release_progress import PROGRESS_SCHEMA_VERSION


@dataclass(frozen=True)
class ConfigRecovery:
    restore_before: bool
    reason: str
    live_state_sha256: str | None = None


def _config_matches(state: Mapping[str, Any], config: AdminConfig) -> bool:
    return (
        isinstance(state.get("admin_config"), dict)
        and canonical_sha256(state["admin_config"]) == config.sha256()
        and state.get("admin_config_sha256") == config.sha256()
        and state.get("admin_config_role_sha256") == config.role_sha256()
    )


def _committed(state: Mapping[str, Any]) -> bool:
    return (
        state.get("phase") == "complete"
        and state.get("transaction_committed") is True
        and state.get("release_lifecycle") == "COMMITTED"
        and "rollback_result" not in state
        and "rollback_failure" not in state
    )


def _previous_matches(state: Mapping[str, Any], apply: AdminConfigApply) -> bool:
    previous = state.get("previous")
    return (
        isinstance(previous, dict)
        and previous.get("release_id") == apply.record["release_identity"]["release_id"]
        and canonical_sha256(previous.get("admin_config")) == apply.before.sha256()
        and state.get("previous_snapshot_sha256") == canonical_sha256(previous)
    )


def _unstarted_config_release(
    state: Mapping[str, Any], apply: AdminConfigApply
) -> bool:
    early = {"preflight", "uploaded", "candidate-preflight-ready"}
    phases = state.get("completed_phases")
    progress = state.get("component_progress")
    diff = state.get("release_diff")
    if (
        not isinstance(state.get("phase"), str)
        or state["phase"] not in {*early, "failed"}
        or not isinstance(phases, list)
        or any(not isinstance(phase, str) or phase not in early for phase in phases)
        or state.get("completed_cluster_ids") != []
        or not isinstance(progress, dict)
        or type(progress.get("schema_version")) is not int
        or progress
        != {"schema_version": PROGRESS_SCHEMA_VERSION, "global": {}, "clusters": {}}
        or not isinstance(diff, dict)
        or diff.get("kind") != ReleaseChangeKind.CONTROL_PLANE_ONLY.value
    ):
        return False
    changed = diff.get("changed")
    if (
        not isinstance(changed, list)
        or not changed
        or any(
            not isinstance(field, str)
            or field
            not in ADMIN_CONFIG_CHANGE_FIELDS
            | {"release_delivery", "rendered_manifests"}
            for field in changed
        )
        or set(changed) & ADMIN_CONFIG_CHANGE_FIELDS
        != {"admin_config_" + role for role in apply.affected_roles}
    ):
        return False
    expected = build_execution_plan(
        ReleaseDiff(ReleaseChangeKind.CONTROL_PLANE_ONLY, frozenset(changed))
    )
    return state.get("execution_plan") == expected.as_dict()


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("release recovery timestamp is missing")
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("release recovery timestamp has no timezone")
    return timestamp


def _synced_rollback(
    site_file: Path,
    apply: AdminConfigApply,
    started_at: datetime,
    state: Mapping[str, Any],
) -> bool:
    # Management sync clears the live rollback journal. Only this invocation's
    # completed driver receipt can explain the resulting committed old roles.
    expected = replace(apply.before, aurora=apply.desired.aurora)
    if (
        not _committed(state)
        or not _config_matches(state, expected)
        or state.get("release_diff") != {"kind": "NOOP", "changed": []}
        or state.get("previous") is not None
    ):
        return False
    release_id = str(apply.record["release_identity"]["release_id"])
    path = site_file.parent / "release-deploy" / release_id / "state.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
            or record.get("release_id") != release_id
            or record.get("site_file") != str(site_file)
            or record.get("phase") != "FAILED"
        ):
            return False
        rollback = record.get("rollback")
        return (
            isinstance(rollback, dict)
            and rollback.get("status") == "PASSED"
            and rollback.get("management_state_synced") is True
            and started_at
            <= _timestamp(record.get("failed_at"))
            <= _timestamp(rollback.get("started_at"))
            <= _timestamp(rollback.get("completed_at"))
            <= datetime.now(UTC)
        )
    except (OSError, ValueError, TypeError):
        return False


def config_failure_recovery(
    apply: AdminConfigApply,
    *,
    baseline: dict[str, Any],
    read_state: Callable[[], dict[str, Any]],
    site_file: Path,
    started_at: datetime,
) -> ConfigRecovery:
    try:
        state = read_state()
        digest = canonical_sha256(state)
    except Exception:
        return ConfigRecovery(False, "LIVE_STATE_UNREADABLE")
    if state.get("release_id") != apply.record["release_identity"]["release_id"]:
        return ConfigRecovery(False, "RELEASE_IDENTITY_CHANGED", digest)
    if _committed(state) and _config_matches(state, apply.desired):
        return ConfigRecovery(False, "TARGET_COMMITTED", digest)
    if state == baseline and _committed(state) and _config_matches(state, apply.before):
        return ConfigRecovery(True, "UNCHANGED_COMMITTED_BASELINE", digest)
    if (
        state.get("transaction_committed") is False
        and _config_matches(state, apply.desired)
        and _previous_matches(state, apply)
    ):
        result = state.get("rollback_result")
        if (
            state.get("phase") == "rolled-back"
            and state.get("release_lifecycle") == "ROLLED_BACK"
            and state.get("rollback_cleanup_completed") is True
            and isinstance(result, dict)
            and result.get("status") == "PASSED"
        ):
            return ConfigRecovery(True, "VERIFIED_ROLLBACK", digest)
        if (
            "rollback_result" not in state
            and "rollback_failure" not in state
            and _unstarted_config_release(state, apply)
        ):
            return ConfigRecovery(True, "CONFIG_ROLLOUT_NOT_STARTED", digest)
    if _synced_rollback(site_file, apply, started_at, state):
        return ConfigRecovery(True, "VERIFIED_SYNCED_ROLLBACK", digest)
    return ConfigRecovery(False, "RELEASE_OUTCOME_UNPROVEN", digest)


def record_config_failure(
    state_dir: Path,
    apply: AdminConfigApply,
    *,
    release_id: str,
    error: str,
    recovery: ConfigRecovery,
    restore_aurora: Callable[[], dict[str, Any] | None],
) -> AdminConfigError | None:
    details: dict[str, Any] = {"release_recovery": asdict(recovery)}
    rollback_error: Exception | None = None
    if recovery.restore_before:
        try:
            rollback = restore_aurora()
            if rollback is not None:
                details["aurora_rollback"] = rollback
        except Exception as exc:
            rollback_error = exc
            error += f"; Aurora rollback failed: {type(exc).__name__}: {exc}"
    restore_before = recovery.restore_before and rollback_error is None
    if not restore_before:
        error += f"; pending configuration target retained ({recovery.reason})"
        if recovery.reason != "RELEASE_SUCCEEDED":
            error += (
                "; no unproven compensation is permitted; reconcile the release with "
                f"gpu-fault-admin deploy --state-dir {state_dir}, then rerun "
                f"{config_command(state_dir)}"
            )
    complete_admin_config_apply(
        state_dir,
        config_sha256=apply.config_sha256,
        release_id=release_id,
        success=False,
        error=error,
        details=details,
        restore_before=restore_before,
    )
    return AdminConfigError(error) if not restore_before else None
