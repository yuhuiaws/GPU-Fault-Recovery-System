from __future__ import annotations

from typing import Any

from gpu_fault_release.regional_release_config import (
    PartialClusterRolloutError,
    ReleaseError,
)
from gpu_fault_release.regional_release_timing import mark_rollback_complete


def _loaded_state(release: Any) -> dict[str, Any]:
    state = getattr(release, "state", None)
    return dict(state) if state else release._load_state()


def commit_release(release: Any) -> None:
    loaded = _loaded_state(release)
    if loaded.get("phase") != "complete":
        raise ReleaseError("only a complete release can be committed")
    previous = loaded.get("previous")
    common = {
        "previous": previous,
        "release_diff": loaded.get("release_diff"),
        "execution_plan": loaded.get("execution_plan"),
        "completed_phases": loaded.get("completed_phases", []),
        "completed_cluster_ids": loaded.get("completed_cluster_ids", []),
        "transaction_committed": True,
        "release_lifecycle": "COMMITTED",
    }
    if not loaded.get("transaction_committed"):
        release._save_state(
            "complete",
            **common,
            commit_cleanup_completed=False,
        )
    if loaded.get("commit_cleanup_completed") is True:
        return
    if isinstance(previous, dict):
        release._delete_release_secret_backups(previous)
    cleanup_snapshots = getattr(release, "_cleanup_previous_snapshots", None)
    if cleanup_snapshots is not None:
        cleanup_snapshots()
    release._save_state(
        "complete",
        **common,
        commit_cleanup_completed=True,
    )


def finalize_rollback(
    release: Any,
    *,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    rollback_plan: dict[str, Any],
    rollback_timing: dict[str, Any],
    original_failure: object,
    rollback_result: dict[str, Any],
) -> None:
    common = {
        "previous": previous,
        "rollback_completed_phases": sorted(completed_phases),
        "rollback_completed_cluster_ids": sorted(completed_clusters),
        "rollback_plan": rollback_plan,
        "rollback_timing": rollback_timing,
        "release_lifecycle": "ROLLED_BACK",
        "original_failure": original_failure,
        "rollback_result": rollback_result,
    }
    loaded = _loaded_state(release)
    if loaded.get("phase") != "rolled-back":
        release._save_state(
            "rolled-back",
            **common,
            rollback_cleanup_completed=False,
        )
    if loaded.get("rollback_cleanup_completed") is True:
        return
    release._delete_release_secret_backups(previous)
    release._save_state(
        "rolled-back",
        **common,
        rollback_cleanup_completed=True,
    )


def raise_automatic_rollback_failure(
    release: Any,
    *,
    upgrade_error: Exception,
    rollback_error: Exception,
    previous: dict[str, Any],
    release_diff: dict[str, Any],
    execution_plan: dict[str, Any],
) -> None:
    rollback_state = release._load_state()
    rollback_result = rollback_state.get("rollback_result")
    if (
        rollback_state.get("phase") == "rolled-back"
        and isinstance(rollback_result, dict)
        and rollback_result.get("status") == "PASSED"
    ):
        raise ReleaseError(
            "release upgrade failed; automatic rollback completed "
            "but post-rollback cleanup is pending: "
            f"{type(rollback_error).__name__}: {rollback_error}"
        ) from upgrade_error
    release._save_state(
        "rollback-failed",
        previous=previous,
        release_diff=release_diff,
        execution_plan=execution_plan,
        original_failure=f"{type(upgrade_error).__name__}: {upgrade_error}",
        rollback_failure=f"{type(rollback_error).__name__}: {rollback_error}",
    )
    raise ReleaseError(
        "release upgrade failed and automatic rollback also failed: "
        f"upgrade={type(upgrade_error).__name__}: {upgrade_error}; "
        f"rollback={type(rollback_error).__name__}: {rollback_error}"
    ) from upgrade_error


def record_upgrade_failure(
    release: Any,
    *,
    error: Exception,
    release_diff: dict[str, Any],
    execution_plan: dict[str, Any],
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> None:
    partial = isinstance(error, PartialClusterRolloutError)
    release._save_state(
        "partial-convergence" if partial else "failed",
        previous=previous,
        release_diff=release_diff,
        execution_plan=execution_plan,
        completed_phases=sorted(completed_phases),
        completed_cluster_ids=sorted(completed_clusters),
        registry_staged=registry_staged,
        partial_convergence=partial,
        release_lifecycle="PAUSED" if partial else "FAILED",
        original_failure=f"{type(error).__name__}: {error}",
    )


def complete_rollback(
    release: Any,
    *,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    rollback_plan: dict[str, Any],
    rollback_timing: dict[str, Any],
    original_failure: object,
) -> None:
    if "completed_at_epoch" not in rollback_timing:
        mark_rollback_complete(rollback_timing)
    finalize_rollback(
        release,
        previous=previous,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
        rollback_plan=rollback_plan,
        rollback_timing=rollback_timing,
        original_failure=original_failure,
        rollback_result={
            "status": "PASSED",
            "t_safe_seconds": rollback_timing.get("t_safe_seconds"),
            "t_full_seconds": rollback_timing.get("t_full_seconds"),
        },
    )
