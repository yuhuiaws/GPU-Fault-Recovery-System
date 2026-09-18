from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_progress as progress
from gpu_fault_release import regional_release_timing as timing
from gpu_fault_release import regional_release_transaction as transaction
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component


def test_attempt_and_component_progress_reject_unknown_states_and_empty_batches() -> (
    None
):
    with pytest.raises(ValueError, match="unsupported cluster attempt"):
        progress.cluster_attempts_with({}, "gpu-a", "UNKNOWN")
    with pytest.raises(ValueError, match="unsupported component progress"):
        progress.update_component_progress({}, Component.EXECUTOR, "UNKNOWN")
    with pytest.raises(ValueError, match="cannot be empty"):
        progress.update_components_progress({}, [], "STARTED")


@pytest.mark.parametrize("prior", ["PAUSED", "FAILED", "ROLLED_BACK"])
def test_restarted_cluster_attempt_advances_generation_without_mutating_checkpoint(
    prior: str,
) -> None:
    state = {
        "cluster_attempts": {
            "gpu-a": {
                "state": prior,
                "attempt_generation": 3,
                "details": {"original": "kept"},
            }
        }
    }
    before = copy.deepcopy(state)
    attempts = progress.cluster_attempts_with(
        state, "gpu-a", "RUNNING", details={"current": "retry"}
    )
    assert attempts["gpu-a"]["attempt_generation"] == 4
    assert attempts["gpu-a"]["details"] == {"original": "kept", "current": "retry"}
    assert state == before


def test_compensation_ignores_unknown_progress_but_retains_known_started_component() -> (
    None
):
    state = {
        "execution_plan": {"nodes": ["executor"]},
        "component_progress": {
            "schema_version": 1,
            "global": {},
            "clusters": {
                "gpu-a": {
                    "executor": {"status": "STARTED"},
                    "future-component": {"status": "STARTED"},
                    "watcher": {"status": "UNKNOWN"},
                    "invalid": None,
                }
            },
        },
    }
    plan = progress.build_rollback_compensation_plan(state, ["gpu-a"])
    assert plan.for_cluster("gpu-a") == frozenset({Component.EXECUTOR})
    assert plan.conservative is False


@pytest.mark.parametrize("stage", ["cpu-finalize", "cpu-stage"])
def test_legacy_checkpoints_conservatively_restore_unstamped_cpu_stage(
    stage: str,
) -> None:
    completed = (
        ["uploaded", "data-converged"]
        if stage == "cpu-finalize"
        else ["uploaded", "registry-staged"]
    )
    state = {
        "execution_plan": {"nodes": [stage, "registry", "executor"]},
        "completed_phases": completed,
    }
    plan = progress.build_rollback_compensation_plan(state, ["gpu-a"])
    assert plan.global_has(Component(stage)), (
        "legacy CPU mutation must remain compensable"
    )
    assert plan.conservative is True


def test_legacy_profile_checkpoint_is_required_before_gpu_compensation() -> None:
    state = {
        "execution_plan": {"nodes": ["runtime-profile", "executor"]},
        "completed_phases": ["uploaded"],
    }
    plan = progress.build_rollback_compensation_plan(state, ["gpu-a"])
    assert plan.for_cluster("gpu-a") == frozenset()
    state["completed_phases"].append("profile-ready")
    plan = progress.build_rollback_compensation_plan(state, ["gpu-a"])
    assert plan.for_cluster("gpu-a") == frozenset({Component.EXECUTOR})


def test_rollback_wave_failure_retains_details_and_elapsed_time() -> None:
    release = SimpleNamespace()
    timing.record_rollback_wave_event(
        release,
        cluster_id="gpu-a",
        wave=("node-a",),
        event="started",
        observed_at_epoch=10,
        details={"started": "known"},
    )
    timing.record_rollback_wave_event(
        release,
        cluster_id="gpu-a",
        wave=("node-a",),
        event="failed",
        observed_at_epoch=15,
        details={"error": "fixture"},
    )
    report: dict[str, Any] = {}
    timing.merge_rollback_wave_timings(release, report)
    wave = report["clusters"]["gpu-a"]["waves"]["node-a"]
    assert wave["status"] == "FAILED"
    assert wave["duration_seconds"] == 5
    assert wave["details"] == {"started": "known", "error": "fixture"}


class Transaction:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = copy.deepcopy(state)
        self.saves: list[dict[str, Any]] = []
        self.cleanup: list[str] = []

    def _load_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def _save_state(self, phase: str, **updates: Any) -> None:
        self.state.update(phase=phase, **updates)
        self.saves.append(copy.deepcopy(self.state))

    def _cleanup_previous_snapshots(self) -> None:
        self.cleanup.append("snapshots")

    def _delete_release_secret_backups(self, _previous: Any) -> None:
        self.cleanup.append("backups")


def test_commit_refuses_incomplete_release_and_does_not_repeat_completed_cleanup() -> (
    None
):
    release = Transaction({"phase": "failed"})
    with pytest.raises(ReleaseError, match="only a complete release"):
        transaction.commit_release(release)
    assert release.saves == []
    release.state = {
        "phase": "complete",
        "transaction_committed": True,
        "commit_cleanup_completed": True,
    }
    transaction.commit_release(release)
    assert release.cleanup == []
    assert release.saves == []


def test_commit_without_backup_still_cleans_up_previous_snapshots() -> None:
    release = Transaction({"phase": "complete", "transaction_committed": True})
    transaction.commit_release(release)
    assert release.cleanup == ["snapshots"]
    assert len(release.saves) == 1
    assert release.state["commit_cleanup_completed"] is True


def test_rollback_completion_does_not_rewrite_completed_cleanup_or_timing() -> None:
    release = Transaction({"phase": "rolled-back", "rollback_cleanup_completed": True})
    measured = {"completed_at_epoch": 3, "t_safe_seconds": 1, "t_full_seconds": 2}
    transaction.complete_rollback(
        release,
        previous={},
        completed_phases={"rollback-verified"},
        completed_clusters=set(),
        rollback_plan={},
        rollback_timing=measured,
        original_failure="fixture",
    )
    assert measured == {
        "completed_at_epoch": 3,
        "t_safe_seconds": 1,
        "t_full_seconds": 2,
    }
    assert release.cleanup == []
    assert release.saves == []


def test_automatic_rollback_cleanup_error_preserves_successful_rollback_state() -> None:
    release = Transaction(
        {"phase": "rolled-back", "rollback_result": {"status": "PASSED"}}
    )
    upgrade = ReleaseError("upgrade failed")
    with pytest.raises(
        ReleaseError, match="post-rollback cleanup is pending"
    ) as caught:
        transaction.raise_automatic_rollback_failure(
            release,
            upgrade_error=upgrade,
            rollback_error=ReleaseError("cleanup failed"),
            previous={},
            release_diff={},
            execution_plan={},
        )
    assert caught.value.__cause__ is upgrade
    assert release.saves == []
