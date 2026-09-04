from __future__ import annotations

from pathlib import Path

import pytest

from tests._script_loader import load_script_module

ROOT = Path(__file__).resolve().parents[2]


class Release:
    def __init__(self) -> None:
        self.loaded = {
            "phase": "complete",
            "previous": {"secret_backups": {"cpu": {"backup": "backup"}}},
            "release_diff": {"kind": "CONTROL_PLANE_ONLY", "changed": ["cpu"]},
            "execution_plan": {"components": ["cpu"]},
            "completed_phases": ["complete"],
            "completed_cluster_ids": [],
        }
        self.events: list[tuple[str, object]] = []
        self.fail_cleanup = False

    def _load_state(self):
        return dict(self.loaded)

    def _save_state(self, phase, **updates):
        self.events.append(("save", updates.get("commit_cleanup_completed")))
        self.loaded.update({"phase": phase, **updates})

    def _delete_release_secret_backups(self, previous):
        self.events.append(("cleanup", previous))
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")


def transaction_module():
    return load_script_module(
        ROOT / "deploy/control-plane/regional/regional_release_transaction.py"
    )


def test_commit_persists_before_deleting_rollback_backups() -> None:
    release = Release()

    transaction_module().commit_release(release)

    assert [event[0] for event in release.events] == ["save", "cleanup", "save"]
    assert release.loaded["transaction_committed"] is True
    assert release.loaded["commit_cleanup_completed"] is True


def test_commit_cleanup_failure_is_retryable_without_uncommitting() -> None:
    release = Release()
    release.fail_cleanup = True

    with pytest.raises(RuntimeError, match="cleanup failed"):
        transaction_module().commit_release(release)

    assert release.loaded["transaction_committed"] is True
    assert release.loaded["commit_cleanup_completed"] is False

    release.fail_cleanup = False
    transaction_module().commit_release(release)

    assert release.loaded["commit_cleanup_completed"] is True


def test_rollback_cleanup_failure_is_retryable_after_rollback_is_persisted() -> None:
    release = Release()
    release.loaded = {"phase": "rollback-verifying"}
    release.fail_cleanup = True
    arguments = {
        "previous": {"secret_backups": {"cpu": {"backup": "backup"}}},
        "completed_phases": {"rollback-verified"},
        "completed_clusters": {"gpu-a"},
        "rollback_plan": {"components": ["cpu"]},
        "rollback_timing": {"t_full_seconds": 1.0},
        "original_failure": "upgrade failed",
        "rollback_result": {"status": "PASSED"},
    }

    with pytest.raises(RuntimeError, match="cleanup failed"):
        transaction_module().finalize_rollback(release, **arguments)

    assert release.loaded["phase"] == "rolled-back"
    assert release.loaded["rollback_cleanup_completed"] is False

    release.fail_cleanup = False
    transaction_module().finalize_rollback(release, **arguments)

    assert release.loaded["rollback_cleanup_completed"] is True
