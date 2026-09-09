from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_transaction
from gpu_fault_release.regional_release_config import ReleaseError

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

    def _delete_stale_release_secret_backups(self, previous):
        # Commit keeps the backups the committed release still needs for its
        # own rollback and sweeps only older ones; the double records the call
        # under the same event name so the ordering assertions read the same.
        self.events.append(("cleanup", previous))
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")


def transaction_module():
    return regional_release_transaction


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


# --- Secret backup lifecycle -------------------------------------------------
#
# A release's backup Secrets are what its rollback restores the CPU email and
# GPU connection Secrets from. They used to be deleted the moment the release
# committed, which made `deploy --rollback` of a committed release fail with
# "release Secret backup is missing". They now live one transaction longer: the
# commit of release N keeps the backups N's own `previous` snapshot references
# and deletes every other labelled backup (release N-1's, or leftovers of a
# transaction that never committed).

BACKUP_LABEL = f"{FLEET.RELEASE_SECRET_BACKUP_LABEL}=true"


class KubeStore:
    """Labelled backup Secrets per kubectl context, plus every command issued."""

    def __init__(self, secrets: dict[str, dict[str, dict]]) -> None:
        self.secrets = {context: dict(items) for context, items in secrets.items()}
        self.commands: list[list[str]] = []

    def get_json(self, arguments: list[str]) -> dict:
        context = arguments[0]
        if "-l" in arguments:
            assert arguments[arguments.index("-l") + 1] == BACKUP_LABEL, (
                "the sweep must list by the backup label only"
            )
            return {
                "items": [
                    {"metadata": {"name": name}}
                    for name in sorted(self.secrets.get(context, {}))
                ]
            }
        name = arguments[-1]
        return dict(self.secrets.get(context, {}).get(name) or {})

    def run(self, arguments: list[str], **kwargs) -> str:
        self.commands.append(list(arguments))
        if "delete" in arguments:
            context = arguments[0]
            names = arguments[arguments.index("secret") + 1 :]
            for name in names:
                if not name.startswith("--"):
                    self.secrets.get(context, {}).pop(name, None)
        if arguments[-3:] == ["apply", "-f", "-"]:
            document = json.loads(kwargs["input_text"])
            self.secrets.setdefault(arguments[0], {})[document["metadata"]["name"]] = (
                document
            )
        return ""


def _secret(name: str, data: dict[str, str]) -> dict:
    return {"metadata": {"name": name}, "type": "Opaque", "data": data}


def fleet_release(store: KubeStore, *cluster_ids: str, **extra: Any) -> SimpleNamespace:
    """A release double over ``store``; ``extra`` adds the seams a test needs."""

    return SimpleNamespace(
        release_id="release-2",
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            clusters=tuple(SimpleNamespace(cluster_id=item) for item in cluster_ids),
        ),
        runner=SimpleNamespace(run=store.run, dry_run=False),
        _cpu=lambda *args: ["cpu", *args],
        _gpu=lambda target, *args: [target.cluster_id, *args],
        _get_json=store.get_json,
        **extra,
    )


def _previous_with_backups(*cluster_ids: str, release_id: str = "release-2") -> dict:
    return {
        "release_id": "release-1",
        "secret_backups": {
            "cpu": {
                "source": "gpu-fault-email",
                "backup": f"gpu-fault-email-rollback-{release_id}",
            },
            "clusters": {
                cluster_id: {
                    "source": "gpu-fault-regional-connection",
                    "backup": f"gpu-fault-regional-connection-rollback-{release_id}",
                }
                for cluster_id in cluster_ids
            },
        },
    }


def test_commit_keeps_the_committed_releases_backups_and_sweeps_older_ones() -> None:
    store = KubeStore(
        {
            "cpu": {
                "gpu-fault-email-rollback-release-1": _secret("old", {"k": "1"}),
                "gpu-fault-email-rollback-release-2": _secret("new", {"k": "2"}),
            },
            "gpu-a": {
                "gpu-fault-regional-connection-rollback-release-1": _secret(
                    "old", {"token": "1"}
                ),
                "gpu-fault-regional-connection-rollback-release-2": _secret(
                    "new", {"token": "2"}
                ),
            },
        }
    )
    release = fleet_release(store, "gpu-a")

    deleted = FLEET.delete_stale_release_secret_backups(
        release, _previous_with_backups("gpu-a")
    )

    assert sorted(deleted) == [
        "gpu-fault-email-rollback-release-1",
        "gpu-fault-regional-connection-rollback-release-1",
    ], "only the older release's backups are swept"
    assert set(store.secrets["cpu"]) == {"gpu-fault-email-rollback-release-2"}, (
        "the committed release's own CPU backup must survive its commit"
    )
    assert set(store.secrets["gpu-a"]) == {
        "gpu-fault-regional-connection-rollback-release-2"
    }, "the committed release's own cluster backup must survive its commit"
    deletes = [command for command in store.commands if "delete" in command]
    assert all("--ignore-not-found" in command for command in deletes), (
        "a backup another operator already removed must not fail the commit"
    )


def test_commit_sweep_keeps_every_referenced_name_in_every_context() -> None:
    # A CPU and a GPU context may be the same EKS cluster (single-cluster
    # sites): the cluster backup must not be swept from the CPU pass just
    # because the CPU snapshot only names the email backup.
    store = KubeStore(
        {
            "cpu": {
                "gpu-fault-email-rollback-release-2": _secret("new", {"k": "2"}),
                "gpu-fault-regional-connection-rollback-release-2": _secret(
                    "shared", {"token": "2"}
                ),
            },
            "gpu-a": {},
        }
    )
    release = fleet_release(store, "gpu-a")

    deleted = FLEET.delete_stale_release_secret_backups(
        release, _previous_with_backups("gpu-a")
    )

    assert deleted == [], "nothing older than the committed release exists"
    assert len(store.secrets["cpu"]) == 2, "referenced backups are kept everywhere"


def test_commit_sweep_ignores_a_cluster_removed_since_the_backup_was_taken() -> None:
    # gpu-b left the site between the backup and the commit: its context is
    # unreachable and its backup name is still referenced, so the sweep must
    # neither touch it nor fail on it.
    store = KubeStore(
        {
            "cpu": {"gpu-fault-email-rollback-release-2": _secret("new", {"k": "2"})},
            "gpu-a": {
                "gpu-fault-regional-connection-rollback-release-1": _secret(
                    "old", {"token": "1"}
                ),
                "gpu-fault-regional-connection-rollback-release-2": _secret(
                    "new", {"token": "2"}
                ),
            },
            "gpu-b": {
                "gpu-fault-regional-connection-rollback-release-2": _secret(
                    "orphan", {"token": "2"}
                )
            },
        }
    )
    release = fleet_release(store, "gpu-a")

    deleted = FLEET.delete_stale_release_secret_backups(
        release, _previous_with_backups("gpu-a", "gpu-b")
    )

    assert deleted == ["gpu-fault-regional-connection-rollback-release-1"], (
        "only the current clusters' older backups are swept"
    )
    assert "gpu-b" not in {command[0] for command in store.commands}, (
        "a cluster no longer in the site must not be contacted"
    )
    assert set(store.secrets["gpu-b"]) == {
        "gpu-fault-regional-connection-rollback-release-2"
    }, "the removed cluster's backup is left alone"


def test_commit_sweep_does_nothing_when_the_snapshot_recorded_no_backups() -> None:
    # A snapshot without `secret_backups` cannot say which backups are still
    # needed; deleting everything on that basis would be a guess.
    store = KubeStore(
        {"cpu": {"gpu-fault-email-rollback-release-1": _secret("old", {"k": "1"})}}
    )
    release = fleet_release(store, "gpu-a")

    deleted = FLEET.delete_stale_release_secret_backups(release, {"release_id": "x"})

    assert deleted == [], "an unreferenced snapshot sweeps nothing"
    assert store.commands == [], "no kubectl call is made without a reference"


def test_rollback_after_commit_restores_from_the_retained_backups() -> None:
    """The end-to-end reason for the lifecycle change.

    Release 2 commits (sweeping release 1's backups), then is rolled back: the
    CPU and cluster restores must find release 2's backups and put the previous
    Secret contents back; a backup that genuinely does not exist still fails
    closed.
    """

    store = KubeStore(
        {
            "cpu": {
                "gpu-fault-email-rollback-release-1": _secret("old", {"k": "1"}),
                "gpu-fault-email-rollback-release-2": _secret("new", {"k": "2"}),
            },
            "gpu-a": {
                "gpu-fault-regional-connection-rollback-release-2": _secret(
                    "new", {"token": "2"}
                )
            },
        }
    )
    previous = _previous_with_backups("gpu-a")
    state = {
        "phase": "complete",
        "previous": previous,
        "release_diff": {"kind": "CONTROL_PLANE_ONLY", "changed": ["cpu"]},
        "execution_plan": {"components": ["cpu"]},
        "completed_phases": ["complete"],
        "completed_cluster_ids": ["gpu-a"],
    }
    # No `_delete_stale_release_secret_backups` seam on purpose: the commit must
    # reach the real fleet sweep the way the engine's release does.
    release = fleet_release(
        store,
        "gpu-a",
        _load_state=lambda: dict(state),
        _save_state=lambda phase, **updates: state.update({"phase": phase, **updates}),
    )

    regional_release_transaction.commit_release(release)

    assert state["commit_cleanup_completed"] is True, "commit finished"
    assert "gpu-fault-email-rollback-release-1" not in store.secrets["cpu"], (
        "the older release's backup is gone after the commit"
    )

    cpu = previous["secret_backups"]["cpu"]
    FLEET.restore_secret(release, ["cpu"], source=cpu["source"], backup=cpu["backup"])
    cluster = previous["secret_backups"]["clusters"]["gpu-a"]
    FLEET.restore_secret(
        release, ["gpu-a"], source=cluster["source"], backup=cluster["backup"]
    )

    assert store.secrets["cpu"]["gpu-fault-email"]["data"] == {"k": "2"}, (
        "the CPU Secret is restored from the retained backup"
    )
    assert store.secrets["gpu-a"]["gpu-fault-regional-connection"]["data"] == {
        "token": "2"
    }, "the cluster Secret is restored from the retained backup"
    with pytest.raises(ReleaseError, match="release Secret backup is missing"):
        FLEET.restore_secret(
            release,
            ["cpu"],
            source="gpu-fault-email",
            backup="gpu-fault-email-rollback-release-1",
        )
