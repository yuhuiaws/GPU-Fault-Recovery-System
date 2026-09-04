from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cluster_join as JOIN
from gpu_fault.admin import cluster_join_commit as COMMIT
from gpu_fault.admin import cluster_removal as REMOVAL
from gpu_fault.admin.cluster_join_evidence import join_activation_is_irreversible
from gpu_fault.installation_resources import InstallationResourceStatus


def _registry(cluster_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_sha256="e" * 64,
        resources=[
            SimpleNamespace(resource_key=key, status=InstallationResourceStatus.ACTIVE)
            for key in (
                f"cluster/{cluster_id}/eks",
                f"cluster/{cluster_id}/hyperpod",
                f"aws/iam/executor/{cluster_id}/role",
            )
        ],
    )


def test_join_activation_is_the_irreversible_commit_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Everything reversible is committed before activation starts.

    Once the rollout activates the cluster the join can no longer be undone, so
    the site file, the release state and the resource registry all have to be
    committed -- and the membership re-verified against them -- while rollback is
    still possible.
    """

    steps: list[str] = []
    site = SimpleNamespace(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        release_config={"clusters": [{"cluster_id": "gpu-a"}]},
    )
    for name, recorded in (
        ("_commit_site", None),
        ("final_membership_identity", None),
        ("validate_verified_membership", "verify"),
    ):
        monkeypatch.setattr(
            COMMIT,
            name,
            lambda *_args, _recorded=recorded, **_kwargs: (
                steps.append(_recorded) if _recorded else {}
            ),
        )
    monkeypatch.setattr(
        COMMIT,
        "complete_step",
        lambda _path, _state, step, _evidence=None: steps.append(step),
    )
    monkeypatch.setattr(
        COMMIT,
        "_sync_registry",
        lambda *_args, **_kwargs: SimpleNamespace(source_sha256="s" * 64),
    )
    monkeypatch.setattr(COMMIT, "_joined_resources", lambda *_a, **_k: [])
    monkeypatch.setattr(COMMIT, "load_site", lambda *_args, **_kwargs: site)
    monkeypatch.setattr(JOIN, "_update_bootstrap_state", lambda *_a, **_k: None)
    monkeypatch.setattr(JOIN, "_sync_join_release_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        JOIN,
        "_run_rollout",
        lambda _candidate, action, **_kwargs: steps.append(f"rollout:{action}"),
    )
    monkeypatch.setattr(
        JOIN, "_fetch_installation_registry", lambda _site: _registry("gpu-a")
    )

    COMMIT.activate_and_commit(
        SimpleNamespace(site=site),
        execution=SimpleNamespace(
            cluster_id="gpu-a",
            candidate=SimpleNamespace(
                source=tmp_path / "candidate.yaml", source_sha256="c" * 64
            ),
            prerequisites={"executor_role": {}, "network": {}},
            discovery={"registry_snapshot": str(tmp_path / "before.json")},
        ),
        state_dir=tmp_path,
        state_path=tmp_path / "state.json",
        state={
            "evidence": {
                "VERIFIED": {
                    "candidate_site_sha256": "c" * 64,
                    "verified_at": "2026-01-01T00:00:00+00:00",
                }
            }
        },
    )

    assert steps == [
        "verify",
        "SITE_UPDATED",
        "RELEASE_STATE_UPDATED",
        "REGISTRY_UPDATED",
        "verify",
        "ACTIVATION_STARTED",
        "rollout:activate-cluster",
        "ACTIVATED",
        "FINAL_VERIFIED",
    ]


def test_join_failure_after_activation_start_is_fail_forward() -> None:
    assert join_activation_is_irreversible(
        {"completed_steps": ["ACTIVATION_STARTED"]}
    ), "activation start was not treated as irreversible"
    assert join_activation_is_irreversible({"completed_steps": ["ACTIVATED"]}), (
        "activated membership was not treated as irreversible"
    )
    assert not join_activation_is_irreversible(
        {"completed_steps": ["SITE_UPDATED", "REGISTRY_UPDATED"]}
    ), "pre-activation commits were incorrectly treated as irreversible"


def test_remove_cluster_uses_the_site_membership_operation_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The membership lock is held before the site is even re-read.

    Two administrators removing clusters at once would otherwise both read the
    pre-removal site and each write back a membership list that is missing the
    other's change.
    """

    class LockHeldElsewhere(RuntimeError):
        pass

    locked: list[object] = []

    class RefusedLock:
        def __init__(self, site: object) -> None:
            locked.append(site)

        def __enter__(self) -> None:
            raise LockHeldElsewhere("another administrator holds the membership lock")

        def __exit__(self, *_arguments: object) -> bool:
            return False

    monkeypatch.setattr(REMOVAL, "membership_operation_lock", RefusedLock)
    monkeypatch.setattr(
        REMOVAL,
        "reload_site_for_mutation",
        lambda _site: pytest.fail("the site was re-read before the lock was held"),
    )
    site = object()

    with pytest.raises(LockHeldElsewhere):
        REMOVAL.remove_cluster(SimpleNamespace(site=site))

    assert locked == [site]
