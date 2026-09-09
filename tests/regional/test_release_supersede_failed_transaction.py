"""``gpu-fault-admin deploy --supersede-failed-transaction``.

Production deploy #7 (2026-09-07): a fail-forward transaction for release
``503c...`` stopped at the data plane in phase ``failed``. The operator fixed
the bug, which produced a *different* candidate id, and ran ``deploy``. The
deploy entrypoint saw a resumable phase, forced ``resume=True``, and the engine
refused with ``resume release_id does not match the candidate`` -- a message
that names neither release nor a way out. There was no product path to open a
new transaction for the fix over the failed one; the operator drove the engine
from a hand-written script.

Now the mismatch is refused with both ids and the flag named, and the flag
opens a *superseding* transaction: a new transaction for the candidate whose
rollback baseline is the failed transaction's ``previous`` (the last committed
release, not the mixed live state), whose diff also re-rolls everything the
failed release moved, and which records the transaction it replaced.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_release_automatic_rollback as AUTOMATIC_ROLLBACK
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_schema_change as SCHEMA_CHANGE
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256

ENV = "GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION"
FLAG = "--supersede-failed-transaction"
FAILED_ID = "503c0ffee0ff"
CANDIDATE_ID = "f3c00c116538"


def _previous() -> dict[str, Any]:
    return {
        "release_id": "committed-release",
        "cpu_wheel": "gpu-fault-wheel-committed",
        "metadata": {
            "required-agent-artifact-sha256": "a" * 64,
            "required-regional-executor-artifact-sha256": "b" * 64,
        },
        "clusters": {
            "gpu-a": {
                "wheel": "executor-committed",
                "reconciler_wheel": "executor-committed",
                "bundle": "bundle-committed",
            }
        },
        "secret_backups": {
            "cpu": None,
            "clusters": {
                "gpu-a": {
                    "source": "gpu-fault-regional-connection",
                    "backup": f"gpu-fault-regional-connection-rollback-{FAILED_ID}",
                }
            },
        },
        "observability": {"adot": {}},
        "endpoint": None,
        "runtime_image": "registry.example/gpu-fault@sha256:" + "c" * 64,
    }


def _failed_state(
    *,
    phase: str = "failed",
    release_id: str = FAILED_ID,
    previous: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    previous = _previous() if previous is None else previous
    return {
        "phase": phase,
        "release_id": release_id,
        "previous": previous,
        "previous_snapshot_sha256": canonical_sha256(previous),
        "cluster_ids": ["gpu-a"],
        "release_diff": {
            "kind": "DATA_PLANE_COMPATIBLE",
            "changed": ["executor_wheel"],
        },
        "execution_plan": {"nodes": ["executor", "verify"]},
        "completed_phases": ["preflight", "uploaded"],
        "completed_cluster_ids": [],
        "original_failure": "ReleaseError: gpu-a executor did not become ready",
        "release_lifecycle": "FAILED",
        "updated_at_epoch": 1757262600,
        "approved_manifest_sha256": "f" * 64,
        **extra,
    }


def _deploy_release(
    state: dict[str, Any] | None, *, release_id: str = CANDIDATE_ID
) -> tuple[SimpleNamespace, list[Any]]:
    calls: list[Any] = []
    release = SimpleNamespace(
        release_id=release_id,
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        ),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: state is not None),
        _load_state=lambda: dict(state or {}),
        pin_approved_manifest_plan=lambda digest: calls.append(("pin", digest)),
        upgrade=lambda **kwargs: calls.append(("upgrade", kwargs)),
        bootstrap=lambda: calls.append(("bootstrap", None)),
        rollback=lambda: calls.append(("rollback", None)),
        commit_release=lambda: calls.append(("commit", None)),
        noop=lambda diff: calls.append(("noop", diff)),
        # A superseding transaction must never read the mixed live state.
        _capture_previous=lambda **_kwargs: pytest.fail(
            "supersede captured the mixed live state as the baseline"
        ),
    )
    return release, calls


# --- the deploy entrypoint -------------------------------------------------


def test_the_cli_and_the_engine_spell_the_variable_and_the_flag_the_same() -> None:
    assert admin_cli.SUPERSEDE_FAILED_TRANSACTION_ENV == ENV
    assert ADMIN.SUPERSEDE_FAILED_TRANSACTION_ENV == ENV
    assert ADMIN.SUPERSEDE_FAILED_TRANSACTION_FLAG == FLAG
    assert ORCHESTRATION.SUPERSEDABLE_PHASES == {"failed", "partial-convergence"}


@pytest.mark.parametrize("phase", ("failed", "partial-convergence"))
def test_a_different_candidate_over_a_failed_transaction_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release, calls = _deploy_release(_failed_state(phase=phase))

    with pytest.raises(ReleaseError) as failure:
        ADMIN.run_deploy(release)

    message = str(failure.value)
    assert FAILED_ID in message
    assert CANDIDATE_ID in message
    assert FLAG in message
    assert phase in message
    assert "resume release_id does not match" not in message
    assert calls == [], "the refusal must happen before anything is pinned or run"


@pytest.mark.parametrize("phase", ("failed", "partial-convergence"))
def test_the_same_release_still_resumes_its_failed_transaction(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    state = _failed_state(phase=phase)
    release, calls = _deploy_release(state, release_id=FAILED_ID)
    expected = ADMIN.diff_from_changed({"executor_wheel"})
    monkeypatch.setattr(ADMIN, "retry_release_diff", lambda _release, _state: expected)

    ADMIN.run_deploy(release)

    assert calls == [("pin", "f" * 64), ("upgrade", {"resume": True, "diff": expected})]


def test_the_flag_opens_a_superseding_transaction_with_the_union_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "1")
    state = _failed_state()
    release, calls = _deploy_release(state)
    # The candidate differs from the failed release only in the control-plane
    # wheel; the failed release had moved the executor wheel. Both roll.
    monkeypatch.setattr(
        ADMIN,
        "classify_release",
        lambda _release, _state: ADMIN.diff_from_changed({"control_plane_wheel"}),
    )
    release.wheel_cm = "gpu-fault-wheel-committed"
    release.executor_wheel_cm = "executor-committed"
    release.bundle_cm = "bundle-committed"
    release.node_wheel_sha = "a" * 64
    release.executor_wheel_sha = "b" * 64

    ADMIN.run_deploy(release)

    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "upgrade"
    assert kwargs["resume"] is False
    assert kwargs["supersede"] is not None
    assert kwargs["supersede"]["release_id"] == FAILED_ID
    diff = kwargs["diff"]
    assert diff.changed == {"control_plane_wheel", "executor_wheel"}
    assert diff.kind is ADMIN.ReleaseChangeKind.DATA_PLANE_COMPATIBLE, (
        "the kind follows the union, not the candidate's control-plane-only view"
    )
    assert not any(name == "pin" for name, _ in calls), (
        "a fresh transaction originates its own manifest digest (M-23); pinning "
        "the failed transaction's would refuse the very fix being deployed"
    )


def test_next_deploy_reports_the_supersede_instead_of_a_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _failed_state()
    release, _calls = _deploy_release(state)
    monkeypatch.setattr(
        ADMIN, "classify_release", lambda _release, _state: ADMIN.diff_from_changed(())
    )
    for name in ("wheel_cm", "executor_wheel_cm", "bundle_cm"):
        setattr(release, name, _previous()["cpu_wheel"])
    release.executor_wheel_cm = "executor-committed"
    release.bundle_cm = "bundle-committed"
    release.node_wheel_sha = "a" * 64
    release.executor_wheel_sha = "b" * 64

    monkeypatch.delenv(ENV, raising=False)
    assert ADMIN.next_deploy(release, state)["resume"] is True

    monkeypatch.setenv(ENV, "1")
    planned = ADMIN.next_deploy(release, state)

    assert planned["resume"] is False
    assert planned["action"] == "upgrade"
    assert planned["supersedes_release_id"] == FAILED_ID
    assert planned["changed"] == ["executor_wheel"]


@pytest.mark.parametrize(
    ("state", "reason"),
    (
        (None, "no release state yet"),
        ({"phase": "complete", "release_id": FAILED_ID}, "no failed transaction"),
        (
            {
                "phase": "complete",
                "release_id": FAILED_ID,
                "transaction_committed": False,
            },
            "still resumable",
        ),
        ({"phase": "cpu-staged", "release_id": FAILED_ID}, "still resumable"),
        ({"phase": "data-plane-progress", "release_id": FAILED_ID}, "still resumable"),
        (
            {"phase": "rollback-data-progress", "release_id": FAILED_ID},
            "rollback is in progress",
        ),
        (
            {"phase": "rollback-failed", "release_id": FAILED_ID},
            "rollback is in progress",
        ),
        ({"phase": "rolled-back", "release_id": FAILED_ID}, "no failed transaction"),
        (
            {"phase": "bootstrap-failed", "release_id": FAILED_ID},
            "no failed transaction",
        ),
    ),
)
def test_the_flag_is_refused_outside_a_terminal_failed_transaction(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, Any] | None, reason: str
) -> None:
    monkeypatch.setenv(ENV, "1")
    release, calls = _deploy_release(state)

    with pytest.raises(ReleaseError) as failure:
        ADMIN.run_deploy(release)

    assert FLAG in str(failure.value)
    assert reason in str(failure.value)
    assert calls == []


def test_the_flag_is_refused_for_the_release_that_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "1")
    release, calls = _deploy_release(_failed_state(), release_id=FAILED_ID)

    with pytest.raises(ReleaseError, match="without the flag to resume"):
        ADMIN.run_deploy(release)

    assert calls == []


def test_resume_refuses_a_different_candidate_with_the_flag_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release, calls = _deploy_release(_failed_state())

    with pytest.raises(ReleaseError) as failure:
        ADMIN.run_resume(release)

    assert FAILED_ID in str(failure.value)
    assert CANDIDATE_ID in str(failure.value)
    assert FLAG in str(failure.value)
    assert calls == []


# --- the CLI ---------------------------------------------------------------


def test_the_flag_travels_to_the_source_preparer_as_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    base = [
        "deploy",
        "--cpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
        "--state-dir",
        str(tmp_path / "state"),
        "--admin-email",
        "operations@example.com",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert ENV not in calls[-1]["extra_environment"], "no flag, no variable"
    assert admin_cli.run(admin_cli.parser().parse_args([*base, FLAG])) == 0
    assert calls[-1]["extra_environment"][ENV] == "1"
    assert ADMIN.supersede_requested(calls[-1]["extra_environment"]) is True


def test_the_flag_wins_over_a_stale_supersede_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leftover ``export`` must not decide consent; the command line does.

    With the flag the variable is set from the flag; without it nothing is
    added and the inherited environment (whatever it says) travels unchanged.
    """

    monkeypatch.setenv(ENV, "1")
    with_flag = admin_cli.parser().parse_args(["deploy", FLAG, "--state-dir", "/tmp/x"])
    without = admin_cli.parser().parse_args(["deploy", "--state-dir", "/tmp/x"])

    assert admin_cli.supersede_environment(with_flag) == {ENV: "1"}
    assert admin_cli.supersede_environment(without) == {}


# --- the engine ------------------------------------------------------------


def _engine_release(
    *, auto_rollback: bool = True, database_schema_version: int = 12
) -> tuple[SimpleNamespace, list[tuple[str, dict[str, Any]]]]:
    saved: list[tuple[str, dict[str, Any]]] = []

    def save_state(phase: str, **updates: Any) -> None:
        release.state.update({"phase": phase, **updates})
        saved.append((phase, dict(updates)))

    release = SimpleNamespace(
        release_id=CANDIDATE_ID,
        _save_state=save_state,
        config=SimpleNamespace(
            auto_rollback=auto_rollback,
            schema_rollback_compatible=False,
            database_schema_version=database_schema_version,
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        ),
        state={"phase": "failed", "release_id": FAILED_ID, "original_failure": "x"},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _apply_rds_ca_bundle=lambda: None,
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _remote_commands_are_idle=lambda: True,
        _load_state=lambda: pytest.fail("a superseding transaction is not a resume"),
        _capture_previous=lambda **_kwargs: pytest.fail(
            "supersede captured the mixed live state as the baseline"
        ),
        _backup_release_secrets=lambda: pytest.fail(
            "supersede backed up the mixed live Secrets"
        ),
    )
    return release, saved


def _union_diff(*changed: str) -> ORCHESTRATION.ReleaseDiff:
    return ADMIN.diff_from_changed({"executor_wheel", *changed})


def test_a_superseding_transaction_inherits_the_failed_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCHEMA_CHANGE.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.setattr(
        ORCHESTRATION, "run_upgrade_phases", lambda *_args, **_kwargs: True
    )
    release, saved = _engine_release()
    failed = _failed_state()

    ORCHESTRATION.upgrade_release(
        release, diff=_union_diff("control_plane_wheel"), supersede=failed
    )

    assert [phase for phase, _ in saved] == ["preflight"]
    _phase, opened = saved[0]
    assert opened["previous"] == _previous()
    assert opened["previous"] is not failed["previous"], "inherited by copy"
    assert opened["previous_snapshot_sha256"] == canonical_sha256(_previous())
    assert opened["previous"]["secret_backups"] == _previous()["secret_backups"]
    assert opened["completed_phases"] == []
    assert opened["completed_cluster_ids"] == []
    assert opened["registry_staged"] is False
    assert opened["transaction_committed"] is False
    assert opened["release_lifecycle"] == "PREPARING"
    assert opened["release_diff"] == _union_diff("control_plane_wheel").as_dict()
    assert opened["cluster_attempts"]["gpu-a"]["state"] == "PENDING"
    record = opened["superseded_transaction"]
    assert record["release_id"] == FAILED_ID
    assert record["phase"] == "failed"
    assert record["updated_at_epoch"] == 1757262600
    assert record["original_failure"] == failed["original_failure"]
    assert record["release_diff"] == failed["release_diff"]
    assert record["superseded_at"].startswith("20"), record
    assert "schema_change_acceptance" not in opened
    # The failed transaction's fields do not leak into the new state.
    assert "original_failure" not in release.state


def test_a_superseding_transaction_carries_the_acceptance_for_the_same_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCHEMA_CHANGE.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.setattr(
        ORCHESTRATION, "run_upgrade_phases", lambda *_args, **_kwargs: True
    )
    acceptance = {
        "mode": "snapshot",
        "accepted_at": "2026-09-07T16:00:00+00:00",
        "database_schema_version": 12,
        "snapshot_id": "gpu-fault-pre-v12-503c0ffee0ff",
        "snapshot_status": "available",
        "cluster_id": "gpu-fault-aurora",
    }
    release, saved = _engine_release(database_schema_version=12)
    failed = _failed_state(schema_change_acceptance=acceptance)

    ORCHESTRATION.upgrade_release(
        release, diff=_union_diff("database_schema"), supersede=failed
    )

    _phase, opened = saved[0]
    assert opened["schema_change_acceptance"] == acceptance
    assert opened["superseded_transaction"]["schema_change_acceptance"] == acceptance


def test_a_further_schema_change_needs_its_own_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCHEMA_CHANGE.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.setattr(
        ORCHESTRATION,
        "run_upgrade_phases",
        lambda *_args, **_kwargs: pytest.fail("refused transactions do not run"),
    )
    release, saved = _engine_release(database_schema_version=13)
    failed = _failed_state(
        schema_change_acceptance={"mode": "snapshot", "database_schema_version": 12}
    )

    with pytest.raises(ReleaseError, match=SCHEMA_CHANGE.ACCEPT_FLAG):
        ORCHESTRATION.upgrade_release(
            release, diff=_union_diff("database_schema"), supersede=failed
        )

    assert saved == []


def test_a_superseding_transaction_follows_the_site_auto_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-forward was the failed transaction's policy, not an inheritance."""

    monkeypatch.delenv(SCHEMA_CHANGE.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.setattr(
        ORCHESTRATION,
        "run_upgrade_phases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReleaseError("gpu-a broke")),
    )
    monkeypatch.setattr(
        AUTOMATIC_ROLLBACK, "record_upgrade_failure", lambda *_args, **_kwargs: None
    )
    release, _saved = _engine_release(auto_rollback=True)
    rolled_back: list[dict[str, Any]] = []
    release.rollback = lambda **kwargs: rolled_back.append(kwargs)

    with pytest.raises(ReleaseError, match="gpu-a broke"):
        ORCHESTRATION.upgrade_release(
            release, diff=_union_diff(), supersede=_failed_state()
        )

    assert [item["state"] for item in rolled_back] == [_previous()], (
        "the automatic rollback lands on the inherited committed baseline"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda state: state.update(phase="cpu-staged"), "only a transaction in"),
        (lambda state: state.update(phase="rollback-failed"), "only a transaction in"),
        (
            lambda state: state.update(release_id=CANDIDATE_ID),
            "failed candidate itself",
        ),
        (lambda state: state.pop("previous"), "no previous baseline"),
        (lambda state: state.pop("previous_snapshot_sha256"), "digest is missing"),
        (
            lambda state: state["previous"].update(cpu_wheel="tampered"),
            "digest drifted",
        ),
        (
            lambda state: state.update(cluster_ids=["gpu-a", "gpu-b"]),
            "membership drifted",
        ),
    ),
)
def test_an_inherited_baseline_is_checked_before_anything_moves(
    mutate: Any, message: str
) -> None:
    release, saved = _engine_release()
    failed = _failed_state()
    mutate(failed)

    with pytest.raises(ReleaseError, match=message):
        ORCHESTRATION.inherit_superseded_previous(release, failed)

    assert saved == []


def test_a_baseline_without_secret_backups_is_refused() -> None:
    release, _saved = _engine_release()
    previous = _previous()
    previous.pop("secret_backups")
    failed = _failed_state(previous=previous)

    with pytest.raises(ReleaseError, match="no Secret backups"):
        ORCHESTRATION.inherit_superseded_previous(release, failed)


def test_a_superseding_transaction_cannot_also_be_a_resume() -> None:
    release, _saved = _engine_release()

    with pytest.raises(ReleaseError, match="cannot also be a resume"):
        ORCHESTRATION.upgrade_release(
            release, resume=True, diff=_union_diff(), supersede=_failed_state()
        )
