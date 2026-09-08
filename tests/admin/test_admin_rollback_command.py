"""`gpu-fault-admin deploy --state-dir X --rollback`: one step back, refused twice.

The command's backend (`gpu_fault.admin.rollback_command`) reads the live
release state, refuses anything but a committed release with a `previous`
snapshot, runs the engine's `rollback` mode, then aligns the management site
and the engine's state the way the automatic-rollback driver does.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import rollback_command as COMMAND
from gpu_fault.admin.site import SiteConfigError, load_site
from gpu_fault_release import regional_admin_commands as ADMIN
from scripts import release_failure_recovery as RECOVERY
from tests.admin.test_admin_site import site_file

RELEASE_ID = "release-b"
PREVIOUS_ID = "release-a"


def _committed_state(**extra: object) -> dict:
    return {
        "phase": "complete",
        "release_id": RELEASE_ID,
        "transaction_committed": True,
        "commit_cleanup_completed": True,
        "release_diff": {"kind": "CONTROL_PLANE_ONLY", "changed": ["cpu_wheel"]},
        "previous": {"release_id": PREVIOUS_ID, "secret_backups": {"clusters": {}}},
        **extra,
    }


# --- refusals ----------------------------------------------------------------


def test_a_committed_release_with_a_previous_snapshot_may_roll_back() -> None:
    assert COMMAND.rollback_refusal(_committed_state()) is None, (
        "the ordinary case must not be refused"
    )


@pytest.mark.parametrize("phase", sorted(ADMIN.RESUMABLE_PHASES))
def test_a_transaction_in_flight_is_refused_and_sent_back_to_deploy(phase: str) -> None:
    refusal = COMMAND.rollback_refusal(_committed_state(phase=phase))

    assert refusal is not None, f"phase {phase} is mid-transaction"
    assert "mid-transaction" in refusal and "deploy --state-dir" in refusal, (
        "the operator is told to finish the transaction with deploy, not rollback"
    )


@pytest.mark.parametrize("phase", sorted(ADMIN.ROLLBACK_PHASES))
def test_a_rollback_in_progress_is_refused_and_sent_back_to_deploy(phase: str) -> None:
    refusal = COMMAND.rollback_refusal(_committed_state(phase=phase))

    assert refusal is not None, f"phase {phase} is a rollback in progress"
    assert "already in progress" in refusal and "deploy --state-dir" in refusal, (
        "deploy resumes the rollback; a second rollback command does not"
    )


def test_a_rolled_back_release_with_pending_cleanup_is_still_in_progress() -> None:
    refusal = COMMAND.rollback_refusal(
        _committed_state(phase="rolled-back", rollback_cleanup_completed=False)
    )

    assert refusal is not None and "already in progress" in refusal, (
        "post-rollback cleanup belongs to the running rollback"
    )


@pytest.mark.parametrize("phase", sorted(ADMIN.BOOTSTRAP_PHASES))
def test_a_bootstrapping_site_has_no_release_to_roll_back_to(phase: str) -> None:
    refusal = COMMAND.rollback_refusal({"phase": phase, "release_id": RELEASE_ID})

    assert refusal is not None and "bootstrapping" in refusal, (
        "bootstrap phases are not transactions with a previous release"
    )


def test_an_uncommitted_complete_release_belongs_to_deploy() -> None:
    refusal = COMMAND.rollback_refusal(_committed_state(transaction_committed=False))

    assert refusal is not None and "mid-transaction" in refusal, (
        "a verified but uncommitted release is finished by deploy (commit)"
    )


def test_pending_commit_cleanup_belongs_to_deploy() -> None:
    refusal = COMMAND.rollback_refusal(_committed_state(commit_cleanup_completed=False))

    assert refusal is not None and "mid-transaction" in refusal, (
        "commit cleanup is deploy's to finish before a rollback opens"
    )


def test_a_second_rollback_in_a_row_is_refused_with_the_way_forward() -> None:
    state = _committed_state(
        phase="rolled-back",
        rollback_cleanup_completed=True,
        rollback_result={"status": "PASSED"},
    )

    refusal = COMMAND.rollback_refusal(state)

    assert refusal is not None, "a rolled-back state has no usable previous"
    assert "already rolled back" in refusal, "the reason names the prior rollback"
    assert PREVIOUS_ID in refusal, "the operator learns which release is live"
    assert "deploy the older commit" in refusal, (
        "going further back is an ordinary deploy, not a rollback"
    )


@pytest.mark.parametrize("previous", (None, {}, "not-a-dict"))
def test_a_release_without_a_previous_snapshot_has_nothing_to_roll_back_to(
    previous: object,
) -> None:
    refusal = COMMAND.rollback_refusal(_committed_state(previous=previous))

    assert refusal is not None and "nothing to roll back to" in refusal, (
        "a first bootstrap or a NOOP release records no previous"
    )


def test_a_schema_change_is_refused_with_the_engines_own_words() -> None:
    state = _committed_state(
        release_diff={"kind": "FULL", "changed": ["database_schema", "cpu_wheel"]},
        schema_change_acceptance={"mode": "snapshot", "snapshot_id": "snap-1"},
    )

    refusal = COMMAND.rollback_refusal(state)

    assert refusal is not None, "the engine refuses to cross a schema version"
    assert "PostgreSQL schema change" in refusal, "the engine's wording is kept"
    assert "snap-1" in refusal, "the pre-schema snapshot is named as the way back"


def test_a_schema_change_without_a_snapshot_says_the_database_is_manual() -> None:
    state = _committed_state(
        release_diff={"kind": "FULL", "changed": ["database_schema"]}
    )

    refusal = COMMAND.rollback_refusal(state)

    assert refusal is not None and "by hand" in refusal, (
        "with no snapshot the operator is told the restore is manual"
    )


def test_a_non_transactional_change_is_refused_by_name() -> None:
    state = _committed_state(release_diff={"kind": "FULL", "changed": ["clusters"]})

    refusal = COMMAND.rollback_refusal(state)

    assert refusal == "rollback is not transactional for: clusters", (
        "cluster membership changes are not rolled back by the engine"
    )


# --- run_rollback ------------------------------------------------------------


class FakeRuns:
    """Records every engine invocation and answers with the configured code."""

    def __init__(self, *, rollback_returncode: int = 0) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []
        self.rollback_returncode = rollback_returncode

    def __call__(self, arguments, *, cwd, env, check):
        assert check is False, "the caller reads the exit code itself"
        self.calls.append((list(arguments), Path(cwd), dict(env)))
        mode = arguments[1]
        code = self.rollback_returncode if mode == "rollback" else 0
        return subprocess.CompletedProcess(arguments, code)

    @property
    def modes(self) -> list[str]:
        return [call[0][1] for call in self.calls]


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    states: list[dict],
    rollback_returncode: int = 0,
    reconcile_error: Exception | None = None,
) -> tuple[Path, object, FakeRuns, list[dict]]:
    site_path = site_file(tmp_path)
    site = load_site(site_path)
    runs = FakeRuns(rollback_returncode=rollback_returncode)
    reads = iter(states)
    monkeypatch.setattr(COMMAND, "live_release_state", lambda _site: next(reads))
    monkeypatch.setattr(COMMAND.subprocess, "run", runs)
    reconciled: list[dict] = []

    def reconcile(path, live_state, *, site_before, source):
        if reconcile_error is not None:
            raise reconcile_error
        reconciled.append(
            {
                "site_file": Path(path),
                "live_state": live_state,
                "site_before": Path(site_before),
                "source": source,
            }
        )
        return site.repository_root

    monkeypatch.setattr(COMMAND, "reconcile_rollback_management", reconcile)
    return site_path, site, runs, reconciled


def _rolled_back_state() -> dict:
    return {
        "phase": "rolled-back",
        "release_id": RELEASE_ID,
        "rollback_cleanup_completed": True,
        "rollback_result": {"status": "PASSED"},
        "previous": {"release_id": PREVIOUS_ID},
    }


def _result(capsys: pytest.CaptureFixture[str]) -> dict:
    out = capsys.readouterr().out.strip().splitlines()
    return json.loads(out[-1])[COMMAND.RESULT_KEY]


def test_rollback_runs_the_engine_then_aligns_management_like_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_path, site, runs, reconciled = _prepare(
        tmp_path, monkeypatch, states=[_committed_state(), _rolled_back_state()]
    )
    record_dir = tmp_path / "release-deploy" / RELEASE_ID
    record_dir.mkdir(parents=True)
    (record_dir / "state.json").write_text(
        json.dumps({"release_id": RELEASE_ID, "phase": "COMPLETED"}), encoding="utf-8"
    )

    status = COMMAND.run_rollback(site, state_dir=tmp_path)

    assert status == 0, capsys.readouterr()
    assert runs.modes == ["rollback", "sync-state"], (
        "the engine rolls back, then the aligned site is synced"
    )
    rollback_arguments, cwd, env = runs.calls[0]
    assert rollback_arguments[0] == str(site.repository_root / COMMAND.ROLLOUT_SCRIPT)
    assert rollback_arguments[2:3] == ["--config"], "the engine gets a config file"
    assert cwd == site.repository_root, "the engine runs from the repository root"
    assert env["GPU_FAULT_REPO_ROOT"] == str(site.repository_root)
    assert env["PYTHONPATH"] == str(site.repository_root / "src")
    assert reconciled == [
        {
            "site_file": site_path,
            "live_state": _rolled_back_state(),
            "site_before": record_dir / "site.before.yaml",
            "source": f"rollback:{RELEASE_ID}",
        }
    ], "management alignment uses the failure driver's inputs"
    result = _result(capsys)
    assert result["status"] == "PASSED", result
    assert result["release_id"] == RELEASE_ID
    assert result["rolled_back_to"] == PREVIOUS_ID
    assert result["management_state_synced"] is True
    record = json.loads((record_dir / "state.json").read_text(encoding="utf-8"))
    assert record["phase"] == "ROLLED_BACK", "the release-deploy record is updated"
    assert record["rollback"]["status"] == "PASSED"
    assert result["release_deploy_record_updated"] is True


def test_rollback_without_a_release_deploy_record_still_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _site_path, site, runs, _reconciled = _prepare(
        tmp_path, monkeypatch, states=[_committed_state(), _rolled_back_state()]
    )

    status = COMMAND.run_rollback(site, state_dir=tmp_path)

    assert status == 0, capsys.readouterr()
    assert runs.modes == ["rollback", "sync-state"], "both engine steps ran"
    result = _result(capsys)
    assert result["release_deploy_record_updated"] is False, (
        "a release deployed elsewhere has no record here; that is not a failure"
    )


def test_a_refused_rollback_touches_nothing_and_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _site_path, site, runs, reconciled = _prepare(
        tmp_path, monkeypatch, states=[_rolled_back_state()]
    )

    status = COMMAND.run_rollback(site, state_dir=tmp_path)

    assert status == 2, "a refusal is a usage error, not an engine failure"
    assert runs.calls == [] and reconciled == [], "nothing runs after a refusal"
    captured = capsys.readouterr()
    assert "rollback refused" in captured.err and "already rolled back" in captured.err
    result = json.loads(captured.out.strip().splitlines()[-1])[COMMAND.RESULT_KEY]
    assert result["status"] == "REFUSED" and "already rolled back" in result["reason"]


def test_an_engine_failure_returns_its_exit_code_and_skips_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _site_path, site, runs, reconciled = _prepare(
        tmp_path, monkeypatch, states=[_committed_state()], rollback_returncode=3
    )

    status = COMMAND.run_rollback(site, state_dir=tmp_path)

    assert status == 3, "the engine's exit code is the command's"
    assert runs.modes == ["rollback"], "no sync-state after a failed rollback"
    assert reconciled == [], "the management site is not rewritten on failure"
    result = _result(capsys)
    assert (
        result["status"] == "FAILED"
        and "release engine rollback failed" in (result["error"])
    )


def test_an_alignment_failure_after_a_good_rollback_is_reported_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _site_path, site, runs, _reconciled = _prepare(
        tmp_path,
        monkeypatch,
        states=[_committed_state(), _rolled_back_state()],
        reconcile_error=SiteConfigError(
            "rolled-back Runtime Profile snapshot is unavailable"
        ),
    )
    record_dir = tmp_path / "release-deploy" / RELEASE_ID
    record_dir.mkdir(parents=True)
    (record_dir / "state.json").write_text(json.dumps({"phase": "COMPLETED"}), "utf-8")

    status = COMMAND.run_rollback(site, state_dir=tmp_path)

    assert status == 1, "the site rolled back but management is not aligned"
    assert runs.modes == ["rollback"], "sync-state never ran"
    captured = capsys.readouterr()
    assert "could not be aligned" in captured.err
    result = json.loads(captured.out.strip().splitlines()[-1])[COMMAND.RESULT_KEY]
    assert result["status"] == "ALIGNMENT_FAILED"
    assert result["management_state_synced"] is False
    assert "Runtime Profile snapshot" in result["error"]
    record = json.loads((record_dir / "state.json").read_text(encoding="utf-8"))
    assert record["rollback"]["status"] == "ALIGNMENT_FAILED", (
        "the release-deploy record says what still needs doing"
    )


def test_a_failed_sync_state_is_an_alignment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _site_path, site, runs, _reconciled = _prepare(
        tmp_path, monkeypatch, states=[_committed_state(), _rolled_back_state()]
    )

    def failing_sync(arguments, *, cwd, env, check):
        runs.calls.append((list(arguments), Path(cwd), dict(env)))
        return subprocess.CompletedProcess(
            arguments, 0 if arguments[1] == "rollback" else 4
        )

    monkeypatch.setattr(COMMAND.subprocess, "run", failing_sync)

    status = COMMAND.run_rollback(site, state_dir=tmp_path)

    assert status == 1, "a sync-state failure is reported as alignment failure"
    result = _result(capsys)
    assert result["status"] == "ALIGNMENT_FAILED"
    assert "sync-state failed (4)" in result["error"]


# --- the failure driver shares the alignment ---------------------------------


def test_the_failure_driver_aligns_through_the_same_function_and_its_own_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """release_failure_recovery must keep its `reconcile_rollback_management`
    monkeypatch seam (tests/admin/test_release_deploy.py relies on it) while the
    body lives in `gpu_fault.admin.rollback_command`."""

    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(
        RECOVERY,
        "reconcile_rollback_management",
        lambda path, state, **kwargs: (seen.append(("reconcile", kwargs)), tmp_path)[1],
    )
    monkeypatch.setattr(
        COMMAND,
        "reconcile_rollback_management",
        lambda *_args, **_kwargs: pytest.fail("the driver's own seam must be used"),
    )

    class Prepared:
        release_id = RELEASE_ID
        state_dir = tmp_path / "release-deploy" / RELEASE_ID

    RECOVERY.align_rolled_back_management(
        Prepared(),
        site_file=tmp_path / "site.yaml",
        root=tmp_path,
        environment={},
        live_state={"phase": "rolled-back"},
        run_release_mode=lambda path, *, mode, root, environment: seen.append(
            (mode, root)
        ),
    )

    assert seen == [
        (
            "reconcile",
            {
                "site_before": Prepared.state_dir / "site.before.yaml",
                "source": f"rollback:{RELEASE_ID}",
            },
        ),
        ("sync-state", tmp_path),
    ], "reconcile first, then sync-state against the reconciled root"
