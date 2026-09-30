from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import source_deploy as admin_source_deploy
from gpu_fault.admin.bootstrap_common import BootstrapError


def repository(path: Path) -> Path:
    path.mkdir()
    (path / "deploy").mkdir()
    scripts = path / "scripts"
    scripts.mkdir()
    (scripts / "staging_deploy.py").write_text("", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    return path


def test_first_deploy_uses_current_repository(tmp_path: Path) -> None:
    source = repository(tmp_path / "repo")

    resolved = admin_source_deploy.resolve_source_repository(
        tmp_path / "state", current_directory=source
    )

    assert resolved == source


def test_recorded_repository_is_reused_outside_checkout(tmp_path: Path) -> None:
    source = repository(tmp_path / "repo")
    state = tmp_path / "state"
    state.mkdir()
    (state / admin_source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps({"schema_version": 1, "source_repository_root": str(source)}),
        encoding="utf-8",
    )

    resolved = admin_source_deploy.resolve_source_repository(
        state, current_directory=tmp_path
    )

    assert resolved == source


def test_current_checkout_wins_over_a_recorded_one_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A first deploy resumed from a corrected checkout, or an upgrade run from a
    fresh clone, must build the checkout the operator is standing in, not the
    directory an earlier attempt recorded."""

    earlier = repository(tmp_path / "attempt-1")
    current = repository(tmp_path / "attempt-2")
    state = tmp_path / "state"
    state.mkdir()
    (state / admin_source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps({"schema_version": 1, "source_repository_root": str(earlier)}),
        encoding="utf-8",
    )

    resolved = admin_source_deploy.resolve_source_repository(
        state, current_directory=current
    )

    assert resolved == current, "the operator's checkout is the source"
    notice = capsys.readouterr().err
    assert str(current) in notice and str(earlier) in notice, (
        "the switch away from the recorded source is announced"
    )
    assert (
        admin_source_deploy.resolve_source_repository(state, current_directory=earlier)
        == earlier
        and capsys.readouterr().err == ""
    ), "no notice when the current directory is the recorded one"


def test_invalid_recorded_repository_fails_closed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / admin_source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps(
            {"schema_version": 1, "source_repository_root": str(tmp_path / "missing")}
        ),
        encoding="utf-8",
    )

    with pytest.raises(BootstrapError, match="recorded source repository"):
        admin_source_deploy.resolve_source_repository(state, current_directory=tmp_path)


def test_source_deploy_hides_release_and_artifact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository(tmp_path / "repo")
    commands: list[tuple[list[str], dict[str, object]]] = []

    def run(command, **kwargs):
        commands.append(([str(item) for item in command], kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(admin_source_deploy, "run_driver", run)

    result = admin_source_deploy.run_source_deploy(
        cpu_cluster_arn="cpu",
        gpu_cluster_arns=("gpu-a", "gpu-b"),
        state_dir=tmp_path / "state",
        admin_email="operations@example.com",
        impact_base="origin/main",
        current_directory=source,
    )

    assert result == 0
    command, options = commands[0]
    assert command[1].endswith("scripts/staging_deploy.py"), (
        "public deploy did not invoke the private source preparer"
    )
    assert command.count("--gpu-cluster-arn") == 2
    assert "--quiet" in command
    assert "--release-ref" not in command
    assert "--site" not in command
    assert "--prebuilt-attestation" not in command
    assert "--profile-approval" not in command
    assert options["cwd"] == source
    assert options["env"]["PYTHONPATH"] == str(source / "src")
