"""BOOT-029 README procedure: block parsing edges, the pristine copy guard and the CLI."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import boot029_readme as readme
from scripts.e2e.regional.boot029_readme import ReadmeProcedureError

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "e2e" / "regional"
GPU = readme.GPU_PLACEHOLDER


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "copy"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.name", "Fixture")
    git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "fixture")
    return repository


def test_deploy_section_runs_to_the_end_without_a_next_level_two_heading() -> None:
    text = f"# README\n{readme.DEPLOY_HEADING}\nfirst\n### detail\nlast"

    assert readme.deploy_section(text) == "first\n### detail\nlast"


@pytest.mark.parametrize("body", ["```bash\necho never closed\n", "no block at all\n"])
def test_fenced_bash_block_refuses_a_missing_or_unterminated_block(body: str) -> None:
    with pytest.raises(ReadmeProcedureError, match="README has no ```bash block under"):
        readme.fenced_bash_block("## Build\n" + body, "## Build")


def test_logical_commands_keep_continuations_and_drop_blank_lines() -> None:
    assert readme.logical_commands("make a \\\n  b\n\n   \nmake c") == [
        "make a \\\n  b",
        "make c",
    ]
    with pytest.raises(ReadmeProcedureError, match="ends in a line continuation"):
        readme.logical_commands("make a \\\n  b \\")


def test_shell_words_refuse_a_command_that_is_not_valid_shell() -> None:
    assert readme.shell_words("make check  'two words'") == [
        "make",
        "check",
        "two words",
    ]
    with pytest.raises(ReadmeProcedureError, match="README command is not valid shell"):
        readme.shell_words("echo 'unterminated")


def test_expand_gpu_lines_needs_a_gpu_cluster() -> None:
    with pytest.raises(ReadmeProcedureError, match="at least one GPU cluster"):
        readme.expand_gpu_lines(f"deploy \\\n  --gpu-cluster-arn {GPU}", [], [])


def test_expand_gpu_lines_on_the_last_line_continues_every_line_but_the_last() -> None:
    block = f"gpu-fault-admin deploy \\\n  --gpu-cluster-arn {GPU}"

    assert readme.expand_gpu_lines(block, ["arn-a", "arn-b"], ["--profile eks-v1"]) == (
        "gpu-fault-admin deploy \\\n"
        "  --gpu-cluster-arn arn-a \\\n"
        "  --gpu-cluster-arn arn-b \\\n"
        "  --profile eks-v1"
    )


@pytest.mark.parametrize(
    ("dirty", "commit", "fragment"),
    [
        (True, None, "pristine source copy is not clean"),
        (False, "0" * 40, "pristine source copy commit changed"),
    ],
)
def test_pristine_source_refuses_a_dirty_or_moved_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty: bool,
    commit: str | None,
    fragment: str,
) -> None:
    copy = committed_repository(tmp_path)
    if dirty:
        (copy / "stray.txt").write_text("left behind\n", encoding="utf-8")
    checkout = SimpleNamespace(
        repository_root=copy, git_commit=commit or git(copy, "rev-parse", "HEAD")
    )
    calls: list[tuple[Path, Path]] = []

    def prepare(repository: Path, *, state_dir: Path) -> Any:
        calls.append((repository, state_dir))
        return checkout

    monkeypatch.setattr(readme, "prepare_source_checkout", prepare)

    with pytest.raises(ReadmeProcedureError, match=fragment):
        readme.pristine_source(tmp_path / "repo", tmp_path / "work")

    assert calls == [(tmp_path / "repo", tmp_path / "work")]


def cli_arguments(tmp_path: Path, *extra: str, attempt: str = "2") -> list[str]:
    return [
        "boot029_readme",
        "--readme",
        str(tmp_path / "README.md"),
        "--repo",
        str(tmp_path / "repo"),
        "--work-dir",
        str(tmp_path / "work"),
        "--attempt",
        attempt,
        "--cpu-arn",
        "arn:aws:eks:us-west-2:000000000000:cluster/cpu",
        "--gpu-arn",
        "arn:aws:eks:us-west-2:000000000000:cluster/gpu-a",
        "--gpu-arn",
        "arn:aws:eks:us-west-2:000000000000:cluster/gpu-b",
        "--state-dir",
        str(tmp_path / "state"),
        "--admin-email",
        "ops@example.invalid",
        "--receipt",
        str(tmp_path / "receipt.jsonl"),
        *extra,
    ]


def test_cli_hands_the_run_its_arguments_and_exits_with_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(readme, "run", lambda **kwargs: calls.append(kwargs) or 3)
    monkeypatch.setattr(
        sys,
        "argv",
        cli_arguments(
            tmp_path,
            "--joined-gpu-arn",
            "arn:aws:sagemaker:us-west-2:000000000000:cluster/hp",
            "--first-provider",
            "eks-managed",
        ),
    )

    with pytest.raises(SystemExit) as stop:
        readme.main()

    assert stop.value.code == 3
    [call] = calls
    assert call["repository"] == (tmp_path / "repo").resolve()
    assert call["state_dir"] == (tmp_path / "state").resolve()
    assert call["attempt"] == 2
    assert call["gpu_arns"] == [
        "arn:aws:eks:us-west-2:000000000000:cluster/gpu-a",
        "arn:aws:eks:us-west-2:000000000000:cluster/gpu-b",
    ]
    assert call["joined_arns"] == [
        "arn:aws:sagemaker:us-west-2:000000000000:cluster/hp"
    ]
    assert call["first_provider"] == "eks-managed"
    assert call["receipt"] == tmp_path / "receipt.jsonl"


def test_cli_reports_a_refused_procedure_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(**_kwargs: Any) -> int:
        raise ReadmeProcedureError("README lacks the heading '## 部署：先区分角色'")

    monkeypatch.setattr(readme, "run", refuse)
    monkeypatch.setattr(sys, "argv", cli_arguments(tmp_path))

    with pytest.raises(SystemExit) as stop:
        readme.main()

    assert stop.value.code == 1
    assert capsys.readouterr().err == (
        "FAIL: README procedure refused: README lacks the heading '## 部署：先区分角色'\n"
    )


def test_program_refuses_a_non_positive_attempt_before_running_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", cli_arguments(tmp_path, attempt="0"))

    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SCRIPTS / "boot029_readme.py"), run_name="__main__")

    assert stop.value.code == 2
    assert "attempt must be positive" in capsys.readouterr().err
    assert not (tmp_path / "work").exists(), "nothing is created for a refused run"
