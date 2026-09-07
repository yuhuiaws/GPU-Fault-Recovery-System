"""Declared documentation facts must match the code, and stale numbers must go.

``make docs-check`` proved that referenced paths exist and generated files are
in sync; it never asked whether a sentence in the prose was still true. Two
design documents carried a hand-counted ``tests/`` file total that had drifted
by a factor of two, and ``COLLECTORS.md`` still said the application performs
no bearer-token validation long after the regional middleware started doing
exactly that. ``scripts/check-doc-facts.py`` holds a small declarative table of
the statements that are cheap to verify by machine.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-doc-facts.py"
MODULE = lazy_script_module(SCRIPT)
DESIGN = "docs/详细设计.md"
DESIGN_V2 = "docs/详细设计-v2.md"
COLLECTORS = "COLLECTORS.md"


def _copy_doc(root: Path, relative: str, replace: tuple[str, str] | None) -> Path:
    text = (ROOT / relative).read_text(encoding="utf-8")
    if replace is not None:
        old, new = replace
        assert old in text, f"{relative} no longer carries {old!r}"
        text = text.replace(old, new, 1)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _facts(*ids: str) -> list[object]:
    selected = [fact for fact in MODULE.FACTS if fact.id in ids]
    assert {fact.id for fact in selected} == set(ids), "unknown fact id"
    return selected


def test_repository_documents_satisfy_every_declared_fact() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_literal_test_file_count_fails_in_a_stale_copy(tmp_path: Path) -> None:
    _copy_doc(
        tmp_path,
        DESIGN,
        ("`tests/` 按领域分目录", "`tests/` 共 223 个 Python 文件，按领域分目录"),
    )
    _copy_doc(tmp_path, DESIGN_V2, None)

    failures = MODULE.check_facts(tmp_path, _facts("test-file-count-not-literal"))

    assert len(failures) == 1
    assert DESIGN in failures[0]
    assert "223 个 Python 文件" in failures[0]


def test_the_largest_test_directory_claim_is_recomputed(tmp_path: Path) -> None:
    _copy_doc(tmp_path, DESIGN, None)
    (tmp_path / "tests/regional").mkdir(parents=True)
    (tmp_path / "tests/regional/test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "tests/store").mkdir()
    for name in ("test_b.py", "test_c.py"):
        (tmp_path / "tests/store" / name).write_text("", encoding="utf-8")

    failures = MODULE.check_facts(tmp_path, _facts("largest-test-directory"))

    assert len(failures) == 1
    assert "tests/store" in failures[0]


def test_a_removed_statement_is_reported_not_skipped(tmp_path: Path) -> None:
    _copy_doc(tmp_path, DESIGN, ("`tests/regional/` 是其中最大的一组", "（略）"))
    (tmp_path / "tests/regional").mkdir(parents=True)
    (tmp_path / "tests/regional/test_a.py").write_text("", encoding="utf-8")

    failures = MODULE.check_facts(tmp_path, _facts("largest-test-directory"))

    assert len(failures) == 1
    assert "statement not found" in failures[0]


def test_the_old_no_bearer_validation_sentence_fails(tmp_path: Path) -> None:
    _copy_doc(
        tmp_path,
        COLLECTORS,
        (
            "区域模式下控制面自身校验集群 bearer token",
            "当前应用本身尚未实现 bearer token 校验，"
            "区域模式下控制面自身校验集群 bearer token",
        ),
    )

    failures = MODULE.check_facts(tmp_path, _facts("collectors-no-stale-auth-claim"))

    assert len(failures) == 1
    assert "尚未实现 bearer token 校验" in failures[0]


def test_the_auth_statement_is_backed_by_the_middleware_source(tmp_path: Path) -> None:
    _copy_doc(tmp_path, COLLECTORS, None)
    for relative in (
        "src/gpu_fault/app/middleware/auth.py",
        "src/gpu_fault/app/factory.py",
        "src/gpu_fault/regional.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            textwrap.dedent(
                """
                def unrelated() -> None:
                    return None
                """
            ),
            encoding="utf-8",
        )

    failures = MODULE.check_facts(tmp_path, _facts("collectors-bearer-auth"))

    assert len(failures) == 1
    assert "compare_digest" in failures[0]
    assert "regional cluster bearer token is required" in failures[0]


def test_cli_reports_and_exits_nonzero_on_a_stale_root(tmp_path: Path) -> None:
    _copy_doc(
        tmp_path,
        COLLECTORS,
        (
            "区域模式下控制面自身校验集群 bearer token",
            "当前应用本身尚未实现 bearer token 校验",
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--fact",
            "collectors-no-stale-auth-claim",
            "--fact",
            "collectors-bearer-auth",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "collectors-no-stale-auth-claim" in result.stdout
    assert "collectors-bearer-auth" in result.stdout
