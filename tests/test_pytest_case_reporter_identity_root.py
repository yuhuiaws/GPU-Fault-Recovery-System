"""The case reporter can bind receipts to a named git tree when pytest runs in a copy without .git."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools import pytest_case_reporter as reporter
from tools.pytest_result_identity import source_identity

ROOT = Path(__file__).resolve().parents[1]


def _run_child(
    tmp_path: Path, *, identity_root: Path | None
) -> subprocess.CompletedProcess[str]:
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "test_tiny.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    report = tmp_path / "report.json"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        reporter.REPORT_ENV: str(report),
    }
    environment.pop(reporter.IDENTITY_ROOT_ENV, None)
    if identity_root is not None:
        environment[reporter.IDENTITY_ROOT_ENV] = str(identity_root)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tools.pytest_case_reporter",
            "-o",
            "addopts=",
            "test_tiny.py",
        ],
        cwd=copy,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_reporter_binds_a_git_less_copy_to_the_named_source_tree(
    tmp_path: Path,
) -> None:
    # Live 2026-09-20 (BOOT-018): the artifact pytest ran inside a copy of the
    # checkout that carries no .git; the reporter's git identity lookup failed
    # with INTERNALERROR "not a git repository". Naming the source tree the copy
    # came from binds the receipt to that tree's identity instead.
    without = _run_child(tmp_path, identity_root=None)
    assert without.returncode != 0
    assert "not a git repository" in without.stderr + without.stdout
    bound = (
        _run_child(tmp_path / "second", identity_root=ROOT)
        if (tmp_path / "second").mkdir() is None
        else None
    )
    assert bound is not None and bound.returncode == 0, (
        bound.stderr[-800:] if bound else "no run"
    )
    payload = json.loads(
        (tmp_path / "second" / "report.json").read_text(encoding="utf-8")
    )
    assert payload["source_identity"] == source_identity(ROOT)
    assert payload["session"]["source_identity"] == source_identity(ROOT)
