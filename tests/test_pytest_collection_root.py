from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import pytest_result_identity
from tools.pytest_result_identity import repository_root, source_identity
from tools.scenario_requirements import Check
from tools.scenario_test_evidence import load_test_evidence

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = "tests/test_origin.py::test_same_name"


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize(
    "location", ["source-root", "nested-copy", "real-tests-root", "nested-cwd"]
)
def test_actual_collection_paths_cannot_alias_an_unexecuted_source_test(
    tmp_path: Path, workers: int, location: str
) -> None:
    root = tmp_path / "source"
    copied_root = root / ".ignored"
    actual = root / "tests/test_origin.py"
    copied = copied_root / "tests/test_origin.py"
    for path in (actual, copied):
        path.parent.mkdir(parents=True, exist_ok=True)
    executes_source = location not in {"nested-copy", "nested-cwd"}
    actual.write_text(
        f"def test_same_name():\n    assert {executes_source!r}, 'source test'\n",
        encoding="ascii",
    )
    copied.write_text(
        "def test_same_name():\n    assert True, 'ignored copy'\n", encoding="ascii"
    )
    (root / ".gitignore").write_text(
        ".ignored/\n__pycache__/\n.pytest_cache/\n", encoding="ascii"
    )
    for arguments in (
        ["git", "init", "-q"],
        ["git", "add", ".gitignore", "tests/test_origin.py"],
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(arguments, cwd=root, check=True, capture_output=True)
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True
        ).stdout
        == b""
    ), "the fixture must cover clean-checkout identity, not rely on dirty input drift"
    identity = source_identity(root)
    report_path = copied_root / "results.json"
    collection_root = (
        root
        if location == "source-root"
        else copied_root
        if location in {"nested-copy", "nested-cwd"}
        else actual.parent
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-p",
        "tools.pytest_case_reporter",
        "-n",
        str(workers),
        "--rootdir",
        str(collection_root),
        str(actual if executes_source else copied),
    ]
    completed = subprocess.run(
        command,
        cwd=copied_root if location == "nested-cwd" else root,
        env={
            "HOME": str(tmp_path),
            "PATH": os.defpath,
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_GPU_FAULT_CASE_REPORT": str(report_path),
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": "/dev/null",
        },
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, (
        "the selected real or copied positive child must finish normally",
        completed.stdout,
        completed.stderr,
    )
    assert source_identity(root) == identity, (
        "the child must not change the source snapshot used by the consumer"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["session"]["exitstatus"] == 0, "the receipt must be from a real pass"
    evidence = load_test_evidence(
        [report_path],
        expected_identity=identity,
        now=datetime.now(timezone.utc),
        root=root,
    )
    assert evidence.confirms(Check(nodeid=SELECTOR)) is executes_source, (
        "only execution of the actual source test can verify that source selector",
        location,
        workers,
        payload["session"]["collected_nodeids"],
    )


def test_missing_git_root_is_not_replaced_with_the_working_directory(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="git command failed"):
        repository_root(tmp_path)


def test_relative_git_root_output_is_not_an_authoritative_source(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, b"relative\n", b"")

    monkeypatch.setattr(pytest_result_identity, "subprocess", SimpleNamespace(run=run))
    with pytest.raises(RuntimeError, match="absolute repository root"):
        repository_root(tmp_path)
    assert calls == [(["git", "rev-parse", "--show-toplevel"], tmp_path.resolve())], (
        "root discovery must use the explicit local Git repository query"
    )
