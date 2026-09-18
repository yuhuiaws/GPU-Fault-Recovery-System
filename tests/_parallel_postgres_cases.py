"""Explicit subprocess-only cases; ordinary pytest discovery does not select this file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tools.pytest_result_identity import parse_pytest_receipt, source_identity

MODE = os.environ.get("PARALLEL_POSTGRES_CASE_MODE", "")
CANARY = os.environ.get("PARALLEL_POSTGRES_CANARY", "example-private-output")
ROOT = Path(__file__).resolve().parents[1]

if MODE == "collection-skip":
    pytest.skip(CANARY, allow_module_level=True)
if MODE == "collection-error":
    raise RuntimeError(CANARY)


@pytest.fixture
def phases() -> Iterator[None]:
    if MODE == "setup":
        pytest.fail(CANARY)
    yield
    if MODE == "teardown":
        pytest.fail(CANARY)


@pytest.mark.parametrize("number", range(24))
def test_variant(number: int, phases: None) -> None:
    del phases
    print(CANARY)
    if MODE == "failure":
        pytest.fail(CANARY)
    if MODE == "skip":
        pytest.skip(CANARY)
    assert number >= 0


def test_nested_pytest_does_not_inherit_partition_or_report(tmp_path: Path) -> None:
    if MODE != "nested":
        return
    from scripts.e2e.regional.focused_pytest import prepare_focused_pytest
    from tests.regional._cov95_focused_mock_receipts import write_focused_receipt

    assert (
        not {
            "PYTEST_GPU_FAULT_CASE_REPORT",
            "PYTEST_GPU_FAULT_PARTITION_COUNT",
            "PYTEST_GPU_FAULT_PARTITION_INDEX",
            "PYTEST_GPU_FAULT_CI_CONTEXT",
            "PYTEST_GPU_FAULT_LOCAL_POSTGRES_FAILURE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        }
        & os.environ.keys()
    ), "local receipt controls leaked into nested pytest"
    nodeid = "tests/_parallel_postgres_cases.py::test_variant[0]"
    with prepare_focused_pytest(
        [sys.executable, "-m", "pytest", nodeid], cwd=ROOT, environment=os.environ
    ) as focused:
        assert focused is not None
        write_focused_receipt(
            focused.command, environment=focused.environment, cwd=ROOT, nodeids=[nodeid]
        )
        checked = focused.verify(
            subprocess.CompletedProcess(focused.command, 0, "", "")
        )
        assert checked.returncode == 0, (
            "a nested runner could not use the public reporter contract"
        )
    child = tmp_path / "test_nested.py"
    child.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('number', range(17))\n"
        "def test_nested(number):\n"
        "    assert number >= 0\n",
        encoding="utf-8",
    )
    report = tmp_path / "nested.json"
    environment = {**os.environ, "PYTEST_GPU_FAULT_CASE_REPORT": str(report)}
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-n",
            "0",
            "-p",
            "tools.pytest_case_reporter",
            "-p",
            "no:cacheprovider",
            "--confcutdir",
            str(tmp_path),
            str(child),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, "nested pytest failed"
    receipt = parse_pytest_receipt(
        json.loads(report.read_text()),
        root=ROOT,
        expected_identity=source_identity(ROOT),
        require_session=True,
        require_passed=True,
    )
    assert len(receipt.records) == 17, "nested pytest silently ran a parent partition"
    assert (
        receipt.session is not None
        and receipt.session["selection"]["partition"] is None
    )
