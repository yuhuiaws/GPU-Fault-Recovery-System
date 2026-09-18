from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSTGRES_URL_ENV = "GPU_FAULT_TEST_POSTGRES_URL"


@pytest.fixture
def invoke_make(tmp_path):
    interpreter = tmp_path / "record-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], "
        f"'postgres_configured': bool(os.getenv({POSTGRES_URL_ENV!r}))}}))\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o700)

    def run(target):
        completed = subprocess.run(
            [
                "make",
                "--no-print-directory",
                target,
                f"PYTHON={interpreter}",
                "PYTEST_XDIST_WORKERS=16",
                "POSTGRES_TESTS=tests/unit-placeholder.py",
            ],
            cwd=ROOT,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(tmp_path),
                POSTGRES_URL_ENV: "postgresql://unused/postgres",
            },
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])

    return run


def test_postgres_make_target_is_serial_even_when_ordinary_workers_are_sixteen(
    invoke_make,
) -> None:
    result = invoke_make("test-postgres")
    assert result["args"] == [
        "-m",
        "pytest",
        "tests/unit-placeholder.py",
        "-n",
        "0",
        "--durations=50",
    ], "external PostgreSQL tests must not inherit the ordinary xdist budget"
    assert result["postgres_configured"] is True, (
        "the serial database selection retains its explicit test configuration"
    )


def test_ordinary_make_target_keeps_sixteen_workers_and_clears_database_access(
    invoke_make,
) -> None:
    result = invoke_make("test-parallel")
    assert result["args"] == [
        "-m",
        "pytest",
        "-n",
        "16",
        "--dist=worksteal",
        "--durations=50",
    ], "ordinary tests preserve the requested parallel worker budget"
    assert result["postgres_configured"] is False, (
        "ordinary workers cannot inherit the external PostgreSQL URL"
    )
