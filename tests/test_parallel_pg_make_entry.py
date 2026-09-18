from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("parallel", [False, True])
def test_make_keeps_external_serial_and_local_parallel_contracts(
    tmp_path: Path, parallel: bool
) -> None:
    capture = tmp_path / "command.json"
    python = tmp_path / "record-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PG_ENTRY_CAPTURE']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:],\n"
        "    'workers': os.environ.get('GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS'),\n"
        "    'rounds': os.environ.get('GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS'),\n"
        "}))\n"
    )
    python.chmod(0o700)
    tests = [
        "tests/store/test_postgres_store.py",
        "tests/store/test_store_contracts.py",
    ]
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "test-postgres-stress-parallel" if parallel else "test-postgres-stress",
            f"PYTHON={python}",
            "POSTGRES_TEST_WORKERS=8",
            "PYTEST_DURATIONS=7",
            "POSTGRES_TESTS=" + " ".join(tests),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PG_ENTRY_CAPTURE": str(capture),
            "GPU_FAULT_TEST_POSTGRES_URL": "isolated-test-reference",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, "the Make test entry failed"
    command = json.loads(capture.read_text())
    assert command["workers"] == "8" and command["rounds"] == "40", (
        "the stress workload was reduced"
    )
    assert command["argv"] == (
        [
            "scripts/run_postgres_shards.py",
            "--python",
            str(python),
            "--workers",
            "8",
            "--durations",
            "7",
            "--tests",
            *tests,
        ]
        if parallel
        else ["-m", "pytest", *tests, "-n", "0", "--durations=7"]
    ), "the test inventory, isolation mode or worker budget changed"
