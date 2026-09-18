from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script",
    ["run_ha007_control_worker_shutdown.py", "run_ha008_processor_exit_acceptance.py"],
)
def test_local_ha_script_imports_without_inherited_pythonpath(
    tmp_path: Path, script: str
) -> None:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/e2e/regional" / script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--run-dir" in completed.stdout
    assert "--release-id" in completed.stdout
    assert "--cluster-id" in completed.stdout
    assert not list(tmp_path.iterdir()), "help must not create acceptance artifacts"
