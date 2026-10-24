from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.run_fault_test_cases import build_isolated_environment


def test_efa_counter_harness_runs_its_real_main_with_isolated_api(tmp_path):
    root = Path(__file__).resolve().parents[2]
    report = tmp_path / "efa-report.json"
    env = build_isolated_environment()
    env.update(
        PYTHONPATH=os.pathsep.join((str(root), str(root / "src"))),
        GPU_FAULT_EFA_E2E_DB=str(tmp_path / "efa.db"),
        GPU_FAULT_EFA_E2E_REPORT=str(report),
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.e2e.hyperpod.run_hyperpod_efa_traffic_e2e"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["verdict"] == "PASS", data
    phases = {sample["name"]: sample["findings"] for sample in data["samples"]}
    assert phases["spike"][0]["severity"] == "warning"
    assert phases["hung-suspected"][0]["severity"] == "critical"
    assert data["hung_workflow"]["step_executions"] == [], (
        "this test verifies detection and planning, not physical recovery"
    )
