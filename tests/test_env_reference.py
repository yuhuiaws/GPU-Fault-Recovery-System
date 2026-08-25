from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_PATTERN = re.compile(r"\bGPU_FAULT_[A-Z0-9_]+\b")
DEPLOY_ENV_SOURCES = (
    ROOT / "deploy" / "node" / "install-gpu-fault-collector.sh",
    ROOT / "deploy" / "node" / "verify-gpu-fault-collector.sh",
    *(ROOT / "deploy" / "systemd").glob("*"),
)


def test_environment_reference_matches_source() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate-env-reference.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_deployment_environment_is_known_to_python_processes() -> None:
    inventory = json.loads(
        (ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    known = set(inventory["variables"])
    deployed = {
        name
        for path in DEPLOY_ENV_SOURCES
        if path.is_file()
        for name in NAME_PATTERN.findall(path.read_text(encoding="utf-8"))
    }
    assert deployed <= known, (
        "deployment code uses GPU_FAULT_* variables absent from "
        f"the runtime inventory: {sorted(deployed - known)}"
    )
