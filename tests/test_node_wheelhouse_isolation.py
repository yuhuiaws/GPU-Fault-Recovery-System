from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_import_and_inventory_validation_leave_application_logging_unchanged():
    program = """
import json,logging
from pathlib import Path
before = logging.getLevelNamesMapping()
from scripts.node_wheelhouse import validate_wheelhouse_inventory
from gpu_fault.logging_setup import resolve_level
try:
    validate_wheelhouse_inventory(Path.cwd(), {})
except ValueError:
    pass
print(json.dumps({
    "unchanged": logging.getLevelNamesMapping() == before,
    "level": resolve_level(),
    "expected": logging.INFO,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env={**os.environ, "GPU_FAULT_LOG_LEVEL": "VERBOSE"},
        text=True,
        capture_output=True,
        check=False,
        timeout=35,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["unchanged"] is True
    assert report["level"] == report["expected"]
