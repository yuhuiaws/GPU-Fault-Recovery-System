from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mixin_contracts_are_explicit() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-mixin-contracts.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
