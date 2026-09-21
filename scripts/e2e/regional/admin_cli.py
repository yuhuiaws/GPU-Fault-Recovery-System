"""Which ``gpu-fault-admin`` may act on a managed state directory.

A managed state dir carries its own deployer venv, and the checkout's module
CLI refuses to act on it ("<state-dir> has its own deploy-host CLI; run
<state-dir>/deployer-venv/bin/gpu-fault-admin ... instead", rc 2 — live
2026-09-20, AUTH-016 a1 and the BOOT-020 public config drill). Sites without one
are driven through this interpreter's module form.
"""

from __future__ import annotations

import sys
from pathlib import Path


def admin_command(state_dir: Path) -> list[str]:
    """The deploy-host CLI that owns ``state_dir``, or the module form."""

    own = Path(state_dir) / "deployer-venv" / "bin" / "gpu-fault-admin"
    if own.is_file():
        return [str(own)]
    return [sys.executable, "-m", "gpu_fault.admin.cli"]
