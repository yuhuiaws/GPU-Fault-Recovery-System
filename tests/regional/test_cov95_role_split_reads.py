from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "deploy/control-plane/tools/verify_control_plane_role_split.py"


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        (1, "Error from server (Forbidden): deployments.apps is forbidden"),
        (1, "Unable to connect to the server: i/o timeout"),
        (1, ""),
        (1, 'Error from server (NotFound): deployments.apps "another-role" not found'),
        (1, 'Error from server (NotFound): secrets "gpu-fault-api-ha" not found'),
        (
            124,
            'Error from server (NotFound): deployments.apps "gpu-fault-api-ha" not found',
        ),
    ],
    ids=["forbidden", "timeout", "empty", "wrong-name", "wrong-kind", "wrong-exit"],
)
def test_role_split_read_errors_cannot_prove_a_missing_deployment(
    tmp_path: Path, returncode: int, stderr: str
) -> None:
    kubectl = tmp_path / "kubectl"
    kubectl.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(stderr)} >&2\nexit {returncode}\n",
        encoding="utf-8",
    )
    kubectl.chmod(0o755)
    completed = subprocess.run(
        [sys.executable, str(VERIFIER)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode == 1, completed
    assert "role Deployments could not be read" in completed.stdout, completed.stdout
    assert "is missing" not in completed.stdout, (
        "an unreadable Deployment produced a confirmed-missing report"
    )
    assert not completed.stderr, (
        "structured verifier failures should not need a traceback"
    )
