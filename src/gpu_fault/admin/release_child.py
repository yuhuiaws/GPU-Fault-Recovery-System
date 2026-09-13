"""Run one release-deploy under the administrator's held site lock.

``gpu-fault-admin config`` holds the site operation lock across its whole
apply and rolls the control-plane roles through ``scripts/release_deploy.py``,
which takes the same lock itself. The child is handed the held descriptor the
way the deploy host script hands it to its children (live 2026-09-13: without
it, release-deploy refused its own parent as "another administrator mutation").
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    inherited_lock_pass_fds,
)


def run_automatic_release(
    *,
    repository_root: Path,
    site_file: Path,
    state_dir: Path,
    staging_only_release: bool,
    lock_fd: int | None = None,
) -> int:
    command = [
        sys.executable,
        str(repository_root / "scripts/release_deploy.py"),
        "--site",
        str(site_file),
        "--prebuilt-attestation",
        str(repository_root / "dist/current-attestation.json"),
        "--prebuilt-bundle",
        str(repository_root / "dist/current-attestation.bundle.json"),
        "--cosign-key",
        str(state_dir / "release-signing/cosign.pub"),
    ]
    if staging_only_release:
        command.append("--allow-staging-release")
    # release-deploy takes the site lock itself; hand it the one this command
    # holds (the deploy host script does the same for its children), or it
    # refuses its own parent as "another administrator mutation".
    lock_environment = (
        {SITE_OPERATION_LOCK_FD_ENV: str(lock_fd)} if lock_fd is not None else {}
    )
    pass_fds = tuple(
        sorted(
            {*inherited_lock_pass_fds(), *([lock_fd] if lock_fd is not None else [])}
        )
    )
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env={
            **os.environ,
            **lock_environment,
            "PYTHONPATH": str(repository_root / "src"),
        },
        check=False,
        pass_fds=pass_fds,
    )
    if completed.returncode:
        return completed.returncode
    return 0
