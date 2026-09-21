"""Resume a wedged uninstall with cleanup tools from another reviewed tree.

A bound site runs only the source snapshot it deployed, and ``deploy`` refuses
while an uninstall is in progress. An uninstall stopped by a defect in its own
cleanup tools therefore resumes with ``--repo-root <reviewed tree>`` plus
``--accept-repository-root-override``; everything that changes is journaled.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall_types import UninstallRequest

CLEANUP_ACCEPT_CONFIG_ENV = "GPU_FAULT_CLEANUP_ACCEPT_CONFIG_SHA256"


def tools_tree_sha256(root: Path) -> str:
    """Digest of the ``deploy/`` and ``scripts/`` trees an uninstall executes."""

    digest = hashlib.sha256()
    for directory in ("deploy", "scripts"):
        base = root / directory
        if not base.is_dir():
            raise BootstrapError(f"uninstall tool tree lacks {directory}/: {root}")
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            digest.update(b"\0path\0" + path.relative_to(root).as_posix().encode())
            digest.update(f"\0mode\0{path.stat().st_mode & 0o777:04o}".encode())
            digest.update(b"\0file\0" + path.read_bytes())
    return digest.hexdigest()


def record_repository_root_override(
    request: UninstallRequest,
    state_path: Path,
    state: dict[str, Any],
    *,
    resumed: bool,
) -> None:
    """Refuse, or journal, cleanup tools that are not the deployed snapshot's.

    A bound site runs the snapshot it deployed. ``deploy`` refuses while an
    uninstall is in progress, so an uninstall stopped by a defect in its own
    cleanup tools resumes with ``--repo-root <reviewed tree>`` plus
    ``--accept-repository-root-override``; the tree and the digest of the tools
    it contributes go into the journal so the run stays auditable.
    """

    actual = request.site.repository_root.expanduser().resolve()
    configured = load_site(request.site.source).repository_root
    if actual == configured:
        return
    if not request.repository_root_override:
        raise BootstrapError(
            "uninstall runs the deployed snapshot's cleanup tools; resume with "
            "--repo-root <reviewed tree> --accept-repository-root-override only "
            "for an uninstall that is already in progress"
        )
    if not resumed or state.get("phase") == "COMPLETED":
        raise BootstrapError(
            "--accept-repository-root-override needs an uninstall already in progress"
        )
    record = {
        "path": str(actual),
        "deployed_repository_root": str(configured),
        "tools_sha256": tools_tree_sha256(actual),
        "phase": state["phase"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    overrides = state.get("repository_root_overrides")
    if not isinstance(overrides, list):
        overrides = []
    latest = overrides[-1] if overrides and isinstance(overrides[-1], dict) else {}
    if (latest.get("path"), latest.get("tools_sha256")) != (
        record["path"],
        record["tools_sha256"],
    ):
        overrides.append(record)
    state["repository_root_overrides"] = overrides
    write_json_atomic(state_path, state)


def cleanup_shell_environment(
    request: UninstallRequest, config_path: Path
) -> dict[str, str]:
    """The environment of the Kubernetes cleanup shell and tools.

    Under ``--accept-repository-root-override`` the release config is
    re-materialized from the override tree, so its manifest path (and nothing
    else) differs from the digest the cleanup journal recorded; the cleanup
    tools accept exactly this config's digest and journal the change.
    """

    environment = {**os.environ, **request.site.environment}
    # The cleanup tools import ``gpu_fault`` from the site's source tree, as
    # ``prepare-clean-redeploy.sh`` arranges for its own invocations; the
    # uninstall's direct invocations need the same path.
    source = str(request.site.repository_root / "src")
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{source}:{inherited}" if inherited else source
    if request.repository_root_override:
        environment[CLEANUP_ACCEPT_CONFIG_ENV] = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
    return environment
