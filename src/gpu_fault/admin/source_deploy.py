from __future__ import annotations

from gpu_fault.admin.execution import run_driver

import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import BootstrapError

SOURCE_DEPLOY_STATE = "source-deploy.json"
REPOSITORY_MARKERS = (
    "pyproject.toml",
    "deploy",
    "scripts/staging_deploy.py",
)


def _require_repository(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not all((resolved / marker).exists() for marker in REPOSITORY_MARKERS):
        raise BootstrapError(f"{description} is not a GPU Fault repository")
    return resolved


def _repository_from_source_state(state_dir: Path) -> Path | None:
    state_file = state_dir / SOURCE_DEPLOY_STATE
    if not state_file.is_file():
        return None
    try:
        value = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError("internal source deployment state is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BootstrapError("internal source deployment state schema is invalid")
    root = value.get("source_repository_root")
    if not isinstance(root, str) or not root:
        raise BootstrapError("internal source deployment state has no repository")
    return _require_repository(
        Path(root),
        description="recorded source repository",
    )


def _repository_from_site(state_dir: Path) -> Path | None:
    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        return None
    try:
        value = yaml.safe_load(site_file.read_text(encoding="utf-8"))
        root = value["spec"]["repositoryRoot"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as exc:
        raise BootstrapError("generated site has no valid repository root") from exc
    if not isinstance(root, str) or not root:
        raise BootstrapError("generated site has no valid repository root")
    return _require_repository(
        Path(root),
        description="generated site repository",
    )


def resolve_source_repository(
    state_dir: Path,
    *,
    current_directory: Path,
) -> Path:
    """The checkout a deploy builds from.

    The operator's current directory wins whenever it is a repository: that is
    how an upgrade from a fresh clone, or a first deploy resumed after a fix,
    picks up the operator's changes. A recorded source (``source-deploy.json``)
    or the generated site's root only stands in when the command runs from
    elsewhere (a bound CLI invoked from any directory). A recorded source that
    cannot be read fails closed before any fallback. Live 2026-09-22: a first
    deploy resumed from a corrected checkout silently rebuilt the earlier
    attempt's recorded copy and failed its release gate a second time.
    """

    resolved_state = state_dir.expanduser().resolve()
    recorded = _repository_from_source_state(resolved_state)
    try:
        current = _require_repository(
            current_directory,
            description="current directory",
        )
    except BootstrapError:
        current = None
    if current is not None:
        if recorded is not None and recorded != current:
            print(
                f"gpu-fault-admin: deploying the source in the current directory "
                f"{current}; the earlier deploy of this site ran from {recorded}",
                file=sys.stderr,
                flush=True,
            )
        return current
    if recorded is not None:
        return recorded
    existing = _repository_from_site(resolved_state)
    if existing is not None:
        return existing
    raise BootstrapError("first deploy must run from the cloned GPU Fault repository")


def run_source_deploy(
    *,
    cpu_cluster_arn: str,
    gpu_cluster_arns: Sequence[str],
    state_dir: Path,
    admin_email: str,
    impact_base: str,
    current_directory: Path,
    extra_environment: Mapping[str, str] | None = None,
    wait_for_email_confirmation: int = 0,
) -> int:
    repository_root = resolve_source_repository(
        state_dir,
        current_directory=current_directory,
    )
    script = repository_root / "scripts/staging_deploy.py"
    command = [
        sys.executable,
        str(script),
        "--cpu-cluster-arn",
        cpu_cluster_arn,
    ]
    for arn in gpu_cluster_arns:
        command.extend(("--gpu-cluster-arn", arn))
    command.extend(
        (
            "--state-dir",
            str(state_dir.expanduser()),
            "--admin-email",
            admin_email,
            "--base",
            impact_base,
            "--repo-root",
            str(repository_root),
            "--quiet",
        )
    )
    if wait_for_email_confirmation > 0:
        command.extend(
            ("--wait-for-email-confirmation", str(wait_for_email_confirmation))
        )
    completed = run_driver(
        command,
        cwd=repository_root,
        env={
            **os.environ,
            # Release-engine settings the operator gave on this command (the
            # schema-change acceptance); every later hop inherits them.
            **(extra_environment or {}),
            "PYTHONPATH": str(repository_root / "src"),
        },
        check=False,
    )
    return completed.returncode
