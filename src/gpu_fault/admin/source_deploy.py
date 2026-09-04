from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml  # type: ignore[import-untyped]

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
    resolved_state = state_dir.expanduser().resolve()
    recorded = _repository_from_source_state(resolved_state)
    if recorded is not None:
        return recorded
    try:
        return _require_repository(
            current_directory,
            description="current directory",
        )
    except BootstrapError:
        existing = _repository_from_site(resolved_state)
        if existing is not None:
            return existing
        raise BootstrapError(
            "first deploy must run from the cloned GPU Fault repository"
        ) from None


def run_source_deploy(
    *,
    cpu_cluster_arn: str,
    gpu_cluster_arns: Sequence[str],
    state_dir: Path,
    admin_email: str,
    impact_base: str,
    current_directory: Path,
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
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env={
            **os.environ,
            "PYTHONPATH": str(repository_root / "src"),
        },
        check=False,
    )
    return completed.returncode
