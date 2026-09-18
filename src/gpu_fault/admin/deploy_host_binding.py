"""Check deploy-host and managed-site identities before opening command logs."""

from __future__ import annotations

import argparse
from collections.abc import Collection
import json
import os
from pathlib import Path
import stat
import sys
from typing import cast

from gpu_fault.admin.site import SiteConfigError, load_site

DEPLOY_HOST_STATE_BINDING = "gpu-fault-managed-state-dir.json"
_MAX_BINDING_BYTES = 16 * 1024


def _canonical(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SiteConfigError("deploy-host state-dir binding path is invalid") from exc


def _binding_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate binding field")
        value[key] = item
    return value


def bound_deploy_host_state_dir(prefix: Path | None = None) -> Path | None:
    """Return a validated binding; only actual absence means an unbound host."""

    try:
        directory = (prefix or Path(sys.prefix)).expanduser()
        try:
            directory.lstat()
        except FileNotFoundError:
            return None
        binding = directory.resolve(strict=True) / DEPLOY_HOST_STATE_BINDING
        try:
            descriptor = os.open(binding, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or not info.st_mode & stat.S_IRUSR
            ):
                raise ValueError("binding must be a private readable regular file")
            raw = stream.read(_MAX_BINDING_BYTES + 1)
            after = os.fstat(stream.fileno())
            if (
                len(raw) > _MAX_BINDING_BYTES
                or info.st_size != after.st_size
                or info.st_mtime_ns != after.st_mtime_ns
                or info.st_ctime_ns != after.st_ctime_ns
                or binding.lstat() != after
            ):
                raise ValueError("binding changed while being read")
        value = json.loads(raw, object_pairs_hook=_binding_fields)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SiteConfigError("deploy-host state-dir binding is invalid") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or set(value) - {"schema_version", "state_dir"}
    ):
        raise SiteConfigError("deploy-host state-dir binding schema is invalid")
    state_dir = value.get("state_dir")
    if not isinstance(state_dir, str) or not state_dir.strip():
        raise SiteConfigError("deploy-host state-dir binding has no state directory")
    try:
        path = Path(state_dir).expanduser()
        if not path.is_absolute() or "\0" in state_dir:
            raise ValueError("state directory must be absolute")
    except (RuntimeError, ValueError) as exc:
        raise SiteConfigError("deploy-host state-dir binding path is invalid") from exc
    return _canonical(path)


def bound_state_dir(prefix: Path | None = None) -> Path | None:
    """Compatible short name for the deploy-host binding reader."""

    if prefix is None:
        return bound_deploy_host_state_dir()
    return bound_deploy_host_state_dir(prefix)


def site_bound_admin(state_dir: Path) -> Path | None:
    """Locate a site's bound CLI without treating a broken installation as absent."""

    state_dir = _canonical(state_dir)
    venv = state_dir / "deployer-venv"
    try:
        venv.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SiteConfigError("site deploy-host binding is unreadable") from exc
    bound = bound_deploy_host_state_dir(venv)
    if bound != state_dir:
        raise SiteConfigError(
            "site deploy-host binding is missing or targets another state"
        )
    admin = venv / "bin" / "gpu-fault-admin"
    try:
        if not stat.S_ISREG(admin.stat().st_mode) or not os.access(admin, os.X_OK):
            raise ValueError("bound CLI is not executable")
    except (OSError, ValueError) as exc:
        raise SiteConfigError("site deploy-host CLI is unavailable") from exc
    return admin


def is_readonly_command(
    arguments: argparse.Namespace, *, readonly_commands: Collection[str] = ()
) -> bool:
    command = str(getattr(arguments, "command", "") or "")
    if command == "config":
        action = getattr(arguments, "config_action", None)
        if action == "spare":
            return not getattr(arguments, "declare", False) and not getattr(
                arguments, "release", False
            )
        return action is None and getattr(arguments, "dry_run", False) is True
    if command == "workflow-reconcile":
        return getattr(arguments, "dry_run", False) is True
    if command == "submit-remediation":
        return getattr(arguments, "plan", False) is True
    if command == "failure-domain-map":
        return getattr(arguments, "output", None) is None
    return command in readonly_commands


def enforce_deploy_host_state_dir(
    arguments: argparse.Namespace, *, readonly_commands: Collection[str] = ()
) -> None:
    bound = bound_state_dir()
    provided = cast(Path | None, getattr(arguments, "state_dir", None))
    explicit = cast(Path | None, getattr(arguments, "file", None))
    command = str(getattr(arguments, "command", "") or "")
    state_dir = _canonical(provided) if provided is not None else None
    source = _canonical(explicit) if explicit is not None else None
    if bound is not None and (
        (state_dir is not None and state_dir != bound)
        or (source is not None and source != bound / "site.yaml")
    ):
        raise SiteConfigError(
            f"installed deploy-host is bound to --state-dir {bound}; "
            "refusing a different managed state or site file"
        )
    if state_dir is None and source is None:
        if bound is not None:
            raise SiteConfigError(
                f"installed deploy-host is bound to --state-dir {bound}; "
                f"{command} requires that managed state"
            )
        return
    if state_dir is None:
        assert source is not None
        state_dir = source.parent
    if source is None:
        source = _canonical(state_dir / "site.yaml")
    if provided is not None and source != state_dir / "site.yaml":
        raise SiteConfigError("--state-dir and the canonical site file conflict")
    admin = site_bound_admin(state_dir)
    if admin is not None and source != state_dir / "site.yaml":
        raise SiteConfigError(
            "site deploy-host binding requires its canonical site.yaml"
        )
    source_deploy = (
        command == "deploy"
        and explicit is None
        and not getattr(arguments, "rollback", False)
        and not getattr(arguments, "prepared_source_release", False)
    )
    readonly = is_readonly_command(arguments, readonly_commands=readonly_commands)
    if bound is None and admin is not None and not source_deploy and not readonly:
        raise SiteConfigError(
            f"{state_dir} has its own deploy-host CLI; run "
            f"{admin} {command} ... instead of this checkout's gpu-fault-admin"
        )
    repository_root = cast(Path | None, getattr(arguments, "repo_root", None))
    if (
        (bound is not None or admin is not None)
        and repository_root is not None
        and not source_deploy
        and not (
            command == "deploy"
            and explicit is None
            and not getattr(arguments, "rollback", False)
            and getattr(arguments, "prepared_source_release", False)
        )
        and _canonical(repository_root) != load_site(source).repository_root
    ):
        raise SiteConfigError(
            "deploy-host binding refuses a different site repository root"
        )
