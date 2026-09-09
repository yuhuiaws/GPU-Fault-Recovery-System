"""The administrator's editable ``admin-config.yaml`` and the first-deploy import.

The YAML ``spec`` is the schema in ``config`` spelled in camelCase; reading it
is ``config_patch.apply_patch`` and writing it is ``config.admin_config_spec``,
so this module only knows the document envelope and the file permissions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.config import (
    ADMIN_CONFIG_API_VERSION,
    ADMIN_CONFIG_KIND,
    AdminConfig,
    AdminConfigError,
    admin_config_desired_path,
    admin_config_spec,
    admin_config_write_lock,
    default_admin_config,
    load_desired_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.config_parser import mapping_field
from gpu_fault.admin.config_patch import apply_patch

ADMIN_CONFIG_EDITABLE = Path("admin-config.yaml")


def load_admin_config_file(
    path: Path,
    *,
    base: AdminConfig | None = None,
    require_private: bool = True,
) -> AdminConfig:
    source = path.expanduser().resolve()
    try:
        if require_private and source.stat().st_mode & 0o077:
            raise AdminConfigError(
                f"admin config must not grant group/other permissions: {source}"
            )
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except AdminConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise AdminConfigError(f"cannot read admin config {source}: {exc}") from exc
    data = mapping_field(
        document,
        "admin config file",
        allowed={"apiVersion", "kind", "spec"},
    )
    if data.get("apiVersion") != ADMIN_CONFIG_API_VERSION:
        raise AdminConfigError(
            f"admin config apiVersion must be {ADMIN_CONFIG_API_VERSION}"
        )
    if data.get("kind") != ADMIN_CONFIG_KIND:
        raise AdminConfigError(f"admin config kind must be {ADMIN_CONFIG_KIND}")
    return apply_patch(base or default_admin_config(), data.get("spec") or {})


def admin_config_file_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_EDITABLE


def admin_config_file_document(config: AdminConfig) -> dict[str, object]:
    return {
        "apiVersion": ADMIN_CONFIG_API_VERSION,
        "kind": ADMIN_CONFIG_KIND,
        "spec": admin_config_spec(config),
    }


def write_admin_config_file(
    path: Path,
    config: AdminConfig,
    *,
    overwrite: bool,
) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                admin_config_file_document(config),
                handle,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def initialize_desired_admin_config(
    state_dir: Path,
    *,
    config_file: Path | None = None,
    permit_change: bool = False,
) -> AdminConfig:
    with admin_config_write_lock(state_dir):
        permit_change = (
            permit_change
            and not (state_dir.expanduser().resolve() / "site.yaml").is_file()
        )
        desired_path = admin_config_desired_path(state_dir)
        editable = admin_config_file_path(state_dir)
        current = load_desired_admin_config(
            state_dir,
            migrate_legacy=True,
        )
        source_file = config_file
        if source_file is None and permit_change and editable.is_file():
            source_file = editable
        desired = (
            load_admin_config_file(source_file, base=current)
            if source_file is not None
            else current
        )
        if desired != current and not permit_change:
            raise AdminConfigError(
                "existing site admin config differs from --config; use "
                f"gpu-fault-admin config --state-dir {state_dir}"
            )
        if not desired_path.is_file() or desired != current:
            persist_desired_admin_config(
                state_dir,
                config=desired,
                source=(
                    f"file:{source_file.expanduser().resolve()}"
                    if source_file is not None
                    else "release-defaults"
                ),
            )
        write_admin_config_file(
            editable,
            desired,
            overwrite=(source_file is not None and source_file != editable),
        )
        return desired
