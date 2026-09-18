"""Read-only local installation records, shared without deploy-host dependencies."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

INSTALLATION_FILE = "installation-lifecycle.json"
RETIREMENT_FILE = "installation-retirement.json"


class InstallationLifecycleError(ValueError):
    pass


def content_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise InstallationLifecycleError(
            "installation lifecycle record is missing or unsafe"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise InstallationLifecycleError(
            "installation lifecycle record is unreadable"
        ) from None
    if not isinstance(value, dict):
        raise InstallationLifecycleError(
            "installation lifecycle record is not an object"
        )
    return value


def site_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in ("site_name", "aws_region", "cpu_eks_arn", "namespace")
    }


def require_installation(value: Mapping[str, Any]) -> str:
    installation_id = value.get("installation_id")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or not isinstance(installation_id, str)
        or re.fullmatch(r"[a-f0-9]{32}", installation_id) is None
    ):
        raise InstallationLifecycleError("installation identity is incomplete")
    return installation_id


def require_retirement_complete(state_dir: Path) -> None:
    path = state_dir / RETIREMENT_FILE
    if path.exists() and read_record(path).get("phase") != "COMPLETED":
        raise InstallationLifecycleError(
            "installation retirement is incomplete; resume deploy"
        )


def generation_registry_site_id(site_name: str, installation_id: str) -> str:
    if re.fullmatch(r"[a-f0-9]{32}", installation_id) is None:
        raise InstallationLifecycleError("installation identity is incomplete")
    return f"{site_name}:installation:{installation_id}"


def require_registry_site_id(value: Mapping[str, Any], site_name: str) -> str:
    installation_id = require_installation(value)
    registry_site_id = value.get("registry_site_id")
    if registry_site_id is None and "retained_uninstall" not in value:
        # Enrollment of an existing installation must not hide its legacy rows.
        return site_name
    if (
        not isinstance(registry_site_id, str)
        or registry_site_id
        not in {
            site_name,
            generation_registry_site_id(site_name, installation_id),
        }
        or ("retained_uninstall" in value and registry_site_id == site_name)
    ):
        raise InstallationLifecycleError("installation registry scope is unbound")
    return str(registry_site_id)


def registry_site_id(source: Path, config: Mapping[str, Any]) -> str:
    return release_lifecycle_inputs(source, config).get(
        "registry_site_id", str(config["site_name"])
    )


def release_lifecycle_inputs(source: Path, config: Mapping[str, Any]) -> dict[str, str]:
    state_dir = source.parent
    require_retirement_complete(state_dir)
    path = state_dir / INSTALLATION_FILE
    if not path.exists():
        if path.is_symlink() or (state_dir / RETIREMENT_FILE).exists():
            raise InstallationLifecycleError("current installation record is missing")
        return {}
    value = read_record(path)
    identity = value.get("site_identity")
    if identity is not None and identity != site_identity(config):
        raise InstallationLifecycleError("release installation belongs to another site")
    result = {
        "installation_id": require_installation(value),
        "registry_site_id": require_registry_site_id(value, str(config["site_name"])),
    }
    retirement_path = state_dir / RETIREMENT_FILE
    if retirement_path.exists():
        previous = read_record(retirement_path).get("next_installation")
        if not isinstance(previous, dict) or any(
            previous.get(key) != result[key]
            for key in ("installation_id", "registry_site_id")
        ):
            raise InstallationLifecycleError(
                "current installation registry scope drifted"
            )
    if value.get("retained_uninstall") is not None:
        result["retained_database_handoff"] = str(path.resolve())
    return result
