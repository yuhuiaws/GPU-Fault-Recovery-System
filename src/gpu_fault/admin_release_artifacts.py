from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_fault.admin_bootstrap_common import (
    DEFAULT_DCGM_IMAGE,
    DEFAULT_NODE_INSTALLER_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    BootstrapError,
    CommandRunner,
    compute_agent_config_digest,
)


DEFAULT_ADOT_IMAGE_AMD64 = (
    "public.ecr.aws/aws-observability/aws-otel-collector@"
    "sha256:bb72328152c72fb9662056759b275f7cc85e115db12bbb114fbea9f68dc4816c"
)


def load_prebuilt_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    runtime_profile: str,
) -> dict[str, Any]:
    manifest = repository_root / "dist/current-release.json"
    if runner.dry_run:
        return {
            "manifest": str(manifest),
            "images": {
                "runtime": DEFAULT_RUNTIME_IMAGE,
                "node_installer": DEFAULT_NODE_INSTALLER_IMAGE,
                "dcgm_exporter": DEFAULT_DCGM_IMAGE,
                "adot": DEFAULT_ADOT_IMAGE_AMD64,
            },
            "agent_config_digest": "0" * 64,
        }
    if not manifest.is_file():
        raise BootstrapError(
            "dist/current-release.json is missing; build and sign the release in CI"
        )
    try:
        release = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"release manifest is invalid: {exc}") from exc
    if (
        int(release.get("schema_version", 0)) < 3
        or release.get("deployable") is not True
    ):
        raise BootstrapError(
            "legacy foundation bootstrap requires a deployable schema v3 release"
        )
    images = dict((release.get("delivery") or {}).get("images") or {})
    required_images = {
        name: str((images.get(name) or {}).get("reference") or "")
        for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
    }
    if any("@sha256:" not in value for value in required_images.values()):
        raise BootstrapError("release manifest image identity is incomplete")
    return {
        "manifest": str(manifest),
        "images": required_images,
        "agent_config_digest": compute_agent_config_digest(
            runner,
            repository_root=repository_root,
            runtime_profile_version=runtime_profile,
        ),
    }
