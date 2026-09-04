from __future__ import annotations

import json
import subprocess
from typing import Any

from gpu_fault.admin.site import RenderedSite, SiteConfigError
from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    hydrate_previous_snapshot,
)

REGIONAL_RELEASE_STATE_CONFIG_MAP = "gpu-fault-regional-release-state"


def live_release_state(site: RenderedSite) -> dict[str, Any]:
    kubeconfig = str(site.release_config["cpu_kubeconfig"])
    namespace = str(site.release_config["namespace"])

    def read_config_map(name: str) -> dict[str, Any]:
        completed = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kubeconfig,
                "-n",
                namespace,
                "get",
                "configmap",
                name,
                "-o",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode:
            raise SiteConfigError(f"cannot read live ConfigMap {name}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SiteConfigError(f"live ConfigMap {name} is invalid") from exc
        if not isinstance(value, dict):
            raise SiteConfigError(f"live ConfigMap {name} is invalid")
        return value

    document = read_config_map(REGIONAL_RELEASE_STATE_CONFIG_MAP)
    try:
        raw = (document.get("data") or {})["state.json"]
        state = json.loads(raw)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SiteConfigError("live regional release state is invalid") from exc
    if not isinstance(state, dict):
        raise SiteConfigError("live regional release state must be an object")
    try:
        return hydrate_previous_snapshot(state, read_config_map)
    except ReleaseStateSnapshotError as exc:
        raise SiteConfigError(
            "live regional release previous snapshot is invalid"
        ) from exc
