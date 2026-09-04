from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from gpu_fault.admin.site import load_site
from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    hydrate_previous_snapshot,
)

RELEASE_STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
RELEASE_STATE_NOT_FOUND_PATTERN = re.compile(
    r'^Error from server \(NotFound\): configmaps? "'
    + re.escape(RELEASE_STATE_CONFIG_MAP)
    + r'" not found$'
)


class ReleaseStateReadError(RuntimeError):
    pass


class ReleaseStateNotFound(ReleaseStateReadError):
    pass


def read_live_release_state(
    site_file: Path,
    *,
    runner=subprocess.run,
) -> dict[str, Any]:
    site = load_site(site_file)
    completed = runner(
        [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "configmap",
            RELEASE_STATE_CONFIG_MAP,
            "-o",
            "json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        error = completed.stderr.strip()
        if RELEASE_STATE_NOT_FOUND_PATTERN.fullmatch(error):
            raise ReleaseStateNotFound(f"{RELEASE_STATE_CONFIG_MAP} does not exist")
        raise ReleaseStateReadError(error or f"cannot read {RELEASE_STATE_CONFIG_MAP}")
    try:
        config_map = json.loads(completed.stdout)
        value = json.loads(config_map["data"]["state.json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReleaseStateReadError("regional release state is invalid") from exc
    if not isinstance(value, dict):
        raise ReleaseStateReadError("regional release state must be a mapping")

    def read_snapshot_config_map(name: str) -> dict[str, Any]:
        snapshot = runner(
            [
                "kubectl",
                "--kubeconfig",
                str(site.release_config["cpu_kubeconfig"]),
                "-n",
                str(site.release_config["namespace"]),
                "get",
                "configmap",
                name,
                "-o",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if snapshot.returncode:
            raise ReleaseStateReadError(
                snapshot.stderr.strip()
                or f"cannot read previous snapshot ConfigMap {name}"
            )
        try:
            document = json.loads(snapshot.stdout)
        except json.JSONDecodeError as exc:
            raise ReleaseStateReadError(
                f"previous snapshot ConfigMap {name} is invalid"
            ) from exc
        if not isinstance(document, dict):
            raise ReleaseStateReadError(
                f"previous snapshot ConfigMap {name} is invalid"
            )
        return document

    try:
        return hydrate_previous_snapshot(value, read_snapshot_config_map)
    except ReleaseStateSnapshotError as exc:
        raise ReleaseStateReadError(str(exc)) from exc
