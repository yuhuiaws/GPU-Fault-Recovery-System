from __future__ import annotations

import json
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    cpu_ingress_deployment_installed,
    exec_cpu_ingress_probe,
)

# What the in-Pod probe reports when nothing can be running: a first bootstrap
# has no control plane yet, so no destructive workflow can be active either.
NOT_INSTALLED_SNAPSHOT: dict[str, Any] = {
    "blocker_count": 0,
    "blockers": [],
    "resolved_blocked_count": 0,
    "resolved_blocked": [],
    "control_plane": "not installed",
}


def workflow_safety_snapshot(release: Any) -> dict[str, Any]:
    if not cpu_ingress_deployment_installed(release):
        return dict(NOT_INSTALLED_SNAPSHOT)
    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("workflow_safety"),
        failure="the workflow safety check",
        sensitive=True,
        interactive=False,
    )
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError("workflow safety check returned invalid evidence") from exc
    if not isinstance(result, dict):
        raise ReleaseError("workflow safety check returned non-object evidence")
    if int(result.get("blocker_count") or 0):
        raise ReleaseError(
            "active destructive workflows block release: "
            + json.dumps(result, sort_keys=True)
        )
    return result
