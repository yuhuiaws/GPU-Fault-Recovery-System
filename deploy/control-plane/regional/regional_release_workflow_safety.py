from __future__ import annotations

import json
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError
from regional_release_probes import probe_source
from regional_release_runtime_identity import CONTROL_PLANE_PYTHON


def workflow_safety_snapshot(release: Any) -> dict[str, Any]:
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("no Running CPU ingress Pod")
    raw = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            pod,
            "--",
            CONTROL_PLANE_PYTHON,
            "-c",
            probe_source("workflow_safety"),
        ),
        capture=True,
        sensitive=True,
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
