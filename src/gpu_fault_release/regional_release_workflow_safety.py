from __future__ import annotations

import json
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    exec_cpu_ingress_probe,
)


# The one blocker class the operator can clear without a node: a record of a
# node HyperPod replaced (BLOCKED never-changed, or PENDING never dispatched)
# is closed by the audited administrator reconcile, never by a row edit.
DEPARTED_NODE_REMEDY = (
    "; a blocker whose node left Kubernetes and HyperPod (spot replacement) is "
    "closed with gpu-fault-admin workflow-reconcile --state-dir <state-dir> "
    "--dry-run, then --reference <change>; see administrator operations 10.2"
)


def workflow_safety_snapshot(release: Any) -> dict[str, Any]:
    # Missing/scaled-down Pods do not prove the durable Store is empty.
    # First bootstrap uses the separate, identity-bound database proof.
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
            + DEPARTED_NODE_REMEDY
        )
    return result
