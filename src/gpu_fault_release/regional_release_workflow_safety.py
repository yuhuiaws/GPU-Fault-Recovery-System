from __future__ import annotations

import json
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    exec_cpu_ingress_probe,
)
from gpu_fault_release.regional_verify_scope import gating_count, listed


# The one blocker class the operator can clear without a node: a record of a
# node its provider replaced or removed (BLOCKED never-changed, or PENDING never
# dispatched) is closed by the audited administrator reconcile, never by a row
# edit.
DEPARTED_NODE_REMEDY = (
    "; a blocker whose node left Kubernetes and its provider (the HyperPod node "
    "list or EC2; a spot replacement, a scale-in) is closed with gpu-fault-admin "
    "workflow-reconcile --state-dir <state-dir> --dry-run, then --reference "
    "<change>; see administrator operations 10.2"
)


def workflow_safety_snapshot(release: Any) -> dict[str, Any]:
    """The blockers' evidence; raises while one of them blocks this release.

    Every blocker gates a deploy, a ``config`` and every unscoped call. The
    candidate preflight of a ``join-cluster`` is scoped to the joining
    cluster(s) (``regional_verify_scope``): another site cluster's blockers --
    the record the product opens for a node a spot reclaim took, a
    remediation in progress there -- are findings, because nothing the join
    changes can drop work in flight on another cluster: its registry
    revisions change the joining cluster's registration only and are ACKed by
    CPU processes, its waves restart the joining cluster's Agents only, the
    AMP rules are alerting only, and the control-worker roll of the
    failure-domain map is the lease handover every control-worker restart is
    (the dispatch lease and each workflow's execution lease, the node action
    running in the Agent). Blockers of the joining cluster, of a cluster the
    site does not name, and those the probe could not attribute still gate.
    """
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
    ids = result.get("blockers_by_cluster")
    if gating_count(
        release,
        result,
        check="workflow_safety",
        total=int(result.get("blocker_count") or 0),
        by_cluster=result.get("blocker_count_by_cluster"),
        finding=lambda cluster, count: (
            f"{count} active destructive workflow(s): {listed(ids, cluster)}"
        ),
    ):
        raise ReleaseError(
            "active destructive workflows block release: "
            + json.dumps(result, sort_keys=True)
            + DEPARTED_NODE_REMEDY
        )
    return result
