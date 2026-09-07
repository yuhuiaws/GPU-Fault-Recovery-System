"""Fleet deployment operations, executed inside a running control-plane Pod.

Request: ``{"operation": "<name>", ...}`` on stdin.
Response: one JSON object on stdout.
"""

import json
import sys
from typing import Any

from gpu_fault import __version__
from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import FleetDeploymentRequest
from gpu_fault.policy import load_xid_policy
from gpu_fault.store import NotFoundError

TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED"})


def stranded_release_rollouts(store: Any, release_id: str) -> list[str]:
    """Ids of this release's non-terminal rollouts that a rollback supersedes.

    A rollback holds the release lock, so no rollout of the same release can
    legitimately still be in flight. Anything non-terminal is a record the
    upgrade abandoned -- and a ``PLANNED`` record is not inert: the destructive
    workflow fence in ``execution/fleet_preflight.py`` treats any non-terminal
    deployment for the cluster as an in-flight rollout and holds every
    destructive remediation behind it, indefinitely.

    Matching is by release id rather than by re-deriving the deployment id from
    the candidate digests. The digest-keyed lookup this replaces silently
    resolved to ``ABSENT`` whenever any input to the id hash had moved since
    the upgrade minted it.
    """

    stranded = []
    for deployment in store.list_fleet_deployments():
        head, _, _identity = str(deployment.deployment_id).rpartition("-")
        if not head.startswith("release-") or not head.endswith(f"-{release_id}"):
            continue
        phase = head[len("release-") : -len(release_id) - 1]
        if phase == "rollback":
            continue
        if str(deployment.status) in TERMINAL_STATUSES:
            continue
        stranded.append(str(deployment.deployment_id))
    return sorted(stranded)


def active_cluster_rollouts(store: Any, cluster_id: str) -> list[str]:
    """Ids of every non-terminal rollout of ``cluster_id``, whatever its release.

    Called by the release engine right before it creates or resumes its own
    fleet deployment (which the caller names and keeps). The engine holds the
    release lock, so no other rollout of any release can legitimately still be
    in flight for the cluster: whatever else is non-terminal was abandoned by a
    transaction that died or was superseded without a rollback. The by-release
    sweep above cannot see those -- on 2026-09-07 two ``IN_PROGRESS`` records
    from two abandoned transactions (one of a different release id) fenced every
    destructive remediation for the cluster for four hours after the successor
    had rolled out cleanly, until they were cancelled by hand.
    """

    return sorted(
        str(deployment.deployment_id)
        for deployment in store.list_active_fleet_deployments(cluster_id)
    )


def main() -> int:
    payload = json.load(sys.stdin)
    context = ApplicationContext.from_environment()
    registry = context.fleet_registry
    if registry is None:
        raise RuntimeError("fleet registry is disabled")
    operation = payload["operation"]
    # The branches return different models -- FleetDeployment for most,
    # DeploymentWaveLease for next-wave -- and the caller only reads the JSON.
    result: Any
    if operation == "create":
        request = dict(payload["request"])
        request.setdefault("desired_agent_version", __version__)
        request.setdefault("desired_policy_version", load_xid_policy().mapping_version)
        result = registry.create_deployment(
            FleetDeploymentRequest.model_validate(request)
        )
    elif operation == "get":
        result = context.store.get_fleet_deployment(payload["deployment_id"])
    elif operation == "next-wave":
        result = registry.start_next_wave(payload["deployment_id"])
    elif operation == "retry-failed":
        result = registry.retry_failed_deployment(payload["deployment_id"])
    elif operation == "normalize-records":
        deployments = context.store.list_fleet_deployments()
        for deployment in deployments:
            context.store.save_fleet_deployment(deployment)
        print(json.dumps({"normalized": len(deployments)}, sort_keys=True))
        return 0
    elif operation == "cancel-if-present":
        try:
            result = registry.cancel_deployment(
                payload["deployment_id"],
                reason=payload["reason"],
            )
        except NotFoundError:
            print(json.dumps({"status": "ABSENT"}))
            return 0
    elif operation == "terminalize-release-rollouts":
        stranded = stranded_release_rollouts(context.store, payload["release_id"])
        for deployment_id in stranded:
            registry.cancel_deployment(deployment_id, reason=payload["reason"])
        print(json.dumps({"terminalized": stranded}, sort_keys=True))
        return 0
    elif operation == "terminalize-cluster-rollouts":
        stranded = [
            deployment_id
            for deployment_id in active_cluster_rollouts(
                context.store, payload["cluster_id"]
            )
            if deployment_id != payload.get("keep_deployment_id")
        ]
        for deployment_id in stranded:
            registry.cancel_deployment(deployment_id, reason=payload["reason"])
        print(json.dumps({"terminalized": stranded}, sort_keys=True))
        return 0
    else:
        raise ValueError(f"unsupported fleet operation: {operation}")
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
