"""Fleet deployment operations, executed inside a running control-plane Pod.

Request: ``{"operation": "<name>", ...}`` on stdin.
Response: one JSON object on stdout.

``exclude-departed`` (``{"deployment_id": ..., "node_ids": [...]}``) takes nodes
the engine proved departed (``regional_release_departure_absorption``) out of
a non-terminal rollout record -- out of its ``nodes`` and its ``waves`` -- so the
record ends ``SUCCEEDED`` over the nodes that remain instead of holding the
cluster's destructive-workflow fence open on a node that can never report. Only
a ``PENDING`` or ``INSTALLING`` node is taken out; a ``READY`` one converged
before it left and stays. The Agent's lease is re-read on the registry's clock
at write time, and a node whose Agent holds a live lease is refused: the engine's
proof is older than this write. When the record has no ``READY`` node yet -- the
departed node was the canary -- the first remaining wave that has not started is
split so a single node still goes first. The record keeps the shape every
deployed control plane parses (no new status value, no new field), so a
rollback's older control plane still reads it.
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
EXCLUDABLE_NODE_STATUSES = frozenset({"PENDING", "INSTALLING"})


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


def _status(value: Any) -> str:
    return str(getattr(value, "value", value))


def _keep_canary(waves: list[list[str]], statuses: dict[str, str]) -> list[list[str]]:
    """``waves`` with a one-node first wave again while nothing has converged."""

    if not waves or "READY" in statuses.values() or len(waves[0]) < 2:
        return waves
    if any(statuses.get(node_id) != "PENDING" for node_id in waves[0]):
        return waves
    first = sorted(waves[0])
    return [first[:1], first[1:], *waves[1:]]


def exclude_departed_nodes(registry: Any, store: Any, payload: Any) -> dict[str, Any]:
    """Take ``payload["node_ids"]`` out of the rollout record; see the module."""

    from gpu_fault.fleet_deployment import deployment_status

    deployment_id = str(payload["deployment_id"])
    wanted = sorted({str(node_id) for node_id in payload["node_ids"]})
    for _ in range(5):
        deployment = store.get_fleet_deployment(deployment_id)
        if _status(deployment.status) in TERMINAL_STATUSES:
            raise ValueError(f"deployment {deployment_id} is already terminal")
        by_node = {item.node_id: item for item in deployment.nodes}
        absent = [node_id for node_id in wanted if node_id not in by_node]
        drop = [
            node_id
            for node_id in wanted
            if node_id in by_node
            and _status(by_node[node_id].status) in EXCLUDABLE_NODE_STATUSES
        ]
        kept = [
            node_id for node_id in wanted if node_id in by_node and node_id not in drop
        ]
        now = registry.now()
        live = sorted(
            agent.node_id
            for agent in store.list_agents(deployment.cluster_id)
            if agent.node_id in drop
            and agent.lease_expires_at is not None
            and agent.lease_expires_at > now
            and _status(agent.lifecycle_state) != "REVOKED"
        )
        if live:
            raise ValueError(
                "nodes still hold a live Agent lease and are not departed: "
                + ",".join(live)
            )
        if not drop:
            replacement = deployment
        else:
            nodes = [item for item in deployment.nodes if item.node_id not in drop]
            if not nodes:
                raise ValueError(f"every node of deployment {deployment_id} departed")
            waves = [
                [node_id for node_id in wave if node_id not in drop]
                for wave in deployment.waves
            ]
            waves = _keep_canary(
                [wave for wave in waves if wave],
                {item.node_id: _status(item.status) for item in nodes},
            )
            replacement = deployment.model_copy(
                update={
                    "nodes": nodes,
                    "waves": waves,
                    "status": deployment_status(nodes),
                    "updated_at": now,
                }
            )
            if not store.replace_fleet_deployment_if_matches(replacement, deployment):
                continue
        return {
            "deployment": json.loads(replacement.model_dump_json()),
            "excluded": drop,
            "absent": absent,
            "kept": kept,
        }
    raise ValueError("fleet deployment conflicted with concurrent updates")


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
    elif operation == "exclude-departed":
        print(
            json.dumps(
                exclude_departed_nodes(registry, context.store, payload),
                sort_keys=True,
            )
        )
        return 0
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
