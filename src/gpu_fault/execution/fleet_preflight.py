from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from gpu_fault.fleet_deployment import DeploymentStatus, FleetDeployment
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    OperationAdapter,
    operations_for_adapter,
)

_PREFLIGHTED_NODE_ACTION_OPERATIONS = operations_for_adapter(
    OperationAdapter.NODE_ACTION
) - {WorkflowOperation.COLLECT_HUNG_TRIAGE}
LOGGER = logging.getLogger(__name__)


def superseded_fence_deployments(
    store: Any,
    cluster_id: str,
    active: Sequence[FleetDeployment],
) -> frozenset[str]:
    """Ids among ``active`` that provably hold nothing and cannot be resumed.

    The fence treats every non-terminal fleet deployment for the cluster as an
    in-flight rollout. That is right for a rollout that is actually running, and
    wrong for a record that was abandoned: ``DeploymentStatus`` has no terminal
    "superseded" value, so an upgrade that created its record and then died
    fences every destructive remediation for the cluster forever. On 2026-09-03 a
    ``PLANNED`` record did exactly that for 36 hours.

    A record is dropped only when *both* of these hold, which together leave no
    reading under which it is still live:

    1. Its status is ``PLANNED``. That is derived state -- ``deployment_status``
       returns it only when every node is still ``PENDING`` -- so no wave has
       ever been started, no node is draining, and nothing is mid-install.
    2. A strictly newer fleet deployment exists for the same cluster. Whatever
       created this record has moved on; a rollout waiting to start its first
       wave is never overtaken by a later deployment of the same cluster.

    Deliberately no staleness threshold. "Old" is not evidence of abandonment --
    a wave can legitimately sit for a long time behind a drain -- and a threshold
    would eventually release the fence on a rollout that is genuinely running.

    Both conditions fail closed. Anything past ``PLANNED``, and anything that is
    the newest record for its cluster, keeps fencing.

    This is the read-side safety net, not the cure. The cure is
    ``supersede_never_started_deployments``, which terminalizes the record when
    the successor is created, so it ages out of the store on the normal retention
    schedule. That matters here: the retention drain only collects
    ``SUCCEEDED``/``FAILED`` records, so if an abandoned record were left for the
    fence to route around, its newer siblings would eventually be collected and
    it would become the newest record for the cluster -- closing the fence again,
    with nothing left to prove it dead.
    """

    planned = {
        str(deployment.deployment_id): deployment.created_at
        for deployment in active
        if deployment.status is DeploymentStatus.PLANNED
    }
    if not planned:
        return frozenset()
    newest = max(
        (
            deployment.created_at
            for deployment in store.list_fleet_deployments()
            if deployment.cluster_id == cluster_id
        ),
        default=None,
    )
    if newest is None:
        return frozenset()
    return frozenset(
        deployment_id
        for deployment_id, created_at in planned.items()
        if created_at < newest
    )


def fleet_rollout_fence_deployment_ids(store: Any, cluster_id: str) -> list[str]:
    """Ids of the rollouts that must fence destructive work on ``cluster_id``.

    The whole fence verdict, in one call against one store. Both enforcement
    points read it through here so neither can drift from the other: the
    control-plane executor holds the store directly, and the data-plane
    executor asks the control plane over
    ``GET /v1/regional/executors/fleet-rollout-fence``, which calls this.
    """

    active = list(store.list_active_fleet_deployments(cluster_id))
    if not active:
        return []
    try:
        superseded = superseded_fence_deployments(store, cluster_id, active)
    except Exception as exc:  # noqa: BLE001 - keep fencing, do not open
        # Losing the supersession evidence is not grounds to release the
        # fence, so this degrades to the pre-supersession behaviour.
        LOGGER.warning(
            "fleet rollout supersession check failed, fence stays closed: "
            "cluster=%s error=%s: %s",
            cluster_id,
            type(exc).__name__,
            exc,
        )
        superseded = frozenset()
    live = [
        str(deployment.deployment_id)
        for deployment in active
        if str(deployment.deployment_id) not in superseded
    ]
    if superseded:
        LOGGER.warning(
            "fleet rollout fence ignored superseded deployment(s): "
            "cluster=%s deployments=%s still_fencing=%d",
            cluster_id,
            ",".join(sorted(superseded)),
            len(live),
        )
    return live


def fleet_rollout_fence(registry: Any) -> Callable[[str], Sequence[str]] | None:
    """Resolve how *this* registry answers the rollout fence, or ``None``.

    A registry that owns a store answers it locally. The regional proxy owns
    no store -- it sets ``store = self`` so shared code can call a handful of
    store-shaped methods on it -- so it answers with a control-plane round
    trip instead, and says so by implementing
    ``fleet_rollout_fence_deployments``.

    Resolving the capability explicitly is the point. Reading
    ``registry.store.list_active_fleet_deployments`` unconditionally is what
    broke the data plane on 2026-09-04: the proxy has no such method, the
    ``AttributeError`` came back as "fence is unavailable", and because this
    fence fails closed every destructive remote command on every regional
    cluster was held forever. A missing capability is a static property of the
    registry, not a transient read failure, and it must not be discovered by
    catching an exception at the mutation boundary.
    """

    remote: Callable[[str], Sequence[str]] | None = getattr(
        registry, "fleet_rollout_fence_deployments", None
    )
    if remote is not None:
        return remote
    store = getattr(registry, "store", None)
    if store is None or store is registry:
        return None
    return lambda cluster_id: fleet_rollout_fence_deployment_ids(store, cluster_id)


def fleet_preflight_reason(
    registry: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> str | None:
    """Explain why a workflow cannot safely enter its destructive phase."""

    completed = set(workflow.completed_step_indexes)
    if any(
        index in completed and step.operation in DESTRUCTIVE_OPERATIONS
        for index, step in enumerate(steps)
    ):
        # Do not strand compensation after a mutation. Remaining node
        # actions still enforce their normal generation/readiness gate.
        return None

    pending = [step for index, step in enumerate(steps) if index not in completed]
    if not any(step.operation in DESTRUCTIVE_OPERATIONS for step in pending):
        return None
    fence = fleet_rollout_fence(registry)
    if fence is not None:
        try:
            fencing = fence(incident.cluster_id)
        except Exception as exc:  # noqa: BLE001 - fail closed before mutation
            return (
                "fleet rollout fence is unavailable before destructive workflow "
                f"steps: {type(exc).__name__}: {exc}"
            )
        if fencing:
            return (
                "fleet rollout fence blocked destructive workflow while the "
                f"cluster has {len(fencing)} active deployment(s)"
            )
    node_action_steps = [
        step
        for step in pending
        if step.operation in _PREFLIGHTED_NODE_ACTION_OPERATIONS
    ]
    if not node_action_steps:
        return None
    node_ids = list(
        dict.fromkeys(
            node_id for step in node_action_steps for node_id in step.node_ids
        )
    )
    if not node_ids:
        return (
            "fleet compatibility preflight blocked destructive workflow: "
            "a pending node action has no explicit node target"
        )
    try:
        report = registry.readiness(incident.cluster_id, node_ids)
    except Exception as exc:  # noqa: BLE001 - fail closed before mutation
        return (
            "fleet compatibility preflight is unavailable before "
            f"destructive workflow steps: {type(exc).__name__}: {exc}"
        )
    if report.ready:
        return None
    reasons = [
        f"{node.node_id}: {reason}" for node in report.nodes for reason in node.reasons
    ]
    return "fleet compatibility preflight blocked destructive workflow: " + (
        "; ".join(reasons) or "fleet readiness returned false"
    )


def command_requires_fleet_preflight(
    operation: WorkflowOperation,
) -> bool:
    """Return true only at a remote destructive-action boundary."""

    return operation in DESTRUCTIVE_OPERATIONS


def held_workflow_result(
    executor: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> Any | None:
    registry = getattr(executor, "fleet_registry", None)
    if registry is None:
        return None
    reason = fleet_preflight_reason(
        registry,
        workflow,
        incident,
        steps,
    )
    if reason is None:
        return None
    LOGGER.warning(
        "workflow held before destructive steps: "
        "workflow=%s incident=%s cluster=%s reason=%s",
        workflow.request_id,
        incident.incident_id,
        incident.cluster_id,
        reason,
    )
    return executor._result(workflow, incident, error=reason)
