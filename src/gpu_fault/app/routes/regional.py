from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.execution.fleet_preflight import (
    fleet_rollout_fence_deployment_ids,
)
from gpu_fault.fleet import AgentRecord, AgentTransitionRequest
from gpu_fault.hyperpod import HyperPodSubmissionRecord
from gpu_fault.markers import (
    blocking_spare_markers,
    describe_blocking_marker,
)
from gpu_fault.models import (
    AdvisoryNotification,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.regional import (
    RegionalExecutorReadinessReport,
    RegionalExecutorReadinessRequest,
    RemoteActionCommand,
    RemoteAdvisoryNotificationRequest,
    RemoteCommandClaim,
    RemoteCommandClaimRequest,
    RemoteCommandLeaseRenewal,
    RemoteCommandProgress,
    RemoteCommandResult,
    RemoteEvidenceCaptureRequest,
    RemoteFleetRolloutFence,
    RemoteHyperPodSubmissionRequest,
    RemoteHyperPodSubmissionReservation,
    RemoteIncidentOwnershipReport,
    RemoteSpareHealthReport,
    RemoteSpareHealthRequest,
)
from gpu_fault.regional_compatibility import (
    REMOTE_STEP_BATCHING_PROTOCOL_VERSION,
    RegionalExecutorCompatibilityPolicy,
)
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import RawEvidenceRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionalRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor
    auth_registry: Any
    max_unclaimed_seconds: float
    max_claim_age_seconds: float
    executor_compatibility: RegionalExecutorCompatibilityPolicy
    # ``RemoteCommandWakeupHub`` (or None): the long-poll wait behind an empty
    # claim with ``wait_seconds > 0``. None keeps every claim immediate.
    remote_command_wakeups: Any | None = None


def get_regional_dependencies() -> RegionalRouterDependencies:
    raise RuntimeError("regional router dependencies are not configured")


router = APIRouter(prefix="/v1/regional", tags=["regional"])


async def _store_call(
    dependencies: RegionalRouterDependencies,
    function: Callable,
    /,
    *args,
    retry_after: str = "2",
    **kwargs,
):
    try:
        return await dependencies.store_io.run(function, *args, **kwargs)
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": retry_after},
        ) from exc


def _require_cluster(cluster_id: str | None) -> str:
    if not cluster_id:
        raise HTTPException(
            status_code=401,
            detail="X-GPU-Fault-Cluster-ID is required",
        )
    return cluster_id


@router.get("/clusters")
@authorization_bucket("execution-token")
async def list_regional_clusters(
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> list[dict]:
    ctx = dependencies.context
    if (
        not ctx.execution_token
        or not execution_token
        or not secrets.compare_digest(execution_token, ctx.execution_token)
    ):
        raise HTTPException(
            status_code=403,
            detail="invalid workflow execution token",
        )
    registrations = await _store_call(dependencies, ctx.store.list_regional_clusters)
    redacted = []
    for registration in registrations:
        item = registration.model_dump(mode="json")
        digest = item.pop("token_sha256", "") or ""
        item["token_sha256_present"] = bool(digest)
        item["token_sha256_length"] = len(digest)
        # The retiring digest is redacted the same way, but whether a rotation is
        # armed and when its window closes stay visible: that is what an operator
        # needs to decide whether the rotation can be completed.
        retiring = item.pop("retiring_token_sha256", None) or ""
        item["retiring_token_sha256_present"] = bool(retiring)
        item["retiring_token_sha256_length"] = len(retiring)
        redacted.append(item)
    return redacted


@router.post(
    "/executors/readiness",
    response_model=RegionalExecutorReadinessReport,
)
@authorization_bucket("cluster-token")
async def regional_executor_readiness(
    probe: RegionalExecutorReadinessRequest,
    response: Response,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RegionalExecutorReadinessReport:
    cluster_id = _require_cluster(cluster_id)
    health = await _store_call(
        dependencies,
        dependencies.context.store.remote_command_cluster_health,
        cluster_id,
        retry_after="5",
    )
    advertised = set(probe.execution_owners)
    unsupported = sorted(
        owner for owner in health["pending_owner_counts"] if owner not in advertised
    )
    reasons = []
    protocol_reason = dependencies.executor_compatibility.rejection_reason(
        probe.executor_protocol_version,
        probe.executor_artifact_sha256,
        probe.executor_compatibility_digest,
    )
    if protocol_reason is not None:
        reasons.append(protocol_reason)
    if not advertised:
        reasons.append(
            "executor advertised no execution owners, so it can claim nothing"
        )
    if unsupported:
        reasons.append(
            "open backlog needs execution owners this executor does "
            f"not advertise: {', '.join(unsupported)}"
        )
    stalled = [owner for owner in health["pending_owner_counts"] if owner in advertised]
    oldest = float(health["oldest_unclaimed_age_seconds"])
    if stalled and oldest >= dependencies.max_unclaimed_seconds:
        reasons.append(
            "a command this executor advertises has been unclaimed "
            f"for {oldest:.0f}s (limit "
            f"{dependencies.max_unclaimed_seconds:.0f}s)"
        )
    age = probe.last_successful_claim_age_seconds
    if age is not None and age > dependencies.max_claim_age_seconds:
        reasons.append(
            f"last successful claim was {age:.0f}s ago (limit "
            f"{dependencies.max_claim_age_seconds:.0f}s)"
        )
    report = RegionalExecutorReadinessReport(
        ready=not reasons,
        cluster_id=cluster_id,
        executor_id=probe.executor_id,
        executor_protocol_version=probe.executor_protocol_version,
        executor_artifact_sha256=probe.executor_artifact_sha256,
        executor_compatibility_digest=probe.executor_compatibility_digest,
        registered=True,
        execution_owners=sorted(advertised),
        unsupported_execution_owners=unsupported,
        open_commands=int(health["open_total"]),
        pending_commands=int(health["pending_total"]),
        oldest_unclaimed_age_seconds=oldest,
        last_successful_claim_age_seconds=age,
        reasons=reasons,
    )
    if reasons:
        LOGGER.warning(
            "executor %s in cluster %s is not ready: %s",
            probe.executor_id,
            cluster_id,
            "; ".join(reasons),
        )
        response.status_code = 503
    return report


@router.post("/executors/claim", response_model=RemoteCommandClaim)
@authorization_bucket("cluster-token")
async def claim_remote_commands(
    claim: RemoteCommandClaimRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteCommandClaim:
    cluster_id = _require_cluster(cluster_id)
    protocol_reason = dependencies.executor_compatibility.rejection_reason(
        claim.executor_protocol_version,
        claim.executor_artifact_sha256,
        claim.executor_compatibility_digest,
    )
    if protocol_reason is not None:
        raise HTTPException(
            status_code=503,
            detail=protocol_reason,
            headers={"Retry-After": "5"},
        )
    owners = set(claim.execution_owners) or {"gpu-fault-kubernetes-adapter"}
    # The mint-time gate (``RemoteStepBatchingPolicy``) should already keep
    # compound commands out of a cluster that still admits older executors;
    # this is the hard stop for a row minted before the pins were widened
    # again: an executor that cannot decode ``batched_steps`` never sees one.
    accept_batched_steps = (
        claim.executor_protocol_version >= REMOTE_STEP_BATCHING_PROTOCOL_VERSION
    )

    async def claim_once() -> list[RemoteActionCommand]:
        commands: list[RemoteActionCommand] = await _store_call(
            dependencies,
            dependencies.context.store.claim_remote_commands,
            cluster_id,
            claim.executor_id,
            limit=claim.max_commands,
            lease_seconds=claim.lease_seconds,
            execution_owners=owners,
            accept_batched_steps=accept_batched_steps,
        )
        return commands

    hub = dependencies.remote_command_wakeups
    if claim.wait_seconds <= 0 or hub is None:
        return RemoteCommandClaim(commands=await claim_once())
    # Long-poll. Subscribe before the first claim so a command written between
    # that claim and the wait still wakes this request. The wait holds no
    # store I/O thread; the hub bounds it (server cap, and the old poll
    # interval while its listener is not connected). A wait that timed out or
    # lost its listener still claims again: a wakeup is a hint, and a hint
    # that never came must cost one wait, not a command. A waiter the hub did
    # not admit (per-cluster cap) answers with the first claim at once.
    async with hub.subscribe(cluster_id) as waiter:
        commands = await claim_once()
        if commands or not waiter.admitted:
            return RemoteCommandClaim(commands=commands)
        await waiter.wait(claim.wait_seconds)
        return RemoteCommandClaim(commands=await claim_once())


@router.post(
    "/executors/{command_id}/renew",
    response_model=RemoteActionCommand,
)
@authorization_bucket("cluster-token")
async def renew_remote_command(
    command_id: str,
    renewal: RemoteCommandLeaseRenewal,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteActionCommand:
    return await _store_call(
        dependencies,
        dependencies.context.store.renew_remote_command_lease,
        _require_cluster(cluster_id),
        command_id,
        renewal.executor_id,
        renewal.lease_token,
        lease_seconds=renewal.lease_seconds,
    )


@router.post(
    "/executors/{command_id}/progress",
    response_model=RemoteActionCommand,
)
@authorization_bucket("cluster-token")
async def report_remote_command_progress(
    command_id: str,
    progress: RemoteCommandProgress,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteActionCommand:
    """Per-step results of a compound command, before its terminal result.

    A separate route rather than a field on the renewal: the renewal runs on
    the executor's heartbeat thread at its own cadence, while progress is
    posted by the executing thread the moment a step settles, and the terminal
    ``/result`` stays exactly what it was. Lease-fenced like the renewal.
    """

    ctx = dependencies.context
    command: RemoteActionCommand = await _store_call(
        dependencies,
        ctx.store.record_remote_command_progress,
        _require_cluster(cluster_id),
        command_id,
        progress.executor_id,
        progress.lease_token,
        batched_results={
            index: result.model_dump(mode="json")
            for index, result in progress.batched_results.items()
        },
    )
    # A settled step lets the control-plane loop mark it and dispatch the next
    # covered one (which then waits on this same command); no need to wait
    # for the poll interval to notice.
    ctx.dispatcher.wake()
    return command


@router.post(
    "/executors/{command_id}/result",
    response_model=RemoteActionCommand,
)
@authorization_bucket("cluster-token")
async def complete_remote_command(
    command_id: str,
    result: RemoteCommandResult,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteActionCommand:
    ctx = dependencies.context
    command = await _store_call(
        dependencies,
        ctx.store.complete_remote_command,
        _require_cluster(cluster_id),
        command_id,
        result,
    )
    await _store_call(
        dependencies,
        ctx.advisory_notifications.dispatch_remote_completion,
        command,
    )
    ctx.dispatcher.wake()
    return command


def _registered_submission(
    dependencies: RegionalRouterDependencies,
    cluster_id: str | None,
    cluster_name: str,
):
    registration = dependencies.auth_registry.get(cluster_id or "")
    if registration is None or cluster_name != registration.hyperpod_cluster_name:
        raise HTTPException(
            status_code=403,
            detail=(
                "submission cluster_name does not match the "
                "registered HyperPod cluster for this cluster_id"
            ),
        )
    return registration


@router.post(
    "/executors/hyperpod-submissions/reserve",
    response_model=RemoteHyperPodSubmissionReservation,
)
@authorization_bucket("cluster-token")
async def reserve_remote_hyperpod_submission(
    request: RemoteHyperPodSubmissionRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteHyperPodSubmissionReservation:
    if cluster_id != request.cluster_id:
        raise HTTPException(
            status_code=403,
            detail="authenticated cluster does not match request",
        )
    _registered_submission(dependencies, cluster_id, request.record.cluster_name)
    record, reserved = await _store_call(
        dependencies,
        dependencies.context.store.reserve_hyperpod_submission,
        request.record,
    )
    return RemoteHyperPodSubmissionReservation(reserved=reserved, record=record)


@router.get(
    "/executors/hyperpod-submissions",
    response_model=HyperPodSubmissionRecord | None,
)
@authorization_bucket("cluster-token")
async def get_remote_hyperpod_submission(
    cluster_name: str,
    idempotency_key: str,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> HyperPodSubmissionRecord | None:
    _registered_submission(dependencies, cluster_id, cluster_name)
    try:
        return await _store_call(
            dependencies,
            dependencies.context.store.get_hyperpod_submission,
            cluster_name,
            idempotency_key,
        )
    except NotFoundError:
        return None


@router.get(
    "/executors/incident-ownership",
    response_model=RemoteIncidentOwnershipReport,
)
@authorization_bucket("cluster-token")
async def get_remote_incident_ownership(
    incident_id: str,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteIncidentOwnershipReport:
    cluster_id = _require_cluster(cluster_id)

    def lookup() -> RemoteIncidentOwnershipReport:
        unknown = RemoteIncidentOwnershipReport(incident_id=incident_id)
        try:
            incident = dependencies.context.store.get_incident(incident_id)
        except (NotFoundError, KeyError):
            return unknown
        if incident.cluster_id != cluster_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    "incident belongs to another cluster than the authenticated one"
                ),
            )
        if not incident.workflow_request_id:
            return unknown
        try:
            workflow = dependencies.context.store.get_workflow(
                incident.workflow_request_id
            )
        except (NotFoundError, KeyError):
            return RemoteIncidentOwnershipReport(
                incident_id=incident_id,
                known=False,
                workflow_request_id=incident.workflow_request_id,
            )
        terminal = workflow.status in {
            WorkflowStatus.BLOCKED,
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.SUPERSEDED,
        }
        quarantine_hold = (
            incident.state is IncidentState.QUARANTINED
            or (
                WorkflowOperation.QUARANTINE in workflow.completed_operations
                and WorkflowOperation.RESTORE_SCHEDULING
                not in workflow.completed_operations
            )
            or any(
                execution.operation is WorkflowOperation.REPLACE_NODE
                and execution.details.get("action") == "SPARE_FAILOVER"
                for execution in workflow.step_executions
            )
        )
        return RemoteIncidentOwnershipReport(
            incident_id=incident_id,
            known=True,
            workflow_request_id=workflow.request_id,
            workflow_status=workflow.status.value,
            terminal=terminal,
            incident_state=incident.state.value,
            quarantine_hold=quarantine_hold,
        )

    return await _store_call(dependencies, lookup)


@router.get(
    "/executors/fleet-rollout-fence",
    response_model=RemoteFleetRolloutFence,
)
@authorization_bucket("cluster-token")
async def get_remote_fleet_rollout_fence(
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteFleetRolloutFence:
    """Answer the rollout fence for the authenticated cluster only.

    The data-plane executor is the last gate before a destructive step, and it
    is the *only* gate once a remote command exists: the control-plane fence
    runs before dispatch, and holding a workflow there does not revoke a
    command already in the store. So the executor has to be able to ask, and
    in the default regional deployment it owns no store to ask.

    The cluster comes from the authenticated header, never from a query
    parameter, so this cannot be turned into a cross-cluster read.
    """

    cluster_id = _require_cluster(cluster_id)
    fencing = await _store_call(
        dependencies,
        fleet_rollout_fence_deployment_ids,
        dependencies.context.store,
        cluster_id,
    )
    return RemoteFleetRolloutFence(
        cluster_id=cluster_id,
        fencing_deployment_ids=list(fencing),
    )


@router.post(
    "/executors/hyperpod-submissions/outcome",
    response_model=RemoteHyperPodSubmissionReservation,
)
@authorization_bucket("cluster-token")
async def record_remote_hyperpod_submission(
    request: RemoteHyperPodSubmissionRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteHyperPodSubmissionReservation:
    if cluster_id != request.cluster_id:
        raise HTTPException(
            status_code=403,
            detail="authenticated cluster does not match request",
        )
    _registered_submission(dependencies, cluster_id, request.record.cluster_name)

    def record() -> RemoteHyperPodSubmissionReservation:
        try:
            existing = dependencies.context.store.get_hyperpod_submission(
                request.record.cluster_name,
                request.record.idempotency_key,
            )
        except NotFoundError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    "cannot record an outcome for an unreserved HyperPod submission"
                ),
            ) from exc
        if existing.request_identity != request.record.request_identity:
            raise HTTPException(
                status_code=409,
                detail=(
                    "recorded outcome does not match the reserved HyperPod request"
                ),
            )
        dependencies.context.store.save_hyperpod_submission(request.record)
        return RemoteHyperPodSubmissionReservation(
            reserved=False, record=request.record
        )

    return await _store_call(dependencies, record)


@router.post(
    "/executors/spares/health",
    response_model=RemoteSpareHealthReport,
)
@authorization_bucket("cluster-token")
async def regional_spare_health(
    health: RemoteSpareHealthRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteSpareHealthReport:
    if cluster_id != health.cluster_id:
        raise HTTPException(
            status_code=403,
            detail="authenticated cluster does not match request",
        )

    def evaluate() -> RemoteSpareHealthReport:
        ctx = dependencies.context
        if ctx.fleet_registry is None:
            raise HTTPException(
                status_code=503,
                detail="agent registry is disabled",
            )
        aliases = set(health.node_aliases)
        agents = [
            agent
            for agent in ctx.store.list_agents(cluster_id)
            if agent.node_id in aliases
        ]
        reasons = []
        if len(agents) != 1:
            reasons.append(f"expected one matching agent, found {len(agents)}")
        elif not ctx.fleet_registry.readiness(cluster_id, [agents[0].node_id]).ready:
            reasons.append("node agent is not fleet-ready")
        markers = blocking_spare_markers(
            ctx.store,
            aliases,
            cluster_id=cluster_id,
            observed_after=health.observed_after,
        )
        if markers:
            reasons.append(
                "active trusted node fault marker exists: "
                + ", ".join(describe_blocking_marker(marker) for marker in markers)
            )
        findings = [
            finding
            for alias in aliases
            for finding in ctx.store.list_gpu_findings(
                cluster_id, alias, active_only=True
            )
            if (
                health.observed_after is None
                or finding.observed_at > health.observed_after
            )
        ]
        if findings:
            reasons.append("active GPU health finding exists")
        return RemoteSpareHealthReport(ready=not reasons, reasons=reasons)

    return await _store_call(dependencies, evaluate)


@router.post(
    "/executors/advisory-notifications",
    response_model=AdvisoryNotification,
)
@authorization_bucket("cluster-token")
async def regional_save_advisory_notification(
    request: RemoteAdvisoryNotificationRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> AdvisoryNotification:
    if (
        cluster_id != request.cluster_id
        or request.notification.cluster_name != cluster_id
    ):
        raise HTTPException(
            status_code=403,
            detail=("notification cluster does not match the authenticated cluster"),
        )

    def save() -> AdvisoryNotification:
        ctx = dependencies.context
        saved = ctx.store.save_notification_if_absent(request.notification)
        ctx.advisory_notifications.send(saved.notification_id)
        return saved

    return await _store_call(dependencies, save)


@router.post(
    "/executors/evidence",
    response_model=RawEvidenceRecord,
)
@authorization_bucket("cluster-token")
async def regional_capture_evidence(
    request: RemoteEvidenceCaptureRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RawEvidenceRecord:
    if cluster_id != request.cluster_id:
        raise HTTPException(
            status_code=403,
            detail="authenticated cluster does not match request",
        )
    return await _store_call(
        dependencies,
        dependencies.context.evidence.capture,
        record_id=request.record_id,
        cluster_id=request.cluster_id,
        node_id=request.node_id,
        kind=request.kind,
        observed_at=request.observed_at,
        attempt_ids=request.attempt_ids,
        payload=request.payload,
    )


async def _agent_transition(
    action: str,
    cluster_id: str | None,
    node_id: str,
    request: AgentTransitionRequest,
    dependencies: RegionalRouterDependencies,
) -> AgentRecord:
    cluster_id = _require_cluster(cluster_id)
    registry = dependencies.context.fleet_registry
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    try:
        return await _store_call(
            dependencies,
            getattr(registry, f"{action}_agent"),
            cluster_id,
            node_id,
            request,
        )
    except (NotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/executors/agents/{node_id}/drain",
    response_model=AgentRecord,
)
@authorization_bucket("cluster-token")
async def regional_drain_agent(
    node_id: str,
    request: AgentTransitionRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> AgentRecord:
    return await _agent_transition("drain", cluster_id, node_id, request, dependencies)


@router.post(
    "/executors/agents/{node_id}/revoke",
    response_model=AgentRecord,
)
@authorization_bucket("cluster-token")
async def regional_revoke_agent(
    node_id: str,
    request: AgentTransitionRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> AgentRecord:
    return await _agent_transition("revoke", cluster_id, node_id, request, dependencies)
