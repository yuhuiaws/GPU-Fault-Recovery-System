from __future__ import annotations

import hmac
import logging
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    AsyncStoreExecutor,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from gpu_fault.cluster_scope import ClusterProvider, cluster_provider
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
from gpu_fault.fleet_compatibility import NODE_ACTION_KEY_VERSION_DERIVED
from gpu_fault.node_action_keys import (
    derive_node_action_secret,
    resolve_node_action_secret,
)
from gpu_fault.node_key_attestation import (
    AttestedNodeActionKeyRequest,
    EksNodeAttestation,
    eks_attestation_refusal,
)
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
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
    RemoteNodeActionKeys,
    RemoteSpareHealthReport,
    RemoteSpareHealthRequest,
)
from gpu_fault.regional_compatibility import (
    ACTIVATION_INHIBITION_VERSION,
    REMOTE_STEP_BATCHING_PROTOCOL_VERSION,
    RegionalExecutorCompatibilityPolicy,
)
from gpu_fault.orchestration.escalation import escalation_descends_from
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import RawEvidenceRecord
from gpu_fault.workflow_quarantine import has_terminal_quarantine_hold

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
activation_inhibition_version = ACTIVATION_INHIBITION_VERSION


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
        raise store_io_rejection(exc, retry_after=retry_after) from exc


def store_io_rejection(
    exc: StoreIoCapacityExceeded, *, retry_after: str = "2"
) -> HTTPException:
    """The 503 a store I/O refusal becomes, named for its real cause: a request
    that ran out of budget is a deadline, not store capacity (the two need
    different remedies -- a shorter wait versus a larger pool)."""

    return HTTPException(
        status_code=503,
        detail=(
            "request deadline exceeded"
            if isinstance(exc, RequestDeadlineExceeded)
            else "store I/O capacity exceeded"
        ),
        headers={"Retry-After": retry_after},
    )


# What a long-poll must leave of the request budget for the claim that follows
# the wait: the store I/O admission timeout (2 s by default) plus a margin.
CLAIM_WAIT_RESERVE_SECONDS = 3.0


def bounded_claim_wait_seconds(
    requested: float,
    *,
    deadline: float | None,
    now: float,
    reserve_seconds: float = CLAIM_WAIT_RESERVE_SECONDS,
) -> float:
    """How long a claim may be held before its request budget runs out.

    The request carries one deadline (``REQUEST_DEADLINE``, the backpressure
    middleware's budget, 15 s by default) and the claim that ends the wait is a
    store call bound by it. An executor asking for a 20 s wait therefore had
    every idle claim answered 503 "deadline" -- live 2026-09-09: both executor
    replicas failed readiness for as long as the queue was empty. The wait is
    the smaller of what was asked, the server cap (applied by the hub) and what
    the budget still allows after the reserve; nothing left means no wait.
    """

    if requested <= 0:
        return 0.0
    if deadline is None:
        return requested
    remaining = deadline - now - reserve_seconds
    return max(0.0, min(requested, remaining))


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
            executor_protocol_version=claim.executor_protocol_version,
        )
        return commands

    hub = dependencies.remote_command_wakeups
    wait_seconds = bounded_claim_wait_seconds(
        claim.wait_seconds, deadline=REQUEST_DEADLINE.get(), now=time.monotonic()
    )
    if wait_seconds <= 0 or hub is None:
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
        await waiter.wait(wait_seconds)
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
    No notification is dispatched here: a batched step's completion mail lands
    with the terminal report (``dispatch_remote_completion``), never mid-run.
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


def provider_cluster_name(registration: RegionalClusterRegistration) -> str:
    """The cluster scope a provider submission names for ``registration``:
    the HyperPod cluster name, or -- a plain EKS cluster has none -- the
    cluster id itself, which is what ``EksProviderConfig.cluster_name`` sends
    (live 2026-10-05 an eks-managed RESTART_NODE was refused 403 against the
    empty HyperPod name)."""

    if cluster_provider(registration.provider) is ClusterProvider.EKS_MANAGED:
        return registration.cluster_id
    return registration.hyperpod_cluster_name


def _registered_submission(
    dependencies: RegionalRouterDependencies,
    cluster_id: str | None,
    cluster_name: str,
):
    registration = dependencies.auth_registry.get(cluster_id or "")
    if registration is None or cluster_name != provider_cluster_name(registration):
        raise HTTPException(
            status_code=403,
            detail=(
                "submission cluster_name does not match the registered provider "
                "cluster for this cluster_id (the HyperPod cluster name, or the "
                "cluster id on an eks-managed cluster)"
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
    successor_workflow_id: str | None = None,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteIncidentOwnershipReport:
    cluster_id = _require_cluster(cluster_id)
    store = dependencies.context.store

    def lineage() -> bool | None:
        # Only a workflow of the authenticated cluster may be named as the
        # asking successor; anything else reads as "not a descendant".
        if not successor_workflow_id:
            return None
        try:
            successor = store.get_incident(
                store.get_workflow(successor_workflow_id).incident_id
            )
        except (NotFoundError, KeyError):
            return False
        if successor.cluster_id != cluster_id:
            return False
        return escalation_descends_from(store, successor_workflow_id, incident_id)

    def lookup() -> RemoteIncidentOwnershipReport:
        unknown = RemoteIncidentOwnershipReport(incident_id=incident_id)
        try:
            incident = store.get_incident(incident_id)
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
            workflow = store.get_workflow(incident.workflow_request_id)
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
        terminal_quarantine = has_terminal_quarantine_hold(workflow) or any(
            execution.operation is WorkflowOperation.REPLACE_NODE
            and execution.details.get("action") == "SPARE_FAILOVER"
            for execution in workflow.step_executions
        )
        quarantine_hold = (
            incident.state is IncidentState.QUARANTINED or terminal_quarantine
        )
        return RemoteIncidentOwnershipReport(
            incident_id=incident_id,
            known=True,
            workflow_request_id=workflow.request_id,
            workflow_status=workflow.status.value,
            terminal=terminal,
            incident_state=incident.state.value,
            quarantine_hold=quarantine_hold,
            terminal_quarantine=terminal_quarantine,
            escalation_lineage=lineage(),
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


def _node_known_to_cluster(
    store: Any,
    cluster_id: str,
    node_id: str,
    *,
    hyperpod_identities: list[Any],
) -> bool:
    """Whether data-plane evidence already places ``node_id`` in ``cluster_id``.

    Any one of: an Agent record (a node that heartbeated before, e.g. a
    re-imaged node), a GPU inventory snapshot or GPU metric (posted by the
    host collectors an install starts, minutes before its Agent registers), or
    a HyperPod node identity whose current Kubernetes name or alias is this
    node. Retired aliases do not count: a replaced node's old name must not be
    re-keyed. A plain EKS node that has never been installed has none of these
    (``gpu_fault.node_key_attestation``).
    """

    try:
        store.get_agent(cluster_id, node_id)
        return True
    except (NotFoundError, KeyError):
        pass
    if store.get_gpu_inventory_snapshot(cluster_id, node_id) is not None:
        return True
    if store.list_gpu_metrics_latest(cluster_id, node_id):
        return True
    return any(
        identity.kubernetes_node_name == node_id or node_id in identity.aliases
        for identity in hyperpod_identities
    )


#: Registrations whose evidence no longer claims a node name: their token is
#: refused (``RegionalClusterRegistration.accepts_token``) and their removal or
#: rollback owns whatever records they left behind.
_RETIRED_REGISTRATIONS = frozenset(
    {RegionalClusterLifecycle.REVOKED, RegionalClusterLifecycle.ROLLED_BACK}
)


def _attested_node_refusal(
    store: Any,
    registration: RegionalClusterRegistration | None,
    node_secrets: Mapping[str, str],
    fleet_secret: str,
    attestation: EksNodeAttestation,
) -> tuple[int, str] | None:
    """``(status, reason)`` when an attested node gets no key, else ``None``.

    403 when the statement does not place the node in the authenticated
    cluster (``eks_attestation_refusal``). 409 when the name is taken, the two
    ways the deploy host's provisioning refuses it too: the CPU node-key mirror
    holds a key for the name that is not this cluster's (``CPU NodeName key
    conflict``), or another live registration already reports the name. Node
    names key one site-wide map, so exclusive ownership is an operator's
    reconciliation, never something the sync decides.
    """

    if registration is None:
        return 403, "the authenticated cluster's registration is unavailable"
    refusal = eks_attestation_refusal(registration, attestation)
    if refusal is not None:
        return 403, refusal
    cluster_id, node_id = registration.cluster_id, attestation.node_id
    mirrored = node_secrets.get(node_id) or node_secrets.get(f"{cluster_id}/{node_id}")
    derived = derive_node_action_secret(fleet_secret, cluster_id, node_id)
    if mirrored is not None and not hmac.compare_digest(
        mirrored.encode(), derived.encode()
    ):
        return 409, (
            f"CPU NodeName key conflict for {node_id}; exclusive cluster "
            "ownership requires reconciliation"
        )
    for other in store.list_regional_clusters():
        if (
            other.cluster_id == cluster_id
            or other.lifecycle_state in _RETIRED_REGISTRATIONS
        ):
            continue
        identities = (
            list(store.list_hyperpod_node_identities(other.hyperpod_cluster_name))
            if other.hyperpod_cluster_name
            else []
        )
        if _node_known_to_cluster(
            store, other.cluster_id, node_id, hyperpod_identities=identities
        ):
            # Which cluster stays in the control plane's log: the caller is
            # authenticated as one cluster and never reads another's records.
            LOGGER.warning(
                "node name %s of cluster %s is already known to cluster %s",
                node_id,
                cluster_id,
                other.cluster_id,
            )
            return 409, (
                f"node name {node_id} is already known to another cluster; "
                "exclusive cluster ownership requires reconciliation"
            )
    return None


@router.post("/node-action-keys", response_model=RemoteNodeActionKeys)
@authorization_bucket("cluster-token")
async def regional_node_action_keys(
    request: AttestedNodeActionKeyRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RemoteNodeActionKeys:
    """Node-scoped Node Action keys for nodes this cluster already has.

    ``gpu-fault-admin deploy``/``join-cluster`` write one version-2 key per
    node into the GPU Secret ``gpu-fault-node-action-keys`` for the nodes
    present at that time. A node the provider replaces later has none, so every
    installer Job for it fails on the missing Secret key (live 2026-09-30) and
    the node never gets an Agent. The cluster executor asks here for the
    missing names and patches only those keys in.

    The fleet master never leaves the control plane: only the derived,
    node-scoped value is returned, only for the authenticated cluster (the
    payload ``cluster_id`` is bound by the middleware and re-checked here), and
    only for node ids the control plane can place in that cluster: its own
    data-plane evidence, or -- for a plain EKS node that was never installed,
    which has none -- the EC2 identity the executor attests in ``nodes``, when
    it binds to the cluster's registration and the name is nobody else's
    (``_attested_node_refusal``). One refusal withholds every key of the
    request. Key values are never logged.
    """

    cluster_id = _require_cluster(cluster_id)
    if cluster_id != request.cluster_id:
        raise HTTPException(
            status_code=403,
            detail="authenticated cluster does not match request",
        )
    registry = dependencies.context.fleet_registry
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    fleet_secret = str(getattr(registry, "secret", "") or "")
    if len(fleet_secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="node action fleet secret is unavailable",
        )
    auth_registry = dependencies.auth_registry
    registration = auth_registry.get(cluster_id) if auth_registry is not None else None
    hyperpod_cluster_name = getattr(registration, "hyperpod_cluster_name", None)
    attested = {item.node_id: item for item in request.nodes}

    def issue() -> RemoteNodeActionKeys:
        store = dependencies.context.store
        identities = (
            list(store.list_hyperpod_node_identities(hyperpod_cluster_name))
            if hyperpod_cluster_name
            else []
        )
        unknown = [
            node_id
            for node_id in request.node_ids
            if not _node_known_to_cluster(
                store, cluster_id, node_id, hyperpod_identities=identities
            )
        ]
        unplaced = [node_id for node_id in unknown if node_id not in attested]
        if unplaced:
            LOGGER.warning(
                "refused node action keys for cluster %s: %d of %d node ids are "
                "not known to this cluster",
                cluster_id,
                len(unplaced),
                len(request.node_ids),
            )
            raise HTTPException(
                status_code=404,
                detail=(
                    "one or more node_ids are not known to the control plane "
                    "for the authenticated cluster"
                ),
            )
        for node_id in unknown:
            attestation = attested[node_id]
            refused = _attested_node_refusal(
                store, registration, registry.node_secrets, fleet_secret, attestation
            )
            if refused is not None:
                LOGGER.warning(
                    "refused node action keys for cluster %s: node %s: %s",
                    cluster_id,
                    node_id,
                    refused[1],
                )
                raise HTTPException(status_code=refused[0], detail=refused[1])
            # The audit line of an admission by statement: which instance,
            # node group and Node UID the executor vouched for.
            LOGGER.info(
                "admitted node %s of cluster %s by its EC2 identity: uid=%s "
                "instance=%s nodegroup=%s address=%s",
                node_id,
                cluster_id,
                attestation.node_uid,
                attestation.instance_id,
                attestation.nodegroup,
                attestation.private_ip,
            )
        keys: dict[str, str] = {}
        for node_id in request.node_ids:
            try:
                keys[node_id] = resolve_node_action_secret(
                    fleet_secret,
                    registry.node_secrets,
                    cluster_id,
                    node_id,
                    NODE_ACTION_KEY_VERSION_DERIVED,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="node action key derivation is unavailable",
                ) from exc
        # The audit line: which cluster asked for which nodes. Never the values.
        LOGGER.info(
            "issued node action keys: cluster=%s nodes=%s",
            cluster_id,
            ",".join(request.node_ids),
        )
        return RemoteNodeActionKeys(cluster_id=cluster_id, keys=keys)

    issued: RemoteNodeActionKeys = await _store_call(dependencies, issue)
    return issued


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


@router.post(
    "/executors/agents/{node_id}/reactivate",
    response_model=AgentRecord,
)
@authorization_bucket("cluster-token")
async def regional_reactivate_agent(
    node_id: str,
    request: AgentTransitionRequest,
    cluster_id: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> AgentRecord:
    """The executor undoes the revocation it made for a provider mutation that
    was never submitted (``AgentTransitionRequest.transition_id`` must be the
    revoking one; the registry refuses anything else)."""

    return await _agent_transition(
        "reactivate", cluster_id, node_id, request, dependencies
    )
