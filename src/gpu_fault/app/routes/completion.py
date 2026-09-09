from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.app.ingest.workload_observations import (
    ingest_workload_observation,
)
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.channel_registry import (
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.models import (
    CompletionDecision,
    OperationResult,
    RecoveryPlan,
    RestartBudgetState,
    TerminalEvent,
)
from gpu_fault.telemetry import (
    ATTEMPT_COVERAGE_PATH,
    WorkloadCoverageHeartbeat,
)
from gpu_fault.watcher import (
    AttemptObservation,
    FailureContainmentDecision,
    FailureDetectedEvent,
)


@dataclass(frozen=True)
class CompletionRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor


def get_completion_dependencies() -> CompletionRouterDependencies:
    raise RuntimeError("completion router dependencies are not configured")


router = APIRouter(tags=["completion"])


async def _store_call(
    dependencies: CompletionRouterDependencies,
    function: Callable,
    /,
    *args,
    **kwargs,
):
    try:
        return await dependencies.store_io.run(function, *args, **kwargs)
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": "2"},
        ) from exc


@router.post(
    WORKLOAD_OBSERVATIONS_PATH,
    response_model=AttemptObservation,
)
@authorization_bucket("cluster-token")
async def observe_workload(
    observation: AttemptObservation,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> AttemptObservation:
    await _store_call(
        dependencies,
        ingest_workload_observation,
        dependencies.context,
        observation,
    )
    return observation


@router.post(
    "/v1/attempts/failure-detected",
    response_model=FailureContainmentDecision,
)
@authorization_bucket("cluster-token")
async def failure_detected(
    event: FailureDetectedEvent,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> FailureContainmentDecision:
    try:
        return await _store_call(
            dependencies,
            dependencies.context.completion.handle_failure_detected,
            event,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/v1/attempts/terminal",
    response_model=CompletionDecision,
)
@authorization_bucket("cluster-token")
async def terminal(
    event: TerminalEvent,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> CompletionDecision:
    return await _store_call(
        dependencies,
        dependencies.context.completion.handle_terminal,
        event,
    )


@router.post(
    ATTEMPT_COVERAGE_PATH,
    response_model=WorkloadCoverageHeartbeat,
    status_code=202,
)
@authorization_bucket("cluster-token")
async def observe_coverage(
    heartbeat: WorkloadCoverageHeartbeat,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> WorkloadCoverageHeartbeat:
    """Record that a watcher completed a full pass over one cluster.

    Weak, self-superseding evidence: the watcher sends it through its ordinary
    sink, so a lost heartbeat is simply replaced by the next pass. 202 says
    exactly that -- the statement was accepted, and an older one that lost the
    race to a newer row is not an error the watcher can act on.
    """

    await _store_call(
        dependencies,
        dependencies.context.topology.observe_coverage,
        heartbeat,
    )
    return heartbeat


@router.get(
    "/v1/attempts/{cluster_id}/{attempt_id}/decision",
    response_model=CompletionDecision,
)
@authorization_bucket("execution-token")
async def get_decision(
    cluster_id: str,
    attempt_id: str,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> CompletionDecision:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_decision_by_attempt,
        cluster_id,
        attempt_id,
    )


@router.get(
    "/v1/restart-budgets/{cluster_id}/{job_id}",
    response_model=RestartBudgetState,
)
@authorization_bucket("execution-token")
async def get_restart_budget(
    cluster_id: str,
    job_id: str,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> RestartBudgetState:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_restart_budget,
        cluster_id,
        job_id,
    )


@router.get(
    "/v1/recovery-plans/{plan_id}",
    response_model=RecoveryPlan,
)
@authorization_bucket("execution-token")
async def get_plan(
    plan_id: str,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> RecoveryPlan:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_plan,
        plan_id,
    )


@router.post(
    "/v1/recovery-plans/{plan_id}/simulate",
    response_model=OperationResult,
)
@authorization_bucket("execution-token")
async def simulate_plan(
    plan_id: str,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> OperationResult:
    plan = await _store_call(
        dependencies,
        dependencies.context.store.get_plan,
        plan_id,
    )
    return await _store_call(
        dependencies,
        dependencies.context.executor.execute,
        plan,
    )
