from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.channel_registry import (
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.models import (
    CompletionDecision,
    OperationResult,
    RecoveryPlan,
    RestartBudgetState,
    TerminalEvent,
    TriageReport,
)
from gpu_fault.app.ingest.workload_observations import (
    ingest_workload_observation,
)
from gpu_fault.service import CompletionPendingError
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
    try:
        return await _store_call(
            dependencies,
            dependencies.context.completion.handle_terminal,
            event,
        )
    except CompletionPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/v1/triage-results",
    response_model=CompletionDecision,
)
@authorization_bucket("cluster-token")
async def triage(
    report: TriageReport,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
) -> CompletionDecision:
    try:
        return await _store_call(
            dependencies,
            dependencies.context.completion.handle_triage,
            report,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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


@router.get("/v1/diagnostic-requests/{request_id}")
@authorization_bucket("execution-token")
async def get_diagnostic(
    request_id: str,
    dependencies: CompletionRouterDependencies = Depends(get_completion_dependencies),
):
    return await _store_call(
        dependencies,
        dependencies.context.store.get_diagnostic,
        request_id,
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
