from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.models import (
    WorkflowDispatchReport,
    WorkflowExecutionRequest,
    WorkflowExecutionResult,
    WorkflowRequest,
)
from gpu_fault.orchestration import WorkflowFencingError


@dataclass(frozen=True)
class WorkflowRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor


def get_workflow_dependencies() -> WorkflowRouterDependencies:
    raise RuntimeError("workflow router dependencies are not configured")


router = APIRouter(prefix="/v1/workflows", tags=["workflows"])


def _require_execution_token(expected: str | None, supplied: str | None) -> None:
    if not expected or not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=403,
            detail="invalid workflow execution token",
        )


async def _store_call(
    dependencies: WorkflowRouterDependencies,
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


@router.post("/dispatch", response_model=WorkflowDispatchReport)
@authorization_bucket("execution-token")
async def dispatch_workflows(
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: WorkflowRouterDependencies = Depends(get_workflow_dependencies),
) -> WorkflowDispatchReport:
    _require_execution_token(dependencies.context.execution_token, execution_token)
    return await _store_call(
        dependencies,
        dependencies.context.dispatcher.run_once,
    )


@router.get("/{request_id}", response_model=WorkflowRequest)
@authorization_bucket("execution-token")
async def get_workflow(
    request_id: str,
    dependencies: WorkflowRouterDependencies = Depends(get_workflow_dependencies),
) -> WorkflowRequest:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_workflow,
        request_id,
    )


@router.post(
    "/{request_id}/simulate",
    response_model=WorkflowExecutionResult,
)
@authorization_bucket("execution-token")
async def simulate_workflow(
    request_id: str,
    execution: WorkflowExecutionRequest,
    dependencies: WorkflowRouterDependencies = Depends(get_workflow_dependencies),
) -> WorkflowExecutionResult:
    try:
        return await _store_call(
            dependencies,
            dependencies.context.orchestrator.simulate,
            request_id,
            execution.expected_fencing_token,
        )
    except WorkflowFencingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{request_id}/execute",
    response_model=WorkflowExecutionResult,
)
@authorization_bucket("execution-token")
async def execute_workflow(
    request_id: str,
    execution: WorkflowExecutionRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: WorkflowRouterDependencies = Depends(get_workflow_dependencies),
) -> WorkflowExecutionResult:
    ctx = dependencies.context
    _require_execution_token(ctx.execution_token, execution_token)
    try:
        return await _store_call(
            dependencies,
            ctx.workflow_executor.execute,
            request_id,
            execution,
        )
    except WorkflowFencingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
