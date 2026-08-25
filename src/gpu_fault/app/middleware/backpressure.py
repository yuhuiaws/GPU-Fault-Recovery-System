from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    remaining_budget,
)


@dataclass(frozen=True)
class IngressBackpressureDependencies:
    service_role: str
    requires_processor: Callable[[Request], bool]
    replay_authorized: Callable[[Request], bool]
    is_fault_path: Callable[[str], bool]
    fault_semaphore: asyncio.Semaphore
    normal_semaphore: asyncio.Semaphore
    fault_wait_seconds: float
    normal_wait_seconds: float
    rejections: dict[str, int]


def install_ingress_backpressure(
    app: FastAPI,
    dependencies: IngressBackpressureDependencies,
) -> None:
    @app.middleware("http")
    async def ingress_backpressure(request: Request, call_next):
        if (
            dependencies.service_role != "ingress"
            or not dependencies.requires_processor(request)
            or dependencies.replay_authorized(request)
        ):
            return await call_next(request)
        fault = dependencies.is_fault_path(request.url.path)
        scope = "fault" if fault else "normal"
        semaphore = (
            dependencies.fault_semaphore if fault else dependencies.normal_semaphore
        )
        wait_started = time.monotonic()
        timeout = remaining_budget(
            dependencies.fault_wait_seconds
            if fault
            else dependencies.normal_wait_seconds
        )
        if timeout <= 0:
            dependencies.rejections[scope] += 1
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "1"},
                content={
                    "detail": "request deadline exceeded at ingress",
                    "scope": scope,
                },
            )
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        except TimeoutError:
            dependencies.rejections[scope] += 1
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "1"},
                content={
                    "detail": "ingress concurrency capacity exceeded",
                    "scope": scope,
                },
            )
        request.scope.setdefault(
            "gpu_fault_server_timing",
            {},
        )[f"backpressure_{scope}"] = (time.monotonic() - wait_started) * 1000
        try:
            return await call_next(request)
        finally:
            semaphore.release()


def install_request_deadline(
    app: FastAPI,
    *,
    request_budget_seconds: float,
) -> None:
    if request_budget_seconds < 0:
        raise ValueError("GPU_FAULT_REQUEST_BUDGET_SECONDS must not be negative")

    @app.middleware("http")
    async def request_deadline(request: Request, call_next):
        started = time.monotonic()
        if request_budget_seconds <= 0:
            response = await call_next(request)
        else:
            token = REQUEST_DEADLINE.set(time.monotonic() + request_budget_seconds)
            try:
                response = await call_next(request)
            finally:
                REQUEST_DEADLINE.reset(token)
        total_ms = (time.monotonic() - started) * 1000
        timings = {
            "total": total_ms,
            **request.scope.get(
                "gpu_fault_server_timing",
                {},
            ),
        }
        value = ", ".join(
            f"gpu_fault_{name};dur={duration:.3f}" for name, duration in timings.items()
        )
        existing = response.headers.get("Server-Timing")
        response.headers["Server-Timing"] = (
            f"{existing}, {value}" if existing else value
        )
        response.headers["X-GPU-Fault-Server-Duration-Ms"] = f"{total_ms:.3f}"
        return response
