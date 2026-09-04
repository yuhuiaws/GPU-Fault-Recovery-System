from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gpu_fault.async_store import StoreIoCapacityExceeded
from gpu_fault.regional_registry_runtime import regional_cluster_request_allowed


@dataclass(frozen=True)
class RegionalAuthDependencies:
    context: Any
    replay_authorized: Callable[[Request], bool]
    authorization_bucket: Callable
    authenticate_cluster: Callable[[str | None, str | None], Any]
    decode_io: Any
    decode_json_body: Callable
    payload_cluster_ids: Callable[[Any], set[str]]
    processor_max_request_bytes: int


def install_regional_authorization(
    app: FastAPI,
    dependencies: RegionalAuthDependencies,
) -> None:
    @app.middleware("http")
    async def regional_default_deny_authorization(request: Request, call_next):
        ctx = dependencies.context
        if not ctx.regional_mode:
            return await call_next(request)
        if dependencies.replay_authorized(request):
            return await call_next(request)
        bucket = dependencies.authorization_bucket(
            request.url.path,
            request.headers,
            request.client.host if request.client else None,
            request.method,
        )
        if bucket is None:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": ("regional route has no declared authorization bucket")
                },
            )
        if bucket != "execution-token":
            return await call_next(request)
        supplied = request.headers.get("X-GPU-Fault-Execution-Token")
        if not supplied and request.url.path == "/metrics":
            authorization = request.headers.get("Authorization") or ""
            if authorization.startswith("Bearer "):
                supplied = authorization.removeprefix("Bearer ").strip()
        if (
            not ctx.execution_token
            or not supplied
            or not secrets.compare_digest(supplied, ctx.execution_token)
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "regional mode requires a valid "
                        "X-GPU-Fault-Execution-Token for this endpoint"
                    )
                },
            )
        return await call_next(request)

    @app.middleware("http")
    async def regional_cluster_authentication(request: Request, call_next):
        if dependencies.replay_authorized(request):
            return await call_next(request)
        if (
            not dependencies.context.regional_mode
            or dependencies.authorization_bucket(
                request.url.path,
                request.headers,
                None,
                request.method,
            )
            != "cluster-token"
        ):
            return await call_next(request)
        cluster_id = request.headers.get("X-GPU-Fault-Cluster-ID")
        try:
            registration = dependencies.authenticate_cluster(
                cluster_id, request.headers.get("Authorization")
            )
            request.state.regional_cluster_registration = registration
            if not regional_cluster_request_allowed(
                registration,
                path=request.url.path,
                method=request.method,
            ):
                return JSONResponse(
                    status_code=423,
                    content={
                        "detail": (
                            "regional cluster lifecycle "
                            f"{registration.lifecycle_state.value} "
                            "does not allow this request"
                        )
                    },
                )
            if request.method in {"POST", "PUT", "PATCH"}:
                body = await request.body()
                payload = request.scope.get("gpu_fault_json_payload")
                if payload is None:
                    try:
                        body, payload = await dependencies.decode_io.run(
                            dependencies.decode_json_body,
                            body,
                            request.headers.get("Content-Encoding", ""),
                        )
                    except StoreIoCapacityExceeded:
                        return JSONResponse(
                            status_code=503,
                            headers={"Retry-After": "2"},
                            content={"detail": ("request decode capacity exceeded")},
                        )
                    except OverflowError:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": ("processor request body is too large"),
                                "max_bytes": (dependencies.processor_max_request_bytes),
                            },
                        )
                    except (
                        OSError,
                        EOFError,
                        json.JSONDecodeError,
                        ValueError,
                    ):
                        return JSONResponse(
                            status_code=400,
                            content={"detail": "invalid JSON request body"},
                        )
                    # The decoded bytes, not the wire bytes, are what the route
                    # must read. Assigning them here is the whole mechanism:
                    # once the body has been read, Starlette replays this cache
                    # downstream and never consults the request's receive
                    # channel again -- see tests/test_body_replay_contract.py.
                    request._body = body
                    request.scope["gpu_fault_body_decompressed"] = True
                    request.scope["gpu_fault_json_payload"] = payload
                if body:
                    try:
                        payload_clusters = dependencies.payload_cluster_ids(payload)
                    except ValueError as exc:
                        raise HTTPException(status_code=422, detail=str(exc)) from exc
                    if payload_clusters - {cluster_id}:
                        raise HTTPException(
                            status_code=403,
                            detail=(
                                "authenticated cluster does not match "
                                "all payload cluster_id values"
                            ),
                        )
        except (HTTPException, json.JSONDecodeError) as exc:
            status_code = getattr(exc, "status_code", 400)
            detail = getattr(exc, "detail", "invalid JSON request")
            return JSONResponse(
                status_code=status_code,
                content={"detail": detail},
                headers=getattr(exc, "headers", None),
            )
        return await call_next(request)
