from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gpu_fault.app.admission_runtime import NulInRequestBody, declared_body_oversize
from gpu_fault.async_store import RequestDeadlineExceeded, StoreIoCapacityExceeded
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
    # Everything below is optional so an assembly that predates it keeps
    # working; ``factory.py`` passes all of them.
    #
    # A fault report is decoded on the fault pool, the way ``from_http`` and
    # the admission wait already are, so a telemetry burst that fills the
    # general decode pool cannot delay an XID at the first hop (A-4).
    fault_decode_io: Any | None = None
    is_fault_path: Callable[[str], bool] | None = None
    # ``ProcessorDispatchState``: the 413 counted here is the same rejection
    # ``dispatch.py`` counts in single-cluster mode (A-3).
    dispatch_state: Any | None = None
    # ``{"nul": n}`` shared with dispatch (E-3).
    decode_rejections: dict[str, int] | None = None
    retry_after_seconds: int = 2


class _PayloadScanRejected(Exception):
    """``payload_cluster_ids`` refused the payload (too structurally complex).

    Not a ``ValueError`` on purpose: it is raised inside the same pool call as
    the JSON decode, whose ``ValueError`` means "not JSON" and answers 400,
    while this one answers 422 as it always has.
    """


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

    def decode_pool(path: str) -> Any:
        if (
            dependencies.fault_decode_io is not None
            and dependencies.is_fault_path is not None
            and dependencies.is_fault_path(path)
        ):
            return dependencies.fault_decode_io
        return dependencies.decode_io

    def scan_clusters(payload: Any) -> set[str]:
        try:
            return dependencies.payload_cluster_ids(payload)
        except ValueError as exc:
            raise _PayloadScanRejected(str(exc)) from exc

    def decode_and_scan(
        body: bytes, content_encoding: str
    ) -> tuple[bytes, Any, set[str]]:
        """One pool call: decompress, parse, and walk for cluster ids.

        The walk used to run on the event loop after the pool returned
        (27 ms for a 5 MiB node-logs batch, A-8); it belongs with the parse
        it depends on.
        """

        body, payload = dependencies.decode_json_body(body, content_encoding)
        clusters = scan_clusters(payload) if body else set()
        return body, payload, clusters

    def oversize_response() -> JSONResponse:
        if dependencies.dispatch_state is not None:
            dependencies.dispatch_state.oversize_rejections += 1
        return JSONResponse(
            status_code=413,
            content={
                "detail": "processor request body is too large",
                "max_bytes": dependencies.processor_max_request_bytes,
            },
        )

    def capacity_response(exc: StoreIoCapacityExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": str(dependencies.retry_after_seconds)},
            content={
                "detail": (
                    "request deadline exceeded"
                    if isinstance(exc, RequestDeadlineExceeded)
                    else "request decode capacity exceeded"
                )
            },
        )

    def nul_response(exc: NulInRequestBody) -> JSONResponse:
        rejections = dependencies.decode_rejections
        if rejections is not None:
            rejections["nul"] = rejections.get("nul", 0) + 1
        return JSONResponse(status_code=422, content={"detail": str(exc)})

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
                # This is the first ``await request.body()`` on the regional
                # path, so the declared-length gate has to sit here: behind
                # it, dispatch's copy measured a body already resident (A-3).
                if declared_body_oversize(
                    request.headers, dependencies.processor_max_request_bytes
                ):
                    return oversize_response()
                body = await request.body()
                payload = request.scope.get("gpu_fault_json_payload")
                pool = decode_pool(request.url.path)
                try:
                    if payload is None:
                        body, payload, payload_clusters = await pool.run(
                            decode_and_scan,
                            body,
                            request.headers.get("Content-Encoding", ""),
                        )
                        # The decoded bytes, not the wire bytes, are what the
                        # route must read. Assigning them here is the whole
                        # mechanism: once the body has been read, Starlette
                        # replays this cache downstream and never consults
                        # the request's receive channel again -- see
                        # tests/test_body_replay_contract.py.
                        request._body = body
                        request.scope["gpu_fault_body_decompressed"] = True
                        request.scope["gpu_fault_json_payload"] = payload
                    elif body:
                        payload_clusters = await pool.run(scan_clusters, payload)
                    else:
                        payload_clusters = set()
                except StoreIoCapacityExceeded as exc:
                    return capacity_response(exc)
                except OverflowError:
                    return oversize_response()
                except NulInRequestBody as exc:
                    return nul_response(exc)
                except _PayloadScanRejected as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
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
