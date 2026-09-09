from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import random
import secrets
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from gpu_fault.app.admission_runtime import (
    NulInRequestBody,
    declared_body_oversize,
)
from gpu_fault.app.runtime import ProcessorDispatchState
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from gpu_fault.processor import (
    ProcessorRequest,
    ProcessorRequestStatus,
)
from gpu_fault.processor.models import is_reserved_tier
from gpu_fault.processor_diagnostics import (
    bind_processor_replay,
    report_processor_replay_phase,
    reset_processor_replay,
)

# A synchronous caller waits for the processor to complete its request by
# polling the store. A fixed 100ms interval cost every waiter ~1150 store reads
# across the full timeout and made all of them retry in lockstep, so a burst of
# waiters kept the store I/O pool busy answering "not yet". Poll fast at first
# (most requests complete in a few tens of milliseconds), then back off to a
# tenth of that rate, with jitter so concurrent waiters spread out.
#
# The interval is also the upper bound on how long a waiter sits on a response
# that is already in the store, which is why the poll is now a floor rather than
# the only mechanism: when the request is completed by a worker of this same
# process, ``ProcessorCompletionSignals`` ends the interval early. Nothing here
# depends on that -- see the class for the cases it cannot cover.
RESPONSE_POLL_INITIAL_SECONDS = 0.01
RESPONSE_POLL_MAX_SECONDS = 0.25
RESPONSE_POLL_GROWTH = 1.6
RESPONSE_POLL_JITTER = 0.25

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"


def idempotent_request_id(
    *,
    cluster_id: str | None,
    path: str,
    key: str,
    body: bytes,
) -> str:
    """The queue row a request with this ``Idempotency-Key`` belongs to (F-E4).

    ``request_id`` used to be a fresh ``uuid4()`` on every POST. The data plane
    already sends ``Idempotency-Key`` (event id, batch id, ``attempt_id/time``)
    and retries the same body when the 202 is lost to a timeout, but nothing
    on this side read it, so every retry became a second, independently
    executed processor request. Every store returns the existing row for a
    known ``request_id``, so deriving the id here is the whole dedup.

    The derivation is scoped to the cluster and path, and to the body itself:
    a key reused by mistake for a different payload gets its own row instead of
    another request's receipt, and no cluster can address another's rows.
    """

    digest = hashlib.sha256()
    for part in (cluster_id or "", path, key):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    digest.update(hashlib.sha256(body).digest())
    return f"processor-idem-{digest.hexdigest()[:40]}"


@dataclass(frozen=True)
class ProcessorDispatchDependencies:
    context: Any
    processor: Any | None
    state: ProcessorDispatchState
    store_io: Any
    decode_io: Any
    fault_store_io: Any
    fault_decode_io: Any
    processor_admission_batcher: Any
    fault_admission_batcher: Any
    evidence_admission_batcher: Any
    telemetry_spool_batcher: Any
    processor_replay_tracker: Any
    requires_processor: Callable[[Request], bool]
    replay_authorized: Callable[[Request], bool]
    returns_processor_receipt: Callable[[str], bool]
    is_fault_ingress_path: Callable[[str], bool]
    decode_json_body: Callable
    processor_max_queue_depth: int
    processor_max_cluster_queue_depth: int
    processor_fault_reserved_queue_depth: int
    processor_fault_reserved_cluster_depth: int
    processor_global_admission_guard: int
    processor_max_request_bytes: int
    processor_retry_after_seconds: int
    processor_response_timeout_seconds: float
    processor_queue_bypass_enabled: bool
    processor_queue_bypass_paths: set[str]
    processor_admission_rejections: dict[str, int]
    processor_admission_rejections_by_path: dict[str, int]
    processor_queue_bypasses_by_path: dict[str, int]
    telemetry_spool_enabled: bool
    telemetry_spool_max_item_bytes: int
    telemetry_spool_rejections: dict[str, int]
    telemetry_spool_admitted_by_path: dict[str, int]
    telemetry_request_budget_seconds: float
    # Optional so an assembly that predates it keeps working; the factory
    # passes the runtime's dict so /metrics sees the count (E-3).
    decode_rejections: dict[str, int] = field(default_factory=lambda: {"nul": 0})


@dataclass(frozen=True)
class PreparedProcessorRequest:
    item: ProcessorRequest
    body: bytes
    store_pool: Any
    server_timing: dict[str, float]


async def _handle_replay(
    request: Request,
    call_next,
    dependencies: ProcessorDispatchDependencies,
) -> Response | None:
    processor = dependencies.processor
    if request.headers.get("X-GPU-Fault-Processor-Replay") is None:
        return None
    if not dependencies.replay_authorized(request):
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "processor replay is accepted only from "
                    "loopback with a valid replay secret"
                )
            },
        )
    try:
        request_id = request.headers["X-GPU-Fault-Processor-Request-ID"]
        lane_owner = request.headers["X-GPU-Fault-Processor-Owner-ID"]
        if not request_id or len(request_id) > 256:
            raise ValueError("invalid request ID")
        lane_epoch = None
        lane_token = None
        lane_key = None
        if processor.active_consumers:
            lane_epoch = int(request.headers["X-GPU-Fault-Processor-Lane-Epoch"])
            lane_token = request.headers["X-GPU-Fault-Processor-Lane-Token"]
            lane_key = base64.urlsafe_b64decode(
                request.headers["X-GPU-Fault-Processor-Lane-Key"].encode("ascii")
            ).decode("utf-8")
    except (KeyError, ValueError, UnicodeDecodeError):
        return JSONResponse(
            status_code=400,
            content={"detail": "invalid processor replay headers"},
        )
    tracker = dependencies.processor_replay_tracker
    tracker.start(
        request_id,
        owner_id=lane_owner,
        path=request.url.path,
        lane_epoch=lane_epoch,
        phase=(
            "lane_validation" if processor.active_consumers else "leadership_validation"
        ),
    )
    replay_context = bind_processor_replay(tracker, request_id)
    try:
        if processor.active_consumers:
            lane_valid = await dependencies.store_io.run(
                dependencies.context.store.validate_processor_lane,
                lane_key,
                lane_owner,
                lane_epoch,
                lane_token,
            )
            if not lane_valid:
                return JSONResponse(
                    status_code=409,
                    content={"detail": "processor lane lease changed"},
                    headers={"X-GPU-Fault-Processor-Retry": ("lane-lease-changed")},
                )
        elif not processor.is_leader():
            return JSONResponse(
                status_code=409,
                content={"detail": "processor leadership changed"},
            )
        report_processor_replay_phase("handler_dispatch")
        response = await call_next(request)
        report_processor_replay_phase("response_ready")
        return response
    finally:
        reset_processor_replay(replay_context)
        tracker.finish(request_id)


async def _prepare_request(
    request: Request,
    dependencies: ProcessorDispatchDependencies,
) -> PreparedProcessorRequest | Response:
    server_timing = request.scope.setdefault("gpu_fault_server_timing", {})
    oversize = _declared_oversize(request, dependencies)
    if oversize is not None:
        dependencies.state.oversize_rejections += 1
        return oversize
    decode_started = time.monotonic()
    body = await request.body()
    fault_ingress = dependencies.is_fault_ingress_path(request.url.path)
    decode_pool = (
        dependencies.fault_decode_io if fault_ingress else dependencies.decode_io
    )
    store_pool = dependencies.fault_store_io if fault_ingress else dependencies.store_io
    payload = request.scope.get("gpu_fault_json_payload")
    if payload is None:
        try:
            body, payload = await decode_pool.run(
                dependencies.decode_json_body,
                body,
                (
                    ""
                    if request.scope.get("gpu_fault_body_decompressed", False)
                    else request.headers.get("Content-Encoding", "")
                ),
            )
        except StoreIoCapacityExceeded as exc:
            return _decode_capacity_response(exc, dependencies)
        except OverflowError:
            dependencies.state.oversize_rejections += 1
            return _oversize_response(dependencies)
        except NulInRequestBody as exc:
            return _nul_response(exc, dependencies.decode_rejections)
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
        request.scope["gpu_fault_json_payload"] = payload
        request._body = body
    server_timing["decode"] = (time.monotonic() - decode_started) * 1000
    if len(body) > dependencies.processor_max_request_bytes:
        dependencies.state.oversize_rejections += 1
        return _oversize_response(dependencies)
    request_build_started = time.monotonic()
    try:
        item = await decode_pool.run(
            ProcessorRequest.from_http,
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            body=body,
            content_type=request.headers.get("Content-Type"),
            cluster_id=request.headers.get("X-GPU-Fault-Cluster-ID"),
            execution_authorized=bool(
                dependencies.context.execution_token
                and request.headers.get("X-GPU-Fault-Execution-Token")
                and secrets.compare_digest(
                    request.headers["X-GPU-Fault-Execution-Token"],
                    dependencies.context.execution_token,
                )
            ),
            parsed_payload=payload,
        )
    except StoreIoCapacityExceeded as exc:
        return _decode_capacity_response(exc, dependencies)
    idempotency_key = request.headers.get(IDEMPOTENCY_KEY_HEADER, "").strip()
    if idempotency_key:
        item = item.model_copy(
            update={
                "request_id": idempotent_request_id(
                    cluster_id=item.cluster_id,
                    path=item.path,
                    key=idempotency_key,
                    body=body,
                )
            }
        )
    server_timing["request_build"] = (time.monotonic() - request_build_started) * 1000
    return PreparedProcessorRequest(
        item=item,
        body=body,
        store_pool=store_pool,
        server_timing=server_timing,
    )


def _nul_response(exc: NulInRequestBody, rejections: dict[str, int]) -> JSONResponse:
    """A 422: the sink treats it as terminal and dead-letters the sample."""

    rejections["nul"] = rejections.get("nul", 0) + 1
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _retry_after_headers(dependencies: ProcessorDispatchDependencies) -> dict[str, str]:
    return {"Retry-After": str(dependencies.processor_retry_after_seconds)}


def _decode_capacity_response(
    exc: StoreIoCapacityExceeded,
    dependencies: ProcessorDispatchDependencies,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        headers=_retry_after_headers(dependencies),
        content={
            "detail": (
                "request deadline exceeded"
                if isinstance(exc, RequestDeadlineExceeded)
                else "request decode capacity exceeded"
            )
        },
    )


def _declared_oversize(
    request: Request,
    dependencies: ProcessorDispatchDependencies,
) -> JSONResponse | None:
    """Reject on the declared length before buffering the body.

    The size check used to run after ``await request.body()``, so a caller that
    announced 200MB got 200MB read into memory and decoded first, and every
    concurrent oversize request held that memory until the same 413 came back.

    A compressed body used to be exempt from this check altogether, on the
    correct observation that gzip output can exceed its input and so the wire
    length is not itself the limit. That left it with no bound at all: an
    announced 200MB gzip body was still buffered in full. It now gets the loosest
    length that could still decode within budget
    (:func:`max_compressed_request_bytes`), and the decoder remains authoritative
    for the decompressed size.

    In regional mode the cluster authentication middleware reads the body
    first, so it runs the same check (:func:`declared_body_oversize`) before
    its own ``await request.body()`` (A-3); this one then covers the
    single-cluster assembly, where dispatch is the first reader.
    """

    if not declared_body_oversize(
        request.headers, dependencies.processor_max_request_bytes
    ):
        return None
    return _oversize_response(dependencies)


def _oversize_response(
    dependencies: ProcessorDispatchDependencies,
) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": "processor request body is too large",
            "max_bytes": dependencies.processor_max_request_bytes,
        },
    )


async def _try_spool(
    prepared: PreparedProcessorRequest,
    dependencies: ProcessorDispatchDependencies,
) -> Response | None:
    item = prepared.item
    if not (
        dependencies.telemetry_spool_enabled
        and len(prepared.body) <= dependencies.telemetry_spool_max_item_bytes
        and item.spoolable()
    ):
        return None
    # Tighten the request's deadline, never extend it: ``set`` replaced a 15 s
    # budget with the 30 s spool budget (F-E1).
    spool_deadline = time.monotonic() + dependencies.telemetry_request_budget_seconds
    existing_deadline = REQUEST_DEADLINE.get()
    if existing_deadline is not None:
        spool_deadline = min(spool_deadline, existing_deadline)
    token = REQUEST_DEADLINE.set(spool_deadline)
    try:
        try:
            spooled, result = await dependencies.telemetry_spool_batcher.submit(item)
        except StoreIoCapacityExceeded as exc:
            return JSONResponse(
                status_code=503,
                headers={
                    "Retry-After": str(dependencies.processor_retry_after_seconds)
                },
                content={
                    "detail": (
                        "request deadline exceeded"
                        if isinstance(exc, RequestDeadlineExceeded)
                        else "store I/O capacity exceeded"
                    ),
                    "processor_request_id": item.request_id,
                },
            )
    finally:
        REQUEST_DEADLINE.reset(token)
    if result in {"global", "cluster"}:
        dependencies.telemetry_spool_rejections[result] += 1
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(dependencies.processor_retry_after_seconds)},
            content={
                "detail": "telemetry spool capacity exceeded",
                "scope": result,
            },
        )
    if spooled is None:
        raise RuntimeError("telemetry spool admission returned no request")
    dependencies.state.telemetry_spool_admitted += 1
    admitted = dependencies.telemetry_spool_admitted_by_path
    admitted[item.path] = admitted.get(item.path, 0) + 1
    if result == "coalesced":
        dependencies.state.telemetry_spool_coalesced += 1
    return JSONResponse(
        status_code=202,
        content={
            "accepted": True,
            "spooled": True,
            "coalesced": result == "coalesced",
        },
    )


async def _enqueue(
    prepared: PreparedProcessorRequest,
    dependencies: ProcessorDispatchDependencies,
) -> tuple[ProcessorRequest, str | None] | Response:
    item = prepared.item
    started = time.monotonic()
    try:
        priority = item.queue_priority()
        if priority == 100:
            queued, result = await dependencies.processor_admission_batcher.submit(item)
        elif is_reserved_tier(priority):
            queued, result = await dependencies.fault_admission_batcher.submit(item)
        elif priority == 50:
            queued, result = await dependencies.evidence_admission_batcher.submit(item)
        else:
            queued, result = await prepared.store_pool.run(
                dependencies.context.store.try_enqueue_processor_request,
                item,
                max_depth=dependencies.processor_max_queue_depth,
                max_cluster_depth=(dependencies.processor_max_cluster_queue_depth),
                reserved_fault_depth=(
                    dependencies.processor_fault_reserved_queue_depth
                ),
                reserved_cluster_fault_depth=(
                    dependencies.processor_fault_reserved_cluster_depth
                ),
                global_admission_guard=(dependencies.processor_global_admission_guard),
            )
        prepared.server_timing["admission"] = (time.monotonic() - started) * 1000
    except StoreIoCapacityExceeded as exc:
        prepared.server_timing["admission"] = (time.monotonic() - started) * 1000
        # Same answer shape as ``_poll_for_response`` and ``_try_spool``: a
        # budget the request spent waiting is a deadline, not store capacity,
        # and the receipt id lets a caller without one find its row (A-1).
        return JSONResponse(
            status_code=503,
            headers=_retry_after_headers(dependencies),
            content={
                "detail": (
                    "request deadline exceeded"
                    if isinstance(exc, RequestDeadlineExceeded)
                    else "store I/O capacity exceeded"
                ),
                "processor_request_id": item.request_id,
            },
        )
    if result in {
        "global",
        "cluster",
        "global_reserved",
        "cluster_reserved",
    }:
        dependencies.processor_admission_rejections[result] += 1
        if result in {"cluster", "cluster_reserved"}:
            cluster_key = f"{result}\x1f{item.cluster_id}"
            rejections = dependencies.processor_admission_rejections
            rejections[cluster_key] = rejections.get(cluster_key, 0) + 1
        by_path = dependencies.processor_admission_rejections_by_path
        by_path[item.path] = by_path.get(item.path, 0) + 1
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(dependencies.processor_retry_after_seconds)},
            content={
                "detail": "processor queue capacity exceeded",
                "scope": result,
            },
        )
    if queued is None:
        raise RuntimeError("processor admission returned no queued request")
    if result == "coalesced":
        dependencies.state.telemetry_coalesced += 1
    return queued, result


async def _wait_for_response(
    item: ProcessorRequest,
    dependencies: ProcessorDispatchDependencies,
    store_io: Any | None = None,
) -> Response:
    signals = getattr(dependencies.processor, "completion_signals", None)
    registration = (
        nullcontext(None) if signals is None else signals.waiting(item.request_id)
    )
    with registration as wake:
        return await _poll_for_response(item, dependencies, wake, store_io=store_io)


async def _poll_for_response(
    item: ProcessorRequest,
    dependencies: ProcessorDispatchDependencies,
    wake: asyncio.Event | None,
    *,
    store_io: Any | None = None,
) -> Response:
    # The pool the request was admitted on: a fault-ingress request waited on
    # the general pool, so the fault path's isolation ended at admission and a
    # telemetry burst could starve the wait for a fault decision (F-E5).
    pool = dependencies.store_io if store_io is None else store_io
    deadline = time.monotonic() + dependencies.processor_response_timeout_seconds
    # The request's own budget (REQUEST_DEADLINE, 15/30 s) is the fourth and
    # tightest clamp; without it the configured response timeout could never
    # take effect and the client saw a capacity 503 instead (F-E1).
    request_deadline = REQUEST_DEADLINE.get()
    if request_deadline is not None:
        deadline = min(deadline, request_deadline)
    retry_after = {"Retry-After": str(dependencies.processor_retry_after_seconds)}
    interval = RESPONSE_POLL_INITIAL_SECONDS
    while time.monotonic() < deadline:
        if wake is not None:
            # Cleared *before* the read: a completion landing between the two is
            # then seen by this read, and one landing after it leaves the event
            # set, so neither can be missed. Clearing afterwards would drop a
            # signal raised while the read was in flight.
            wake.clear()
        try:
            current = await pool.run(
                dependencies.context.store.get_processor_request,
                item.request_id,
            )
        except StoreIoCapacityExceeded as exc:
            if isinstance(exc, RequestDeadlineExceeded) or time.monotonic() >= deadline:
                break
            return JSONResponse(
                status_code=503,
                headers=retry_after,
                content={
                    "detail": "store I/O capacity exceeded",
                    "processor_request_id": item.request_id,
                },
            )
        if current.status is ProcessorRequestStatus.COMPLETED:
            headers = {}
            if current.response_content_type:
                headers["Content-Type"] = current.response_content_type
            return Response(
                content=current.response_body(),
                status_code=current.response_status or 500,
                headers=headers,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        jitter = 1.0 + random.uniform(-RESPONSE_POLL_JITTER, RESPONSE_POLL_JITTER)
        await _wait_before_next_poll(wake, min(interval * jitter, remaining))
        # Grown even when the wait ended in a signal: a request released for
        # retry is signalled too, and its waiter must not drop back to reading
        # the store every 10ms for the rest of a long retry backoff.
        interval = min(interval * RESPONSE_POLL_GROWTH, RESPONSE_POLL_MAX_SECONDS)
    return JSONResponse(
        status_code=503,
        headers=retry_after,
        content={
            "detail": "processor response timed out",
            "processor_request_id": item.request_id,
        },
    )


async def _wait_before_next_poll(wake: asyncio.Event | None, delay: float) -> None:
    """Hold for ``delay``, or until this request is finished in this process."""

    if wake is None:
        await asyncio.sleep(delay)
        return
    try:
        await asyncio.wait_for(wake.wait(), timeout=delay)
    except TimeoutError:
        # The ordinary case for a request completed elsewhere: the interval
        # expired, so poll again.
        pass


async def dispatch_processor_request(
    request: Request,
    call_next,
    dependencies: ProcessorDispatchDependencies,
) -> Response:
    processor = dependencies.processor
    if processor is None or not dependencies.requires_processor(request):
        return await call_next(request)
    replay = await _handle_replay(request, call_next, dependencies)
    if replay is not None:
        return replay
    prepared = await _prepare_request(request, dependencies)
    if isinstance(prepared, Response):
        return prepared
    if (
        dependencies.processor_queue_bypass_enabled
        and request.url.path in dependencies.processor_queue_bypass_paths
    ):
        by_path = dependencies.processor_queue_bypasses_by_path
        by_path[request.url.path] = by_path.get(request.url.path, 0) + 1
        return await call_next(request)
    spooled = await _try_spool(prepared, dependencies)
    if spooled is not None:
        return spooled
    admitted = await _enqueue(prepared, dependencies)
    if isinstance(admitted, Response):
        return admitted
    item, result = admitted
    if dependencies.returns_processor_receipt(request.url.path):
        return JSONResponse(
            status_code=202,
            content={
                "accepted": True,
                "processor_request_id": item.request_id,
                "status_url": (f"/v1/processor/requests/{item.request_id}"),
                "coalesced": result == "coalesced",
            },
        )
    return await _wait_for_response(item, dependencies, prepared.store_pool)


def install_processor_dispatch(
    app: FastAPI,
    dependencies: ProcessorDispatchDependencies,
) -> None:
    @app.middleware("http")
    async def processor_dispatch(request: Request, call_next):
        return await dispatch_processor_request(request, call_next, dependencies)
