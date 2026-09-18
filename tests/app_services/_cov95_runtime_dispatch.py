from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from gpu_fault.app.middleware.dispatch import (
    ProcessorDispatchDependencies,
    dispatch_processor_request,
)
from gpu_fault.processor_diagnostics import ProcessorReplayTracker
from tests.app_services._cov95_runtime_admission import Admission


class ImmediateIo:
    def __init__(self, *, error: Exception | None = None, fail_at: int = 1) -> None:
        self.calls = 0
        self.error = error
        self.fail_at = fail_at

    async def run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.error is not None and self.calls == self.fail_at:
            raise self.error
        return function(*args, **kwargs)


def dependencies(admission: Admission, **changes: Any) -> ProcessorDispatchDependencies:
    runtime = admission.runtime
    io = ImmediateIo()
    base = ProcessorDispatchDependencies(
        context=admission.context,
        processor=SimpleNamespace(active_consumers=True, is_leader=lambda: True),
        state=runtime.dispatch_state,
        store_io=io,
        decode_io=io,
        fault_store_io=io,
        fault_decode_io=io,
        processor_admission_batcher=runtime.admission_batcher,
        fault_admission_batcher=runtime.fault_batcher,
        evidence_admission_batcher=runtime.evidence_batcher,
        telemetry_spool_batcher=runtime.spool_batcher,
        processor_replay_tracker=ProcessorReplayTracker(),
        requires_processor=lambda request: True,
        replay_authorized=lambda request: False,
        returns_processor_receipt=lambda path: True,
        is_fault_ingress_path=lambda path: False,
        decode_json_body=runtime.decode_json_body,
        processor_max_queue_depth=runtime.max_queue_depth,
        processor_max_cluster_queue_depth=runtime.max_cluster_queue_depth,
        processor_fault_reserved_queue_depth=runtime.fault_reserved_depth,
        processor_fault_reserved_cluster_depth=runtime.fault_reserved_cluster_depth,
        processor_global_admission_guard=runtime.global_admission_guard,
        processor_max_request_bytes=runtime.max_request_bytes,
        processor_retry_after_seconds=2,
        processor_response_timeout_seconds=0.05,
        processor_queue_bypass_enabled=False,
        processor_queue_bypass_paths=set(),
        processor_admission_rejections=runtime.admission_rejections,
        processor_admission_rejections_by_path=runtime.admission_rejections_by_path,
        processor_queue_bypasses_by_path={},
        telemetry_spool_enabled=False,
        telemetry_spool_max_item_bytes=runtime.spool_max_item_bytes,
        telemetry_spool_rejections=runtime.spool_rejections,
        telemetry_spool_admitted_by_path=runtime.spool_admitted_by_path,
        telemetry_request_budget_seconds=30,
    )
    return replace(base, **changes)


def request(
    body: bytes = b"{}",
    *,
    path: str = "/v1/collector-events/gpu-metrics",
    headers: dict[str, str] | None = None,
    cached: dict | None = None,
) -> Request:
    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (key.lower().encode(), value.encode())
            for key, value in (headers or {}).items()
        ],
        "server": ("unit.invalid", 80),
        "client": ("127.0.0.1", 1234),
    }
    if cached is not None:
        scope["gpu_fault_json_payload"] = cached
    return Request(scope, receive)


def dispatch(
    admission: Admission,
    value: Request,
    deps: ProcessorDispatchDependencies,
    *,
    next_call: Any = None,
) -> Any:
    async def default_next(request: Request) -> JSONResponse:
        return JSONResponse({"handler": True})

    async def scenario() -> Any:
        try:
            return await dispatch_processor_request(
                value, next_call or default_next, deps
            )
        finally:
            await admission.close_batchers()

    return asyncio.run(scenario())
