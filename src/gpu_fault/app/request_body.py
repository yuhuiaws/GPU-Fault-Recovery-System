"""Bound request buffering before JSON decoding or persistent admission."""

from __future__ import annotations

import asyncio
import time

from starlette.requests import Request

from gpu_fault.app.admission_runtime import (
    declared_body_oversize,
    max_compressed_request_bytes,
)
from gpu_fault.async_store import RequestDeadlineExceeded, remaining_budget

BODY_READ_TIMEOUT_SECONDS = 30.0


async def read_bounded_request_body(request: Request, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max request bytes must be positive")
    if declared_body_oversize(request.headers, max_bytes):
        raise OverflowError("processor request body is too large")
    compressed = not request.scope.get("gpu_fault_body_decompressed", False) and bool(
        request.headers.get("Content-Encoding", "").strip()
    )
    limit = max_compressed_request_bytes(max_bytes) if compressed else max_bytes
    budget = remaining_budget(BODY_READ_TIMEOUT_SECONDS)
    if budget <= 0:
        raise RequestDeadlineExceeded("request body read deadline exceeded")
    cached = getattr(request, "_body", None)
    if isinstance(cached, bytes):
        if len(cached) > limit:
            raise OverflowError("processor request body is too large")
        return cached

    body = bytearray()
    deadline = time.monotonic() + budget
    try:
        async with asyncio.timeout(budget):
            async for chunk in request.stream():
                if time.monotonic() >= deadline:
                    raise RequestDeadlineExceeded("request body read deadline exceeded")
                if len(chunk) > limit - len(body):
                    raise OverflowError("processor request body is too large")
                body.extend(chunk)
    except TimeoutError as exc:
        raise RequestDeadlineExceeded("request body read deadline exceeded") from exc
    result = bytes(body)
    # Starlette must replay these same bounded bytes to the downstream middleware.
    request._body = result
    return result
