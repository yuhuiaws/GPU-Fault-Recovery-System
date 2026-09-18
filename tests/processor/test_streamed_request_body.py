from __future__ import annotations

import asyncio
import gzip
import time
from collections.abc import AsyncIterator

import pytest
from starlette.requests import ClientDisconnect, Request
from starlette.types import Message

from gpu_fault.app import create_app, request_body
from gpu_fault.app.admission_runtime import max_compressed_request_bytes
from gpu_fault.async_store import REQUEST_DEADLINE, RequestDeadlineExceeded
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, registration


def request_from_chunks(
    chunks: list[bytes], *, encoding: str = "", consumed: list[int] | None = None
) -> Request:
    pending = iter(enumerate(chunks))

    async def receive() -> Message:
        index, chunk = next(pending)
        if consumed is not None:
            consumed.append(index)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/body",
            "headers": [(b"content-encoding", encoding.encode())] if encoding else [],
        },
        receive,
    )


def test_stream_reader_stops_before_buffering_an_oversize_chunk() -> None:
    consumed: list[int] = []
    request = request_from_chunks([b"x" * 64, b"x", b"unread"], consumed=consumed)
    with pytest.raises(OverflowError):
        asyncio.run(request_body.read_bounded_request_body(request, 64))
    assert consumed == [0, 1], (
        "an oversize request must not consume its remaining stream"
    )


@pytest.mark.parametrize("chunks", [[b""], [b"a", b"b"], [b"ab"], [b"", b"ab", b""]])
def test_stream_reader_caches_complete_bytes_for_downstream_readers(
    chunks: list[bytes],
) -> None:
    consumed: list[int] = []
    request = request_from_chunks(chunks, consumed=consumed)

    async def scenario() -> None:
        expected = b"".join(chunks)
        assert await request_body.read_bounded_request_body(request, 2) == expected
        assert await request_body.read_bounded_request_body(request, 2) == expected
        assert await request.body() == expected

    asyncio.run(scenario())
    assert consumed == list(range(len(chunks))), "cached bytes must not reread receive"


def test_cached_bytes_are_checked_against_the_selected_limit() -> None:
    request = request_from_chunks([b"oversize"])
    asyncio.run(request.body())
    with pytest.raises(OverflowError):
        asyncio.run(request_body.read_bounded_request_body(request, 2))


def test_compressed_stream_uses_wire_limit_and_preserves_its_bytes() -> None:
    body = bytes(range(256))
    compressed = gzip.compress(body, compresslevel=0)
    assert len(compressed) > len(body), "fixture must exceed the decompressed limit"
    request = request_from_chunks([compressed[:20], compressed[20:]], encoding="gzip")
    assert (
        asyncio.run(request_body.read_bounded_request_body(request, len(body)))
        == compressed
    )


def test_decompressed_body_cannot_reuse_the_larger_gzip_wire_limit() -> None:
    request = request_from_chunks([b"x" * 65], encoding="gzip")
    request.scope["gpu_fault_body_decompressed"] = True
    with pytest.raises(OverflowError):
        asyncio.run(request_body.read_bounded_request_body(request, 64))


@pytest.mark.parametrize("maximum", [0, -1])
def test_stream_reader_rejects_invalid_limits(maximum: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        asyncio.run(
            request_body.read_bounded_request_body(request_from_chunks([b""]), maximum)
        )


def test_expired_budget_refuses_before_reading_any_bytes() -> None:
    consumed: list[int] = []
    request = request_from_chunks([b"{}"], consumed=consumed)
    token = REQUEST_DEADLINE.set(time.monotonic() - 1)
    try:
        with pytest.raises(RequestDeadlineExceeded):
            asyncio.run(request_body.read_bounded_request_body(request, 64))
    finally:
        REQUEST_DEADLINE.reset(token)
    assert not consumed, "an expired request must not start receiving its body"


@pytest.mark.parametrize("request_budget", [False, True])
def test_stalled_receive_is_cancelled_by_the_remaining_read_budget(
    monkeypatch: pytest.MonkeyPatch, request_budget: bool
) -> None:
    monkeypatch.setattr(request_body, "BODY_READ_TIMEOUT_SECONDS", 0.02)
    cancelled: list[bool] = []

    async def receive() -> Message:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
        raise AssertionError("a stalled read should never complete")

    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": []}, receive
    )

    async def scenario() -> None:
        token = REQUEST_DEADLINE.set(
            time.monotonic() + 0.01 if request_budget else None
        )
        try:
            with pytest.raises(RequestDeadlineExceeded):
                await request_body.read_bounded_request_body(request, 64)
        finally:
            REQUEST_DEADLINE.reset(token)

    asyncio.run(scenario())
    assert cancelled == [True], "deadline handling must cancel the in-flight receive"


def test_disconnect_does_not_return_a_successful_partial_body() -> None:
    async def receive() -> Message:
        return {"type": "http.disconnect"}

    request = Request({"type": "http", "headers": []}, receive)
    with pytest.raises(ClientDisconnect):
        asyncio.run(request_body.read_bounded_request_body(request, 64))


@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("encoding", ["", "gzip"])
def test_chunked_flood_is_rejected_before_decode_or_processor_admission(
    monkeypatch: pytest.MonkeyPatch, regional: bool, encoding: str
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("POD_UID", "bounded-body-pod")
    context = build_context(execution_token="e" * 32, processor_replay_secret="r" * 32)
    context.regional_mode = regional
    if regional:
        context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    app = create_app(context)
    consumed: list[int] = []
    limit = max_compressed_request_bytes(1024) if encoding else 1024

    async def chunks() -> AsyncIterator[bytes]:
        consumed.append(1)
        yield b"x" * limit
        consumed.append(2)
        yield b"x"
        pytest.fail("the middleware consumed bytes after the request exceeded its cap")

    async def scenario() -> None:
        async with asgi_client(app) as client:
            response = await client.post(
                "/v1/collector-events/node-logs",
                content=chunks(),
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": encoding,
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
        assert response.status_code == 413, response.text
        assert response.json()["max_bytes"] == 1024

    asyncio.run(scenario())
    assert consumed == [1, 2], "only bounded chunks should have been consumed"
    assert context.store.processor_queue_stats()["depth"] == 0, (
        "oversize input must not reach durable processor admission"
    )


@pytest.mark.parametrize("regional", [False, True])
def test_slow_chunked_request_returns_retryable_failure_without_admission(
    monkeypatch: pytest.MonkeyPatch, regional: bool
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setattr(request_body, "BODY_READ_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setenv("POD_UID", "bounded-body-pod")
    context = build_context(execution_token="e" * 32, processor_replay_secret="r" * 32)
    context.regional_mode = regional
    if regional:
        context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    app = create_app(context)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"{"
        await asyncio.Event().wait()
        raise AssertionError("the stalled body must be cancelled")

    async def scenario() -> None:
        async with asgi_client(app) as client:
            response = await client.post(
                "/v1/collector-events/node-logs",
                content=chunks(),
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
        assert response.status_code == 503, response.text
        assert "deadline" in response.json()["detail"]
        assert response.headers["Retry-After"]

    asyncio.run(scenario())
    assert context.store.processor_queue_stats()["depth"] == 0, (
        "a partial body must never be admitted"
    )
