"""Bounds on the request body an ingress replica is willing to hold.

The wire limit (``GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES``) was enforced on the
*decompressed* size only after the whole thing existed in memory, and compressed
bodies skipped the pre-buffer length check entirely. Both halves of that are
memory exhaustion an authenticated collector could trigger with a small request,
so both get a test here rather than only the plain-body case that already lived
in ``tests/processor/_processor_leadership_cases_4.py``.
"""

from __future__ import annotations

import asyncio
import gzip
import zlib

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.admission_runtime import (
    inflate_bounded,
    max_compressed_request_bytes,
)
from tests._builders import asgi_client


def _gzip_bomb(output_bytes: int) -> bytes:
    """A gzip frame that inflates to ``output_bytes`` of zeros.

    Zeros compress about 1000:1, so the frame stays trivially small while the
    output is whatever this asks for. That ratio is the whole attack.
    """

    return gzip.compress(b"\0" * output_bytes, compresslevel=9)


def test_a_bomb_never_allocates_more_than_the_limit() -> None:
    # 8 MiB of output against a 1 KiB budget. The assertion that matters is not
    # the exception type but that the exception arrives without the 8 MiB ever
    # being built; the wire frame here is a few kilobytes.
    bomb = _gzip_bomb(8 * 1024 * 1024)
    assert len(bomb) < 16 * 1024, "fixture is not a bomb if the frame is large"
    with pytest.raises(OverflowError):
        inflate_bounded(bomb, 1024)


def test_a_body_exactly_at_the_limit_still_decodes() -> None:
    payload = b"\0" * 1024
    assert inflate_bounded(gzip.compress(payload), 1024) == payload


def test_one_byte_over_the_limit_is_rejected() -> None:
    with pytest.raises(OverflowError):
        inflate_bounded(gzip.compress(b"\0" * 1025), 1024)


def test_corrupt_gzip_raises_a_client_error_not_zlib_error() -> None:
    """``zlib.error`` must not escape.

    ``middleware/auth.py`` and ``middleware/dispatch.py`` both map
    ``OSError``/``EOFError``/``ValueError`` to 400 and let anything else become a
    500. ``zlib.error`` is neither, so the incremental inflate has to translate
    it or a corrupt body would report a server fault.
    """

    frame = bytearray(gzip.compress(b'{"cluster_id": "cluster-a"}'))
    frame[12] ^= 0xFF
    with pytest.raises(ValueError) as raised:
        inflate_bounded(bytes(frame), 1024)
    assert not isinstance(raised.value, zlib.error), (
        "a zlib.error reaching the middleware is reported as a 500 server fault: "
        f"{raised.value!r}"
    )


def test_truncated_gzip_raises_eof_like_the_stdlib_did() -> None:
    frame = gzip.compress(b"\0" * 512)
    with pytest.raises(EOFError):
        inflate_bounded(frame[: len(frame) - 4], 1024)


def test_trailing_data_after_the_stream_is_rejected() -> None:
    """Concatenated members are refused rather than partly decoded.

    ``gzip.decompress`` decodes every member, which is an unbounded number of
    them; a single ``decompressobj`` decodes the first and leaves the rest in
    ``unused_data``. Returning just the first member would silently drop caller
    data, so this is an error instead.
    """

    with pytest.raises(ValueError):
        inflate_bounded(gzip.compress(b"{}") + gzip.compress(b"{}"), 1024)


def test_the_compressed_wire_bound_admits_incompressible_input() -> None:
    """A body at the limit that does not compress must still fit under the bound.

    This is the property that makes the pre-buffer check safe to apply to
    compressed requests at all: worst case, deflate stores the bytes verbatim and
    adds block headers plus the gzip envelope.
    """

    limit = 1024 * 1024
    stored = zlib.compressobj(level=0, wbits=16 + zlib.MAX_WBITS)
    frame = stored.compress(b"\0" * limit) + stored.flush()
    assert len(frame) > limit, "level 0 should be storing, not compressing"
    assert len(frame) <= max_compressed_request_bytes(limit)


def test_the_compressed_wire_bound_rejects_a_declared_flood() -> None:
    assert max_compressed_request_bytes(16 * 1024 * 1024) < 17 * 1024 * 1024


def test_the_compressed_wire_bound_requires_a_positive_limit() -> None:
    with pytest.raises(ValueError):
        max_compressed_request_bytes(0)


def test_processor_rejects_a_gzip_bomb_over_the_wire(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-gzip-bomb")
    # 64 KiB rather than the 1 KiB the unit tests above use, so the bomb's wire
    # frame fits under the pre-buffer length bound and the request actually
    # reaches the decoder. That is the path under test; a smaller limit would be
    # answered by the length check and prove nothing about decompression.
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", "65536")
    token = "processor-bomb-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)
    bomb = _gzip_bomb(8 * 1024 * 1024)

    async def scenario():
        async with asgi_client(app) as client:
            rejected = await client.post(
                "/v1/collector-events/node-logs",
                content=bomb,
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/json",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            metrics = await client.get("/metrics")
            return rejected, metrics

    rejected, metrics = asyncio.run(scenario())

    # The frame is well inside the wire bound, so this 413 can only come from the
    # decoder stopping at the output cap.
    assert len(bomb) <= max_compressed_request_bytes(65536)
    assert rejected.status_code == 413
    assert rejected.json()["max_bytes"] == 65536
    assert context.store.processor_queue_stats()["depth"] == 0
    assert "gpu_fault_processor_oversize_rejections_total 1" in metrics.text


def test_processor_rejects_a_declared_oversize_compressed_body(monkeypatch) -> None:
    """The wire length is checked before the body is buffered.

    A compressed request used to be exempt from the pre-buffer check, so a
    declared 200MB gzip body was read into memory in full and only then measured.
    """

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-gzip-declared")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", "1024")
    token = "processor-declared-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)
    oversize = b"x" * (max_compressed_request_bytes(1024) + 1)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.post(
                "/v1/collector-events/node-logs",
                content=oversize,
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/json",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )

    rejected = asyncio.run(scenario())

    # Not a valid gzip frame at all: reaching the decoder would be a 400, so a
    # 413 proves the declared length was rejected first.
    assert rejected.status_code == 413
    assert rejected.json()["max_bytes"] == 1024


def test_processor_still_accepts_a_compressed_body_within_budget(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-gzip-ok")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", str(1024 * 1024))
    token = "processor-ok-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.post(
                "/v1/collector-events/node-logs",
                content=gzip.compress(b'{"events": []}'),
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/json",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )

    accepted = asyncio.run(scenario())

    assert accepted.status_code != 413, accepted.text
    assert accepted.status_code != 400, accepted.text
