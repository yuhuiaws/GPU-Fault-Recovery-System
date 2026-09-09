"""The regional cluster-authentication middleware is the first hop that touches
a request body in production, so the body guards have to live there too.

A-3: ``_declared_oversize`` in ``dispatch.py`` ran *after* ``auth.py`` had
already ``await request.body()``-ed the whole announcement, so a 200 MB
``Content-Length`` from a holder of a valid cluster token was buffered in full
before anything measured it -- the "reject on the declared length before
buffering" docstring described dead code in regional mode.

A-4: the fault path's dedicated decode pool only ever saw ``from_http``; the
decompress-and-parse step for an XID report queued on the general pool behind
telemetry, undoing the isolation F-E5 asked for at the very first hop.

A-8: ``payload_cluster_ids`` walked the whole payload on the event loop.

E-3: a body carrying ``\\u0000`` is refused with a 422 here, before it can
reach a ``::jsonb`` column and fail a whole spool batch of other clusters'
requests with SQLSTATE 22P05.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request

from gpu_fault.app.admission_runtime import (
    AdmissionRuntimeFactory,
    max_compressed_request_bytes,
)
from gpu_fault.app.middleware.auth import (
    RegionalAuthDependencies,
    install_regional_authorization,
)
from gpu_fault.app.runtime import ProcessorDispatchState
from gpu_fault.async_store import RequestDeadlineExceeded, StoreIoCapacityExceeded
from gpu_fault.regional import RegionalClusterRegistration

FAULT_PATH = "/v1/collector-events/nvidia-kernel"
ROUTINE_PATH = "/v1/collector-events/node-logs"
MAX_BYTES = 4096


class _Runner:
    """Stands in for an ``AsyncStoreExecutor``; records what ran on it."""

    def __init__(self, name: str = "runner") -> None:
        self.name = name
        self.calls: list[Any] = []
        self.active = False

    async def run(self, function, *args, **kwargs):
        self.calls.append(function)
        self.active = True
        try:
            return function(*args, **kwargs)
        finally:
            self.active = False


class _Saturated:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def run(self, *_args, **_kwargs):
        raise self.error


def _registration() -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id="cluster-a",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/a",
        token_sha256="a" * 64,
        allowed_namespaces=["training"],
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
    )


def _payload_cluster_ids(payload: dict) -> set[str]:
    return {payload["cluster_id"]} if "cluster_id" in payload else set()


def _app_with_auth(
    *,
    decode_io=None,
    fault_decode_io=None,
    is_fault_path=None,
    payload_cluster_ids=_payload_cluster_ids,
    dispatch_state=None,
    decode_rejections=None,
    retry_after_seconds=2,
) -> FastAPI:
    app = FastAPI()

    @app.post(FAULT_PATH)
    @app.post(ROUTINE_PATH)
    async def echo(request: Request) -> dict[str, Any]:
        return {"json": await request.json()}

    install_regional_authorization(
        app,
        RegionalAuthDependencies(
            context=SimpleNamespace(regional_mode=True, execution_token=None),
            replay_authorized=lambda _request: False,
            authorization_bucket=lambda *_arguments: "cluster-token",
            authenticate_cluster=lambda *_arguments: _registration(),
            decode_io=decode_io or _Runner("general"),
            decode_json_body=AdmissionRuntimeFactory._decode_json_body(MAX_BYTES),
            payload_cluster_ids=payload_cluster_ids,
            processor_max_request_bytes=MAX_BYTES,
            fault_decode_io=fault_decode_io,
            is_fault_path=is_fault_path,
            dispatch_state=dispatch_state,
            decode_rejections=decode_rejections,
            retry_after_seconds=retry_after_seconds,
        ),
    )
    return app


def _post(
    app: FastAPI, path: str, *, headers: dict[str, str], body: bytes = b"", receive=None
) -> tuple[int, Any, dict[str, str]]:
    """Drive the ASGI app directly so the test owns ``receive``."""

    header_items = [
        (b"host", b"control-plane"),
        (b"x-gpu-fault-cluster-id", b"cluster-a"),
        (b"authorization", b"Bearer cluster-a-token"),
        (b"content-type", b"application/json"),
    ]
    for name, value in headers.items():
        header_items.append((name.lower().encode(), value.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": header_items,
        "root_path": "",
        "scheme": "http",
        "server": ("control-plane", 80),
        "client": ("10.0.0.5", 4242),
    }
    messages: list[dict] = []

    async def default_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive or default_receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    raw = b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )
    response_headers = {k.decode(): v.decode() for k, v in start["headers"]}
    return start["status"], (json.loads(raw) if raw else None), response_headers


def _cluster_body(**extra: Any) -> bytes:
    return json.dumps({"cluster_id": "cluster-a", **extra}).encode()


# --- A-3 ----------------------------------------------------------------------


def test_regional_authentication_rejects_a_declared_oversize_body_unread() -> None:
    async def receive():
        pytest.fail("the oversize body was buffered before the size check")

    state = ProcessorDispatchState()
    app = _app_with_auth(dispatch_state=state)

    status, body, _ = _post(
        app,
        ROUTINE_PATH,
        headers={"Content-Length": str(MAX_BYTES + 1)},
        receive=receive,
    )

    assert status == 413, body
    assert body["max_bytes"] == MAX_BYTES
    assert state.oversize_rejections == 1


def test_regional_authentication_bounds_a_declared_compressed_body_too() -> None:
    async def receive():
        pytest.fail("the oversize compressed body was buffered before the check")

    app = _app_with_auth()
    beyond = max_compressed_request_bytes(MAX_BYTES) + 1

    status, _, _ = _post(
        app,
        ROUTINE_PATH,
        headers={"Content-Length": str(beyond), "Content-Encoding": "gzip"},
        receive=receive,
    )

    assert status == 413


def test_regional_authentication_still_admits_a_compressed_body_within_budget() -> None:
    app = _app_with_auth()
    wire = gzip.compress(_cluster_body())

    status, body, _ = _post(
        app,
        ROUTINE_PATH,
        headers={"Content-Length": str(len(wire)), "Content-Encoding": "gzip"},
        body=wire,
    )

    assert status == 200, body
    assert body["json"] == {"cluster_id": "cluster-a"}


# --- A-4 ----------------------------------------------------------------------


def test_a_fault_report_is_decoded_on_the_fault_pool() -> None:
    general = _Runner("general")
    fault = _Runner("fault")
    app = _app_with_auth(
        decode_io=general,
        fault_decode_io=fault,
        is_fault_path=lambda path: path == FAULT_PATH,
    )

    status, _, _ = _post(app, FAULT_PATH, headers={}, body=_cluster_body())
    assert status == 200
    assert len(fault.calls) == 1 and general.calls == []

    status, _, _ = _post(app, ROUTINE_PATH, headers={}, body=_cluster_body())
    assert status == 200
    assert len(general.calls) == 1 and len(fault.calls) == 1


def test_without_a_fault_pool_every_path_uses_the_general_pool() -> None:
    """Backwards compatible with a factory that has not adopted the field."""

    general = _Runner("general")
    app = _app_with_auth(decode_io=general, is_fault_path=lambda path: True)

    status, _, _ = _post(app, FAULT_PATH, headers={}, body=_cluster_body())

    assert status == 200
    assert len(general.calls) == 1


def test_decode_pool_saturation_reads_the_configured_retry_after() -> None:
    app = _app_with_auth(
        decode_io=_Saturated(StoreIoCapacityExceeded("saturated")),
        retry_after_seconds=7,
    )

    status, body, headers = _post(app, ROUTINE_PATH, headers={}, body=_cluster_body())

    assert status == 503
    assert body["detail"] == "request decode capacity exceeded"
    assert headers["retry-after"] == "7"


def test_an_expired_deadline_at_decode_is_named_as_such() -> None:
    app = _app_with_auth(decode_io=_Saturated(RequestDeadlineExceeded("expired")))

    status, body, _ = _post(app, ROUTINE_PATH, headers={}, body=_cluster_body())

    assert status == 503
    assert body["detail"] == "request deadline exceeded"


# --- A-8 ----------------------------------------------------------------------


def test_the_payload_cluster_walk_runs_on_the_decode_pool() -> None:
    general = _Runner("general")
    seen: list[bool] = []

    def payload_cluster_ids(payload: dict) -> set[str]:
        seen.append(general.active)
        return {payload["cluster_id"]}

    app = _app_with_auth(decode_io=general, payload_cluster_ids=payload_cluster_ids)

    status, _, _ = _post(app, ROUTINE_PATH, headers={}, body=_cluster_body())

    assert status == 200
    assert seen == [True], "payload_cluster_ids ran on the event loop"


def test_a_too_complex_payload_is_still_a_422_from_the_pool() -> None:
    def payload_cluster_ids(_payload: dict) -> set[str]:
        raise ValueError("JSON payload is too structurally complex")

    app = _app_with_auth(payload_cluster_ids=payload_cluster_ids)

    status, body, _ = _post(app, ROUTINE_PATH, headers={}, body=_cluster_body())

    assert status == 422
    assert "too structurally complex" in body["detail"]


def test_a_foreign_cluster_id_is_still_refused() -> None:
    app = _app_with_auth()

    status, body, _ = _post(
        app,
        ROUTINE_PATH,
        headers={},
        body=json.dumps({"cluster_id": "cluster-b"}).encode(),
    )

    assert status == 403
    assert "does not match" in body["detail"]


# --- E-3 ----------------------------------------------------------------------


def test_a_body_carrying_a_nul_escape_is_refused_with_422_and_counted() -> None:
    rejections = {"nul": 0}
    app = _app_with_auth(decode_rejections=rejections)
    body = _cluster_body(message="kernel\u0000panic")
    assert b"\\u0000" in body

    status, response, _ = _post(app, ROUTINE_PATH, headers={}, body=body)

    assert status == 422, response
    assert "NUL" in response["detail"]
    assert rejections == {"nul": 1}


def test_an_escaped_backslash_before_u0000_is_not_a_nul() -> None:
    """The bytes heuristic must be confirmed on the parsed payload."""

    rejections = {"nul": 0}
    app = _app_with_auth(decode_rejections=rejections)
    body = _cluster_body(message="literal \\u0000 text")
    assert b"\\\\u0000" in body

    status, response, _ = _post(app, ROUTINE_PATH, headers={}, body=body)

    assert status == 200, response
    assert rejections == {"nul": 0}


def test_the_dispatch_decoder_refuses_a_nul_the_same_way() -> None:
    """Single-cluster mode has no authentication hop; dispatch must agree."""

    from gpu_fault.app.middleware import dispatch
    from tests.processor.test_processor_response_wait import (
        _dependencies,
        _no_call_next,
        _request,
    )

    dependencies = _dependencies(SimpleNamespace(), timeout_seconds=5.0)
    dependencies = dispatch.ProcessorDispatchDependencies(
        **{
            **dependencies.__dict__,
            "decode_json_body": AdmissionRuntimeFactory._decode_json_body(MAX_BYTES),
            "decode_rejections": {"nul": 0},
        }
    )
    body = json.dumps({"message": "a\u0000b"}).encode()

    response = asyncio.run(
        dispatch.dispatch_processor_request(_request(body), _no_call_next, dependencies)
    )

    assert response.status_code == 422
    assert dependencies.decode_rejections == {"nul": 1}
