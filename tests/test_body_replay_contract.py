"""What the ingress middleware needs from Starlette's private request state.

`auth.py` and `dispatch.py` decode a request body once -- decompress it, parse the
JSON, size it -- and then hand the decoded bytes on to the route by assigning
``request._body``. That attribute is private to a library FastAPI pins only as
``starlette>=0.46``, so `pyproject.toml` carries a narrow direct bound and these
tests say what that bound is protecting: an upgrade that kept the attribute name
but changed how ``BaseHTTPMiddleware`` replays it would surface as routes
receiving compressed bytes or an empty body, not as an import error.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Request

from gpu_fault.app.middleware.auth import (
    RegionalAuthDependencies,
    install_regional_authorization,
)
from gpu_fault.regional import RegionalClusterRegistration
from tests._builders import asgi_client


def _echo_app() -> FastAPI:
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, Any]:
        # Reads the body itself, exactly as the routes behind the middleware do.
        return {"body": (await request.body()).decode(), "json": await request.json()}

    return app


def _post(app: FastAPI, **keywords: Any) -> Any:
    async def scenario() -> Any:
        async with asgi_client(app) as client:
            return await client.post("/echo", **keywords)

    return asyncio.run(scenario())


def test_substituted_body_is_what_the_route_reads() -> None:
    """A decoded body assigned to ``_body`` must replace the wire body."""

    app = _echo_app()

    @app.middleware("http")
    async def decompress(request: Request, call_next):
        request._body = gzip.decompress(await request.body())
        return await call_next(request)

    response = _post(
        app,
        content=gzip.compress(json.dumps({"cluster_id": "cluster-a"}).encode()),
        headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["json"] == {"cluster_id": "cluster-a"}


def test_the_cached_body_and_not_receive_is_what_is_replayed() -> None:
    """Once the body has been read, ``_body`` wins and ``_receive`` is not read.

    This is why the middleware substitutes only ``_body``: a middleware that also
    installs a replaying ``_receive`` is writing code that never runs, and the
    reader is left believing the replay is what carries the payload. The
    assertion runs the two against each other so a Starlette that reversed the
    precedence would be caught rather than silently making the dead branch live.
    """

    app = _echo_app()

    @app.middleware("http")
    async def replay(request: Request, call_next):
        await request.body()
        request._body = json.dumps({"from": "body"}).encode()

        async def receive() -> dict[str, Any]:
            return {
                "type": "http.request",
                "body": json.dumps({"from": "receive"}).encode(),
                "more_body": False,
            }

        request._receive = receive
        return await call_next(request)

    response = _post(
        app,
        content=json.dumps({"from": "wire"}).encode(),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["json"] == {"from": "body"}


def test_reading_the_body_twice_returns_the_same_bytes() -> None:
    """The middleware reads the body, then the route reads it again.

    Nothing in the stack buffers it for them; both reads land on Starlette's own
    cache, and a Starlette that stopped caching would hand the route an empty
    body instead of raising.
    """

    app = _echo_app()
    seen: list[bytes] = []

    @app.middleware("http")
    async def observe(request: Request, call_next):
        seen.append(await request.body())
        return await call_next(request)

    payload = json.dumps({"read": "twice"}).encode()
    response = _post(app, content=payload, headers={"Content-Type": "application/json"})

    assert response.status_code == 200, response.text
    assert seen == [payload]
    assert response.json()["body"] == payload.decode()


class _Runner:
    """Stand in for the decode thread pool."""

    async def run(self, function, *arguments, **keywords):
        return function(*arguments, **keywords)


def _registration() -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id="cluster-a",
        region="us-east-1",
        hyperpod_cluster_name="gpu-a",
        eks_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        token_sha256="a" * 64,
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
    )


def _decode(body: bytes, content_encoding: str) -> tuple[bytes, dict[str, Any]]:
    if content_encoding.lower() == "gzip":
        body = gzip.decompress(body)
    return body, json.loads(body) if body else {}


def test_cluster_authentication_hands_the_route_a_decompressed_body() -> None:
    """The whole point of the substitution, through the real middleware.

    Collectors gzip their payloads above a size threshold, so on a cluster-token
    route the bytes on the wire are not JSON. Authentication has to decode them to
    check the payload's `cluster_id` values, and the route then reads the body
    again -- this asserts it gets what authentication decoded, not the gzip frame.
    """

    app = _echo_app()
    install_regional_authorization(
        app,
        RegionalAuthDependencies(
            context=SimpleNamespace(regional_mode=True, execution_token=None),
            replay_authorized=lambda _request: False,
            authorization_bucket=lambda *_arguments: "cluster-token",
            authenticate_cluster=lambda *_arguments: _registration(),
            decode_io=_Runner(),
            decode_json_body=_decode,
            payload_cluster_ids=lambda payload: {payload["cluster_id"]},
            processor_max_request_bytes=1024,
        ),
    )

    response = _post(
        app,
        content=gzip.compress(json.dumps({"cluster_id": "cluster-a"}).encode()),
        headers={
            "Content-Encoding": "gzip",
            "Content-Type": "application/json",
            "X-GPU-Fault-Cluster-ID": "cluster-a",
            "Authorization": "Bearer cluster-a-token",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["json"] == {"cluster_id": "cluster-a"}


def test_cluster_authentication_rejects_a_foreign_payload_cluster_id() -> None:
    """The decoded payload is what the cluster check runs on, not the wire bytes."""

    app = _echo_app()
    install_regional_authorization(
        app,
        RegionalAuthDependencies(
            context=SimpleNamespace(regional_mode=True, execution_token=None),
            replay_authorized=lambda _request: False,
            authorization_bucket=lambda *_arguments: "cluster-token",
            authenticate_cluster=lambda *_arguments: _registration(),
            decode_io=_Runner(),
            decode_json_body=_decode,
            payload_cluster_ids=lambda payload: {payload["cluster_id"]},
            processor_max_request_bytes=1024,
        ),
    )

    response = _post(
        app,
        content=gzip.compress(json.dumps({"cluster_id": "cluster-b"}).encode()),
        headers={
            "Content-Encoding": "gzip",
            "Content-Type": "application/json",
            "X-GPU-Fault-Cluster-ID": "cluster-a",
            "Authorization": "Bearer cluster-a-token",
        },
    )

    assert response.status_code == 403, response.text
    assert "does not match" in response.json()["detail"]
