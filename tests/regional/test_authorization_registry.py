from __future__ import annotations

import asyncio

import pytest
from fastapi import APIRouter, FastAPI
from starlette.datastructures import Headers

from gpu_fault.app import create_app
from gpu_fault.app.authorization import (
    ExplicitAuthorizationRegistry,
    authorization_bucket,
    iter_api_routes,
    validate_direct_client_identity_environment,
)
from gpu_fault.app.cluster_binding import CLUSTER_IDENTIFIER_FIELDS, payload_cluster_ids
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, registration

DUAL_CREDENTIAL_PATH = "/v1/gpu-metrics/cluster-a/node-1/latest"


def test_every_application_route_declares_authorization_bucket() -> None:
    app = create_app(build_context())
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    application_paths = {
        route.path
        for route in iter_api_routes(app.routes)
        if route.path.startswith("/v1/")
        or route.path in {"/healthz", "/livez", "/metrics"}
    }
    assert set(registry.inventory) == application_paths
    assert (
        registry.declared("/v1/processor/requests/request-a", "GET") == "cluster-token"
    )
    assert registry.declared("/v1/capabilities/operations", "GET") == "execution-token"


def test_unannotated_route_fails_closed_at_startup() -> None:
    router = APIRouter()

    @router.get("/v1/unclassified")
    async def unclassified():
        return {}

    app = FastAPI()
    app.include_router(router)
    with pytest.raises(RuntimeError, match="explicit authorization"):
        ExplicitAuthorizationRegistry().load(app.routes)


def test_route_exists_tells_an_unserved_path_from_a_declared_route() -> None:
    """The default deny answers 403 for a path it cannot bucket; a path no route
    serves must read 404 instead -- a collector's outbox dead-letters a 404 and
    replays a 403 forever (NET-008, 2026-09-10)."""
    app = create_app(build_context())
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    served = ["/v1/capabilities/operations", DUAL_CREDENTIAL_PATH, "/healthz"]
    unserved = ["/v1/collector-events/retired-acceptance-channel", "/v1/no-such-route"]
    assert [registry.route_exists(path) for path in served] == [True] * 3, served
    assert [registry.route_exists(path) for path in unserved] == [False] * 2, unserved


def test_no_write_route_is_public_or_metrics() -> None:
    app = create_app(build_context())
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    unsafe = []
    for route in iter_api_routes(app.routes):
        writes = set(route.methods or ()) & {"POST", "PUT", "PATCH", "DELETE"}
        if not writes:
            continue
        bucket = registry.inventory.get(route.path)
        if bucket in {None, "public", "metrics"}:
            unsafe.append((route.path, sorted(writes), bucket))

    assert unsafe == []


def test_included_router_routes_are_loaded_recursively() -> None:
    router = APIRouter()

    @router.get("/healthz")
    @authorization_bucket("public")
    async def healthz():
        return {}

    included = type("IncludedRouter", (), {"original_router": router})()
    registry = ExplicitAuthorizationRegistry()

    registry.load([included])

    assert registry.declared("/healthz") == "public"


def test_authorization_decorator_rejects_unknown_bucket() -> None:
    with pytest.raises(ValueError, match="invalid authorization"):
        authorization_bucket("inherited-prefix")


def test_cluster_identifier_aliases_are_all_bound() -> None:
    assert CLUSTER_IDENTIFIER_FIELDS == {"cluster_id", "clusterId", "cluster"}
    assert payload_cluster_ids(
        {
            "cluster_id": "a",
            "items": [{"clusterId": "b"}, {"metadata": {"cluster": "c"}}],
        }
    ) == {"a", "b", "c"}


def test_declared_bucket_resolution_is_memoised_and_reset_by_load() -> None:
    """Every request resolves its bucket twice, so the Starlette match is cached.

    Both regional middlewares ask for the bucket of the same request, and each
    unmemoised answer is a linear match over every registered route. The cache
    has to be dropped on ``load`` or a reloaded application would keep serving
    buckets from routes it no longer has.
    """

    app = create_app(build_context())
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    first = registry.declared(DUAL_CREDENTIAL_PATH, "GET")
    second = registry.declared(DUAL_CREDENTIAL_PATH, "GET")
    other_method = registry.declared(DUAL_CREDENTIAL_PATH, "POST")
    info = registry.declared_cache_info()

    assert (first, second) == ("dual-credential", "dual-credential")
    assert other_method is None, "the bucket cache ignored the request method"
    assert (info.hits, info.misses) == (1, 2)

    registry.load(app.routes)

    assert registry.declared_cache_info().hits == 0, "load kept a stale bucket cache"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, "execution-token"),
        ({"X-GPU-Fault-Execution-Token": "operator-token"}, "execution-token"),
        (
            {"Authorization": "Bearer cluster-token", "X-GPU-Fault-Cluster-ID": "c"},
            "cluster-token",
        ),
        ({"X-GPU-Fault-Cluster-ID": "c"}, "execution-token"),
        ({"X-GPU-Fault-Cluster-ID": " "}, "execution-token"),
        ({"Authorization": "Bearer cluster-token"}, "execution-token"),
        (
            {"Authorization": "Basic abc", "X-GPU-Fault-Cluster-ID": "c"},
            "execution-token",
        ),
        (
            {
                "X-GPU-Fault-Execution-Token": "operator-token",
                "Authorization": "Bearer cluster-token",
                "X-GPU-Fault-Cluster-ID": "c",
            },
            "execution-token",
        ),
    ],
)
def test_dual_credential_bucket_follows_the_presented_credential(
    headers: dict[str, str], expected: str
) -> None:
    """A caller cannot choose its own authorization class with a hint header.

    ``X-GPU-Fault-Cluster-ID`` is a routing hint, not a credential. While the
    bucket keyed off its presence alone, anyone could send it and be handed to
    per-cluster authentication instead of the execution-token check.
    """

    app = create_app(build_context())
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    assert registry.effective(DUAL_CREDENTIAL_PATH, Headers(headers)) == expected


def test_cluster_hint_without_a_credential_never_reaches_cluster_auth() -> None:
    """A credential-less probe gets one uniform denial, not the cluster path.

    Per-cluster authentication answers 401 when the bearer token is missing and
    503 with ``Retry-After`` while the registry snapshot is stale, so a caller
    that only sent the hint header could tell those states apart on a route that
    has an operator credential available to fail closed with instead.
    """

    context = build_context()
    context.regional_mode = True
    context.execution_token = "operator-token"
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))

    async def scenario() -> None:
        async with asgi_client(context) as client:
            registered = await client.get(
                DUAL_CREDENTIAL_PATH, headers={"X-GPU-Fault-Cluster-ID": "cluster-a"}
            )
            unregistered = await client.get(
                DUAL_CREDENTIAL_PATH, headers={"X-GPU-Fault-Cluster-ID": "cluster-zz"}
            )

            assert registered.status_code == 403, registered.text
            assert unregistered.json() == registered.json()
            assert "X-GPU-Fault-Execution-Token" in registered.json()["detail"], (
                "the cluster hint still routed the probe into cluster authentication"
            )

    asyncio.run(scenario())


def test_proxy_derived_client_identity_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
    with pytest.raises(RuntimeError, match="FORWARDED_ALLOW_IPS"):
        validate_direct_client_identity_environment()


@pytest.mark.parametrize("token", ["1", "yes", "on"])
def test_proxy_headers_switch_is_refused_however_it_is_spelled(
    monkeypatch, token: str
) -> None:
    """``GPU_FAULT_PROXY_HEADERS_ENABLED=1`` used to slip past the guard."""

    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    monkeypatch.setenv("GPU_FAULT_PROXY_HEADERS_ENABLED", token)
    with pytest.raises(RuntimeError, match="proxy-derived"):
        validate_direct_client_identity_environment()
