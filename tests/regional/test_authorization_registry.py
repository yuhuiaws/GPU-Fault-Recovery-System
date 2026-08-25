from __future__ import annotations

import pytest
from fastapi import APIRouter, FastAPI

from gpu_fault.app import create_app
from gpu_fault.app.authorization import (
    ExplicitAuthorizationRegistry,
    authorization_bucket,
    iter_api_routes,
    validate_direct_client_identity_environment,
)
from gpu_fault.app.cluster_binding import CLUSTER_IDENTIFIER_FIELDS, payload_cluster_ids
from tests._builders import build_context


def test_every_application_route_declares_authorization_bucket() -> None:
    app = create_app(build_context())
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    application_paths = {
        route.path
        for route in iter_api_routes(app.routes)
        if route.path.startswith("/v1/") or route.path in {"/healthz", "/metrics"}
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


def test_proxy_derived_client_identity_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
    with pytest.raises(RuntimeError, match="FORWARDED_ALLOW_IPS"):
        validate_direct_client_identity_environment()
