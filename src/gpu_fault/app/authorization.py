from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple

from fastapi.routing import APIRoute
from starlette.routing import Match

from gpu_fault.env import env_bool

AUTHORIZATION_BUCKET_ATTRIBUTE = "__gpu_fault_authorization_bucket__"
CLUSTER_ID_HEADER = "X-GPU-Fault-Cluster-ID"
EXECUTION_TOKEN_HEADER = "X-GPU-Fault-Execution-Token"
LOOPBACK_CLIENT_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
UNDOCUMENTED_PUBLIC_PATHS = frozenset(
    {
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
)
# Both regional middlewares resolve the bucket for every request, and the
# resolution is a linear Starlette match over ~90 routes. The cache is bounded
# because the key includes the concrete request path, which callers control:
# unbounded memoisation would turn random 404 paths into unbounded memory.
DECLARED_CACHE_SIZE = 2048
AUTHORIZATION_BUCKETS = frozenset(
    {
        "public",
        "metrics",
        "cluster-token",
        "dual-credential",
        "execution-token",
    }
)


def validate_direct_client_identity_environment() -> None:
    if os.getenv("FORWARDED_ALLOW_IPS", "").strip():
        raise RuntimeError(
            "FORWARDED_ALLOW_IPS is forbidden: replay and metrics "
            "loopback authorization require the direct socket peer"
        )
    if env_bool("GPU_FAULT_PROXY_HEADERS_ENABLED", False):
        raise RuntimeError("proxy-derived client identity is forbidden")


def authorization_bucket(bucket: str) -> Callable:
    if bucket not in AUTHORIZATION_BUCKETS:
        raise ValueError(f"invalid authorization bucket: {bucket}")

    def decorate(endpoint):
        setattr(endpoint, AUTHORIZATION_BUCKET_ATTRIBUTE, bucket)
        return endpoint

    return decorate


def iter_api_routes(routes):
    visited = set()

    def collect(items):
        for route in items:
            identity = id(route)
            if identity in visited:
                continue
            visited.add(identity)
            if isinstance(route, APIRoute):
                yield route
            included = getattr(route, "original_router", None)
            if included is not None:
                yield from collect(getattr(included, "routes", ()))

    yield from collect(routes)


def dual_credential_bucket(headers: Mapping[str, str]) -> str:
    """Pick the credential class a dual-credential request actually presented.

    These routes accept either the operator execution token or a per-cluster
    token. Selecting on ``X-GPU-Fault-Cluster-ID`` alone let the caller pick:
    that header is a routing hint, not a credential, so sending it skipped the
    execution-token check and handed the request to per-cluster authentication,
    which answers 401 without a bearer token and 503 with ``Retry-After`` while
    the registry snapshot is stale. It also denied an operator who legitimately
    named a cluster while holding only the execution token.

    So the bucket follows the credential the request actually carries, and a
    request carrying neither falls back to the operator bucket, whose denial is
    the same 403 for every cluster id.
    """

    if (headers.get(EXECUTION_TOKEN_HEADER) or "").strip():
        return "execution-token"
    authorization = (headers.get("Authorization") or "").strip()
    if (
        authorization.startswith("Bearer ")
        and (headers.get(CLUSTER_ID_HEADER) or "").strip()
    ):
        return "cluster-token"
    return "execution-token"


class DeclaredCacheStats(NamedTuple):
    hits: int
    misses: int
    size: int


@dataclass(frozen=True)
class AuthorizedRoute:
    route: APIRoute
    bucket: str


class ExplicitAuthorizationRegistry:
    def __init__(self) -> None:
        self._routes: list[AuthorizedRoute] = []
        self.inventory: dict[str, str] = {}
        self._reset_declared_cache()

    def _reset_declared_cache(self) -> None:
        self._declared = lru_cache(maxsize=DECLARED_CACHE_SIZE)(self._match_declared)

    def declared_cache_info(self) -> DeclaredCacheStats:
        """Expose the memoisation counters so tests and operators can see hits."""

        info = self._declared.cache_info()
        return DeclaredCacheStats(info.hits, info.misses, info.currsize)

    def load(self, routes) -> None:
        entries = []
        inventory = {}
        for route in iter_api_routes(routes):
            path = route.path
            if not (
                path.startswith("/v1/") or path in {"/healthz", "/livez", "/metrics"}
            ):
                continue
            bucket = getattr(
                route.endpoint,
                AUTHORIZATION_BUCKET_ATTRIBUTE,
                None,
            )
            if bucket not in AUTHORIZATION_BUCKETS:
                raise RuntimeError(f"route {path} has no explicit authorization bucket")
            entries.append(AuthorizedRoute(route, bucket))
            inventory[path] = bucket
        self._routes = entries
        self.inventory = inventory
        self._reset_declared_cache()

    def _match_declared(self, path: str, method: str) -> str | None:
        scope = {
            "type": "http",
            "path": path,
            "root_path": "",
            "method": method,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("localhost", 80),
            "client": ("127.0.0.1", 1),
        }
        for entry in self._routes:
            match, _ = entry.route.matches(scope)
            if match is Match.FULL:
                return entry.bucket
        if path in UNDOCUMENTED_PUBLIC_PATHS:
            return "public"
        return None

    def declared(
        self,
        path: str,
        method: str = "GET",
    ) -> str | None:
        return self._declared(path, method)

    def effective(
        self,
        path: str,
        headers,
        client_host: str | None = None,
        method: str = "GET",
    ) -> str | None:
        declared = self.declared(path, method)
        if declared == "metrics":
            if client_host in LOOPBACK_CLIENT_HOSTS:
                return "public"
            return "execution-token"
        if declared == "dual-credential":
            return dual_credential_bucket(headers)
        return declared
