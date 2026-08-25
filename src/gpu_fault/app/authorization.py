from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

from fastapi.routing import APIRoute
from starlette.routing import Match


AUTHORIZATION_BUCKET_ATTRIBUTE = "__gpu_fault_authorization_bucket__"
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
    if os.getenv("GPU_FAULT_PROXY_HEADERS_ENABLED", "false").strip().lower() == "true":
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


@dataclass(frozen=True)
class AuthorizedRoute:
    route: APIRoute
    bucket: str


class ExplicitAuthorizationRegistry:
    def __init__(self) -> None:
        self._routes: list[AuthorizedRoute] = []
        self.inventory: dict[str, str] = {}

    def load(self, routes) -> None:
        entries = []
        inventory = {}
        for route in iter_api_routes(routes):
            path = route.path
            if not (path.startswith("/v1/") or path in {"/healthz", "/metrics"}):
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

    def declared(
        self,
        path: str,
        method: str = "GET",
    ) -> str | None:
        for entry in self._routes:
            match, _ = entry.route.matches(
                {
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
            )
            if match is Match.FULL:
                return entry.bucket
        if path in {
            "/openapi.json",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
        }:
            return "public"
        return None

    def effective(
        self,
        path: str,
        headers,
        client_host: str | None = None,
        method: str = "GET",
    ) -> str | None:
        declared = self.declared(path, method)
        if declared == "metrics":
            return (
                "public"
                if client_host in {"127.0.0.1", "::1", "localhost"}
                else "execution-token"
            )
        if declared == "dual-credential":
            return (
                "cluster-token"
                if headers.get("X-GPU-Fault-Cluster-ID")
                else "execution-token"
            )
        return declared
