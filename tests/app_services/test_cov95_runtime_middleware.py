from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from gpu_fault.app.authorization import (
    ExplicitAuthorizationRegistry,
    authorization_bucket,
)
from gpu_fault.app.cluster_binding import payload_cluster_ids
from gpu_fault.app.middleware.auth import (
    RegionalAuthDependencies,
    install_regional_authorization,
)
from gpu_fault.app.middleware.backpressure import (
    IngressBackpressureDependencies,
    install_ingress_backpressure,
    install_request_deadline,
)
from gpu_fault.async_store import REQUEST_DEADLINE
from tests.app_services._cov95_runtime_admission import Admission
from tests.app_services._cov95_runtime_admission import (
    admission_fixture as admission_fixture,
)
from tests.app_services._cov95_runtime_dispatch import ImmediateIo
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime
from tests.regional._regional_support import TOKEN_A, registration

PATH = "/v1/unit-observation"
HEADERS = {"X-GPU-Fault-Cluster-ID": "cluster-a", "Authorization": f"Bearer {TOKEN_A}"}


def auth_app(
    admission: Admission, *, cached: Any = ..., **changes: Any
) -> tuple[FastAPI, ImmediateIo, list[str]]:
    app = FastAPI()
    served = []

    @app.post(PATH)
    @authorization_bucket("cluster-token")
    async def handle() -> dict:
        served.append("served")
        return {"accepted": True}

    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)
    cluster = registration("cluster-a", TOKEN_A)
    io = ImmediateIo()

    def authenticate(cluster_id: str | None, authorization: str | None) -> Any:
        if cluster_id != cluster.cluster_id or not authorization:
            raise HTTPException(status_code=401)
        scheme, _, token = authorization.partition(" ")
        if scheme != "Bearer" or cluster.matched_token_slot(token) is None:
            raise HTTPException(status_code=403)
        return cluster

    deps = RegionalAuthDependencies(
        context=SimpleNamespace(regional_mode=True, execution_token=None),
        replay_authorized=lambda value: False,
        authorization_bucket=registry.effective,
        authenticate_cluster=authenticate,
        decode_io=io,
        decode_json_body=admission.runtime.decode_json_body,
        payload_cluster_ids=payload_cluster_ids,
        processor_max_request_bytes=1_000_000,
    )
    install_regional_authorization(app, replace(deps, **changes))
    if cached is not ...:

        @app.middleware("http")
        async def cached_payload(value: Request, call_next: Any) -> Any:
            value.scope["gpu_fault_json_payload"] = cached
            return await call_next(value)

    return app, io, served


@pytest.mark.parametrize(
    ("cached", "body", "status", "calls"),
    [
        ({}, b"", 200, 0),
        ({"nested": [{"cluster": "cluster-a"}]}, b"{}", 200, 1),
        ({"nested": [{"clusterId": "cluster-b"}]}, b"{}", 403, 1),
        ({"nested": [{"cluster_id": ""}]}, b"{}", 422, 1),
        ({"nested": [{"cluster_id": 1}]}, b"{}", 422, 1),
    ],
)
def test_cached_payload_is_rescanned_for_nested_cluster_identity(
    admission: Admission, cached: Any, body: bytes, status: int, calls: int
) -> None:
    app, io, served = auth_app(admission, cached=cached)
    with TestClient(app) as client:
        response = client.post(PATH, content=body, headers=HEADERS)
    assert response.status_code == status
    assert io.calls == calls
    assert served == (["served"] if status == 200 else [])
    if status == 403:
        assert "all payload cluster_id values" in response.json()["detail"]
    if status == 422:
        assert "non-empty strings" in response.json()["detail"]


def test_cluster_identity_scan_rejects_excessive_structure_before_the_handler(
    admission: Admission,
) -> None:
    app, io, served = auth_app(admission)
    body = json.dumps({"nested": [0] * 100_001}).encode()
    with TestClient(app) as client:
        response = client.post(PATH, content=body, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["detail"] == "JSON payload is too structurally complex"
    assert io.calls == 1
    assert served == []


def test_nul_refusal_works_without_an_optional_rejection_counter(
    admission: Admission,
) -> None:
    app, _io, served = auth_app(admission, decode_rejections=None)
    with TestClient(app) as client:
        response = client.post(PATH, json={"nested": ["\x00"]}, headers=HEADERS)
    assert response.status_code == 422
    assert "NUL" in response.json()["detail"]
    assert served == []


@pytest.mark.parametrize("request_budget,fault_budget", [(-1, None), (1, -1)])
def test_negative_request_budgets_refuse_middleware_installation(
    request_budget: float, fault_budget: float | None
) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        install_request_deadline(
            FastAPI(),
            request_budget_seconds=request_budget,
            fault_request_budget_seconds=fault_budget,
        )


def test_disabled_deadline_preserves_existing_server_timing() -> None:
    app = FastAPI()

    @app.get("/unit")
    async def handle() -> JSONResponse:
        assert REQUEST_DEADLINE.get() is None
        return JSONResponse({"ok": True}, headers={"Server-Timing": "db;dur=1"})

    install_request_deadline(app, request_budget_seconds=0)
    with TestClient(app) as client:
        response = client.get("/unit")
    assert response.status_code == 200
    assert response.headers["server-timing"].startswith(
        "db;dur=1, gpu_fault_total;dur="
    ), response.headers["server-timing"]
    assert float(response.headers["X-GPU-Fault-Server-Duration-Ms"]) >= 0


@pytest.mark.parametrize("fault", [False, True])
def test_expired_ingress_budget_rejects_before_acquiring_capacity_or_calling_handler(
    fault: bool,
) -> None:
    app = FastAPI()
    served = []

    @app.post("/unit")
    async def handle() -> dict:
        served.append("served")
        return {}

    rejections = {"fault": 0, "normal": 0}
    normal, reserved = asyncio.Semaphore(1), asyncio.Semaphore(1)
    install_ingress_backpressure(
        app,
        IngressBackpressureDependencies(
            service_role="ingress",
            requires_processor=lambda value: True,
            replay_authorized=lambda value: False,
            is_fault_path=lambda path: fault,
            fault_semaphore=reserved,
            normal_semaphore=normal,
            fault_wait_seconds=1,
            normal_wait_seconds=1,
            rejections=rejections,
        ),
    )

    @app.middleware("http")
    async def expired(value: Request, call_next: Any) -> Any:
        token = REQUEST_DEADLINE.set(time.monotonic() - 1)
        try:
            return await call_next(value)
        finally:
            REQUEST_DEADLINE.reset(token)

    with TestClient(app) as client:
        response = client.post("/unit")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["detail"] == "request deadline exceeded at ingress"
    assert rejections == {"fault": int(fault), "normal": int(not fault)}
    assert not normal.locked() and not reserved.locked(), (
        "an expired request must not consume either admission lane"
    )
    assert served == []
