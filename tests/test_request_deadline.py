from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from gpu_fault.app.middleware.backpressure import install_request_deadline
from gpu_fault.async_store import remaining_budget
from tests._builders import asgi_client


def test_fault_path_uses_its_own_request_budget() -> None:
    app = FastAPI()

    @app.get("/normal")
    async def normal() -> dict[str, float]:
        return {"remaining": remaining_budget(60)}

    @app.get("/fault")
    async def fault() -> dict[str, float]:
        return {"remaining": remaining_budget(60)}

    install_request_deadline(
        app,
        request_budget_seconds=1,
        fault_request_budget_seconds=5,
        is_fault_path=lambda path: path == "/fault",
    )

    async def scenario() -> tuple[float, float]:
        async with asgi_client(app) as client:
            normal_response = await client.get("/normal")
            fault_response = await client.get("/fault")
        return (
            float(normal_response.json()["remaining"]),
            float(fault_response.json()["remaining"]),
        )

    normal_remaining, fault_remaining = asyncio.run(scenario())

    assert 0 < normal_remaining <= 1, normal_remaining
    assert 4 < fault_remaining <= 5, fault_remaining


def test_fault_request_budget_rejects_negative_values() -> None:
    with pytest.raises(
        ValueError, match="GPU_FAULT_FAULT_REQUEST_BUDGET_SECONDS must not be negative"
    ):
        install_request_deadline(
            FastAPI(),
            request_budget_seconds=1,
            fault_request_budget_seconds=-1,
            is_fault_path=lambda _path: True,
        )
