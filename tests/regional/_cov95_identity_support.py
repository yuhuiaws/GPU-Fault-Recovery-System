"""Offline boundaries and deterministic clocks for the coverage-95 tests."""

from __future__ import annotations

import asyncio
import io
import socket
import subprocess
import urllib.error
import urllib.request
from typing import Any

import httpx
import pytest

from gpu_fault.app import create_app


@pytest.fixture(autouse=True)
def offline_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("coverage tests must not start processes or use a network")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setattr(urllib.request, "urlopen", refused)


class Clock:
    def __init__(self, step: float = 1.0) -> None:
        self.value = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.value += self.step
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class ASGIBridge:
    """Exercise urllib request builders against real auth without a socket."""

    def __init__(self, context: Any) -> None:
        self.app = create_app(context)
        self.requests: list[urllib.request.Request] = []

    def urlopen(self, request: urllib.request.Request, **kwargs: Any) -> io.BytesIO:
        assert kwargs["timeout"] > 0
        self.requests.append(request)

        async def send() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app, client=("10.0.1.5", 40000)),
                base_url="https://unit.invalid",
            ) as client:
                return await client.request(
                    request.get_method(),
                    request.full_url,
                    headers=dict(request.header_items()),
                    content=request.data,
                )

        response = asyncio.run(send())
        stream = io.BytesIO(response.content)
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url, response.status_code, "ASGI denial", {}, stream
            )
        stream.status = response.status_code
        return stream
