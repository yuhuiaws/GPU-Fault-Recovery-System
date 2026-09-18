from __future__ import annotations

import io
import json
from typing import Any
from urllib.request import Request

import pytest

from gpu_fault.cluster_executor import regional_client


class Wire:
    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.calls: list[tuple[Request, dict[str, Any]]] = []

    def __call__(self, request: Request, **kwargs: Any) -> io.BytesIO:
        self.calls.append((request, kwargs))
        response = self.response(request) if callable(self.response) else self.response
        if isinstance(response, BaseException):
            raise response
        body = (
            response
            if isinstance(response, bytes)
            else json.dumps(response, default=str).encode()
        )
        return io.BytesIO(body)


def client(
    monkeypatch: pytest.MonkeyPatch, response: Any = None
) -> tuple[regional_client.RegionalExecutorClient, Wire]:
    wire = Wire(response)
    monkeypatch.setattr(regional_client, "urlopen", wire)
    return regional_client.RegionalExecutorClient(
        "https://unit.invalid",
        "cluster-a",
        "a" * 32,
        executor_artifact_sha256="a" * 64,
        executor_compatibility_digest="b" * 64,
    ), wire
