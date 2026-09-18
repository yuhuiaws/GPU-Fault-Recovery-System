from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.client import HTTPMessage
from threading import get_ident
from typing import Any

import pytest

from gpu_fault.transport import http_client


@dataclass
class Reply:
    status: int = 200
    body: bytes = b"unit-response"
    will_close: bool = False
    version: int = 11
    reason: str = "synthetic response"
    msg: HTTPMessage = field(default_factory=HTTPMessage)

    def read(self) -> bytes:
        return self.body


class Wire:
    def __init__(self) -> None:
        self.replies: list[Reply | BaseException] = []
        self.connections: list[Any] = []
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def connect(self, host: str, port: int, **options: Any) -> Any:
        wire = self

        class Socket:
            def __init__(self) -> None:
                self.timeouts: list[float] = []

            def settimeout(self, timeout: float) -> None:
                self.timeouts.append(timeout)

        class Connection:
            def __init__(self) -> None:
                self.host = host
                self.port = port
                self.options = options
                self.thread_id = get_ident()
                self.timeout = options.get("timeout")
                self.sock: Socket | None = None
                self.closed = False

            def request(
                self,
                method: str,
                path: str,
                *,
                body: bytes | None,
                headers: dict[str, str],
            ) -> None:
                wire.requests.append((method, path, body, headers))
                self.sock = self.sock or Socket()

            def getresponse(self) -> Reply:
                response = wire.replies.pop(0) if wire.replies else Reply()
                if isinstance(response, BaseException):
                    raise response
                return response

            def close(self) -> None:
                self.closed = True
                self.sock = None

        connection = Connection()
        self.connections.append(connection)
        return connection


@pytest.fixture(name="http_wire")
def http_wire_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[Wire]:
    http_client.CONNECTION_POOL.close()
    wire = Wire()
    monkeypatch.setattr(http_client, "HTTPConnection", wire.connect)
    monkeypatch.setattr(http_client, "HTTPSConnection", wire.connect)
    try:
        yield wire
    finally:
        http_client.CONNECTION_POOL.close()
