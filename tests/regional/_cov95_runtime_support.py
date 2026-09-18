"""Process and network boundaries for offline runtime coverage tests."""

from __future__ import annotations

import socket
import subprocess
import urllib.request
from typing import Any

import botocore.session
import pytest


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("runtime coverage must use a fake process or transport")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setattr(urllib.request, "urlopen", refused)
    monkeypatch.setattr(botocore.session.Session, "create_client", refused)


class Clock:
    def __init__(self, step: float = 1.0) -> None:
        self.value = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.value += self.step
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds
