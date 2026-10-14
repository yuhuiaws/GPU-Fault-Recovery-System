"""The in-Pod registry scripts, run in-process against the real control plane.

``resource_registry`` reaches the registry through ``kubectl exec <api-ha Pod>
-- python -c <script>``; the script talks to ``127.0.0.1:8080``. ``PodRegistry``
stands in for both: it runs the script here, with the Pod's environment, and
serves its HTTP requests from the real FastAPI application (routes, store,
exception handlers), so a test sees what the deploy host would see.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import urllib.error
import urllib.request
from email.message import Message
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from gpu_fault.admin import resource_registry as registry
from gpu_fault.app import ApplicationContext, create_app

TOKEN = "registry-pod-execution-token-" + "x" * 32
LOOPBACK = "http://127.0.0.1:8080"


class PodRegistry:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.context = ApplicationContext(execution_token=TOKEN)
        self.context.regional_mode = True
        self.client = TestClient(create_app(self.context))
        self.scripts: list[str] = []
        #: Runs before the script's first POST; a test's concurrent writer.
        self.before_post: Callable[[], None] | None = None
        monkeypatch.setattr(registry, "run_command", self.run)
        monkeypatch.setattr(registry, "_cpu_pod", lambda _site: "cpu-test")

    @property
    def store(self) -> Any:
        return self.context.store

    def urlopen(self, request: Any, timeout: float | None = None) -> io.BytesIO:
        if isinstance(request, str):
            method, url, data, headers = "GET", request, None, {}
        else:
            method, url = request.get_method(), request.full_url
            data, headers = request.data, dict(request.header_items())
        if method == "POST" and self.before_post is not None:
            hook, self.before_post = self.before_post, None
            hook()
        response = self.client.request(
            method, url.removeprefix(LOOPBACK), content=data, headers=headers
        )
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                url,
                response.status_code,
                response.reason_phrase,
                Message(),
                io.BytesIO(response.content),
            )
        return io.BytesIO(response.content)

    def run(
        self, arguments: list[str], *, input_text: str | None = None, **_options: Any
    ) -> subprocess.CompletedProcess[str]:
        script = arguments[-1]
        self.scripts.append(script)
        environment = {"GPU_FAULT_EXECUTION_TOKEN": TOKEN}
        for item in arguments:
            name, _, value = item.partition("=")
            if name == "GPU_FAULT_INSTALLATION_SITE_ID":
                environment[name] = value
        stdout, stderr = io.StringIO(), io.StringIO()
        code: object = 0
        with (
            pytest.MonkeyPatch.context() as patch,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            for name, value in environment.items():
                patch.setenv(name, value)
            patch.setattr(urllib.request, "urlopen", self.urlopen)
            patch.setattr(
                sys, "stdin", io.TextIOWrapper(io.BytesIO((input_text or "").encode()))
            )
            try:
                exec(script, {"__name__": "__main__"})
            except SystemExit as stopped:
                code = stopped.code
        returncode = code if isinstance(code, int) else (0 if code is None else 1)
        return subprocess.CompletedProcess(
            arguments, returncode, stdout.getvalue(), stderr.getvalue()
        )
