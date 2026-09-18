from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from gpu_fault.admin import process_supervisor


def isolated_supervisor(monkeypatch):
    name = "cov95_isolated_supervisor"
    spec = importlib.util.spec_from_file_location(name, process_supervisor.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class CompletionTransport:
    """Feed completion reports through owned pipes without starting a process."""

    def __init__(self):
        self.reports = [
            {"status": "exited", "returncode": 0},
            {"status": "exited", "returncode": 7},
        ]
        self.calls = []
        self.streams = []
        self.pid = 700001
        self.returncode = 0
        self.stdin = self.stdout = self.stderr = None
        self.spawn_error = None
        self.communicate_error = None

    def install(self, module, monkeypatch):
        monkeypatch.setattr(
            module,
            "subprocess",
            SimpleNamespace(**{**vars(subprocess), "Popen": self.spawn}),
        )
        handlers = {}
        monkeypatch.setattr(
            module,
            "signal",
            SimpleNamespace(
                **{
                    **vars(signal),
                    "signal": lambda signum, handler: handlers.update(
                        {signum: handler}
                    ),
                    "getsignal": lambda signum: handlers.get(signum, signal.SIG_DFL),
                }
            ),
        )

    def spawn(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.spawn_error is not None:
            raise self.spawn_error
        for option, document in zip(
            ("--report-fd", "--command-report-fd"), self.reports, strict=True
        ):
            descriptor = int(command[command.index(option) + 1])
            payload = (
                document
                if isinstance(document, bytes)
                else json.dumps(document).encode()
            )
            os.write(descriptor, payload)
        self.stdin = io.StringIO() if kwargs["stdin"] is not None else None
        self.stdout = io.StringIO("output") if kwargs["stdout"] is not None else None
        self.stderr = (
            io.StringIO("diagnostic\n") if kwargs["stderr"] is not None else None
        )
        self.streams = [
            stream
            for stream in (self.stdin, self.stdout, self.stderr)
            if stream is not None
        ]
        return self

    def poll(self):
        return self.returncode

    def communicate(self, input_text=None, timeout=None):
        if self.communicate_error is not None:
            error, self.communicate_error = self.communicate_error, None
            raise error
        if self.stdin is not None and input_text is not None:
            self.stdin.write(input_text)
        return (
            self.stdout.getvalue() if self.stdout else None,
            self.stderr.getvalue() if self.stderr else None,
        )

    def assert_closed(self):
        assert all(stream.closed for stream in self.streams), (
            "owned command stream leaked"
        )
        for command, _options in self.calls:
            for option in ("--report-fd", "--command-report-fd"):
                descriptor = int(command[command.index(option) + 1])
                assert not Path(f"/proc/self/fd/{descriptor}").exists(), (
                    "report writer leaked"
                )
