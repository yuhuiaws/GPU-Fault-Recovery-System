from __future__ import annotations

import io
import os
import selectors
import signal
import subprocess
import sys
from collections import deque
from types import SimpleNamespace

from gpu_fault.admin import api_budget as budget


class Pipe(io.BytesIO):
    def __init__(self, descriptor):
        super().__init__()
        self.descriptor = descriptor

    def fileno(self):
        return self.descriptor


class ShimTransport:
    """A CLI transport whose process, pipes and signals never reach the host."""

    def __init__(self):
        self.calls = []
        self.waits = []
        self.signals = []
        self.handlers = {}
        self.reads = {901: deque([b"out\n", b""]), 902: deque([b"err\n", b""])}
        self.result = 7
        self.returncode = None
        self.wait_failures = deque()
        self.spawn_error = None
        self.missing_pipes = False
        self.pid = 700001
        self.stdout = None
        self.stderr = None
        self.output = io.TextIOWrapper(io.BytesIO(), write_through=True)
        self.errors = io.TextIOWrapper(io.BytesIO(), write_through=True)
        self.os = SimpleNamespace(**vars(os))
        self.os.killpg = self.killpg
        self.os.read = self.read
        self.os.set_blocking = self.set_blocking
        self.signal = SimpleNamespace(**vars(signal))
        self.signal.signal = self.register_handler
        self.subprocess = SimpleNamespace(**vars(subprocess))
        self.subprocess.Popen = self.spawn
        self.selectors = SimpleNamespace(**vars(selectors))
        self.selectors.DefaultSelector = self.selector

    def install(self, monkeypatch, arguments):
        monkeypatch.setattr(budget, "os", self.os)
        monkeypatch.setattr(budget, "signal", self.signal)
        monkeypatch.setattr(budget, "subprocess", self.subprocess)
        monkeypatch.setattr(budget, "selectors", self.selectors)
        monkeypatch.setattr(
            budget,
            "sys",
            SimpleNamespace(
                **{
                    **vars(sys),
                    "argv": ["api-shim", *arguments],
                    "stdout": self.output,
                    "stderr": self.errors,
                }
            ),
        )

    def register_handler(self, signum, handler):
        self.handlers[signum] = handler

    def spawn(self, arguments, **kwargs):
        self.calls.append((arguments, kwargs))
        if self.spawn_error is not None:
            raise self.spawn_error
        if not self.missing_pipes:
            self.stdout = Pipe(901) if kwargs["stdout"] is not None else None
            self.stderr = Pipe(902) if kwargs["stderr"] is not None else None
        return self

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.wait_failures:
            raise self.wait_failures.popleft()
        self.returncode = self.result
        return self.result

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append(signal.SIGTERM)

    def kill(self):
        self.signals.append(signal.SIGKILL)

    def killpg(self, pid, signum):
        assert pid == self.pid, "attempted to signal outside the fake CLI"
        self.signals.append(signum)

    def read(self, descriptor, size):
        assert descriptor in self.reads, "attempted to read a host descriptor"
        item = self.reads[descriptor].popleft()
        if isinstance(item, BaseException):
            raise item
        assert len(item) <= size
        return item

    def set_blocking(self, descriptor, blocking):
        assert descriptor in self.reads and not blocking

    def selector(self):
        registered = {}

        class Selector:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                registered.clear()

            def register(self, descriptor, events):
                registered[descriptor] = events

            def unregister(self, descriptor):
                del registered[descriptor]

            def get_map(self):
                return registered

            def select(self, timeout):
                assert 0 < timeout <= 0.1
                return [
                    (
                        SimpleNamespace(
                            fd=stream if isinstance(stream, int) else stream.fileno(),
                            fileobj=stream,
                        ),
                        events,
                    )
                    for stream, events in registered.items()
                ]

        return Selector()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        for stream in (self.stdout, self.stderr):
            if stream is not None:
                stream.close()
