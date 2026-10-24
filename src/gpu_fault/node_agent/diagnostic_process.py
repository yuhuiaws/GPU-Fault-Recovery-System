"""Byte/time bounded diagnostic subprocesses, including the post-kill wait."""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from typing import Any

from gpu_fault.collectors.process import BoundedProcessRunner

MAX_OUTPUT_BYTES = 1_048_576


class DiagnosticProcessRunner(BoundedProcessRunner):
    def __call__(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if timeout is None or timeout <= 0:
            raise ValueError("diagnostic subprocess requires a timeout")
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        streams = (process.stdout, process.stderr)
        output = [bytearray(), bytearray()]
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        try:
            for index, stream in enumerate(streams):
                if stream is not None:
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                for key, _ in selector.select(min(0.1, remaining)):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    output[key.data].extend(block)
                    if sum(map(len, output)) > MAX_OUTPUT_BYTES:
                        raise ValueError("diagnostic subprocess output exceeds 1 MiB")
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except BaseException:
            self._abandon(argv, process)
            raise
        finally:
            selector.close()
            for stream in streams:
                if stream is not None:
                    stream.close()
        result = subprocess.CompletedProcess(
            argv,
            process.returncode,
            *(bytes(value).decode(errors="replace") for value in output),
        )
        if check:
            result.check_returncode()
        return result


def run_probe(runner: Any, argv: list[str], **kwargs: Any) -> Any:
    bounded = getattr(runner, "diagnostic", None)
    result = bounded(argv, **kwargs) if callable(bounded) else runner(argv, **kwargs)
    if (
        len((result.stdout or "").encode()) + len((result.stderr or "").encode())
        > MAX_OUTPUT_BYTES
    ):
        raise ValueError("diagnostic subprocess output exceeds 1 MiB")
    return result
