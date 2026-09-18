"""Start only this probe's tracer after the parent pins its process identity."""

from __future__ import annotations

import ctypes
import os
import resource
import signal
import sys

MAX_TRACE_BYTES = 4 * 1024 * 1024 + 1


def run(arguments: list[str]) -> int:
    try:
        parent, gate, maximum = map(int, arguments[:3])
        if (
            parent <= 0
            or gate < 0
            or not 0 < maximum <= MAX_TRACE_BYTES
            or len(arguments) < 4
            or arguments[3] != "strace"
        ):
            return 125
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
            return 125
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (maximum, maximum))
        try:
            proceed = os.read(gate, 1) == b"G"
        finally:
            os.close(gate)
        if not proceed or os.getppid() != parent:
            return 125
        os.execvp(arguments[3], arguments[3:])
    except (OSError, ValueError):
        return 125
    return 125


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
