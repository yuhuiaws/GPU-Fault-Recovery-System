"""Own and reap the entire probe subtree, including children of a killed reader."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

from gpu_fault.node_agent import observation_worker


def children() -> list[int]:
    return [
        int(value)
        for value in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
    ]


def reap_tree() -> None:
    while True:
        for pid in children():
            try:
                descriptor = os.pidfd_open(pid)
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                finally:
                    os.close(descriptor)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                break
        time.sleep(0.05)


def main() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish probe subtree ownership")
    stopping = False

    def stop(_signal: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 1
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", observation_worker.__name__],
            stdin=sys.stdin.buffer,
            stdout=sys.stdout.buffer,
            stderr=subprocess.DEVNULL,
        )
        while process.poll() is None and not stopping:
            time.sleep(0.02)
        code = process.returncode if process.returncode is not None else 124
    finally:
        reap_tree()
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
