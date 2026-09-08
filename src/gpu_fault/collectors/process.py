"""Subprocess calls a child in uninterruptible sleep cannot outlive.

Every collector that reads a device reads it through a command -- nvidia-smi,
ethtool, smartctl, ipmitool, nvidia-smi again for discovery at startup -- and
those are exactly the commands that block in the driver when the device is the
thing that is broken. This module owns the bound, so both the host collector
and the cluster-side context builder can share it without either importing the
other.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
from typing import Callable


LOGGER = logging.getLogger(__name__)


def _reap_abandoned_process(argv: list[str], process: subprocess.Popen[str]) -> None:
    """Wait out a killed child that has not left the kernel yet.

    Runs on a daemon thread and touches nothing but its own child: the circuit
    breaker and every other piece of collector state stay owned by the
    collection thread.
    """

    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()
    with contextlib.suppress(Exception):
        process.wait()
    LOGGER.info(
        "abandoned %s (pid %s) finally exited with %s",
        argv[0] if argv else "?",
        process.pid,
        process.returncode,
    )


class BoundedProcessRunner:
    """``subprocess.run`` that a child in uninterruptible sleep cannot outlive.

    CPython implements ``run(timeout=...)`` as ``kill()`` followed by a
    *blocking* ``wait()``. The failure this collector exists to report --
    nvidia-smi wedged inside the driver (XID 79, a GPU off the bus, a
    fabric-manager hang) -- is exactly the case where SIGKILL is not acted on
    until the syscall returns, so ``TimeoutExpired`` never reached the caller:
    the nvidia-smi circuit breaker never engaged, ``collect_once`` never
    returned, and the node sent no host telemetry at all. Here the second wait
    is bounded too and a child that outlives it is handed to a daemon reaper,
    so a call returns within ``timeout + kill_grace_seconds``. Every
    ``self.runner`` call site shares this bound, so smartctl on a dying NVMe
    and ethtool on a wedged EFA netdev cannot hold a tick either.
    """

    def __init__(
        self,
        *,
        popen: Callable[..., subprocess.Popen[str]] | None = None,
        kill_grace_seconds: float = 1.0,
    ) -> None:
        # Deliberately *not* ``popen=subprocess.Popen`` as a default: that
        # binds the real ``Popen`` when this module is imported, so a test that
        # monkeypatches ``subprocess.Popen`` never reaches the runner and its
        # assertions pass vacuously -- which is how the unbounded startup path
        # survived a regression test written for it. ``None`` means "whatever
        # ``subprocess.Popen`` is when the call happens".
        self.popen = popen
        self.kill_grace_seconds = kill_grace_seconds

    def __call__(
        self,
        argv: list[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        pipe = subprocess.PIPE if capture_output else None
        popen = self.popen if self.popen is not None else subprocess.Popen
        process = popen(argv, stdout=pipe, stderr=pipe, text=text)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._abandon(argv, process)
            raise
        completed = subprocess.CompletedProcess(
            argv, process.returncode or 0, stdout, stderr
        )
        if check:
            completed.check_returncode()
        return completed

    def _abandon(self, argv: list[str], process: subprocess.Popen[str]) -> None:
        with contextlib.suppress(OSError):
            process.kill()
        try:
            process.wait(timeout=self.kill_grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        else:
            # ``communicate`` timed out, so its pipes were never drained or
            # closed. A tick runs every 15 s and the collector's RLIMIT_NOFILE
            # is the unit's default, so leaking two descriptors per timed-out
            # call ends in EMFILE -- where every probe fails, not just this one.
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()
            return
        LOGGER.warning(
            "%s did not die within %gs of SIGKILL (uninterruptible sleep in a "
            "driver); abandoning it to a background reaper",
            argv[0] if argv else "?",
            self.kill_grace_seconds,
        )
        threading.Thread(
            target=_reap_abandoned_process,
            args=(argv, process),
            name="gpu-fault-process-reaper",
            daemon=True,
        ).start()
