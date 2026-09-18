"""Owned-child lifecycle; no API accepts a PID or names an existing worker."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from scripts.e2e.regional.ha011_contracts import ProofError


class FileSignal:
    """A killed waiter cannot strand a cross-process condition lock."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def is_set(self) -> bool:
        return self.path.is_file()

    def set(self) -> None:
        self.path.touch()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True


class WorkerProcess:
    def __init__(
        self, target: Callable[..., None], *, role: str, request_id: str
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.signals = TemporaryDirectory(prefix="ha011-owned-worker-")
        root = Path(self.signals.name)
        self.stop = FileSignal(root / "stop")
        self.release = FileSignal(root / "release")
        self.process = context.Process(
            target=target,
            args=(role, request_id, child, self.stop, self.release),
            daemon=False,
        )
        self.events: list[dict[str, Any]] = []
        self.crashed = False
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            self.signals.cleanup()
            raise
        finally:
            child.close()

    def receive(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProofError("owned worker observation deadline expired")
        if not self.connection.poll(min(remaining, 0.1)):
            if not self.process.is_alive():
                raise ProofError("owned worker exited before the required observation")
            return
        try:
            event = self.connection.recv()
        except EOFError:
            raise ProofError("owned worker observation pipe closed") from None
        if not isinstance(event, dict) or event.get("pid") != self.process.pid:
            raise ProofError("observation did not come from the owned child")
        if event.get("kind") == "error":
            raise ProofError("owned worker reported an execution failure")
        self.events.append(event)

    def wait(
        self, kind: str, deadline: float, request_id: str | None = None
    ) -> dict[str, Any]:
        while True:
            for event in self.events:
                if event["kind"] == kind and (
                    request_id is None or event.get("request_id") == request_id
                ):
                    return event
            self.receive(deadline)

    def crash(self) -> int:
        if not self.process.is_alive():
            raise ProofError("the busy owned worker exited before the crash")
        self.process.kill()
        self.process.join(timeout=5)
        if self.process.is_alive() or self.process.exitcode != -9:
            raise ProofError("owned process crash could not be confirmed")
        self.crashed = True
        return self.process.exitcode

    def alive(self) -> bool:
        return bool(self.process.is_alive())

    def finish(self) -> bool:
        if self.process.is_alive():
            self.stop.set()
            self.release.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
            raise ProofError("owned worker did not stop within its grace period")
        if self.process.exitcode != 0 and not (
            self.crashed and self.process.exitcode == -9
        ):
            raise ProofError("owned worker exited unsuccessfully")
        return True

    def close(self) -> None:
        try:
            self.finish()
        finally:
            self.connection.close()
            if not self.process.is_alive():
                self.signals.cleanup()
