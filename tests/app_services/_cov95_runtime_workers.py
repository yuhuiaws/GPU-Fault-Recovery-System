from __future__ import annotations

from typing import Any


class Stop:
    def __init__(self, cycles: int = 1) -> None:
        self.cycles = cycles
        self.stopped = False
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self.stopped

    def set(self) -> None:
        self.stopped = True

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        if self.cycles == 0:
            self.stopped = True
            return True
        self.cycles -= 1
        return False


class ThreadRecorder:
    def __init__(self) -> None:
        self.threads: list[Any] = []

    def __call__(
        self, *, target: Any, args: tuple = (), name: str, daemon: bool
    ) -> Any:
        owner = self

        class Thread:
            started = False

            def __init__(self) -> None:
                self.name = name
                self.daemon = daemon
                owner.threads.append(self)

            def start(self) -> None:
                self.started = True

            def run(self) -> Any:
                return target(*args)

        return Thread()
