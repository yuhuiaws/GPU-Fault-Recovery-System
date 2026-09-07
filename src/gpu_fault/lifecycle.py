from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Callable


def required_processor_shutdown_seconds(
    request_max_execution_seconds: float,
    exit_grace_seconds: float,
    *,
    overhead_seconds: float = 5,
) -> float:
    if (
        request_max_execution_seconds <= 0
        or exit_grace_seconds < 0
        or overhead_seconds < 0
    ):
        raise ValueError("processor shutdown inputs are invalid")
    return request_max_execution_seconds + exit_grace_seconds + overhead_seconds


def validate_lifespan_shutdown_budget(
    configured_seconds: float,
    required_seconds: float,
) -> float:
    if configured_seconds <= 0:
        raise ValueError("lifespan shutdown maximum must be positive")
    if configured_seconds < required_seconds:
        raise RuntimeError(
            "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS must cover "
            "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS plus "
            "GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS and 5 seconds "
            "of shutdown overhead"
        )
    return configured_seconds


@dataclass
class ShutdownCoordinator:
    max_seconds: float
    now: Callable[[], float] = time.monotonic
    failures: list[str] = field(default_factory=list)
    _deadline: float = field(init=False)

    def __post_init__(self) -> None:
        if self.max_seconds <= 0:
            raise ValueError("shutdown coordinator maximum must be positive")
        self._deadline = self.now() + self.max_seconds

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - self.now())

    def join(self, thread: Thread | None, label: str) -> None:
        if thread is None:
            return
        thread.join(timeout=self.remaining_seconds)
        if thread.is_alive():
            self.failures.append(label)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise RuntimeError(
                "lifespan shutdown deadline exceeded: "
                + ", ".join(dict.fromkeys(self.failures))
            )
