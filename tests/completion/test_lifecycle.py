from __future__ import annotations

from threading import Event, Thread

import pytest

from gpu_fault.lifecycle import (
    ShutdownCoordinator,
    required_processor_shutdown_seconds,
    validate_lifespan_shutdown_budget,
)


def test_processor_shutdown_budget_includes_execution_and_exit() -> None:
    required = required_processor_shutdown_seconds(120, 5)

    assert required == 130
    assert validate_lifespan_shutdown_budget(130, required) == 130
    with pytest.raises(RuntimeError, match="must cover"):
        validate_lifespan_shutdown_budget(129, required)


def test_shutdown_coordinator_uses_one_shared_deadline() -> None:
    clock = iter([0.0, 3.0, 8.0])
    coordinator = ShutdownCoordinator(10, now=lambda: next(clock))

    assert coordinator.remaining_seconds == 7
    assert coordinator.remaining_seconds == 2


def test_shutdown_coordinator_reports_live_thread() -> None:
    stop = Event()
    thread = Thread(target=stop.wait, daemon=True)
    thread.start()
    coordinator = ShutdownCoordinator(0.01)

    coordinator.join(thread, "stuck")
    with pytest.raises(RuntimeError, match="stuck"):
        coordinator.raise_if_failed()
    stop.set()
    thread.join(timeout=1)
