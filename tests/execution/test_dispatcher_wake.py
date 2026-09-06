"""A wake request is never lost between the wait and the clear (F-A8).

``run_forever`` used to wait, then clear the wake event, then scan: a wake
that arrived between the wait returning and the clear was swallowed and the
scan it asked for waited a full poll interval. The clear now precedes the
scan, so a wake landing at any point is answered by the scan that follows.
"""

from __future__ import annotations

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from tests._builders import active_workflow_executor, build_store


class _RecordingEvent:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self._set = False

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self.log.append("clear")
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def wait(self, _timeout=None) -> bool:
        self.log.append("wait")
        return self._set


def test_the_wake_event_is_cleared_before_the_scan_not_after_the_wait():
    store = build_store()
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True, poll_interval_seconds=0.001),
    )
    log: list[str] = []
    dispatcher._wake = _RecordingEvent(log)  # type: ignore[assignment]
    cycles = 0

    def run_once():
        nonlocal cycles
        cycles += 1
        log.append("scan")
        # A wake that lands while the scan runs...
        dispatcher._wake.set()
        if cycles == 2:
            dispatcher._stop.set()

    dispatcher.run_once = run_once  # type: ignore[method-assign]
    dispatcher.run_forever()

    # ...is still set when the wait runs, and only cleared right before the
    # scan that answers it; nothing sits between a wait and a scan but the
    # loop head.
    assert log == ["clear", "scan", "wait", "clear", "scan", "wait"]
