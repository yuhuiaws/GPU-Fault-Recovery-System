"""Supervisor deadlines that fall where a test puts them, however slow the host.

The supervisor's deadline tests gave a real worker, or a stand-in for one,
1.5-3 s of real time to reach what they test -- the probe blocked in its read,
the checkpoint published, the slot held -- and failed on a CPU-starved host,
where the start alone (two interpreters, the package's imports) took longer
(b210: 24 busy loops on the test's CPU). Here the run's budget is one it never
reaches on its own (``REAL_WORKER_BUDGET_SECONDS``), and the supervisor's clock
jumps past the deadline at the moment the test has seen that point reached: the
deadline falls exactly where the test wants it, on any host.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from gpu_fault.node_agent import observation_supervisor


class DeadlineClock:
    """``time.monotonic()``, moved forward by ``jump`` from a moment the test picks.

    ``expire()`` moves every thread's reading, ``expire(thread)`` one thread's
    (a holder that has to leave the slot while another caller's wait goes on
    being measured in real time), and ``when`` expires every thread the first
    time it holds -- it is checked on each reading, and the supervisor reads
    its clock at least every 0.1 s while it waits on the worker's pipe.
    ``expired_at`` is the real time of the moment every thread was expired.
    Readings stay on ``time.monotonic``'s timebase: the deadline the
    supervisor hands the worker is still a real one.
    """

    def __init__(self, jump: float, when: Callable[[], bool] | None = None) -> None:
        self.jump = jump
        self.when = when
        self.expired_at: float | None = None
        self._threads: set[int] = set()
        self._lock = threading.Lock()

    def expire(self, thread: threading.Thread | None = None) -> None:
        with self._lock:
            if thread is None:
                if self.expired_at is None:
                    self.expired_at = time.monotonic()
            elif thread.ident is not None:
                self._threads.add(thread.ident)

    def monotonic(self) -> float:
        if self.expired_at is None and self.when is not None and self.when():
            self.expire()
        now = time.monotonic()
        if self.expired_at is not None or threading.get_ident() in self._threads:
            return now + self.jump
        return now

    def install(self, monkeypatch: pytest.MonkeyPatch) -> DeadlineClock:
        """Make this the clock of every supervisor the test runs."""

        monkeypatch.setattr(observation_supervisor, "time", _SupervisorTime(self))
        return self


class _SupervisorTime:
    """The ``time`` module as ``observation_supervisor`` sees it: the clock's
    ``monotonic``, everything else the real module's."""

    def __init__(self, clock: DeadlineClock) -> None:
        self.monotonic = clock.monotonic

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class WatchedSlot:
    """A supervisor's slot that tells the test when a caller starts waiting
    for it, and when the waiter got it; when a caller first found it held
    (``refused_at``) and when it was first given back (``released_at``)."""

    def __init__(self, slot: Any) -> None:
        self.slot = slot
        self.waiting = threading.Event()
        self.acquired_at: float | None = None
        self.refused_at: float | None = None
        self.released_at: float | None = None

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        if not blocking:
            acquired = bool(self.slot.acquire(blocking=False))
            if not acquired and self.refused_at is None:
                self.refused_at = time.monotonic()
            return acquired
        self.waiting.set()
        acquired = bool(self.slot.acquire(timeout=timeout))
        if acquired:
            self.acquired_at = time.monotonic()
        return acquired

    def release(self) -> None:
        if self.released_at is None:
            # Read before the slot is free: a waiter's ``acquired_at`` is
            # read after it got the slot, so the two order the hand-over.
            self.released_at = time.monotonic()
        self.slot.release()

    def wait_for_the_holder(self, result: dict[str, Any], below: float) -> float:
        """The command's wait for a holder, measured here: it found the slot
        held, waited, and got it only once the holder had let go of it, in
        under ``below`` seconds of wall clock.

        Not read off ``slot_wait_seconds`` alone: the supervisor names a wait
        of 10 ms or more, and a preemption can hand the slot over in less.
        The sample's worker dies about 2 ms after the SIGTERM; when the signal
        lands just that long before the guard's 20 ms poll wakes, the guard
        finds it gone, reaps and exits at once, and the command has the slot
        in under 10 ms (9.6-9.97 ms with the signal 1.9-2.2 ms before the
        wake; a focused ``-n 4`` run lost the key that way, 2026-11-05, t305).
        Where the supervisor names the wait, it is the one measured here or a
        little longer: its clock starts at the call, this one at the refused
        first try. Returns the wait.
        """

        # The status and the named wait, not the whole result: a triage's
        # weighs hundreds of kilobytes.
        brief = {key: result.get(key) for key in ("status", "slot_wait_seconds")}
        assert self.refused_at is not None, (
            "the command found the slot free: its holder had already left it",
            brief,
        )
        assert self.released_at is not None and self.acquired_at is not None, (
            "the holder never gave the slot back to the waiting command",
            self.released_at,
            self.acquired_at,
        )
        assert self.refused_at <= self.released_at <= self.acquired_at, (
            "the command got the slot only after the holder let go of it",
            self.refused_at,
            self.released_at,
            self.acquired_at,
        )
        waited = self.acquired_at - self.refused_at
        assert 0 < waited < below, (waited, brief)
        reported = result.get("slot_wait_seconds")
        if reported is None:
            assert waited < 0.01, ("a wait of 10 ms or more is named", waited)
        else:
            assert round(waited, 3) <= reported < below, (reported, waited)
        return waited
