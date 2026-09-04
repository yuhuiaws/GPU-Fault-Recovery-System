from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# A waiter that is refused registration keeps the latency it already had, so the
# cap costs nothing but the handover. It is here because this is per-request
# state reached from the request path, and the queue depth that bounds admission
# does not bound how many callers are waiting on a response at once.
MAX_COMPLETION_WAITERS = 4096


@dataclass(frozen=True, eq=False)
class _Waiter:
    """One synchronous caller, and the loop its event has to be set on."""

    loop: asyncio.AbstractEventLoop
    event: asyncio.Event


class ProcessorCompletionSignals:
    """Hands a finished request straight to the caller waiting on it.

    A synchronous caller waits for its response by polling the store, and in
    queued mode the worker that produces that response runs on a thread of the
    same process. So the poll routinely discovers a row that was already
    complete: at the backed-off end of the interval, up to
    ``RESPONSE_POLL_MAX_SECONDS`` of a request's latency can be nothing but a
    waiter that has not looked yet, plus the store read that tells it to look
    again.

    This is the handover for that case and only that case. The registry is
    in-process: it says nothing about a request completed on another replica, and
    an ingress-only role runs no coordinator at all, so polling stays the
    mechanism rather than a fallback. It is not a delivery guarantee either -- a
    waiter that is never signalled, or that was refused registration, waits out
    its poll interval exactly as it did before.
    """

    def __init__(self, *, max_waiters: int = MAX_COMPLETION_WAITERS) -> None:
        if max_waiters < 1:
            raise ValueError("completion signal waiter cap must be positive")
        self._lock = threading.Lock()
        self._waiters: dict[str, list[_Waiter]] = {}
        self._max_waiters = max_waiters
        self._registered = 0
        self.refused_total = 0

    @property
    def pending_waiters(self) -> int:
        with self._lock:
            return self._registered

    @contextmanager
    def waiting(self, request_id: str) -> Iterator[asyncio.Event | None]:
        """An event that is set when ``request_id`` finishes in this process.

        ``None`` means there is no handover to be had and the caller should rely
        on its poll interval: either no event loop is running to be woken, or the
        waiter cap is reached. Registration is scoped to the block, so a caller
        that times out, is cancelled, or gets its answer from a poll leaves
        nothing behind for the signal to find.
        """

        waiter = self._register(request_id)
        try:
            yield None if waiter is None else waiter.event
        finally:
            if waiter is not None:
                self._unregister(request_id, waiter)

    def signal(self, request_id: str) -> None:
        """Wake every waiter on ``request_id``. Callable from any thread.

        The processor calls this after the response is committed to the store, so
        a woken waiter's next read sees it. Waking one whose request was released
        for retry rather than completed is harmless: it reads once, finds the
        request still pending and resumes waiting.
        """

        with self._lock:
            waiters = list(self._waiters.get(request_id, ()))
        for waiter in waiters:
            try:
                waiter.loop.call_soon_threadsafe(waiter.event.set)
            except RuntimeError:
                # The loop is closed, so the waiter went with it and its
                # ``waiting`` block cannot still be running.
                continue

    def _register(self, request_id: str) -> _Waiter | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        waiter = _Waiter(loop=loop, event=asyncio.Event())
        with self._lock:
            if self._registered >= self._max_waiters:
                self.refused_total += 1
                return None
            self._waiters.setdefault(request_id, []).append(waiter)
            self._registered += 1
        return waiter

    def _unregister(self, request_id: str, waiter: _Waiter) -> None:
        with self._lock:
            remaining = self._waiters.get(request_id)
            if remaining is None:
                return
            if waiter in remaining:
                remaining.remove(waiter)
                self._registered -= 1
            if not remaining:
                self._waiters.pop(request_id, None)
