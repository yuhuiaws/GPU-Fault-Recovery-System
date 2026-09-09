"""Per-cluster wakeups for the long-poll ``POST /v1/regional/executors/claim``.

The data-plane executor used to poll the claim route every 2 s, so every
remediation step paid up to 2 s before the executor even saw its command. With
``wait_seconds`` in the claim body the route holds an empty claim until the
store publishes a ``REMOTE_COMMAND`` wakeup for that cluster, then claims
again. This module is the wait: one listener thread per process runs
``store.run_wakeup_listener(WakeupChannel.REMOTE_COMMAND, ...)`` and resolves,
via ``loop.call_soon_threadsafe``, the future of every request subscribed to
the cluster named in the payload. The request itself awaits that future, so a
held claim costs one coroutine and no store I/O thread.

Liveness rule: a wakeup is a hint, and a hint nobody is listening for is a
stall. While the listener is not connected (never connected, reconnecting,
store without the primitive) a wait is bounded by ``degraded_wait_seconds`` --
the old poll interval -- so a deaf listener degrades to today's polling. A
listener that drops mid-wait releases every waiter at once for the same
reason; the route claims again and the executor's next request is bounded.

The thread starts lazily on the first subscription, so the worker and spool
roles, which serve no executor, never open a LISTEN connection; it stops with
the application lifespan (``wrap_lifespan``). Every uvicorn worker process
holds its own hub and its own LISTEN connection.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any

from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.contracts import WakeupChannel

LOGGER = logging.getLogger(__name__)

# Below the executor's default claim wait (20 s) plus its request timeout, and
# far below anything in front of the API: uvicorn's --timeout-keep-alive only
# bounds an idle connection between requests, and the regional NLB is L4 with
# a 350 s idle timeout.
DEFAULT_MAX_WAIT_SECONDS = 25.0
# The pre-long-poll poll interval (GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS).
DEFAULT_DEGRADED_WAIT_SECONDS = 2.0
# Two executor replicas per cluster is the deployed shape; a restart overlap
# adds one or two more for a few seconds. Anything past that is a client bug,
# and it must not pin request handlers.
DEFAULT_MAX_WAITERS_PER_CLUSTER = 4


class WakeupWaitOutcome(StrEnum):
    WOKEN = "woken"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"
    NOT_ADMITTED = "not_admitted"


class ClaimWaiter:
    """One subscribed claim request. Created by ``RemoteCommandWakeupHub``."""

    def __init__(
        self,
        hub: RemoteCommandWakeupHub,
        cluster_id: str,
        *,
        admitted: bool,
    ) -> None:
        self._hub = hub
        self.cluster_id = cluster_id
        self.admitted = admitted
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[WakeupWaitOutcome] = self._loop.create_future()

    def _resolve(self, outcome: WakeupWaitOutcome) -> None:
        """Called on the waiter's own loop, possibly after it stopped waiting."""

        if not self._future.done():
            self._future.set_result(outcome)

    def resolve_threadsafe(self, outcome: WakeupWaitOutcome) -> None:
        try:
            self._loop.call_soon_threadsafe(self._resolve, outcome)
        except RuntimeError:
            # The loop is closed: the request is long gone.
            pass

    async def wait(self, wait_seconds: float) -> WakeupWaitOutcome:
        """Wait for a PENDING command on this cluster, bounded by the hub.

        Returns at once when the hub did not admit this waiter (cap reached
        or hub closed), so the route can answer with what it has.
        """

        if not self.admitted:
            return WakeupWaitOutcome.NOT_ADMITTED
        timeout = min(max(wait_seconds, 0.0), self._hub.max_wait_seconds)
        if not self._hub.connected:
            timeout = min(timeout, self._hub.degraded_wait_seconds)
        try:
            return await asyncio.wait_for(self._future, timeout=timeout)
        except TimeoutError:
            return WakeupWaitOutcome.TIMEOUT


class RemoteCommandWakeupHub:
    def __init__(
        self,
        store: Any,
        *,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        degraded_wait_seconds: float = DEFAULT_DEGRADED_WAIT_SECONDS,
        max_waiters_per_cluster: int = DEFAULT_MAX_WAITERS_PER_CLUSTER,
        listener_timeout_seconds: float = 1.0,
    ) -> None:
        if max_wait_seconds <= 0 or degraded_wait_seconds <= 0:
            raise ValueError("wakeup wait bounds must be positive")
        if max_waiters_per_cluster < 1:
            raise ValueError("at least one waiter per cluster must be admitted")
        self._store = store
        self.max_wait_seconds = max_wait_seconds
        self.degraded_wait_seconds = degraded_wait_seconds
        self.max_waiters_per_cluster = max_waiters_per_cluster
        self._listener_timeout_seconds = listener_timeout_seconds
        self._lock = threading.Lock()
        self._waiters: dict[str, list[ClaimWaiter]] = {}
        self._connected = False
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Observability counters, read by tests and available to a metrics
        # contributor; not wired to /metrics yet.
        self.wakeups_total = 0
        self.disconnects_total = 0

    # -- state ---------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def listener_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- subscription --------------------------------------------------------

    @asynccontextmanager
    async def subscribe(self, cluster_id: str) -> AsyncIterator[ClaimWaiter]:
        """Register a claim request before its first claim, so a command
        written between that claim and the wait still wakes it."""

        self._ensure_listener()
        with self._lock:
            waiters = self._waiters.setdefault(cluster_id, [])
            admitted = not self._closed and len(waiters) < self.max_waiters_per_cluster
            waiter = ClaimWaiter(self, cluster_id, admitted=admitted)
            if admitted:
                waiters.append(waiter)
        try:
            yield waiter
        finally:
            if admitted:
                with self._lock:
                    remaining = self._waiters.get(cluster_id, [])
                    if waiter in remaining:
                        remaining.remove(waiter)
                    if not remaining:
                        self._waiters.pop(cluster_id, None)

    # -- listener thread -----------------------------------------------------

    def _ensure_listener(self) -> None:
        with self._lock:
            if self._closed or self.listener_alive:
                return
            run = getattr(self._store, "run_wakeup_listener", None)
            if run is None:
                # A store without the primitive: never connected, every wait
                # bounded by the degraded interval.
                return
            self._thread = threading.Thread(
                target=self._listen,
                args=(run,),
                name="remote-command-wakeups",
                daemon=True,
            )
            self._thread.start()

    def _listen(self, run: Callable[..., None]) -> None:
        try:
            run(
                WakeupChannel.REMOTE_COMMAND,
                self._stop,
                self._on_notification,
                timeout_seconds=self._listener_timeout_seconds,
                on_state=self._on_state,
            )
        except Exception:  # noqa: BLE001 - the thread must report, not vanish
            LOGGER.exception("remote command wakeup listener stopped")
        finally:
            self._on_state(False)

    def _on_state(self, connected: bool) -> None:
        with self._lock:
            was_connected = self._connected
            self._connected = connected
            if connected or not was_connected:
                return
            self.disconnects_total += 1
            released = [
                waiter for waiters in self._waiters.values() for waiter in waiters
            ]
        # Liveness rule: nobody may keep waiting on a channel that is deaf.
        for waiter in released:
            waiter.resolve_threadsafe(WakeupWaitOutcome.DISCONNECTED)
        if released:
            LOGGER.warning(
                "remote command wakeup listener disconnected; released %d "
                "waiting claim(s) to poll",
                len(released),
            )

    def _on_notification(self, payload: dict[str, Any]) -> None:
        if payload.get("status") != RemoteCommandStatus.PENDING.value:
            return
        cluster_id = payload.get("cluster_id")
        if not isinstance(cluster_id, str):
            return
        with self._lock:
            woken = list(self._waiters.get(cluster_id, ()))
            self.wakeups_total += len(woken)
        for waiter in woken:
            waiter.resolve_threadsafe(WakeupWaitOutcome.WOKEN)

    # -- shutdown ------------------------------------------------------------

    def close(self, *, join_seconds: float = 5.0) -> None:
        """Stop the listener and release every waiter. Idempotent."""

        with self._lock:
            self._closed = True
            self._stop.set()
            released = [
                waiter for waiters in self._waiters.values() for waiter in waiters
            ]
            self._waiters.clear()
            thread = self._thread
        for waiter in released:
            waiter.resolve_threadsafe(WakeupWaitOutcome.CLOSED)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_seconds)
            if thread.is_alive():
                LOGGER.warning(
                    "remote command wakeup listener did not stop within %.0fs",
                    join_seconds,
                )

    def wrap_lifespan(self, lifespan: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Close the hub when the wrapped application lifespan exits."""

        @asynccontextmanager
        async def wrapped(app: Any) -> AsyncIterator[None]:
            try:
                async with lifespan(app):
                    yield
            finally:
                await asyncio.to_thread(self.close)

        return wrapped
