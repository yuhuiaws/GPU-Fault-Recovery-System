"""The Completion Watcher loop: list + watch + debounced reconcile.

``run`` serves the metrics endpoint and loops for ever; ``run_watch_cycle``
is one list/watch/reconcile cycle with the debounce timers the Task 6 review
called the debounced reconciler; ``_run_polling`` is the fallback without a
watch client. Split out of ``completion_controller`` as a pure move (F6b).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from threading import RLock, Timer, current_thread
from typing import Any, Callable

from gpu_fault.completion_metrics_server import start_completion_metrics_server
from gpu_fault.completion_observation import (
    completion_list_arguments,
    list_completion_pods,
)
from gpu_fault.completion_pod_parsing import (
    ATTEMPT_LABEL,
    CompletionControllerError,
)

# Borrowed on purpose: every line below used to be logged by
# ``gpu_fault.completion_controller`` and the log format prints the logger
# name, so the split must not rename what operators grep for.
LOGGER = logging.getLogger("gpu_fault.completion_controller")


class CompletionReconcileLoopMixin:
    """Watch/poll loop half of ``KubernetesCompletionController``."""

    # Attributes supplied by the composed concrete implementation.
    _gpu_uuid_cache: dict[str, list[str]]
    _gpu_uuid_failures: dict[str, tuple[int, datetime]]
    _pod_key: Callable[[dict[str, Any]], str]
    _reconcile: Callable[..., list[dict[str, Any]]]
    core_api: Any
    namespace: str | None
    note_progress: Callable[[], None]
    poll_interval_seconds: float
    progress_stall_budget_seconds: float
    reconcile_debounce_seconds: float
    run_once: Callable[[], list[dict[str, Any]]]
    serializer: Callable[[Any], dict[str, Any]]
    watch_factory: Callable[[], Any] | None
    watch_timeout_seconds: int

    def run(self, *, metrics_port: int | None = None) -> None:
        """Serve ``/metrics`` + ``/healthz`` and loop until the process ends.

        ``metrics_port`` overrides the environment setting; ``0`` disables the
        server, which the Deployment must never do -- the liveness probe reads
        that endpoint, so no server means kubelet keeps restarting the Pod.
        """

        metrics_server = start_completion_metrics_server(self, port=metrics_port)
        try:
            self._run_forever()
        finally:
            if metrics_server is not None:
                metrics_server.stop()

    def _run_forever(self) -> None:
        if self.watch_factory is None:
            LOGGER.warning("watch client is unavailable; using polling fallback")
            self._run_polling()
            return

        while True:
            # Reaching the top of the loop is progress, and so is coming back
            # from a failure (I2): an API server that refuses every LIST keeps
            # the loop turning, and CrashLooping the only Completion Watcher
            # during a control-plane outage would only add a cold start to it.
            self.note_progress()
            try:
                self.run_watch_cycle()
            except Exception:
                LOGGER.exception("Kubernetes Pod watch cycle failed; retrying")
                self.note_progress()
                time.sleep(self.poll_interval_seconds)

    def _run_polling(self) -> None:
        while True:
            self.note_progress()
            try:
                self.run_once()
            except Exception:
                LOGGER.exception("Kubernetes completion reconciliation failed")
            self.note_progress()
            time.sleep(self.poll_interval_seconds)

    def run_watch_cycle(self) -> None:
        """One list + watch + reconcile cycle: the loop's unit of work.

        ``run()`` never returns and ``run_once()`` only covers the polling
        fallback, so this is the public seam for anything that wants a single
        watch cycle -- a supervisor, an operator one-shot, or a test that has
        to observe what the cycle's ``finally`` does.
        """

        pods, resource_version = list_completion_pods(self.core_api, self.namespace)
        self._last_resource_version = resource_version
        self.note_progress()
        serialized = [self.serializer(item) for item in pods]
        cache = {self._pod_key(pod): pod for pod in serialized}
        # ``_reconcile`` takes ``reconcile_lock`` itself (F5), which is what
        # keeps a timer that outlived its cycle from running beside this pass.
        self._reconcile(list(cache.values()))
        cache_lock = RLock()
        pending_lock = RLock()
        pending_attempts: set[str] = set()
        pending_timer: list[Timer | None] = [None]
        # Timers stay reachable after they have fired so the ``finally`` can
        # join a flush that is still running; ``pending_timer`` is cleared by
        # the flush itself so ``schedule`` can arm the next one, which means a
        # slow flush and a freshly armed timer can both be live (the reconcile
        # lock serializes them). Finished timers are dropped on every arm, so
        # the list holds at most the live ones.
        started_timers: list[Timer] = []

        def attempt_id(pod: dict[str, Any] | None) -> str | None:
            if not pod:
                return None
            metadata = pod.get("metadata") or {}
            labels = metadata.get("labels") or {}
            value = labels.get(ATTEMPT_LABEL)
            return value if isinstance(value, str) and value else None

        def flush_pending() -> None:
            with pending_lock:
                attempts = set(pending_attempts)
                pending_attempts.clear()
                pending_timer[0] = None
            if not attempts:
                return
            with cache_lock:
                affected_pods = [
                    pod for pod in cache.values() if attempt_id(pod) in attempts
                ]
            self._reconcile(
                affected_pods,
                attempt_filter=attempts,
            )

        def schedule(attempts: set[str]) -> None:
            if not attempts:
                return
            with pending_lock:
                pending_attempts.update(attempts)
                if pending_timer[0] is not None:
                    return
                if self.reconcile_debounce_seconds == 0:
                    pass
                else:
                    timer = Timer(
                        self.reconcile_debounce_seconds,
                        flush_pending,
                    )
                    timer.daemon = True
                    pending_timer[0] = timer
                    started_timers[:] = [
                        item for item in started_timers if item.is_alive()
                    ]
                    started_timers.append(timer)
                    timer.start()
                    return
            flush_pending()

        watcher = self.watch_factory()
        LOGGER.info(
            "starting Kubernetes Pod watch: resource_version=%s cached_pods=%s",
            resource_version or "current",
            len(cache),
        )
        try:
            try:
                for event in watcher.stream(
                    self._list_method(),
                    **completion_list_arguments(self.namespace),
                    resource_version=resource_version or None,
                    timeout_seconds=self.watch_timeout_seconds,
                    allow_watch_bookmarks=True,
                    # F3: without this the client leaves ``timeout=None`` and a
                    # stream the API server drops without an RST (dead
                    # endpoint, NAT idle eviction) parks this thread for ever:
                    # no relist, no reconcile, and every node UNKNOWN after
                    # 600 s. The read budget sits above the server-side
                    # ``timeout_seconds`` so a healthy stream always ends by
                    # the server closing it, never by the client.
                    _request_timeout=(5, self.watch_timeout_seconds + 15),
                ):
                    event_type = str(event.get("type", "")).upper()
                    raw_object = event.get("object")
                    pod = self.serializer(raw_object) if raw_object is not None else {}
                    if event_type == "ERROR":
                        code = pod.get("code")
                        if code == 410:
                            LOGGER.info("Pod watch resourceVersion expired; relisting")
                            return
                        raise CompletionControllerError(
                            f"Kubernetes Pod watch error: {pod}"
                        )
                    if event_type == "BOOKMARK":
                        continue
                    if event_type not in {
                        "ADDED",
                        "MODIFIED",
                        "DELETED",
                    }:
                        LOGGER.warning(
                            "ignoring unknown Kubernetes watch event %s",
                            event_type,
                        )
                        continue
                    # An event delivered and applied is progress: a busy
                    # stream keeps the probe green between relists.
                    self.note_progress()
                    key = self._pod_key(pod)
                    with cache_lock:
                        previous = cache.get(key)
                        if event_type == "DELETED":
                            cache.pop(key, None)
                            self._gpu_uuid_cache.pop(key, None)
                            self._gpu_uuid_failures.pop(key, None)
                        else:
                            cache[key] = pod
                    schedule(
                        {
                            value
                            for value in (
                                attempt_id(previous),
                                attempt_id(pod),
                            )
                            if value is not None
                        }
                    )
            except Exception as exc:
                if getattr(exc, "status", None) == 410:
                    LOGGER.info("Pod watch resourceVersion expired; relisting")
                    return
                raise
        finally:
            with pending_lock:
                timer = pending_timer[0]
                if timer is not None:
                    timer.cancel()
                    pending_timer[0] = None
                in_flight = [item for item in started_timers if item.is_alive()]
                started_timers.clear()
            # Join OUTSIDE every lock (F5). A timer that already fired holds no
            # lock this thread wants, and this thread holds none it wants, so
            # the join cannot deadlock; taking ``reconcile_lock`` here would.
            join_timeout = self.progress_stall_budget_seconds
            for item in in_flight:
                if item is current_thread():
                    continue
                item.join(timeout=join_timeout)
                if item.is_alive():
                    LOGGER.warning(
                        "debounced reconcile still running after %ss; the next "
                        "full pass will wait on the reconcile lock",
                        join_timeout,
                    )
            # F11: the flush is the only step here that can raise (one bad Pod
            # is enough), and skipping ``stop()`` leaked the API-server stream
            # and its thread on every such cycle.
            try:
                flush_pending()
            finally:
                watcher.stop()
        LOGGER.info("Kubernetes Pod watch timed out; resyncing")

    def _list_method(self):
        if self.namespace:
            return self.core_api.list_namespaced_pod
        return self.core_api.list_pod_for_all_namespaces
