from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, RLock
from typing import Any, Protocol

from gpu_fault.channel_registry import (
    ATTEMPT_COVERAGE_PATH,
    BATCHABLE_CHANNEL_PATHS,
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.processor.batching import telemetry_batch_size
from gpu_fault.processor.completion_signals import ProcessorCompletionSignals
from gpu_fault.processor.models import ProcessorRequest

LOGGER = logging.getLogger(__name__)


class SlotWake:
    """Turns finished workers of every pool into one claim-loop wake.

    A finished worker frees a slot, and nothing else announces it: the queue
    trigger notifies only an idle->busy edge, so a standing backlog raises
    none, and otherwise the claim loop sleeps to its nearest stream deadline,
    the notification fallback once the fault stream has backed off with no
    fault-tier row. Each pool's futures live in a set made by ``futures``;
    every future added to it gets a done callback, which runs on the worker
    thread (or inline for a future that is already done), so it only records
    the pool and sets the loop's event: any number of finishes between two
    cycles is one wake. The loop thread ``take``s the pools before it reaps
    futures. A freed fault slot also claims the fault stream at once: under
    the STRICT lane policy it usually frees the lane of the next row too,
    which the trigger notified (if at all) while that lane was still leased.
    Other pools only wake the loop -- a stream that filled its claim has no
    backoff to clear, and one that came back empty keeps it -- unless
    fault-tier work may be due (``_fault_rows_waiting``): then the fault stream
    is claimed first too.
    """

    def __init__(self, work_available: Event) -> None:
        self._work_available = work_available
        self._lock = Lock()
        self._freed: set[str] = set()

    def futures(self, pool: str) -> set[Future[Any]]:
        """An empty set for ``pool``'s futures, each of which wakes the loop."""

        return _WakingFutures(self, pool)

    def worker_finished(self, pool: str) -> None:
        with self._lock:
            self._freed.add(pool)
        self._work_available.set()

    def take(self) -> frozenset[str]:
        with self._lock:
            freed, self._freed = self._freed, set()
        return frozenset(freed)


class _WakingFutures(set[Future[Any]]):
    def __init__(self, wake: SlotWake, pool: str) -> None:
        super().__init__()
        self._wake = wake
        self._pool = pool

    def add(self, future: Future[Any]) -> None:
        future.add_done_callback(self._finished)
        super().add(future)

    def _finished(self, _future: Future[Any]) -> None:
        self._wake.worker_finished(self._pool)


class ProcessorCoordinatorStore(Protocol):
    def acquire_processor_leadership(self, *args: Any, **kwargs: Any) -> Any: ...

    def claim_active_processor_requests(self, *args: Any, **kwargs: Any) -> Any: ...

    def claim_processor_requests(self, *args: Any, **kwargs: Any) -> Any: ...

    def complete_active_processor_request(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def complete_active_processor_requests_batch(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def complete_processor_request(self, *args: Any, **kwargs: Any) -> Any: ...

    def list_attempt_observation_states(self, *args: Any, **kwargs: Any) -> Any: ...

    def list_training_progress_states(self, *args: Any, **kwargs: Any) -> Any: ...

    def release_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None: ...

    def release_processor_request(
        self,
        request_id: str,
        owner_id: str,
        leader_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None: ...

    def renew_active_processor_request(self, *args: Any, **kwargs: Any) -> Any: ...

    def save_attempt_observations_batch(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


class ProcessorLaneRuntimeMixin:
    _OBSERVATION_BATCH_SIZE: int
    _claimed_not_started: dict[str, ProcessorRequest]
    _claimed_not_started_released_total: int
    _in_flight: dict[str, Any]
    _lane_holder_by_path: dict[str, dict[str, float | int]]
    _retry_delay_seconds_max: float
    _retry_rescheduled_by_path: dict[str, int]
    _retry_rescheduled_total: int
    _state_lock: RLock
    active_consumers: bool
    owner_id: str
    retry_backoff_max_seconds: float
    retry_backoff_seconds: float
    store: ProcessorCoordinatorStore
    _retry_max_age: float
    observation_stale_seconds: float
    _completion_failure_releases_total: int
    _retry_horizon_failures_total: int
    _pool_for_request: Callable[[ProcessorRequest], str]
    _process: Callable[[ProcessorRequest], None]
    _process_observation_batch: Callable[[list[ProcessorRequest]], None]
    _process_telemetry_batch: Callable[[list[ProcessorRequest]], None]

    def _configure_retry_backoff(
        self,
        retry_backoff_seconds: float,
        retry_backoff_max_seconds: float,
    ) -> None:
        if (
            retry_backoff_seconds <= 0
            or retry_backoff_max_seconds < retry_backoff_seconds
        ):
            raise ValueError(
                "processor retry backoff must be positive and not exceed its maximum"
            )
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds

    def _initialize_lane_runtime_state(self) -> None:
        # Not declared in the contract block above: the host supplies the state
        # listed there, while this registry is created and owned here.
        self.completion_signals = ProcessorCompletionSignals()
        self._claimed_not_started: dict[str, ProcessorRequest] = {}
        self._claimed_not_started_released_total = 0
        self._lane_holder_by_path: dict[str, dict[str, float | int]] = {}
        self._retry_rescheduled_total = 0
        self._retry_rescheduled_by_path: dict[str, int] = {}
        self._retry_delay_seconds_max = 0.0

    def _retry_horizon_seconds(self, item: ProcessorRequest) -> float:
        """How long a failing row may keep being retried.

        A workload observation holds the fault <-> observation interlock for
        every correlated fault row while it is PENDING or LEASED (F-D3), so
        its horizon is bounded by the age at which the observation is stale
        anyway rather than by the generic retry horizon: past
        ``observation_stale_seconds`` the watcher's next report supersedes
        it, and retrying it further only holds faults back.
        """

        if item.path == WORKLOAD_OBSERVATIONS_PATH:
            return min(self._retry_max_age, self.observation_stale_seconds)
        return self._retry_max_age

    def _release(
        self,
        item: ProcessorRequest,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
        failure: str | None = None,
    ) -> None:
        """Hand a claimed row back to PENDING.

        With ``failure`` set the release is a *retry* and is booked as one
        (F-D4): ``retry_count + 1`` and an exponential ``not_before``, so the
        row does not become the oldest of its priority and re-execute its
        side effects immediately; past the retry horizon it is completed as
        failed instead of released forever. Without ``failure`` (shutdown,
        pool capacity) the row goes back untouched.
        """

        if item.leader_epoch is None or item.lease_token is None:
            raise RuntimeError("claimed processor request has no fencing token")
        if failure is not None and not_before is None and retry_count is None:
            now = datetime.now(timezone.utc)
            age_seconds = max(0.0, (now - item.created_at).total_seconds())
            if age_seconds > self._retry_horizon_seconds(item):
                self._complete_after_repeated_failure(item, failure, age_seconds)
                return
            retry_count = item.retry_count + 1
            delay_seconds = min(
                self.retry_backoff_max_seconds,
                self.retry_backoff_seconds * (2 ** min(item.retry_count, 16)),
            )
            not_before = now + timedelta(seconds=delay_seconds)
            with self._state_lock:
                self._completion_failure_releases_total += 1
            self._observe_retry_schedule(item.path, delay_seconds)
            LOGGER.warning(
                "processor request released after failure; rescheduled "
                "request_id=%s path=%s lane=%s failure=%s retry_count=%s "
                "not_before=%s",
                item.request_id,
                item.path,
                item.ordering_key(),
                failure,
                retry_count,
                not_before.isoformat(),
            )
        with self._state_lock:
            self._claimed_not_started.pop(item.request_id, None)
        if self.active_consumers:
            self.store.release_active_processor_request(
                item.request_id,
                self.owner_id,
                item.leader_epoch,
                item.lease_token,
                not_before=not_before,
                retry_count=retry_count,
            )
        else:
            self.store.release_processor_request(
                item.request_id,
                self.owner_id,
                item.leader_epoch,
                item.lease_token,
                not_before=not_before,
                retry_count=retry_count,
            )
        LOGGER.warning(
            "processor request released request_id=%s path=%s lane=%s "
            "owner=%s epoch=%s",
            item.request_id,
            item.path,
            item.ordering_key(),
            self.owner_id,
            item.leader_epoch,
        )

    def _complete_after_repeated_failure(
        self,
        item: ProcessorRequest,
        failure: str,
        age_seconds: float,
    ) -> None:
        """Past the retry horizon a failing row is finished, not recycled: a
        deterministic failure would otherwise loop "execute -> fail ->
        re-execute" at the head of its lane forever (F-D4)."""

        assert item.leader_epoch is not None and item.lease_token is not None
        body = json.dumps(
            {
                "error": "processor request failed past its retry horizon",
                "failure": failure,
                "age_seconds": round(age_seconds, 3),
                "retry_count": item.retry_count,
            }
        ).encode("utf-8")
        encoded = base64.b64encode(body).decode("ascii")
        with self._state_lock:
            self._retry_horizon_failures_total += 1
            self._claimed_not_started.pop(item.request_id, None)
        if self.active_consumers:
            self.store.complete_active_processor_request(
                item.request_id,
                self.owner_id,
                item.leader_epoch,
                item.lease_token,
                response_status=503,
                response_content_type="application/json",
                response_body_base64=encoded,
                path=item.path,
            )
        else:
            self.store.complete_processor_request(
                item.request_id,
                self.owner_id,
                item.leader_epoch,
                item.lease_token,
                response_status=503,
                response_content_type="application/json",
                response_body_base64=encoded,
            )
        LOGGER.error(
            "processor request completed as failed past its retry horizon "
            "request_id=%s path=%s lane=%s failure=%s age_seconds=%.3f "
            "retry_count=%s horizon_seconds=%.3f",
            item.request_id,
            item.path,
            item.ordering_key(),
            failure,
            age_seconds,
            item.retry_count,
            self._retry_horizon_seconds(item),
        )

    def register_claimed_requests(
        self,
        requests: list[ProcessorRequest],
    ) -> None:
        with self._state_lock:
            for request in requests:
                self._claimed_not_started[request.request_id] = request

    def _dispatch_claimed(
        self,
        requests: list[ProcessorRequest],
        pools: dict[str, ThreadPoolExecutor],
        futures: dict[str, set[Future[Any]]],
        available: dict[str, int],
    ) -> None:
        """Start the workers of claimed rows while their pools have
        ``available`` slots; the rest go back to PENDING. Observation rows run
        in batches of ``_OBSERVATION_BATCH_SIZE``, batchable telemetry in its
        path's batches. A claim cycle calls it twice with one ``available``:
        for the fault stream's rows the moment their sub-claim returns -- a
        P10 worker does not wait for the cycle's observation and evidence
        claims -- and for the other rows once those are done.
        """

        self.register_claimed_requests(requests)
        observation_requests: list[ProcessorRequest] = []
        telemetry_requests: dict[tuple[str, str], list[ProcessorRequest]] = {}
        for item in requests:
            pool_name = self._pool_for_request(item)
            if pool_name == "observation":
                observation_requests.append(item)
                continue
            if pool_name in {"gpu", "host"} and item.path in BATCHABLE_CHANNEL_PATHS:
                telemetry_requests.setdefault((pool_name, item.path), []).append(item)
                continue
            if available[pool_name] <= 0:
                self._release(item)
                continue
            futures[pool_name].add(pools[pool_name].submit(self._process, item))
            available[pool_name] -= 1
        while observation_requests:
            batch = observation_requests[: self._OBSERVATION_BATCH_SIZE]
            del observation_requests[: len(batch)]
            if available["observation"] <= 0:
                for item in batch:
                    self._release(item)
                continue
            futures["observation"].add(
                pools["observation"].submit(
                    self._process_observation_batch,
                    batch,
                )
            )
            available["observation"] -= 1
        for (
            pool_name,
            path,
        ), pending in telemetry_requests.items():
            batch_size = telemetry_batch_size(path)
            while pending:
                batch = pending[:batch_size]
                del pending[: len(batch)]
                if available[pool_name] <= 0:
                    for item in batch:
                        self._release(item)
                    continue
                futures[pool_name].add(
                    pools[pool_name].submit(
                        self._process_telemetry_batch,
                        batch,
                    )
                )
                available[pool_name] -= 1

    def release_unstarted_claims(self) -> None:
        with self._state_lock:
            requests = list(self._claimed_not_started.values())
        for item in requests:
            try:
                self._release(item)
            except Exception:
                LOGGER.exception(
                    "processor shutdown release failed request_id=%s path=%s lane=%s",
                    item.request_id,
                    item.path,
                    item.ordering_key(),
                )
                continue
            with self._state_lock:
                self._claimed_not_started_released_total += 1

    def _mark_request_started(self, request_id: str) -> None:
        with self._state_lock:
            self._claimed_not_started.pop(request_id, None)

    def _request_finished(self, request_id: str) -> None:
        observed = time.monotonic()
        with self._state_lock:
            self._claimed_not_started.pop(request_id, None)
            state = self._in_flight.pop(request_id, None)
            if state is not None:
                path = self._metric_path(state.item.path)
                values = self._lane_holder_by_path.setdefault(
                    path,
                    {"count": 0, "sum": 0.0, "max": 0.0},
                )
                duration = max(0.0, observed - state.started)
                values["count"] += 1
                values["sum"] += duration
                values["max"] = max(float(values["max"]), duration)
        # Every caller of this is the ``finally`` of a path that has already
        # committed the request's outcome to the store, so a caller woken here
        # reads the response rather than another "not yet". Signalled outside the
        # state lock, and unconditionally: the request is equally finished in this
        # process whether or not it was ever counted as in flight.
        self.completion_signals.signal(request_id)

    def _observe_retry_schedule(self, path: str, delay_seconds: float) -> None:
        path = self._metric_path(path)
        with self._state_lock:
            self._retry_rescheduled_total += 1
            self._retry_rescheduled_by_path[path] = (
                self._retry_rescheduled_by_path.get(path, 0) + 1
            )
            self._retry_delay_seconds_max = max(
                self._retry_delay_seconds_max,
                delay_seconds,
            )

    @staticmethod
    def _metric_path(path: str) -> str:
        if path in {
            ATTEMPT_COVERAGE_PATH,
            "/v1/attempts/failure-detected",
            "/v1/attempts/terminal",
            "/v1/advisory-notifications/dispatch",
            "/v1/installation-resources/sync",
            "/v1/installation-resources/replacements",
            "/v1/fleet/agents/heartbeat",
        }:
            return path
        for prefix in (
            "/v1/incidents/",
            "/v1/workflows/",
            "/v1/recovery-plans/",
            "/v1/attempts/",
            "/v1/advisory-notifications/",
            "/v1/installation-resources/",
            "/v1/training-health/",
            "/v1/fleet/agents/",
            "/v1/fleet/deployments/",
        ):
            if path.startswith(prefix):
                return f"{prefix}{{id}}"
        return path
