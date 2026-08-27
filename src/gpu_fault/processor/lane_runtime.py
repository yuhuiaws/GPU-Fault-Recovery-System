from __future__ import annotations

import logging
import time
from datetime import datetime
from threading import RLock
from typing import Any, Protocol

from gpu_fault.processor.models import ProcessorRequest


LOGGER = logging.getLogger(__name__)


class LaneInFlightState(Protocol):
    item: ProcessorRequest
    started: float


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
        self._claimed_not_started: dict[str, ProcessorRequest] = {}
        self._claimed_not_started_released_total = 0
        self._lane_holder_by_path: dict[str, dict[str, float | int]] = {}
        self._retry_rescheduled_total = 0
        self._retry_rescheduled_by_path: dict[str, int] = {}
        self._retry_delay_seconds_max = 0.0

    def _release(
        self,
        item: ProcessorRequest,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None:
        if item.leader_epoch is None or item.lease_token is None:
            raise RuntimeError("claimed processor request has no fencing token")
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

    def register_claimed_requests(
        self,
        requests: list[ProcessorRequest],
    ) -> None:
        with self._state_lock:
            for request in requests:
                self._claimed_not_started[request.request_id] = request

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
            if state is None:
                return
            path = self._metric_path(state.item.path)
            values = self._lane_holder_by_path.setdefault(
                path,
                {"count": 0, "sum": 0.0, "max": 0.0},
            )
            duration = max(0.0, observed - state.started)
            values["count"] += 1
            values["sum"] += duration
            values["max"] = max(float(values["max"]), duration)

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
        for prefix in (
            "/v1/incidents/",
            "/v1/workflows/",
            "/v1/recovery-plans/",
        ):
            if path.startswith(prefix):
                return f"{prefix}{{id}}"
        return path
