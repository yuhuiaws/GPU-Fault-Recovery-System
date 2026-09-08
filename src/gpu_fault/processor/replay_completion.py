from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.processor.rejected_events import record_replay_completion

LOGGER = logging.getLogger(__name__)


def retryable_status(status: int) -> bool:
    """A handler answer the queue retries rather than commits."""

    return status in {408, 425, 429} or status >= 500


def retry_disposition(coordinator: Any, item: Any, *, status: int) -> str | None:
    """How a retryable handler response is booked for this row.

    ``"retry"`` while the row is inside its retry horizon; ``"horizon"``
    once it is past it (counted in ``retry_horizon_failures_total`` and
    logged here, the caller commits the response as the row's final
    answer); ``None`` for a response that is not retryable at all. Shared
    by the single-request replay and the batch paths so a 500 on one item
    of a telemetry batch is booked exactly like a 500 on a replayed request
    (B-4).
    """

    if not retryable_status(status):
        return None
    response_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - item.created_at).total_seconds(),
    )
    # An observation's horizon is its own stale limit, not the generic
    # retry age (F-D3): while it retries it holds correlated faults back.
    retry_horizon_seconds = coordinator._retry_horizon_seconds(item)
    if response_age_seconds <= retry_horizon_seconds:
        return "retry"
    # Retryable, but past the horizon: the response is committed as the
    # row's final answer. Booked like the release path's horizon failures
    # so the two exits of the same bound share a count.
    with coordinator._state_lock:
        coordinator._retry_horizon_failures_total += 1
    LOGGER.error(
        "processor replay returned a retryable response past the retry "
        "horizon; completing as failed request_id=%s path=%s status=%s "
        "age_seconds=%.3f horizon_seconds=%.3f",
        item.request_id,
        item.path,
        status,
        response_age_seconds,
        retry_horizon_seconds,
    )
    return "horizon"


def reschedule_retryable_response(coordinator: Any, item: Any, *, status: int) -> None:
    """Release the row as a booked retry with an exponential backoff (F-D4)."""

    retry_count = item.retry_count + 1
    delay_seconds = min(
        coordinator.retry_backoff_max_seconds,
        coordinator.retry_backoff_seconds * (2 ** min(item.retry_count, 16)),
    )
    not_before = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
    LOGGER.warning(
        "processor replay returned a retryable response; "
        "rescheduling request request_id=%s path=%s status=%s "
        "age_seconds=%.3f retry_count=%s not_before=%s "
        "retry_max_age_seconds=%.3f",
        item.request_id,
        item.path,
        status,
        max(0.0, (datetime.now(timezone.utc) - item.created_at).total_seconds()),
        retry_count,
        not_before.isoformat(),
        coordinator._retry_horizon_seconds(item),
    )
    coordinator._release(
        item,
        not_before=not_before,
        retry_count=retry_count,
    )
    coordinator._observe_retry_schedule(item.path, delay_seconds)


def finalize_replay_response(
    coordinator: Any,
    item: Any,
    *,
    status: int,
    content_type: str | None,
    retry_partition: str | None,
    body: bytes,
    started: float,
) -> None:
    """Retry transient handler responses or fence their completion."""

    outcome = "error"
    if retry_partition == "lane-lease-changed":
        # Somebody else holds the lane now. Booked as a retry (F-D4): a bare
        # release made this row the oldest PENDING of its priority and the
        # lane's next claim re-ran it immediately.
        coordinator._release(item, failure="lane lease changed")
        coordinator._observe_processing(outcome, time.monotonic() - started)
        return
    if retry_disposition(coordinator, item, status=status) == "retry":
        reschedule_retryable_response(coordinator, item, status=status)
        coordinator._observe_processing(outcome, time.monotonic() - started)
        return

    coordinator._set_request_phase(item.request_id, "completion")
    try:
        completion_error = None
        for completion_attempt in range(3):
            try:
                encoded_body = base64.b64encode(body).decode("ascii")
                if coordinator.active_consumers:
                    coordinator.store.complete_active_processor_request(
                        item.request_id,
                        coordinator.owner_id,
                        item.leader_epoch,
                        item.lease_token,
                        response_status=status,
                        response_content_type=content_type,
                        response_body_base64=encoded_body,
                        path=item.path,
                    )
                else:
                    coordinator.store.complete_processor_request(
                        item.request_id,
                        coordinator.owner_id,
                        item.leader_epoch,
                        item.lease_token,
                        response_status=status,
                        response_content_type=content_type,
                        response_body_base64=encoded_body,
                    )
                completion_error = None
                break
            except Exception as exc:
                completion_error = exc
                if completion_attempt < 2:
                    with coordinator._state_lock:
                        coordinator._completion_retries_total += 1
                    time.sleep(0.05 * (2**completion_attempt))
                    continue
                with coordinator._state_lock:
                    coordinator._completion_failures_total += 1
                LOGGER.exception(
                    "processor request completion failed "
                    "request_id=%s path=%s lane=%s owner=%s "
                    "epoch=%s status=%s duration_seconds=%.3f",
                    item.request_id,
                    item.path,
                    item.ordering_key(),
                    coordinator.owner_id,
                    item.leader_epoch,
                    status,
                    time.monotonic() - started,
                )
        if completion_error is not None:
            try:
                coordinator._release(
                    item,
                    failure=f"completion failed: {type(completion_error).__name__}",
                )
            except Exception:
                LOGGER.exception(
                    "processor request release after completion "
                    "failure failed request_id=%s path=%s lane=%s",
                    item.request_id,
                    item.path,
                    item.ordering_key(),
                )
            raise completion_error
        outcome = "success"
        # The row is committed with this status as its final answer; book
        # it where an operator can see it. A 4xx on a fault channel is a
        # fault thrown away after the collector's 202 (G1).
        record_replay_completion(coordinator, item, status=status, body=body)
        LOGGER.info(
            "processor request completed request_id=%s path=%s "
            "lane=%s owner=%s epoch=%s status=%s duration_seconds=%.3f",
            item.request_id,
            item.path,
            item.ordering_key(),
            coordinator.owner_id,
            item.leader_epoch,
            status,
            time.monotonic() - started,
        )
    finally:
        coordinator._observe_processing(outcome, time.monotonic() - started)
