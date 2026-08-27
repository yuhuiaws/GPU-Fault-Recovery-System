from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Any


LOGGER = logging.getLogger(__name__)


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
        coordinator._release(item)
        coordinator._observe_processing(outcome, time.monotonic() - started)
        return
    response_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - item.created_at).total_seconds(),
    )
    if (status in {408, 425, 429} or status >= 500) and (
        response_age_seconds <= coordinator._retry_max_age
    ):
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
            response_age_seconds,
            retry_count,
            not_before.isoformat(),
            coordinator._retry_max_age,
        )
        coordinator._release(
            item,
            not_before=not_before,
            retry_count=retry_count,
        )
        coordinator._observe_retry_schedule(item.path, delay_seconds)
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
                coordinator._release(item)
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
