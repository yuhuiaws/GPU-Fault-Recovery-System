"""A 4xx on a fault channel after the 202 must not complete in silence.

Every collector channel is ``receipt=True``: the ingress answers 202 after
JSON decoding only, and the handler's real verdict arrives when the processor
replays the row. A Pydantic 422 on that replay used to be booked as
``outcome="success"`` at INFO with no per-status counter, so a payload-shape
drift on the kernel XID channel would look like a healthy pipeline whose nodes
had simply gone quiet.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.processor import ProcessorCoordinator, ProcessorRequestStatus
from gpu_fault.processor.rejected_events import (
    REJECTED_EVENT_ERROR_PREFIX,
    is_rejected_event_status,
)
from gpu_fault.processor.replay_completion import finalize_replay_response
from gpu_fault.telemetry import CollectorKind
from tests._builders import build_store, processor_request

REQUEST_LEASE = timedelta(seconds=120)
KERNEL_BODY = (
    b'{"cluster_id":"cluster-a","node_id":"node-a","record_id":"kmsg-1",'
    b'"observed_at":"2026-09-07T10:00:00+00:00","message":"NVRM: Xid"}'
)
PYDANTIC_DETAIL = (
    b'{"detail":[{"type":"missing","loc":["body","observed_at"],'
    b'"msg":"Field required","input":{"message":"NVRM: Xid 79 secret"}}]}'
)


def _claimed(store, path: str, body: bytes):
    request = store.enqueue_processor_request(processor_request(path, body=body))
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    return request, claimed


def _processor(store) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
    )


def _finalize(processor, claimed, status: int, body: bytes = b"{}") -> None:
    finalize_replay_response(
        processor,
        claimed,
        status=status,
        content_type="application/json",
        retry_partition=None,
        body=body,
        started=time.monotonic(),
    )


def test_fault_channel_4xx_is_counted_warned_and_recorded(caplog) -> None:
    store = build_store()
    request, claimed = _claimed(store, NVIDIA_KERNEL_PATH, KERNEL_BODY)
    processor = _processor(store)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.processor"):
        _finalize(processor, claimed, 422, PYDANTIC_DETAIL)

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.COMPLETED
    snapshot = processor.metrics_snapshot()
    assert snapshot["completions_by_path_status"] == {NVIDIA_KERNEL_PATH: {"4xx": 1}}
    assert snapshot["fault_rejections_total"] == 1

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and request.request_id in record.getMessage()
    ]
    assert warnings, "a rejected fault event completed without a WARNING"
    message = warnings[0].getMessage()
    assert NVIDIA_KERNEL_PATH in message and "422" in message
    assert "Field required" in message, "the response detail was not surfaced"
    assert "secret" not in message, "the rejected payload body leaked into the log"

    statuses = store.list_collector_statuses("cluster-a", "node-a")
    rejected = [item for item in statuses if is_rejected_event_status(item)]
    assert [item.collector for item in rejected] == [CollectorKind.NVIDIA_KERNEL]
    assert rejected[0].last_error_at is not None
    assert rejected[0].last_success_at is None, "a rejection was booked as success"
    assert any(
        error.startswith(REJECTED_EVENT_ERROR_PREFIX) for error in rejected[0].errors
    ), rejected[0].errors


def test_rejection_does_not_erase_the_previous_success_time() -> None:
    store = build_store()
    request, claimed = _claimed(store, NVIDIA_KERNEL_PATH, KERNEL_BODY)
    processor = _processor(store)
    _finalize(processor, claimed, 200)
    _, claimed = _claimed(store, NVIDIA_KERNEL_PATH, KERNEL_BODY.replace(b"-1", b"-2"))

    _finalize(processor, claimed, 422, PYDANTIC_DETAIL)

    statuses = store.list_collector_statuses("cluster-a", "node-a")
    assert len(statuses) == 1, statuses
    assert statuses[0].last_error_at is not None
    assert is_rejected_event_status(statuses[0]), statuses[0].errors


def test_routine_channel_4xx_is_counted_without_a_fault_warning(caplog) -> None:
    store = build_store()
    body = b'{"cluster_id":"cluster-a","node_id":"node-a","edge_filter_reasons":["health-summary"]}'
    _, claimed = _claimed(store, GPU_METRICS_PATH, body)
    processor = _processor(store)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.processor"):
        _finalize(processor, claimed, 422, b'{"detail":"bad batch"}')

    snapshot = processor.metrics_snapshot()
    assert snapshot["completions_by_path_status"] == {GPU_METRICS_PATH: {"4xx": 1}}
    assert snapshot["fault_rejections_total"] == 0
    assert not [
        record for record in caplog.records if record.levelno == logging.WARNING
    ], "a routine telemetry 4xx was logged as a fault rejection"


def test_successful_completions_are_counted_by_status_class() -> None:
    store = build_store()
    _, claimed = _claimed(store, NVIDIA_KERNEL_PATH, KERNEL_BODY)
    processor = _processor(store)

    _finalize(processor, claimed, 200, b'{"decisions":[]}')

    snapshot = processor.metrics_snapshot()
    assert snapshot["completions_by_path_status"] == {NVIDIA_KERNEL_PATH: {"2xx": 1}}
    assert store.list_collector_statuses("cluster-a", "node-a") == [], (
        "a successful completion minted a rejected-event status"
    )
