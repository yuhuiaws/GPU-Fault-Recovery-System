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

import pytest

from gpu_fault.channel_registry import (
    ATTEMPT_COVERAGE_PATH,
    GPU_METRICS_PATH,
    NVIDIA_KERNEL_PATH,
)
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


@pytest.mark.parametrize(
    ("request_path", "metric_path"),
    [
        ("/v1/incidents/incident-{id}/close", "/v1/incidents/{id}"),
        ("/v1/workflows/workflow-{id}/simulate", "/v1/workflows/{id}"),
        ("/v1/workflows/workflow-{id}/steps/step-{id}/complete", "/v1/workflows/{id}"),
        ("/v1/recovery-plans/plan-{id}/simulate", "/v1/recovery-plans/{id}"),
        ("/v1/attempts/attempt-{id}/hang-check", "/v1/attempts/{id}"),
        ("/v1/attempts/cluster-{id}/attempt-{id}/decision", "/v1/attempts/{id}"),
        (
            "/v1/advisory-notifications/notification-{id}/send",
            "/v1/advisory-notifications/{id}",
        ),
        (
            "/v1/installation-resources/site-{id}/resource/{id}/nested",
            "/v1/installation-resources/{id}",
        ),
        ("/v1/training-health/cluster-{id}/scan", "/v1/training-health/{id}"),
        ("/v1/fleet/agents/cluster-{id}/node-{id}/drain", "/v1/fleet/agents/{id}"),
        ("/v1/fleet/deployments/deployment-{id}/advance", "/v1/fleet/deployments/{id}"),
    ],
)
def test_dynamic_completion_paths_have_bounded_labels_and_keep_request_identity(
    request_path: str, metric_path: str
) -> None:
    store = build_store()
    processor = _processor(store)
    for index in range(100):
        path = request_path.format(id=index)
        request, claimed = _claimed(store, path, b'{"unchanged":true}')
        _finalize(processor, claimed, 200 if index % 2 == 0 else 422)
        current = store.get_processor_request(request.request_id)
        assert current.path == claimed.path == path
        assert current.body() == b'{"unchanged":true}'
        assert current.status is ProcessorRequestStatus.COMPLETED
    assert processor.metrics_snapshot()["completions_by_path_status"] == {
        metric_path: {"2xx": 50, "4xx": 50}
    }


@pytest.mark.parametrize(
    "path",
    [
        ATTEMPT_COVERAGE_PATH,
        "/v1/attempts/failure-detected",
        "/v1/attempts/terminal",
        "/v1/advisory-notifications/dispatch",
        "/v1/installation-resources/sync",
        "/v1/fleet/agents/heartbeat",
        "/v1/fleet/deployments",
        "/v1/runtime-profiles",
        NVIDIA_KERNEL_PATH,
        GPU_METRICS_PATH,
    ],
)
def test_static_completion_paths_keep_their_labels(path: str) -> None:
    store = build_store()
    _, claimed = _claimed(store, path, b"{}")
    processor = _processor(store)
    _finalize(processor, claimed, 200)
    assert processor.metrics_snapshot()["completions_by_path_status"] == {
        path: {"2xx": 1}
    }
