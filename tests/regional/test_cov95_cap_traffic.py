from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from scripts.e2e.regional import capacity_acceptance_traffic as traffic
from scripts.e2e.regional.capacity_acceptance_base import CapHarnessBase


class StopAfter:
    def __init__(self, samples: int) -> None:
        self.remaining = samples

    def wait(self, seconds: float) -> bool:
        assert seconds == 1
        self.remaining -= 1
        return self.remaining < 0


def metrics(queue: float, rejection: float = 2.0) -> list[Any]:
    a_depth = min(queue, 2.0)
    b_depth = min(queue - a_depth, 1.0)
    return [
        ("gpu_fault_processor_queue_depth", {}, queue),
        (
            "gpu_fault_processor_cluster_queue_depth",
            {"cluster_id": "cap-cluster-000"},
            a_depth,
        ),
        (
            "gpu_fault_processor_cluster_queue_depth",
            {"cluster_id": "cap-cluster-001"},
            b_depth,
        ),
        (
            "gpu_fault_processor_cluster_queue_depth",
            {"cluster_id": "cap-cluster-002"},
            queue - a_depth - b_depth,
        ),
        (
            "gpu_fault_processor_admission_rejections_by_cluster_total",
            {"cluster_id": "cap-cluster-000", "reason": "capacity"},
            rejection,
        ),
        (
            "gpu_fault_processor_admission_rejections_by_cluster_total",
            {"cluster_id": "cap-cluster-000", "reason": "reserved"},
            1.0,
        ),
    ]


def test_monitor_uses_cluster_labels_and_retains_maxima_across_samples() -> None:
    batches = iter([metrics(5), metrics(3)])
    harness = SimpleNamespace(
        metrics=lambda _url: next(batches), metric_value=CapHarnessBase.metric_value
    )
    maxima: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    traffic.cap001_monitor(
        harness,
        SimpleNamespace(url="http://example"),
        StopAfter(2),
        maxima,
        samples,
        errors,
    )
    assert errors == []
    assert len(samples) == 2
    assert [item["queue_depth"] for item in samples] == [5, 3]
    assert maxima == {
        "queue_depth": 5.0,
        "a_depth": 2.0,
        "b_depth": 1.0,
        "a_rejections": 3.0,
        "b_rejections": 0.0,
    }


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_monitor_rejects_invalid_admission_counters_without_claiming_a_sample(
    value: float,
) -> None:
    values = metrics(1, value)
    assert traffic.sparse_cluster_queue_depth(values, "cap-cluster-000") == 1, (
        "the census must be valid so the rejection counter is the only defect"
    )
    harness = SimpleNamespace(
        metrics=lambda _url: values, metric_value=CapHarnessBase.metric_value
    )
    maxima: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    traffic.cap001_monitor(
        harness,
        SimpleNamespace(url="http://example"),
        StopAfter(1),
        maxima,
        samples,
        errors,
    )
    assert errors == ["CapError"]
    assert samples == []
    assert maxima == {}


@pytest.mark.parametrize("defect", ["total", "missing-cluster", "duplicate-cluster"])
def test_monitor_rejects_an_incomplete_or_contradictory_census(defect: str) -> None:
    values = metrics(5)
    if defect == "total":
        values[0] = ("gpu_fault_processor_queue_depth", {}, 4.0)
    elif defect == "missing-cluster":
        values.pop(3)
    else:
        values.append(values[1])
    harness = SimpleNamespace(
        metrics=lambda _url: values, metric_value=CapHarnessBase.metric_value
    )
    maxima: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    traffic.cap001_monitor(
        harness,
        SimpleNamespace(url="http://example"),
        StopAfter(1),
        maxima,
        samples,
        errors,
    )
    assert errors == ["CapError"], "an invalid queue census must be recorded"
    assert samples == [], "an invalid census cannot claim a complete sample"
    assert maxima == {}, "invalid census values cannot enter the observed maxima"


def test_monitor_records_transport_failure_and_continues_observing() -> None:
    reads = []

    def read(url: str) -> list[Any]:
        reads.append(url)
        if len(reads) == 1:
            raise TimeoutError("synthetic metric timeout")
        return metrics(2)

    harness = SimpleNamespace(metrics=read, metric_value=CapHarnessBase.metric_value)
    maxima: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    traffic.cap001_monitor(
        harness,
        SimpleNamespace(url="http://example"),
        StopAfter(2),
        maxima,
        samples,
        errors,
    )
    assert reads == ["http://example", "http://example"]
    assert errors == ["TimeoutError"]
    assert len(samples) == 1
    assert maxima["queue_depth"] == 2


@pytest.mark.parametrize("scheduled", [0.0, 12.0])
@pytest.mark.parametrize("failures,status", [(0, 202), (1, 429), (3, None)])
def test_sender_bounds_transport_retries_and_preserves_request_identity(
    monkeypatch: pytest.MonkeyPatch, scheduled: float, failures: int, status: int | None
) -> None:
    elapsed = [10.0]
    sleeps = []
    requests: list[httpx.Request] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        elapsed[0] += seconds

    def send(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) <= failures:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return httpx.Response(status or 202, headers={"Retry-After": "2"})

    monkeypatch.setattr(
        traffic, "time", SimpleNamespace(perf_counter=lambda: elapsed[0], sleep=sleep)
    )
    harness = SimpleNamespace(
        tokens=["synthetic-a", "synthetic-b"],
        cluster_headers=CapHarnessBase.cluster_headers,
    )
    with httpx.Client(
        transport=httpx.MockTransport(send), base_url="http://example"
    ) as client:
        result = traffic.cap001_send(harness, client, "baseline", 1, 4, scheduled)
    assert len(requests) == min(failures + 1, 3)
    assert result["transport_retries"] == failures
    assert result["status"] == (
        status if status is not None else "transport-error:ReadTimeout"
    )
    assert result["retry_after"] == ("2" if status is not None else None)
    assert result["latency_ms"] == pytest.approx(min(failures, 2) * 50)
    assert sleeps == ([2.0] if scheduled == 12 else []) + [0.05] * min(failures, 2)
    payloads = [json.loads(request.content) for request in requests]
    assert all(payload == payloads[0] for payload in payloads), (
        "transport retry changed the request's deduplication identity"
    )
    assert payloads[0]["batch_id"] == "cap001-baseline-c001-00004"
    assert payloads[0]["cluster_id"] == "cap-cluster-001"
    assert payloads[0]["node_id"] == "node-baseline-c001-00004"
    assert {request.url.path for request in requests} == {
        "/v1/collector-events/host-telemetry"
    }
    assert all(
        request.headers["X-GPU-Fault-Cluster-ID"] == "cap-cluster-001"
        for request in requests
    ), "a retry changed the authenticated cluster binding"
