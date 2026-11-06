"""The runners' metric snapshots keep the families the perf plan reads.

``regional_capacity_suite.scrape_metrics`` keeps only the lines its
``METRIC_PATTERN`` matches. It dropped the processor lane-holder summary the
perf plan asks every round to record (§13, item 5: the path label and the
before/after delta in the summary), and the process-health families the
ingress liveness analysis reads -- the listening sockets' accept queues, GC
collections and pauses, resident memory (§13, item 13) -- so no artifact of a
round carried them.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.app.metric_aggregation import STRATEGIES
from scripts.perf import regional_capacity_suite as capacity

KEPT = [
    'gpu_fault_processor_lane_holder_seconds_count{path="/v1/x",process="0"} 3',
    'gpu_fault_processor_lane_holder_seconds_sum{path="/v1/x",process="0"} 0.5',
    'gpu_fault_processor_lane_holder_seconds_max{path="/v1/x",process="0"} 0.4',
    'gpu_fault_listen_accept_queue_connections{port="8080",process="1"} 12',
    'gpu_fault_listen_accepted_connections_total{port="8080",process="1"} 900',
    'gpu_fault_listen_open_connections{port="8080"} 40',
    "gpu_fault_listen_somaxconn 4096",
    'gpu_fault_gc_collections_total{generation="2",process="0"} 7',
    'gpu_fault_gc_collection_seconds_total{generation="2",process="0"} 1.2',
    'gpu_fault_gc_collection_seconds_max{generation="2"} 0.31',
    "gpu_fault_gc_frozen_objects 120000",
    'gpu_fault_process_resident_memory_bytes{process="0"} 512000000',
    'gpu_fault_event_loop_lag_seconds_bucket{le="0.5",process="0"} 4',
]

# The product's names for the families above (gpu_fault.app.metric_aggregation).
FAMILIES = {
    "gpu_fault_processor_lane_holder_seconds",
    "gpu_fault_listen_accept_queue_connections",
    "gpu_fault_listen_accepted_connections_total",
    "gpu_fault_listen_open_connections",
    "gpu_fault_listen_somaxconn",
    "gpu_fault_gc_collections_total",
    "gpu_fault_gc_collection_seconds_total",
    "gpu_fault_gc_collection_seconds_max",
    "gpu_fault_gc_frozen_objects",
    "gpu_fault_process_resident_memory_bytes",
    "gpu_fault_event_loop_lag_seconds",
}


def test_the_snapshot_keeps_the_lane_holder_and_process_health_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "\n".join(["# HELP noise", "gpu_fault_unrelated_family 1", *KEPT])
    monkeypatch.setattr(capacity.capacity_timeseries, "metrics_port", lambda _pod: 8080)
    monkeypatch.setattr(capacity, "control", lambda *_args, **_kwargs: body)

    snapshot = capacity.scrape_metrics(["gpu-fault-api-ha-a"])

    assert snapshot["gpu-fault-api-ha-a"].splitlines() == KEPT, (
        "every lane-holder and process-health sample reaches metrics-*.json"
    )
    assert FAMILIES <= set(STRATEGIES), (
        f"families the product no longer renders: {sorted(FAMILIES - set(STRATEGIES))}"
    )


def pod(*lines: str) -> str:
    return "\n".join(lines)


def test_the_lane_holder_delta_is_per_path_over_pods_and_processes() -> None:
    before = {
        "api-a": pod(
            'gpu_fault_processor_lane_holder_seconds_count{path="/v1/a",process="0"} 10',
            'gpu_fault_processor_lane_holder_seconds_sum{path="/v1/a",process="0"} 1.0',
            'gpu_fault_processor_lane_holder_seconds_count{path="/v1/a",process="1"} 50',
            'gpu_fault_processor_lane_holder_seconds_sum{path="/v1/a",process="1"} 9.0',
        ),
        "api-b": "# scrape failed: timeout",
    }
    after = {
        "api-a": pod(
            'gpu_fault_processor_lane_holder_seconds_count{path="/v1/a",process="0"} 30',
            'gpu_fault_processor_lane_holder_seconds_sum{path="/v1/a",process="0"} 5.0',
            'gpu_fault_processor_lane_holder_seconds_max{path="/v1/a",process="0"} 0.8',
            # process 1 restarted: its counters start again from zero
            'gpu_fault_processor_lane_holder_seconds_count{path="/v1/a",process="1"} 4',
            'gpu_fault_processor_lane_holder_seconds_sum{path="/v1/a",process="1"} 2.0',
            'gpu_fault_processor_lane_holder_seconds_max{path="/v1/a",process="1"} 1.5',
        ),
        "api-b": pod(
            'gpu_fault_processor_lane_holder_seconds_count{process="0",path="/v1/{id}/b"} 2',
            'gpu_fault_processor_lane_holder_seconds_sum{process="0",path="/v1/{id}/b"} 0.2',
        ),
    }

    delta: dict[str, Any] = capacity.lane_holder_delta(before, after)

    assert delta == {
        "/v1/a": {
            "holds": 24,
            "held_seconds": 6.0,
            "mean_seconds": 0.25,
            "max_seconds": 1.5,
        },
        "/v1/{id}/b": {
            "holds": 2,
            "held_seconds": 0.2,
            "mean_seconds": 0.1,
            "max_seconds": 0.0,
        },
    }, "holds and held seconds added by the round, per path; a reset counts from 0"
    assert capacity.lane_holder_delta({}, {}) == {}, "no samples, no paths"
