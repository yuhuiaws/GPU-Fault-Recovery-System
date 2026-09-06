"""Every channel whose findings can name a node gives its requests node
correlation keys, so the aggregation gate sees them in flight (F-B8).

Only the attempt lane and the two correlated-fault channels used to get
node keys. GPU metrics, host telemetry, node logs and GPU inventory all
create incidents for a node, yet their queued batches were invisible to
``_processor_queue_blocks`` -- the aggregation window opened while the very
evidence it waited for was still in the queue.
"""

from __future__ import annotations

import json

import pytest

from gpu_fault.channel_registry import (
    CHANNEL_REGISTRY,
    COLLECTOR_HEALTH_PATH,
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    NVIDIA_KERNEL_PATH,
    validate_channel_registry,
)
from gpu_fault.processor import ProcessorRequest

NODE_KEY = json.dumps(["cluster-a", "node", "node-a"], separators=(",", ":"))


def _request(path: str) -> ProcessorRequest:
    body = json.dumps(
        {"cluster_id": "cluster-a", "node_id": "node-a", "samples": []}
    ).encode()
    return ProcessorRequest.from_http(
        method="POST",
        path=path,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="cluster-a",
    )


@pytest.mark.parametrize(
    "path", [GPU_METRICS_PATH, HOST_TELEMETRY_PATH, NODE_LOG_PATH, GPU_INVENTORY_PATH]
)
def test_incident_scoped_channels_carry_the_node_key(path: str) -> None:
    request = _request(path)

    assert NODE_KEY in request.correlation_scope_keys, path
    # ...without becoming correlated faults: the lane interlocks stay as they were.
    assert not request.is_correlated_fault(), (
        f"{path} must not join the fault interlock"
    )


def test_collector_health_stays_unscoped() -> None:
    assert _request(COLLECTOR_HEALTH_PATH).correlation_scope_keys == []


def test_a_fault_channel_keeps_its_key_and_its_fault_status() -> None:
    request = _request(NVIDIA_KERNEL_PATH)
    assert NODE_KEY in request.correlation_scope_keys
    assert request.is_correlated_fault(), "kernel faults drive the interlock"


def test_the_registry_refuses_a_node_scoped_channel_without_node_keys(
    monkeypatch,
) -> None:
    from dataclasses import replace

    broken = replace(CHANNEL_REGISTRY[GPU_METRICS_PATH], incident_scoped=False)
    monkeypatch.setitem(CHANNEL_REGISTRY, GPU_METRICS_PATH, broken)

    with pytest.raises(RuntimeError, match="incident nodes without node keys"):
        validate_channel_registry()
