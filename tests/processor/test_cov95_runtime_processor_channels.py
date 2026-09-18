"""Channel registry invariants enforced before producer/processor routing."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from gpu_fault import channel_registry as channels
from tests._builders import processor_request
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize(
    ("path", "updates", "message"),
    [
        (
            channels.GPU_METRICS_PATH,
            {"path": "/different"},
            "invalid processor channel path",
        ),
        (channels.GPU_METRICS_PATH, {"edge_filtered": False}, "mode and flag disagree"),
        (
            channels.GPU_METRICS_PATH,
            {"routine_reasons": frozenset()},
            "no routine vocabulary",
        ),
        (channels.GPU_METRICS_PATH, {"batchable": False}, "must be batchable"),
        (
            channels.GPU_METRICS_PATH,
            {"lane": channels.ChannelLane.NODE},
            "requires EDGE_SUMMARY",
        ),
        (channels.GPU_METRICS_PATH, {"spool_weight": -1}, "cannot be negative"),
        (
            channels.NVIDIA_KERNEL_PATH,
            {"incident_scoped": False},
            "correlated fault channel",
        ),
        (channels.GPU_INVENTORY_PATH, {"incident_scoped": False}, "without node keys"),
    ],
)
def test_invalid_channel_cannot_pass_registry_startup_validation(
    monkeypatch, path, updates, message
):
    monkeypatch.setitem(
        channels.CHANNEL_REGISTRY,
        path,
        replace(channels.CHANNEL_REGISTRY[path], **updates),
    )
    with pytest.raises(RuntimeError, match=message):
        channels.validate_channel_registry()


@pytest.mark.parametrize("change", ["missing", "unregistered", "exact"])
def test_collector_route_inventory_requires_exact_registered_channels(change):
    paths = set(channels.COLLECTOR_CHANNEL_PATHS)
    if change == "missing":
        paths.remove(channels.GPU_METRICS_PATH)
    elif change == "unregistered":
        paths.add("/v1/collector-events/unregistered")
    if change == "exact":
        channels.validate_collector_routes(paths)
    else:
        with pytest.raises(RuntimeError, match="collector channel registry mismatch"):
            channels.validate_collector_routes(paths)


@pytest.mark.parametrize(
    ("path", "payload", "routine", "priority"),
    [
        (channels.GPU_INVENTORY_PATH, {}, True, 100),
        (channels.WORKLOAD_OBSERVATIONS_PATH, {}, False, 50),
        (channels.GPU_METRICS_PATH, {"collection_errors": ["partial"]}, False, 50),
        (
            channels.GPU_METRICS_PATH,
            {"edge_filter_reasons": "health-summary"},
            False,
            50,
        ),
        (channels.GPU_METRICS_PATH, {"edge_filter_reasons": [None]}, False, 50),
        (
            channels.GPU_METRICS_PATH,
            {"edge_filter_reasons": ["baseline:boot"]},
            True,
            100,
        ),
        (channels.NODE_LOG_PATH, {}, False, 50),
    ],
)
def test_channel_routine_vocabulary_controls_request_spooling_and_priority(
    path, payload, routine, priority
):
    channel = channels.channel_for_path(path)
    assert channel.is_routine_payload(payload) is routine
    request = processor_request(path, body=json.dumps(payload).encode())
    assert request.queue_priority() == priority
    assert request.spoolable() is (routine and channel.spoolable)


def test_summary_channel_without_suffix_retains_node_lane(monkeypatch):
    path = channels.HOST_TELEMETRY_PATH
    monkeypatch.setitem(
        channels.CHANNEL_REGISTRY,
        path,
        replace(channels.CHANNEL_REGISTRY[path], summary_lane_suffix=None),
    )
    channels.validate_channel_registry()
    request = processor_request(
        path, body=b'{"node_id":"node-a","edge_filter_reasons":["health-summary"]}'
    )
    assert request.ordering_key() == "cluster-a:node:node-a"


def test_coverage_heartbeat_never_consumes_fault_reserve_under_attempt_prefix():
    request = processor_request(channels.ATTEMPT_COVERAGE_PATH)
    assert request.queue_priority() == 100
    assert request.coalescable(), (
        "cluster coverage is intentionally latest-wins evidence"
    )
    assert not channels.is_fault_path(request.path), (
        "coverage heartbeat is not a fault event"
    )
    assert not channels.is_control_plane_action_path(request.path), (
        "coverage heartbeat must not inherit the attempt mutation tier"
    )
