from __future__ import annotations

from gpu_fault.app.routes.collector_events import router
from gpu_fault.channel_registry import (
    CHANNEL_REGISTRY,
    COLLECTOR_CHANNEL_PATHS,
    GPU_METRICS_PATH,
    NVIDIA_KERNEL_PATH,
    ChannelPool,
    channel_for_path,
    paths_for_pool,
    validate_channel_registry,
    validate_collector_routes,
)


def test_channel_registry_is_internally_consistent() -> None:
    validate_channel_registry()
    assert len(COLLECTOR_CHANNEL_PATHS) == 7
    assert set(CHANNEL_REGISTRY) >= COLLECTOR_CHANNEL_PATHS


def test_edge_filtered_reason_vocabulary_is_centralized() -> None:
    channel = channel_for_path(GPU_METRICS_PATH)
    assert channel is not None
    assert channel.priority({"edge_filter_reasons": ["baseline:gpu"]}) == 100
    assert channel.priority({"edge_filter_reasons": ["candidate-confirmed"]}) == 50
    assert (
        channel.priority(
            {
                "edge_filter_reasons": ["health-summary"],
                "collection_errors": ["dcgm unavailable"],
            }
        )
        == 50
    )


def test_fault_and_pool_classification_come_from_registry() -> None:
    assert channel_for_path(NVIDIA_KERNEL_PATH).priority({}) == 0
    assert GPU_METRICS_PATH in paths_for_pool(ChannelPool.GPU)


def test_every_collector_route_is_registered() -> None:
    paths = {
        route.path
        for route in router.routes
        if route.path.startswith("/v1/collector-events/")
    }
    validate_collector_routes(paths)
    for path in paths:
        channel = CHANNEL_REGISTRY[path]
        assert channel.pool is not None
        assert channel.lane is not None
        assert isinstance(channel.spoolable, bool)
