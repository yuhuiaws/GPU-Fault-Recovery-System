from __future__ import annotations

from types import MappingProxyType

from gpu_fault.channel_registry import (
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
)


GPU_INVENTORY_BATCH_SIZE = 16
GPU_METRICS_BATCH_SIZE = 8
HOST_TELEMETRY_BATCH_SIZE = 8
NODE_LOG_BATCH_SIZE = 8

TELEMETRY_BATCH_SIZE_BY_PATH = MappingProxyType(
    {
        GPU_INVENTORY_PATH: GPU_INVENTORY_BATCH_SIZE,
        GPU_METRICS_PATH: GPU_METRICS_BATCH_SIZE,
        HOST_TELEMETRY_PATH: HOST_TELEMETRY_BATCH_SIZE,
        NODE_LOG_PATH: NODE_LOG_BATCH_SIZE,
    }
)


def telemetry_batch_size(path: str) -> int:
    return TELEMETRY_BATCH_SIZE_BY_PATH[path]
