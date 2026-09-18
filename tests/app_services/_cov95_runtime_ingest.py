from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from gpu_fault.app import ApplicationContext
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricSource,
)
from tests._builders import asgi_client

NOW = datetime(2026, 9, 12, 8, tzinfo=timezone.utc)


def post(context: ApplicationContext, path: str, payload: Any) -> httpx.Response:
    async def request() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(path, json=payload.model_dump(mode="json"))

    return asyncio.run(request())


def inventory(
    context: ApplicationContext,
    *,
    observed_at: datetime = NOW,
    boot: str = "boot-a",
    devices: tuple[tuple[str, str], ...] = (
        ("GPU-a", "0000:b9:00.0"),
        ("GPU-b", "0000:c1:00.0"),
    ),
) -> None:
    context.store.save_gpu_inventory_snapshot(
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=observed_at,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id=boot,
            expected_gpu_count=len(devices),
            devices=[
                GpuInventoryDevice(gpu_index=index, gpu_uuid=uuid, pci_bdf=pci)
                for index, (uuid, pci) in enumerate(devices)
            ],
        )
    )


def legacy_metric(
    context: ApplicationContext,
    *,
    uuid: str | None = "GPU-a",
    pci: str | None = "0000:b9:00.0",
    observed_at: datetime = NOW,
) -> None:
    context.store.observe_gpu_metric(
        ("cluster-a", "node-a", uuid or "unknown", "unit_inventory"),
        GpuMetricLatest(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=observed_at,
            source=GpuMetricSource.NVIDIA_SMI,
            sample=GpuMetricSample(
                metric_name="unit_inventory",
                canonical_name="unit_inventory",
                value=1,
                gpu_uuid=uuid,
                pci_bdf=pci,
            ),
        ),
    )
