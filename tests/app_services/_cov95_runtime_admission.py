from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from gpu_fault.app.admission_runtime import AdmissionRuntime, AdmissionRuntimeFactory
from gpu_fault.processor import ProcessorRequest
from tests._builders import build_context

POOLS = (
    "store_io",
    "decode_io",
    "fault_store_io",
    "evidence_store_io",
    "fault_decode_io",
    "spool_store_io",
)
BATCHERS = ("admission_batcher", "fault_batcher", "evidence_batcher", "spool_batcher")


@dataclass
class Admission:
    context: Any
    runtime: AdmissionRuntime

    async def close_batchers(self) -> None:
        for name in BATCHERS:
            await getattr(self.runtime, name).close()


@pytest.fixture(name="admission")
def admission_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[Admission]:
    for name in (
        "GPU_FAULT_STORE_IO_WORKERS",
        "GPU_FAULT_INGRESS_DECODE_WORKERS",
        "GPU_FAULT_FAULT_STORE_IO_WORKERS",
        "GPU_FAULT_EVIDENCE_STORE_IO_WORKERS",
        "GPU_FAULT_FAULT_DECODE_WORKERS",
        "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS",
    ):
        monkeypatch.setenv(name, "1")
    context = build_context()
    runtime = AdmissionRuntimeFactory(context, None).build()
    try:
        yield Admission(context, runtime)
    finally:
        for name in POOLS:
            getattr(runtime, name).close()


def item(cluster: str = "cluster-a") -> ProcessorRequest:
    return ProcessorRequest.from_http(
        method="POST",
        path="/v1/collector-events/node-logs",
        query="",
        body=b"{}",
        content_type="application/json",
        cluster_id=cluster,
    )
