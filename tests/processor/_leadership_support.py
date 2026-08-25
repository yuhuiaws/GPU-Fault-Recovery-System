"""Shared fixtures for processor leadership shards."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import SqliteStore
from tests._builders import build_store, processor_request

NOW = datetime.now(timezone.utc)

LEADER_LEASE = timedelta(seconds=15)

REQUEST_LEASE = timedelta(seconds=120)


@pytest.fixture(params=["memory", "sqlite"])
def stores(request, tmp_path):
    if request.param == "memory":
        shared = build_store()
        yield shared, shared
        return
    path = str(tmp_path / "processor.db")
    first = SqliteStore(path)
    second = SqliteStore(path)
    try:
        yield first, second
    finally:
        first.close()
        second.close()


def correlated_request(
    path: str, *, job_id: str = "training-job", attempt_id: str = "training-job-a001"
) -> ProcessorRequest:
    return processor_request(
        path, body=f'{{"job_id":"{job_id}","attempt_id":"{attempt_id}"}}'.encode()
    )


def _telemetry(
    node_id: str,
    value: int,
    *,
    path: str = "/v1/collector-events/gpu-metrics",
    reasons: list[str] | None = None,
) -> ProcessorRequest:
    body: dict = {"node_id": node_id, "value": value}
    if reasons is not None:
        body["edge_filter_reasons"] = reasons
    return processor_request(
        path, body=json.dumps(body, separators=(",", ":")).encode()
    )
