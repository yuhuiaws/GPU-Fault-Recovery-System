from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import processor_request

NOW = datetime(2030, 1, 1, tzinfo=UTC)
HOST_PATH = "/v1/collector-events/host-telemetry"
GPU_PATH = "/v1/collector-events/gpu-metrics"
FAULT_PATH = "/v1/gpu-events/xid"
OBSERVATION_PATH = "/v1/workload-observations"


@pytest.fixture(name="compat_store", params=["memory", "sqlite"])
def compat_store_fixture(request, tmp_path):
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "compat.db"))
    )
    try:
        yield store
    finally:
        if isinstance(store, SqliteStore):
            store.close()


def request_model(
    request_id,
    *,
    path=HOST_PATH,
    node_id="node-local",
    cluster_id="cluster-local",
    payload=None,
    **values,
):
    body = {"node_id": node_id, "summary": True} if payload is None else payload
    request = processor_request(
        path, cluster_id=cluster_id, body=json.dumps(body, sort_keys=True).encode()
    )
    return ProcessorRequest.model_validate(
        {
            **request.model_dump(),
            "request_id": request_id,
            "created_at": NOW,
            "updated_at": NOW,
            **values,
        }
    )


def claim_request(
    store,
    request_id,
    *,
    node_id="node-local",
    owner_id="owner",
    now=None,
    lease_duration=timedelta(seconds=120),
):
    at = datetime.now(UTC) if now is None else now
    store.enqueue_processor_request(
        request_model(
            request_id,
            node_id=node_id,
            created_at=at - timedelta(seconds=1),
            updated_at=at - timedelta(seconds=1),
        )
    )
    (claimed,) = store.claim_active_processor_requests(
        owner_id, now=at, lease_duration=lease_duration, limit=1
    )
    return claimed
