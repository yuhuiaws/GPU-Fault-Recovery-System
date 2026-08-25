from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.host_health import HostMetricSample
from gpu_fault.store import SqliteStore
from gpu_fault.telemetry import EvidenceKind, EvidenceService
from gpu_fault.watcher import AttemptObservation
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    build_store,
    container_observation,
    host_telemetry_batch,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def running_observation() -> AttemptObservation:
    return attempt_observation(
        "training-job",
        "attempt-a",
        NOW,
        workload_ids=["training/job/training-job"],
        containers=[
            container_observation("pod-0", "worker-0", 0, "node-0", gpu_uuids=["GPU-0"])
        ],
    )


def capture(service: EvidenceService, index: int, node_id: str = "node-a") -> None:
    service.capture(
        record_id=f"record-{index}",
        cluster_id="cluster-a",
        node_id=node_id,
        kind=EvidenceKind.GPU_METRICS,
        observed_at=NOW + timedelta(seconds=index),
        attempt_ids=["attempt-a"],
        payload={"index": index},
    )


def test_evidence_is_bounded_per_node() -> None:
    store = build_store()
    service = EvidenceService(
        store, retention=timedelta(hours=1), max_records_per_node=2
    )

    capture(service, 1)
    capture(service, 2)
    capture(service, 3)

    records = store.list_raw_evidence("cluster-a", attempt_id="attempt-a")
    assert [item.payload["index"] for item in records] == [3, 2]


def test_evidence_survives_sqlite_restart(tmp_path) -> None:
    path = tmp_path / "evidence.db"
    first = SqliteStore(str(path))
    capture(EvidenceService(first), 1)
    first.close()

    second = SqliteStore(str(path))
    try:
        records = second.list_raw_evidence(
            "cluster-a", node_id="node-a", kind=EvidenceKind.GPU_METRICS
        )
        assert records[0].record_id == "record-1"
    finally:
        second.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_expired_evidence_is_cleaned_without_new_writes(tmp_path, store_kind) -> None:
    store = (
        build_store()
        if store_kind == "memory"
        else SqliteStore(str(tmp_path / "cleanup.db"))
    )
    service = EvidenceService(store, retention=timedelta(hours=1))
    try:
        capture(service, 1)

        deleted = store.cleanup_expired_raw_evidence(
            now=datetime.now(timezone.utc) + timedelta(hours=2)
        )

        assert deleted == 1
        assert store.list_raw_evidence("cluster-a") == []
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def test_host_batch_is_correlated_to_attempt_and_archived() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            await client.post(
                "/v1/workload-observations",
                json=running_observation().model_dump(mode="json"),
            )
            response = await client.post(
                "/v1/collector-events/host-telemetry",
                json=host_telemetry_batch(
                    "host-correlated",
                    NOW,
                    [HostMetricSample(name="cpu_usage_percent", value=99)],
                    node_id="node-0",
                ).model_dump(mode="json"),
            )
            evidence = await client.get(
                "/v1/evidence/cluster-a", params={"attempt_id": "attempt-a"}
            )

        assert response.status_code == 200
        assert response.json()["findings"][0]["affected_workload_ids"] == [
            "training/job/training-job"
        ]
        assert evidence.status_code == 200
        assert evidence.json()[0]["record_id"] == ("host-telemetry/host-correlated")

    asyncio.run(scenario())
