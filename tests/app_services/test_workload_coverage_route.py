"""``POST /v1/workload-coverage`` records the watcher's scan heartbeat.

The heartbeat is the data plane's statement that it scanned the cluster at a
given instant and found ``attempt_count`` managed attempts. It is what lets
the topology service answer IDLE for a cluster with no attempt at all instead
of failing closed to UNKNOWN.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from gpu_fault.channel_registry import (
    CHANNEL_REGISTRY,
    WORKLOAD_COVERAGE_PATH,
    ChannelPriorityMode,
)
from gpu_fault.processor.models import ProcessorRequest
from gpu_fault.watcher import WorkloadCoverageHeartbeat
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)


def _heartbeat(scanned_at: datetime = NOW, attempt_count: int = 0) -> dict:
    return WorkloadCoverageHeartbeat(
        cluster_id="cluster-a", scanned_at=scanned_at, attempt_count=attempt_count
    ).model_dump(mode="json")


def _post(context, body: dict) -> int:
    async def scenario() -> int:
        async with asgi_client(context) as client:
            response = await client.post(WORKLOAD_COVERAGE_PATH, json=body)
        return response.status_code

    return asyncio.run(scenario())


def test_a_heartbeat_is_stored_and_makes_the_cluster_idle() -> None:
    context = build_context()

    assert _post(context, _heartbeat()) == 200

    stored = context.store.get_workload_coverage("cluster-a")
    assert stored is not None and stored.scanned_at == NOW
    assert context.topology.resolve("cluster-a", "node-1", NOW).workload_state == "IDLE"


def test_an_older_heartbeat_is_accepted_but_does_not_move_coverage_back() -> None:
    context = build_context()
    assert _post(context, _heartbeat(NOW)) == 200

    assert _post(context, _heartbeat(NOW - timedelta(minutes=5))) == 200

    assert context.store.get_workload_coverage("cluster-a").scanned_at == NOW


def test_the_channel_is_routine_latest_wins_and_coalesces_per_cluster() -> None:
    channel = CHANNEL_REGISTRY[WORKLOAD_COVERAGE_PATH]
    assert channel.priority_mode is ChannelPriorityMode.ROUTINE
    assert channel.latest_wins is True
    assert channel.receipt is True

    def request(scanned_at: datetime) -> ProcessorRequest:
        return ProcessorRequest.from_http(
            method="POST",
            path=WORKLOAD_COVERAGE_PATH,
            query="",
            body=json.dumps(_heartbeat(scanned_at)).encode(),
            content_type="application/json",
            cluster_id="cluster-a",
        )

    first = request(NOW)
    second = request(NOW + timedelta(seconds=15))
    assert first.queue_priority() == 100
    assert first.ordering_key() == second.ordering_key() == "cluster-a"
