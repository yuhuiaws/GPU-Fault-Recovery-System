"""A collector that fails on time is not silent, and used to be invisible.

Silence was the only observable collector failure: `gpu_fault_collector_silent_nodes`
counts nodes whose last *success* has aged past a threshold. A node whose journal
read fails every cycle still delivers a batch every cycle, so nothing about it is
late -- and the node-logs threshold is a quarter of an hour, so a node that had
stopped reporting kernel events kept looking healthy for that long. These cases
pin the two halves of the fix: that an error-only batch is not booked as a
success, and that the count of such nodes is published.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot, is_erroring
from gpu_fault.collector_requirements import (
    CollectorActiveState,
    CollectorEnabledState,
    CollectorServiceState,
)
from gpu_fault.host_health import NodeLogBatch, NodeLogEntry
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import asgi_client, build_context, build_store

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def _error_only_batch(batch_id: str) -> NodeLogBatch:
    return NodeLogBatch(
        batch_id=batch_id,
        cluster_id="cluster-a",
        node_id="node-a",
        collected_at=NOW,
        runtime_profile_version="simulated-v1",
        edge_filter_reasons=["collection-error"],
        entries=[],
        collection_errors=["journalctl exited 1: Failed to open journal"],
    )


def _post(context, batch: NodeLogBatch) -> None:
    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/node-logs", json=batch.model_dump(mode="json")
            )
        assert response.status_code == 200, (
            f"node-log ingest returned {response.status_code}"
        )

    asyncio.run(scenario())


def _status(context) -> CollectorStatus:
    statuses = [
        item
        for item in context.store.list_collector_statuses("cluster-a", "node-a")
        if item.collector is CollectorKind.NODE_LOGS
    ]
    assert len(statuses) == 1, "expected exactly one node-logs collector status"
    return statuses[0]


def test_an_error_only_node_log_batch_is_not_booked_as_a_success() -> None:
    """The batch arrived; the collection it reports on did not happen.

    Booking it as a success is what let a broken log collector look healthy: the
    silence gauge reads `last_success_at`, so a node erroring every cycle kept
    resetting the only clock that would have caught it.
    """

    context = build_context(store=build_store())

    _post(context, _error_only_batch("logs-error-1"))

    status = _status(context)
    assert status.last_success_at is None, (
        "a batch that carried only collection errors is not a success"
    )
    assert status.last_error_at == NOW, "the failure has to carry a timestamp"
    assert status.errors == ["journalctl exited 1: Failed to open journal"], (
        "the node's own reason is the first thing an operator needs"
    )


def test_a_later_successful_batch_clears_the_error_state() -> None:
    """Self-clearing on real evidence, not on a timer.

    Both timestamps are sticky -- each ingest keeps the value it does not set --
    so nothing but an actual delivered batch can move the node out of the
    erroring count.
    """

    context = build_context(store=build_store())
    _post(context, _error_only_batch("logs-error-1"))
    assert is_erroring(_status(context)), "the node starts out erroring"

    _post(
        context,
        NodeLogBatch(
            batch_id="logs-ok-1",
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW + timedelta(seconds=30),
            runtime_profile_version="simulated-v1",
            edge_filter_reasons=["candidate-confirmed"],
            entries=[
                NodeLogEntry(
                    entry_id="journal-1",
                    source="dmesg",
                    observed_at=NOW + timedelta(seconds=30),
                    message="mce: Hardware Error: Machine check",
                )
            ],
        ),
    )

    status = _status(context)
    assert status.last_error_at == NOW, "the past failure is not rewritten away"
    assert not is_erroring(status), (
        "one successful batch is what clears the error state"
    )


@pytest.mark.parametrize(
    ("last_success_at", "last_error_at", "erroring", "why"),
    [
        (None, None, False, "a node that has never reported anything is not erroring"),
        (None, NOW, True, "errors and no success at all is the worst case"),
        (NOW, NOW + timedelta(seconds=30), True, "the newest state is the failure"),
        (NOW + timedelta(seconds=30), NOW, False, "the failure has been superseded"),
        (NOW, NOW, True, "a tie is resolved as failing, not as healthy"),
    ],
)
def test_erroring_is_decided_by_which_timestamp_is_newer(
    last_success_at: datetime | None,
    last_error_at: datetime | None,
    erroring: bool,
    why: str,
) -> None:
    status = CollectorStatus(
        cluster_id="cluster-a",
        node_id="node-a",
        collector=CollectorKind.NODE_LOGS,
        observed_at=NOW,
        ingested_at=NOW,
        last_success_at=last_success_at,
        last_error_at=last_error_at,
    )

    assert is_erroring(status) is erroring, why


def test_a_node_reporting_errors_on_time_is_counted_but_not_called_silent() -> None:
    """The two gauges have to disagree, or the new one adds nothing.

    The node here delivered a batch seconds ago, so it is nowhere near any
    silence threshold. If the erroring count did not separate from the silent
    count, the alert would just be a slower copy of `GpuFaultCollectorSilent`.
    """

    agent = SimpleNamespace(
        cluster_id="cluster-a",
        node_id="node-a",
        lifecycle_state=SimpleNamespace(value="ACTIVE"),
        last_seen_at=NOW,
        lease_expires_at=None,
        collector_services={
            "gpu-fault-log-collector": CollectorServiceState(
                active=CollectorActiveState.ACTIVE,
                enabled=CollectorEnabledState.ENABLED,
            )
        },
    )
    store = build_store(
        list_agents=lambda cluster_id=None: [agent],
        list_collector_statuses=lambda cluster_id, node_id=None: [
            CollectorStatus(
                cluster_id="cluster-a",
                node_id="node-a",
                collector=CollectorKind.NODE_LOGS,
                observed_at=NOW,
                ingested_at=NOW,
                last_success_at=NOW - timedelta(seconds=20),
                last_error_at=NOW,
            )
        ],
    )
    snapshot = CollectorMetricsSnapshot(
        build_context(store=store), owner_id="owner-a", enabled=True, now=lambda: NOW
    )

    snapshot.refresh()

    node_logs = [line for line in snapshot.lines() if 'channel="NODE_LOGS"' in line]
    assert any(
        line.startswith("gpu_fault_collector_erroring_nodes") and line.endswith(" 1")
        for line in node_logs
    ), f"the erroring count is not published in {node_logs}"
    assert any(
        line.startswith("gpu_fault_collector_silent_nodes") and line.endswith(" 0")
        for line in node_logs
    ), f"a node delivering batches 20 seconds ago is not silent: {node_logs}"
    assert [item["erroring"] for item in snapshot.details() if item["erroring"]] == [
        True
    ], "the per-node details name the erroring node for the runbook"
