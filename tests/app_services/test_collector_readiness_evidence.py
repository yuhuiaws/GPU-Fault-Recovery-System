"""First-report grace never supplies positive collector readiness."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.collector_requirements import (
    COLLECTOR_SYSTEMD_UNITS,
    collector_silent_thresholds,
    required_collectors_for_agent,
)
from gpu_fault.regional import RegionalClusterLifecycle
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import asgi_client, build_context, build_store
from tests.fleet._support import heartbeat, registry, signed
from tests.regional._regional_support import TOKEN_A, registration

EXECUTION_TOKEN = "e" * 32
HEADERS = {"X-GPU-Fault-Execution-Token": EXECUTION_TOKEN}


@pytest.mark.parametrize("report", ["missing", "error-only", "fresh", "stale"])
@pytest.mark.parametrize("service_state", ["active", "unknown", "inactive", "failed"])
def test_readiness_requires_a_fresh_success_not_warmup(
    report: str, service_state: str
) -> None:
    now = datetime.now(timezone.utc)
    store = build_store()
    fleet = registry(store, now=lambda: now)
    kind = CollectorKind.NVIDIA_KERNEL
    services = {
        unit: {
            "active": service_state if collector is kind else "active",
            "enabled": "enabled" if collector is kind else "disabled",
        }
        for collector, unit in COLLECTOR_SYSTEMD_UNITS.items()
    }
    fleet.register(
        signed(heartbeat("node-a", observed_at=now, collector_services=services))
    )
    context = build_context(
        store=store, execution_token=EXECUTION_TOKEN, fleet_registry=fleet
    )
    if report != "missing":
        last = (
            None
            if report == "error-only"
            else now
            - timedelta(
                seconds=collector_silent_thresholds()[kind] + 10
                if report == "stale"
                else 1
            )
        )
        store.save_collector_status(
            CollectorStatus(
                cluster_id="cluster-a",
                node_id="node-a",
                collector=kind,
                observed_at=now,
                ingested_at=now,
                last_success_at=last,
                last_error_at=now if report == "error-only" else None,
                errors=["first collection failed"] if report == "error-only" else [],
            )
        )

    async def scenario() -> dict[str, Any]:
        async with asgi_client(context) as client:
            response = await client.get(
                "/v1/collector-readiness/cluster-a", headers=HEADERS
            )
        assert response.status_code == 200, response.text
        return response.json()

    result = asyncio.run(scenario())
    (node,) = result["nodes"]
    collector = node["collectors"][kind.value]
    running = service_state in {"active", "unknown"}
    expected = report == "fresh" and running
    assert collector["ready"] is expected
    assert node["ready"] is expected and result["ready"] is expected
    assert collector["pending_first_report"] is (
        report in {"missing", "error-only"} and running
    ), "first-report grace remains diagnostic"
    if report in {"missing", "error-only"}:
        assert collector["last_success_at"] is None
        assert collector["age_seconds"] is None
    assert store.list_notifications() == [], (
        "a readiness read must not turn warmup into a silence notification"
    )


def test_partial_collector_reports_hold_readiness_and_pending_claims() -> None:
    now = datetime.now(timezone.utc)
    store = build_store()
    fleet = registry(store, now=lambda: now)
    agents = [
        fleet.register(signed(heartbeat(node, observed_at=now)))
        for node in ("node-a", "node-b")
    ]
    context = build_context(
        store=store, execution_token=EXECUTION_TOKEN, fleet_registry=fleet
    )
    context.regional_mode = True
    store.save_regional_cluster(
        registration("cluster-a", TOKEN_A).model_copy(
            update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
        )
    )

    def report(node_id: str, kind: CollectorKind) -> None:
        store.save_collector_status(
            CollectorStatus(
                cluster_id="cluster-a",
                node_id=node_id,
                collector=kind,
                observed_at=now,
                ingested_at=now,
                last_success_at=now,
            )
        )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for agent in agents:
                for kind in required_collectors_for_agent(agent):
                    if (
                        agent.node_id == "node-b"
                        and kind is CollectorKind.NVIDIA_KERNEL
                    ):
                        continue
                    report(agent.node_id, kind)
            partial = await client.get(
                "/v1/collector-readiness/cluster-a", headers=HEADERS
            )
            assert partial.status_code == 200, partial.text
            result = partial.json()
            nodes = {node["node_id"]: node for node in result["nodes"]}
            assert result["ready"] is False
            assert nodes["node-a"]["ready"] is True
            assert nodes["node-b"]["ready"] is False
            missing = nodes["node-b"]["collectors"]["NVIDIA_KERNEL"]
            assert missing["pending_first_report"] is True
            assert missing["ready"] is False and missing["last_success_at"] is None

            report("node-b", CollectorKind.NVIDIA_KERNEL)
            complete = await client.get(
                "/v1/collector-readiness/cluster-a", headers=HEADERS
            )
            assert complete.status_code == 200, complete.text
            assert complete.json()["ready"] is True
            assert all(
                not item["pending_first_report"]
                for node in complete.json()["nodes"]
                for item in node["collectors"].values()
            ), "fresh reports from every collector must clear pending-first-report"

            claim = await client.post(
                "/v1/regional/executors/claim",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={"executor_id": "pending-executor"},
            )
            assert claim.status_code == 423, (
                "collector observations do not activate a PENDING cluster"
            )

    asyncio.run(scenario())
