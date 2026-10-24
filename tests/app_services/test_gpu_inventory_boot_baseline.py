"""Inventory from a boot the node is not running never replaces its baseline.

GF-GENERAL-LIVE-026 ``stale_boot``: the inventory baseline is keyed on
cluster/node and the store only refuses an older-or-equal ``observed_at``, so a
snapshot that carried another boot id with a fresh timestamp -- a previous
boot's collector or outbox, a boot id that never existed -- became the
baseline and was graded: an identity finding on GPUs the node no longer
reports, and a baseline every later boot comparison (XID product fill, SXID
scope, composite boot checks) then refused. A boot *change* is now accepted
only from the node's current boot as its Agent reports it (ACTIVE, within its
heartbeat lease); without that knowledge the timestamp decides, as before.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.collector_metrics import is_erroring
from gpu_fault.app.ingest.telemetry_context import GPU_IDENTITY_CHANGED_METRIC
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import asgi_client, build_context

CLUSTER = "hp-cluster"
NODE = "worker-1"
PATH = "/v1/collector-events/gpu-inventory"
BATCH_PATH = "/v1/internal/processor/telemetry-batch"
#: tests/conftest.py sets GPU_FAULT_PROCESSOR_REPLAY_SECRET to this value.
REPLAY_TOKEN = "test-processor-replay-secret-" + "r" * 32
EXECUTION_TOKEN = "inventory-boot-baseline-token-" + "x" * 32


def _base() -> datetime:
    """Snapshot times near the wall clock the Agent lease is judged against."""

    return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=10)


def _snapshot(
    snapshot_id: str, uuids: list[str], *, boot: str, observed_at: datetime
) -> dict[str, Any]:
    return GpuInventorySnapshot(
        snapshot_id=snapshot_id,
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=observed_at,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id=boot,
        devices=[
            GpuInventoryDevice(
                gpu_index=index,
                gpu_uuid=uuid,
                pci_bdf=f"0000:{0x59 + index:02x}:00.0",
                product="H100",
            )
            for index, uuid in enumerate(uuids)
        ],
        expected_gpu_count=2,
    ).model_dump(mode="json")


def _agent(
    context: ApplicationContext,
    boot_id: str | None,
    *,
    lifecycle: AgentLifecycleState = AgentLifecycleState.ACTIVE,
    lease: timedelta | None = timedelta(minutes=5),
) -> None:
    now = datetime.now(timezone.utc)
    context.store.save_agent(
        AgentRecord(
            cluster_id=CLUSTER,
            node_id=NODE,
            endpoint="https://worker-1:9099",
            agent_version="test",
            artifact_sha256="a" * 64,
            policy_version="610",
            runtime_profile_version="simulated-v1",
            config_digest="test-config",
            allowed_operations=[],
            boot_id=boot_id,
            first_seen_at=now - timedelta(hours=1),
            last_seen_at=now,
            lease_expires_at=None if lease is None else now + lease,
            lifecycle_state=lifecycle,
        )
    )


def _post_all(target: Any, payloads: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    async def scenario() -> list[tuple[int, Any]]:
        answers = []
        async with asgi_client(target) as client:
            for payload in payloads:
                response = await client.post(PATH, json=payload)
                answers.append((response.status_code, response.json()))
        return answers

    return asyncio.run(scenario())


def _post_batch(app: Any, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = [
        {"request_id": payload["snapshot_id"], "path": PATH, "payload": payload}
        for payload in payloads
    ]

    async def scenario() -> list[dict[str, Any]]:
        async with asgi_client(app) as client:
            response = await client.post(
                BATCH_PATH,
                json={"items": items},
                headers={"X-GPU-Fault-Processor-Replay": REPLAY_TOKEN},
            )
        assert response.status_code == 200, response.text
        return response.json()["results"]

    return asyncio.run(scenario())


def _baseline_id(context: ApplicationContext) -> str | None:
    baseline = context.store.get_gpu_inventory_snapshot(CLUSTER, NODE)
    return None if baseline is None else baseline.snapshot_id


def _identity_incident(context: ApplicationContext, snapshot_id: str):
    return context.store.get_incident_by_event(
        f"{snapshot_id}-{GPU_IDENTITY_CHANGED_METRIC}"
    )


def _inventory_status(context: ApplicationContext) -> CollectorStatus:
    statuses = [
        status
        for status in context.store.list_collector_statuses(CLUSTER, NODE)
        if status.collector is CollectorKind.GPU_INVENTORY
    ]
    assert len(statuses) == 1, statuses
    return statuses[0]


def test_old_boot_inventory_with_a_fresh_timestamp_keeps_the_baseline() -> None:
    context = build_context()
    _agent(context, "boot-current")
    base = _base()
    answers = _post_all(
        context,
        [
            _snapshot(
                "inv-1", ["GPU-a", "GPU-b"], boot="boot-current", observed_at=base
            ),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-stale"],
                boot="boot-old",
                observed_at=base + timedelta(minutes=1),
            ),
        ],
    )
    assert [code for code, _ in answers] == [200, 200], answers
    assert answers[1][1]["snapshot_id"] == "inv-1", (
        "the refused snapshot must answer with the baseline it did not replace"
    )
    assert _baseline_id(context) == "inv-1", (
        "an old-boot inventory with a fresh timestamp replaced the baseline"
    )
    assert _identity_incident(context, "inv-2") is None, (
        "the product graded inventory from a boot the node is not running"
    )
    status = _inventory_status(context)
    assert is_erroring(status), status
    assert status.batch_id == "inv-2", status
    assert any(
        "boot-old" in error and "boot-current" in error for error in status.errors
    ), status.errors

    # The collector reports from the node's boot again: graded against the
    # baseline that was kept, so there is nothing to report and the channel
    # is healthy again.
    answers = _post_all(
        context,
        [
            _snapshot(
                "inv-3",
                ["GPU-a", "GPU-b"],
                boot="boot-current",
                observed_at=base + timedelta(minutes=2),
            )
        ],
    )
    assert [code for code, _ in answers] == [200], answers
    assert _baseline_id(context) == "inv-3"
    assert _identity_incident(context, "inv-3") is None
    status = _inventory_status(context)
    assert not is_erroring(status) and status.errors == [], status


def test_a_reboot_the_agent_reports_moves_the_baseline_and_is_still_graded() -> None:
    """The node came back on a new boot and its Agent says so: the new boot's
    inventory is the baseline, and a GPU lost across the reboot is diagnosed."""

    context = build_context()
    _agent(context, "boot-old")
    base = _base()
    _post_all(
        context,
        [_snapshot("inv-1", ["GPU-a", "GPU-b"], boot="boot-old", observed_at=base)],
    )
    _agent(context, "boot-new")
    answers = _post_all(
        context,
        [
            _snapshot(
                "inv-2",
                ["GPU-a"],
                boot="boot-new",
                observed_at=base + timedelta(minutes=5),
            )
        ],
    )
    assert [code for code, _ in answers] == [200], answers
    assert _baseline_id(context) == "inv-2"
    incident = _identity_incident(context, "inv-2")
    assert incident is not None, "a GPU lost across a reboot was not diagnosed"
    assert "GPU-b" in " ".join(incident.reasons), incident.reasons


def test_inventory_from_a_boot_the_agent_has_not_reported_waits_for_it() -> None:
    """A reboot the Agent has not re-registered yet: its first inventory is
    refused, and the one after the Agent's next heartbeat is the baseline."""

    context = build_context()
    _agent(context, "boot-a")
    base = _base()
    _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a", "GPU-b"], boot="boot-a", observed_at=base),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                boot="boot-b",
                observed_at=base + timedelta(minutes=1),
            ),
        ],
    )
    assert _baseline_id(context) == "inv-1"
    _agent(context, "boot-b")
    _post_all(
        context,
        [
            _snapshot(
                "inv-3",
                ["GPU-a", "GPU-b"],
                boot="boot-b",
                observed_at=base + timedelta(minutes=2),
            )
        ],
    )
    assert _baseline_id(context) == "inv-3"
    status = _inventory_status(context)
    assert not is_erroring(status), status
    for snapshot_id in ("inv-2", "inv-3"):
        assert _identity_incident(context, snapshot_id) is None, snapshot_id


@pytest.mark.parametrize(
    "agent",
    [
        "no_agent",
        "agent_without_boot_id",
        "lapsed_lease",
        "no_lease_recorded",
        "draining",
    ],
)
def test_boot_change_is_ordered_by_time_when_the_current_boot_is_unknown(
    agent: str,
) -> None:
    """Without an authoritative current boot the product cannot call another
    boot old, so the timestamp decides and the change is graded as before."""

    context = build_context()
    if agent == "agent_without_boot_id":
        _agent(context, None)
    elif agent == "lapsed_lease":
        _agent(context, "boot-a", lease=timedelta(minutes=-1))
    elif agent == "no_lease_recorded":
        _agent(context, "boot-a", lease=None)
    elif agent == "draining":
        _agent(context, "boot-a", lifecycle=AgentLifecycleState.DRAINING)
    base = _base()
    answers = _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a", "GPU-b"], boot="boot-a", observed_at=base),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-stale"],
                boot="boot-old",
                observed_at=base + timedelta(minutes=1),
            ),
        ],
    )
    assert [code for code, _ in answers] == [200, 200], answers
    assert _baseline_id(context) == "inv-2"
    assert _identity_incident(context, "inv-2") is not None
    status = _inventory_status(context)
    assert not is_erroring(status), status


def test_only_a_boot_change_is_judged_against_the_agent() -> None:
    """The first baseline, and a newer snapshot of the baseline's own boot,
    keep observed_at ordering even while the Agent reports another boot."""

    context = build_context()
    _agent(context, "boot-new")
    base = _base()
    _post_all(
        context,
        [
            _snapshot("inv-1", ["GPU-a", "GPU-b"], boot="boot-old", observed_at=base),
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                boot="boot-old",
                observed_at=base + timedelta(minutes=1),
            ),
        ],
    )
    assert _baseline_id(context) == "inv-2"
    status = _inventory_status(context)
    assert not is_erroring(status), status


def test_a_batch_judges_each_snapshot_against_the_baseline_it_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One replayed batch: the reboot snapshot moves the baseline to the new
    boot, so the old-boot snapshot behind it is a boot change and is refused
    -- not compared with the pre-batch baseline it happens to share a boot
    with."""

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-inventory-boot-baseline")
    context = ApplicationContext(execution_token=EXECUTION_TOKEN)
    app = create_app(context)
    _agent(context, "boot-new")
    base = _base()
    first = _post_batch(
        app, [_snapshot("inv-1", ["GPU-a", "GPU-b"], boot="boot-old", observed_at=base)]
    )
    assert [item["status"] for item in first] == [200], first
    results = _post_batch(
        app,
        [
            _snapshot(
                "inv-2",
                ["GPU-a", "GPU-b"],
                boot="boot-new",
                observed_at=base + timedelta(minutes=5),
            ),
            _snapshot(
                "inv-3",
                ["GPU-a", "GPU-stale"],
                boot="boot-old",
                observed_at=base + timedelta(minutes=6),
            ),
        ],
    )
    assert [item["status"] for item in results] == [200, 200], results
    assert {item["request_id"]: item["body"]["snapshot_id"] for item in results} == {
        "inv-2": "inv-2",
        "inv-3": "inv-2",
    }, results
    assert _baseline_id(context) == "inv-2"
    assert _identity_incident(context, "inv-3") is None
