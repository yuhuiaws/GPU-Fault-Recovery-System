"""Priority tiers: control-plane actions rank before the device events they end.

FINAL-建议汇总 F-D1 (P0-77A / P0-78B). Tier 0 used to hold both a raw XID from
a node and the ``/v1/workflows/...`` request that ends the storm the XID is part
of; inside one tier the claim window is FIFO by ``created_at``, so during a storm
the control-plane request never entered the window. The tier is split: control
plane actions stay at 0, device events move to 10, and every place that used to
ask ``priority == 0`` to mean "fault, keep in the reserve" now asks the named
predicate ``is_reserved_tier`` so a device event is still admitted inside the
reserved depth and still routed to the fault pool.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.processor.coordinator import ProcessorCoordinator
from gpu_fault.processor.models import (
    CONTROL_PLANE_ACTION_PRIORITY,
    DEVICE_EVENT_PRIORITY,
    EVIDENCE_PRIORITY,
    ROUTINE_PRIORITY,
    is_reserved_tier,
)
from gpu_fault.store import SqliteStore
from tests._builders import build_store, processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

CONTROL_PLANE_PATHS = [
    "/v1/incidents/inc-1/acknowledge",
    "/v1/workflows/wf-1/steps/s1/complete",
    "/v1/attempts/attempt-1/hang-check",
    "/v1/recovery-plans/plan-1/approve",
]
DEVICE_EVENT_PATHS = [
    "/v1/gpu-events/xid",
    "/v1/gpu-events/nvidia-kernel",
    "/v1/provider-events/hyperpod-hma/health",
    "/v1/collector-events/nvidia-kernel",
    "/v1/collector-events/fabric-manager",
]


@pytest.mark.parametrize("path", CONTROL_PLANE_PATHS)
def test_control_plane_actions_are_tier_zero(path: str) -> None:
    request = processor_request(path, body=b'{"node_id":"node-a"}')

    assert request.queue_priority() == CONTROL_PLANE_ACTION_PRIORITY == 0


@pytest.mark.parametrize("path", DEVICE_EVENT_PATHS)
def test_device_events_rank_after_control_plane_actions(path: str) -> None:
    request = processor_request(path, body=b'{"node_id":"node-a"}')

    assert request.queue_priority() == DEVICE_EVENT_PRIORITY
    assert CONTROL_PLANE_ACTION_PRIORITY < DEVICE_EVENT_PRIORITY < EVIDENCE_PRIORITY
    assert request.spoolable() is False


def test_reserved_tier_covers_both_fault_tiers_and_nothing_else() -> None:
    assert is_reserved_tier(CONTROL_PLANE_ACTION_PRIORITY), (
        "expected is_reserved_tier(CONTROL_PLANE_ACTION_PRIORITY) to be true"
    )
    assert is_reserved_tier(DEVICE_EVENT_PRIORITY), (
        "expected is_reserved_tier(DEVICE_EVENT_PRIORITY) to be true"
    )
    assert not is_reserved_tier(EVIDENCE_PRIORITY), (
        "expected is_reserved_tier(EVIDENCE_PRIORITY) to be false"
    )
    assert not is_reserved_tier(ROUTINE_PRIORITY), (
        "expected is_reserved_tier(ROUTINE_PRIORITY) to be false"
    )
    assert not is_reserved_tier(49), "expected is_reserved_tier(49) to be false"


@pytest.mark.parametrize("path", CONTROL_PLANE_PATHS + DEVICE_EVENT_PATHS)
def test_requests_in_both_fault_tiers_are_reserved(path: str) -> None:
    assert processor_request(path, body=b'{"node_id":"node-a"}').is_reserved_tier(), (
        'expected processor_request(path, body=b\'{"node_id":"node-a"}\').is_reserved_tier() to be true'
    )


def test_evidence_and_routine_requests_are_not_reserved() -> None:
    evidence = processor_request("/v1/workload-observations", body=b"{}")
    routine = processor_request(
        "/v1/collector-events/host-telemetry",
        body=b'{"node_id":"node-a","summary":true}',
    )

    assert evidence.queue_priority() == EVIDENCE_PRIORITY
    assert routine.queue_priority() == ROUTINE_PRIORITY
    assert not evidence.is_reserved_tier(), (
        "expected evidence.is_reserved_tier() to be false"
    )
    assert not routine.is_reserved_tier(), (
        "expected routine.is_reserved_tier() to be false"
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def processor_store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        store = SqliteStore(str(tmp_path / "tiers.db"))
        try:
            yield store
        finally:
            store.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    # Leave the shared database as we found it: the store-contract tests that
    # run after this file claim whatever is pending.
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _routine(node: str, cluster_id: str = "cluster-tiers"):
    return processor_request(
        "/v1/collector-events/host-telemetry",
        body=('{"node_id":"' + node + '","summary":true}').encode(),
        cluster_id=cluster_id,
    )


def _fill_to_the_reserve(processor_store, *, limits: dict[str, int]) -> None:
    for index in range(limits["max_depth"] - limits["reserved_fault_depth"]):
        accepted, reason = processor_store.try_enqueue_processor_request(
            _routine(f"node-{index}"), **limits
        )
        assert accepted is not None, reason
    accepted, reason = processor_store.try_enqueue_processor_request(
        _routine("node-one-too-many"), **limits
    )
    assert accepted is None
    assert reason == "global_reserved"


def test_device_events_are_admitted_inside_the_fault_reserve(processor_store) -> None:
    """The reserve is for faults, and a device event is a fault.

    Splitting the tier without changing the admission predicate would 429 the
    storm's own events once routine traffic reached the reserve line -- the
    regression the split must not introduce.
    """

    limits = {
        "max_depth": 10,
        "max_cluster_depth": 10,
        "reserved_fault_depth": 4,
        "reserved_cluster_fault_depth": 4,
    }
    _fill_to_the_reserve(processor_store, limits=limits)

    device_event = processor_request(
        "/v1/gpu-events/xid",
        body=b'{"node_id":"node-storm","xid":79}',
        cluster_id="cluster-tiers",
    )
    accepted, reason = processor_store.try_enqueue_processor_request(
        device_event, **limits
    )

    assert reason is None
    assert accepted is not None
    assert accepted.queue_priority() == DEVICE_EVENT_PRIORITY


def test_control_plane_actions_are_admitted_inside_the_fault_reserve(
    processor_store,
) -> None:
    limits = {
        "max_depth": 10,
        "max_cluster_depth": 10,
        "reserved_fault_depth": 4,
        "reserved_cluster_fault_depth": 4,
    }
    _fill_to_the_reserve(processor_store, limits=limits)

    accepted, reason = processor_store.try_enqueue_processor_request(
        processor_request(
            "/v1/workflows/wf-1/steps/s1/complete",
            body=b'{"node_id":"node-storm"}',
            cluster_id="cluster-tiers",
        ),
        **limits,
    )

    assert reason is None
    assert accepted is not None


def test_fault_backlog_depth_counts_both_reserved_tiers(processor_store) -> None:
    limits = {"max_depth": 10, "max_cluster_depth": 10}
    for path in ("/v1/gpu-events/xid", "/v1/workflows/wf-1/steps/s1/complete"):
        accepted, reason = processor_store.try_enqueue_processor_request(
            processor_request(
                path, body=b'{"node_id":"node-a"}', cluster_id="cluster-tiers"
            ),
            **limits,
        )
        assert accepted is not None, reason
    accepted, reason = processor_store.try_enqueue_processor_request(
        _routine("node-b"), **limits
    )
    assert accepted is not None, reason

    assert processor_store.processor_fault_backlog_depth() == 2


def test_device_events_are_routed_to_the_fault_pool() -> None:
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-tier-token",
        active_consumers=True,
    )
    device_event = processor_request("/v1/gpu-events/xid", body=b'{"node_id":"n"}')
    control = processor_request(
        "/v1/workflows/wf-1/steps/s1/complete", body=b'{"node_id":"n"}'
    )

    assert processor._pool_for_request(device_event) == "fault"
    assert processor._pool_for_request(control) == "fault"
