from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

from gpu_fault.app import create_app
from gpu_fault.models import RecoveryAction
from gpu_fault.policy import (
    ActionDisposition,
    GpuFaultPolicyEngine,
    XidEvent,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.xid_correlation import XidCorrelationCoordinator


NOW = datetime(2026, 8, 13, 18, tzinfo=timezone.utc)


def endpoint_probe() -> int:
    source = inspect.getsource(create_app)
    kernel = source.split("def ingest_nvidia_kernel", 1)[1].split(
        "def ingest_fabric_manager_log", 1
    )[0]
    fabric = source.split("def ingest_fabric_manager_log", 1)[1].split(
        "def ingest_gpu_inventory", 1
    )[0]
    assert "ctx.dispatcher.wake()" in kernel
    assert "ctx.dispatcher.wake()" in fabric
    return 2


def correlation_probe() -> str:
    store = InMemoryStore()
    policy = GpuFaultPolicyEngine()
    current = [NOW]
    finalized = []
    coordinator = XidCorrelationCoordinator(
        store,
        policy,
        lambda _event, decision: (finalized.append(decision) or decision),
        owner="audit-correlator",
        now=lambda: current[0],
    )
    event = XidEvent(
        event_id="audit-pending-xid45",
        cluster_id="audit-cluster",
        node_id="audit-node",
        observed_at=NOW,
        xid=45,
        gpu_uuid="GPU-a",
        pci_bdf="0000:59:00.0",
        product="H200",
        driver_branch=575,
        cuda_version="12.9",
    )
    pending = coordinator.ingest(event)
    assert pending.disposition is ActionDisposition.PENDING_CORRELATION
    current[0] += timedelta(seconds=policy.policy.companion_window_seconds + 1)
    policy.evaluate_xid = lambda *_args, **_kwargs: pending
    assert coordinator.run_once() == 1
    decision = store.get_xid_policy_decision(event.event_id)
    assert decision is not None
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert len(finalized) == 1
    return decision.disposition.value


def main() -> None:
    accepted_events = endpoint_probe()
    disposition = correlation_probe()
    print(
        "PASS",
        {
            "dispatcher_wakes": accepted_events,
            "closed_pending_disposition": disposition,
            "safety_action": RecoveryAction.QUARANTINE.value,
        },
    )


if __name__ == "__main__":
    main()
