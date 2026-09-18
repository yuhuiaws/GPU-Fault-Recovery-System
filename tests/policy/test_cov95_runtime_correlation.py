from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.policy import (
    ActionDisposition,
    DynamicRecoveryAction,
    GpuFaultPolicyEngine,
    XidCorrelationStatus,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.xid_correlation import XidCorrelationCoordinator
from tests.metrics.test_xid_correlation import NOW, xid


def correlation():
    state = SimpleNamespace(
        store=InMemoryStore(), clock=NOW, failures=set(), finalized=[]
    )
    policy = GpuFaultPolicyEngine()

    def finalize(event, decision):
        if event.event_id in state.failures:
            raise RuntimeError("unit finalization storage failure")
        state.finalized.append(event.event_id)
        return decision.model_copy(update={"incident_id": f"inc-{event.event_id}"})

    state.coordinator = XidCorrelationCoordinator(
        state.store, policy, finalize, owner="unit-correlator", now=lambda: state.clock
    )
    state.window = policy.policy.companion_window_seconds
    return state


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("poll_interval_seconds", 0, "poll interval"),
        ("lease_seconds", 4, "at least five"),
        ("batch_size", 0, "batch size"),
        ("retained_windows", 1, "at least two"),
    ],
)
def test_invalid_correlation_windows_are_rejected_before_store_activity(
    field, value, message
) -> None:
    store = InMemoryStore()
    with pytest.raises(ValueError, match=message):
        XidCorrelationCoordinator(
            store,
            GpuFaultPolicyEngine(),
            lambda _event, decision: decision,
            **{field: value},
        )
    assert store.list_xid_events("cluster-a", "node-a") == [], (
        "invalid configuration must not create pending correlation state"
    )


def test_missing_event_load_does_not_finalize_and_can_retry_after_lease_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = correlation()
    event = xid(45, "missing-read")
    pending = state.coordinator.ingest(event)
    assert pending.disposition is ActionDisposition.PENDING_CORRELATION, pending
    state.clock += timedelta(seconds=state.window + 1)
    original = state.store.get_xid_events
    monkeypatch.setattr(state.store, "get_xid_events", lambda _ids: {})
    assert state.coordinator.run_once() == 0, "a missing payload cannot be finalized"
    record = state.store.get_xid_correlation(event.event_id)
    assert (
        record.status is XidCorrelationStatus.PENDING
        and record.lease_owner == "unit-correlator"
    ), record
    assert state.finalized == [], state.finalized
    monkeypatch.setattr(state.store, "get_xid_events", original)
    state.clock += timedelta(seconds=31)
    assert state.coordinator.run_once() == 1, (
        "the released read must be retried after its lease"
    )
    assert state.finalized == [event.event_id], state.finalized


def test_failed_finalizer_does_not_block_the_rest_of_the_batch_or_lose_retry() -> None:
    state = correlation()
    bad = xid(45, "a-failed")
    good = xid(45, "b-good").model_copy(update={"node_id": "node-b"})
    for event in (bad, good):
        assert (
            state.coordinator.ingest(event).disposition
            is ActionDisposition.PENDING_CORRELATION
        ), event
    state.failures.add(bad.event_id)
    state.clock += timedelta(seconds=state.window + 1)
    assert state.coordinator.run_once() == 1, (
        "one finalizer failure must not abandon its sibling"
    )
    assert (
        state.store.get_xid_correlation(bad.event_id).status
        is XidCorrelationStatus.PENDING
    ), bad
    assert (
        state.store.get_xid_correlation(good.event_id).status
        is XidCorrelationStatus.FINALIZED
    ), good
    state.failures.clear()
    state.clock += timedelta(seconds=31)
    assert state.coordinator.run_once() == 1, (
        "the failed finalization must remain retryable"
    )
    assert state.finalized == [good.event_id, bad.event_id], state.finalized


def test_redelivery_after_raw_event_retention_reuses_the_final_decision() -> None:
    state = correlation()
    event = xid(45, "old-finalized")
    state.coordinator.ingest(event)
    state.clock += timedelta(seconds=state.window + 1)
    assert state.coordinator.run_once() == 1, "the initial correlation must close"
    decided = state.store.get_xid_policy_decision(event.event_id)
    newer = xid(
        43, "newer-event", observed_at=NOW + timedelta(seconds=state.window * 9)
    )
    state.coordinator.ingest(newer)
    assert state.store.get_xid_events([event.event_id]) == {}, (
        "the scenario must actually remove the old raw event while retaining its decision"
    )
    replay = state.coordinator.ingest(event)
    assert replay.duplicate is True and replay.incident_id == decided.incident_id, (
        replay
    )
    assert replay.disposition is decided.disposition, replay
    assert state.finalized == [event.event_id], (
        "retention must not open another incident"
    )


def test_already_derived_xid154_action_is_preserved_only_when_the_log_agrees() -> None:
    event = xid(154, "derived").model_copy(
        update={
            "raw_message": "NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU reset required)",
            "xid_154_action": DynamicRecoveryAction.RESET_GPU,
        }
    )
    prepared = XidCorrelationCoordinator.prepare_xid154(event)
    assert prepared is event, (
        "an already-derived matching action needs no record rewrite"
    )
    assert prepared.xid_154_action is DynamicRecoveryAction.RESET_GPU, prepared
    contradictory = event.model_copy(
        update={"xid_154_action": DynamicRecoveryAction.IGNORE}
    )
    corrected = XidCorrelationCoordinator.prepare_xid154(contradictory)
    assert corrected.xid_154_action is DynamicRecoveryAction.RESET_GPU, corrected
