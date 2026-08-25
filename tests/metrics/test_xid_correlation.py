from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.models import RecoveryAction, WorkflowOperation
from gpu_fault.policy import (
    ActionDisposition,
    GpuFaultPolicyEngine,
    XidCorrelationStatus,
    XidEvent,
)
from gpu_fault.store import SqliteStore
from gpu_fault.xid_correlation import XidCorrelationCoordinator
from tests._builders import asgi_client, build_store, copy_model

NOW = datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc)


def xid(
    code: int,
    event_id: str,
    *,
    gpu_uuid: str | None = "GPU-a",
    pci_bdf: str | None = "0000:01:00.0",
    observed_at: datetime = NOW,
) -> XidEvent:
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=observed_at,
        xid=code,
        gpu_uuid=gpu_uuid,
        pci_bdf=pci_bdf,
        product="H200",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
    )


async def post_xid(app, event: XidEvent) -> dict:
    async with asgi_client(app) as client:
        response = await client.post(
            "/v1/gpu-events/xid", json=event.model_dump(mode="json")
        )
        assert response.status_code == 200, response.text
        return response.json()


def test_xid45_waits_and_follows_later_companion_across_replicas():
    store = build_store()
    first = ApplicationContext(store=store)
    second = ApplicationContext(store=store)
    app_a = create_app(first)
    app_b = create_app(second)
    current = [NOW]
    first.xid_correlation.now = lambda: current[0]
    second.xid_correlation.now = lambda: current[0]

    pending = asyncio.run(post_xid(app_a, xid(45, "xid-45")))
    companion = asyncio.run(
        post_xid(app_b, xid(79, "xid-79", observed_at=NOW + timedelta(seconds=5)))
    )

    assert pending["disposition"] == "PENDING_CORRELATION"
    assert pending["incident_id"] is None
    assert companion["action"] == "REBOOT_NODE"
    current[0] += timedelta(seconds=31)
    assert second.xid_correlation.run_once() == 1

    finalized = store.get_xid_policy_decision("xid-45")
    correlation = store.get_xid_correlation("xid-45")
    assert finalized.action is RecoveryAction.REBOOT_NODE
    assert finalized.correlated_event_id == "xid-79"
    assert finalized.incident_id == companion["incident_id"]
    assert correlation.status is XidCorrelationStatus.FINALIZED
    assert len(store.list_workflows(limit=100)) == 1


def test_closed_pending_correlation_finalizes_fail_closed() -> None:
    store = build_store()
    policy = GpuFaultPolicyEngine()
    current = [NOW]
    finalized = []
    coordinator = XidCorrelationCoordinator(
        store,
        policy,
        lambda _event, decision: (finalized.append(decision) or decision),
        owner="correlator-a",
        now=lambda: current[0],
    )
    event = xid(45, "closed-still-pending")
    pending = coordinator.ingest(event)
    assert pending.disposition is ActionDisposition.PENDING_CORRELATION
    current[0] += timedelta(seconds=policy.policy.companion_window_seconds + 1)
    policy.evaluate_xid = lambda *_args, **_kwargs: pending

    assert coordinator.run_once() == 1

    decision = store.get_xid_policy_decision(event.event_id)
    assert decision is not None
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert finalized == [decision]
    assert (
        store.get_xid_correlation(event.event_id).status
        is XidCorrelationStatus.FINALIZED
    )


def test_xid48_correlates_63_across_replicas() -> None:
    """A second replica must see the XID 63 the first one persisted.

    The engine keeps no event history, so ingesting XID 48 on a fresh
    replica has to reach the shared store for candidates; otherwise
    every pod restart silently downgrades DRAIN_AND_RESET to a bare
    RESET_GPU with no cordon.
    """
    store = build_store()
    replica_a = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        owner="replica-a",
        now=lambda: NOW,
    )
    replica_b = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        owner="replica-b",
        now=lambda: NOW,
    )

    replica_a.ingest(xid(63, "xid-63"))
    decision = replica_b.ingest(
        xid(48, "xid-48", observed_at=NOW + timedelta(seconds=1))
    )

    assert decision.official_action == "DRAIN_AND_RESET"
    assert decision.action is RecoveryAction.RESET_GPU
    assert decision.pre_actions == [RecoveryAction.MARK_UNSCHEDULABLE]


def test_xid48_shares_one_reset_with_xid64_but_never_with_xid63() -> None:
    """Dedupe the double reset without ever dropping the reset.

    XID 64 is itself RESET_GPU, so XID 48 landing after it used to
    compile a second eight-step reset of the same GPU. Merging the
    incidents fixes that -- but the merge in ``finalize_xid`` makes
    XID 48 adopt the companion's existing workflow, so merging into
    XID 63 (IGNORE, no workflow at all) would leave the double-bit
    error completely unremediated. Assert both directions here: shared
    with 64, independent from 63, and a reset present either way.
    """
    for companion_xid, expect_shared in ((64, True), (63, False)):
        store = build_store()
        context = ApplicationContext(store=store)
        app = create_app(context)
        context.xid_correlation.now = lambda: NOW

        first = asyncio.run(post_xid(app, xid(companion_xid, f"xid-{companion_xid}")))
        second = asyncio.run(
            post_xid(app, xid(48, "xid-48", observed_at=NOW + timedelta(seconds=6)))
        )

        assert second["official_action"] == "DRAIN_AND_RESET"
        workflows = store.list_workflows(limit=50)
        reset_bearing = [
            workflow
            for workflow in workflows
            if any(
                step.operation is WorkflowOperation.RESET_GPU
                for step in workflow.official_steps
            )
        ]
        # The GPU must always be reset exactly once, never zero times.
        assert len(reset_bearing) == 1
        if expect_shared:
            assert second["correlated_event_id"] == (f"xid-{companion_xid}")
            assert second["incident_id"] == first["incident_id"]
            assert len(workflows) == 1
        else:
            assert second["correlated_event_id"] is None
            assert second["incident_id"] != first["incident_id"]
            assert first["workflow_request_id"] is None
            assert second["workflow_request_id"]


def test_xid48_and_xid64_reset_once_in_either_order() -> None:
    """Arrival order must not decide how many times a GPU is reset.

    ``_workflow_xid_48`` only merges into a companion already in the
    store, so ``48 -> 64`` used to compile two independent eight-step
    resets of the same card. XID 64 now backlinks the earlier XID 48
    instead, which dedupes without holding the uncorrectable error
    open for the 30s companion window.
    """
    for order in (((48, "a-48"), (64, "b-64")), ((64, "a-64"), (48, "b-48"))):
        store = build_store()
        context = ApplicationContext(store=store)
        app = create_app(context)
        context.xid_correlation.now = lambda: NOW

        responses = [
            asyncio.run(
                post_xid(
                    app,
                    xid(code, event_id, observed_at=NOW + timedelta(seconds=index * 6)),
                )
            )
            for index, (code, event_id) in enumerate(order)
        ]

        workflows = store.list_workflows(limit=50)
        reset_bearing = [
            workflow
            for workflow in workflows
            if any(
                step.operation is WorkflowOperation.RESET_GPU
                for step in workflow.official_steps
            )
        ]
        assert len(reset_bearing) == 1, order
        assert len(workflows) == 1, order
        # Both events land on the one incident that owns that reset.
        assert responses[1]["incident_id"] == responses[0]["incident_id"], order
        assert responses[1]["correlated_event_id"] == order[0][1], order


def test_xid45_companion_choice_ignores_arrival_order() -> None:
    """The verdict must not depend on which companion logged last.

    XID 94/95 both say XID 45 "will be seen in relation to this
    error", so a benign XID 63 arriving later must not shadow the
    XID 95 that requires a reset.
    """
    severe = xid(95, "xid-95", observed_at=NOW + timedelta(seconds=1))
    benign = xid(63, "xid-63", observed_at=NOW + timedelta(seconds=2))
    subject = xid(45, "xid-45", observed_at=NOW)
    engine = GpuFaultPolicyEngine()

    severe_last = engine.evaluate_xid(
        subject, companion_events=[benign, severe, subject]
    )
    benign_last = GpuFaultPolicyEngine().evaluate_xid(
        copy_model(subject, event_id="xid-45-b"),
        companion_events=[severe, benign, subject],
    )

    assert severe_last.action is RecoveryAction.RESET_GPU
    assert benign_last.action is RecoveryAction.RESET_GPU
    assert severe_last.correlated_event_id == "xid-95"
    assert benign_last.correlated_event_id == "xid-95"


def test_policy_engine_decision_cache_is_bounded() -> None:
    """A long-lived replica must not grow one entry per event forever."""
    engine = GpuFaultPolicyEngine(decision_cache_size=8)

    for index in range(64):
        engine.evaluate_xid(
            xid(63, f"bounded-{index}", observed_at=NOW + timedelta(seconds=index))
        )

    assert len(engine._decisions) == 8
    assert engine._cached("bounded-63") is not None
    assert engine._cached("bounded-0") is None


def test_persisted_xid_events_are_pruned_past_the_retention_window():
    """The correlation table must not grow for the life of the cluster.

    Only events that can still be companions are worth keeping, so an
    ingest drops the ones that aged out of the retained windows.
    """
    store = SqliteStore(":memory:")
    coordinator = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        owner="replica-a",
        retained_windows=2,
        now=lambda: NOW,
    )
    window = coordinator.policy.policy.companion_window_seconds

    coordinator.ingest(xid(63, "ancient", observed_at=NOW))
    coordinator.ingest(xid(63, "recent", observed_at=NOW + timedelta(seconds=window)))
    coordinator.ingest(
        xid(63, "newest", observed_at=NOW + timedelta(seconds=window * 4))
    )

    surviving = {item.event_id for item in store.list_xid_events("cluster-a", "node-a")}
    assert surviving == {"newest"}


def test_non_utc_producer_still_lands_in_its_own_window() -> None:
    """A +08:00 event must correlate, not sort outside the window.

    Persisted windows are compared lexically on the serialized
    timestamp, so the model normalizes to UTC at the boundary.
    """
    store = SqliteStore(":memory:")
    shifted = copy_model(
        xid(63, "shifted"), observed_at=NOW.astimezone(timezone(timedelta(hours=8)))
    )
    coordinator = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        owner="replica-a",
        now=lambda: NOW,
    )

    coordinator.ingest(shifted)
    decision = coordinator.ingest(
        xid(48, "xid-48", observed_at=NOW + timedelta(seconds=1))
    )

    stored = store.get_xid_event("shifted")
    assert stored.observed_at.utcoffset() == timedelta(0)
    assert stored.observed_at == NOW
    assert decision.official_action == "DRAIN_AND_RESET"


def test_xid45_solo_finalizes_only_after_deadline() -> None:
    store = build_store()
    decisions = []
    current = [NOW]
    coordinator = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decisions.append(decision) or decision,
        owner="replica-a",
        now=lambda: current[0],
    )

    pending = coordinator.ingest(xid(45, "solo"))
    assert pending.disposition is ActionDisposition.PENDING_CORRELATION
    assert coordinator.run_once() == 0

    current[0] += timedelta(seconds=30)
    assert coordinator.run_once() == 1
    assert decisions[0].official_action == "RESTART_FM"
    assert decisions[0].disposition is ActionDisposition.EXECUTABLE


def test_xid45_uses_pci_bdf_fallback_but_not_unknown_gpu() -> None:
    engine = GpuFaultPolicyEngine()
    primary = xid(79, "primary", gpu_uuid=None, pci_bdf="0000:01:00.0")
    matched = engine.evaluate_xid(
        xid(45, "matched", gpu_uuid=None, pci_bdf="0000:01:00.0"),
        companion_events=[primary],
    )
    unknown = engine.evaluate_xid(
        xid(45, "unknown", gpu_uuid=None, pci_bdf=None),
        companion_events=[xid(79, "also-unknown", gpu_uuid=None, pci_bdf=None)],
    )

    assert matched.correlated_event_id == "primary"
    assert matched.action is RecoveryAction.REBOOT_NODE
    assert unknown.correlated_event_id is None
    assert unknown.official_action == "RESTART_FM"


def test_xid45_does_not_correlate_across_boots() -> None:
    engine = GpuFaultPolicyEngine()
    event = copy_model(xid(45, "after-reboot"), source_boot_id="boot-new")
    old = copy_model(xid(79, "before-reboot"), source_boot_id="boot-old")

    decision = engine.evaluate_xid(event, companion_events=[old])

    assert decision.correlated_event_id is None
    assert decision.official_action == "RESTART_FM"


def test_pending_xid45_survives_sqlite_reopen(tmp_path) -> None:
    path = tmp_path / "control-plane.db"
    current = [NOW]
    first = SqliteStore(str(path))
    coordinator = XidCorrelationCoordinator(
        first,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        owner="replica-a",
        now=lambda: current[0],
    )
    coordinator.ingest(xid(45, "restart-safe"))
    first.close()

    second = SqliteStore(str(path))
    finalized = []
    resumed = XidCorrelationCoordinator(
        second,
        GpuFaultPolicyEngine(),
        lambda _event, decision: finalized.append(decision) or decision,
        owner="replica-b",
        now=lambda: current[0],
    )
    current[0] += timedelta(seconds=31)

    assert resumed.run_once() == 1
    assert finalized[0].official_action == "RESTART_FM"
    assert (
        second.get_xid_correlation("restart-safe").status
        is XidCorrelationStatus.FINALIZED
    )
    second.close()


def test_only_one_replica_claims_due_xid45() -> None:
    store = build_store()
    current = [NOW]
    calls = []
    first = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: calls.append("a") or decision,
        owner="replica-a",
        now=lambda: current[0],
    )
    second = XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: calls.append("b") or decision,
        owner="replica-b",
        now=lambda: current[0],
    )
    first.ingest(xid(45, "leased"))
    current[0] += timedelta(seconds=31)

    assert first.run_once() == 1
    assert second.run_once() == 0
    assert calls == ["a"]


def test_xid154_is_parsed_and_applied_to_pending_nvlink5_event() -> None:
    store = build_store()
    first = ApplicationContext(store=store)
    second = ApplicationContext(store=store)
    app_a = create_app(first)
    app_b = create_app(second)
    current = [NOW]
    first.xid_correlation.now = lambda: current[0]
    second.xid_correlation.now = lambda: current[0]
    primary = copy_model(
        xid(145, "xid-145"),
        product="B200",
        driver_branch=575,
        intr_info=4,
        error_status=1,
    )
    summary = copy_model(
        xid(154, "xid-154", observed_at=NOW + timedelta(seconds=5)),
        product="B200",
        raw_message="NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)",
    )

    pending_primary = asyncio.run(post_xid(app_a, primary))
    pending_summary = asyncio.run(post_xid(app_b, summary))

    assert pending_primary["disposition"] == "PENDING_CORRELATION"
    assert pending_summary["disposition"] == "PENDING_CORRELATION"
    current[0] += timedelta(seconds=36)
    assert second.xid_correlation.run_once() == 2

    primary_decision = store.get_xid_policy_decision("xid-145")
    summary_decision = store.get_xid_policy_decision("xid-154")
    assert primary_decision.action is RecoveryAction.REBOOT_NODE
    assert primary_decision.source.value == "NVIDIA_XID_154"
    assert summary_decision.correlated_event_id == "xid-145"
    assert summary_decision.incident_id == primary_decision.incident_id
    assert len(store.list_workflows(limit=100)) == 1


def xid74(
    event_id: str, *, link_id: int, register_index: int = 0, bit: int = 4
) -> XidEvent:
    registers = [0] * 7
    registers[register_index] = 1 << bit
    return copy_model(
        xid(74, event_id),
        registers=registers,
        nvlink_link_id=link_id,
        nvlink_link_identity_source="test-explicit",
    )


def test_xid74_ecc_reset_starts_after_third_same_link_event() -> None:
    store = build_store()
    coordinator = XidCorrelationCoordinator(
        store, GpuFaultPolicyEngine(), lambda _event, decision: decision
    )

    decisions = [
        coordinator.ingest(xid74(f"ecc-{index}", link_id=3)) for index in range(1, 4)
    ]

    assert [item.action for item in decisions[:2]] == [
        RecoveryAction.NO_ACTION,
        RecoveryAction.NO_ACTION,
    ]
    assert decisions[2].action is RecoveryAction.RESET_GPU
    assert decisions[2].nvlink_occurrence_counts == {"register1.bit4": 3}


def test_xid74_counts_are_isolated_by_link_and_register_bit() -> None:
    store = build_store()
    coordinator = XidCorrelationCoordinator(
        store, GpuFaultPolicyEngine(), lambda _event, decision: decision
    )

    coordinator.ingest(xid74("link3-bit4", link_id=3))
    other_link = coordinator.ingest(xid74("link4-bit4", link_id=4))
    other_bit = coordinator.ingest(xid74("link3-bit5", link_id=3, bit=5))

    assert other_link.nvlink_occurrence_counts == {"register1.bit4": 1}
    assert other_bit.nvlink_occurrence_counts == {"register1.bit5": 1}


def test_xid74_repeated_bit_resets_on_second_same_link_event() -> None:
    store = build_store()
    coordinator = XidCorrelationCoordinator(
        store, GpuFaultPolicyEngine(), lambda _event, decision: decision
    )

    first = coordinator.ingest(xid74("repeat-1", link_id=3, bit=27))
    second = coordinator.ingest(xid74("repeat-2", link_id=3, bit=27))

    assert first.action is RecoveryAction.NO_ACTION
    assert second.action is RecoveryAction.RESET_GPU
    assert second.nvlink_occurrence_counts == {"register1.bit27": 2}


def test_xid74_duplicate_event_does_not_increment_counter() -> None:
    store = build_store()
    coordinator = XidCorrelationCoordinator(
        store, GpuFaultPolicyEngine(), lambda _event, decision: decision
    )
    event = xid74("same-event", link_id=2)

    first = coordinator.ingest(event)
    duplicate = coordinator.ingest(event)
    next_event = coordinator.ingest(xid74("next-event", link_id=2))

    assert first.nvlink_occurrence_counts["register1.bit4"] == 1
    assert duplicate.duplicate
    assert next_event.nvlink_occurrence_counts["register1.bit4"] == 2


def test_xid74_sqlite_counter_is_durable_and_atomic(tmp_path) -> None:
    path = str(tmp_path / "xid74.db")
    first = SqliteStore(path)
    second = SqliteStore(path)
    coordinators = [
        XidCorrelationCoordinator(
            first, GpuFaultPolicyEngine(), lambda _event, decision: decision
        ),
        XidCorrelationCoordinator(
            second, GpuFaultPolicyEngine(), lambda _event, decision: decision
        ),
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_two = list(
                pool.map(
                    lambda args: args[0].ingest(args[1]),
                    [
                        (coordinators[0], xid74("sqlite-1", link_id=1)),
                        (coordinators[1], xid74("sqlite-2", link_id=1)),
                    ],
                )
            )
        third = coordinators[0].ingest(xid74("sqlite-3", link_id=1))

        assert sorted(
            item.nvlink_occurrence_counts["register1.bit4"] for item in first_two
        ) == [1, 2]
        assert third.nvlink_occurrence_counts["register1.bit4"] == 3
        assert third.action is RecoveryAction.RESET_GPU
    finally:
        first.close()
        second.close()
