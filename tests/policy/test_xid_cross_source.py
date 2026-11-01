"""The kernel-line / DCGM field-230 twin rule of ``policy/xid_cross_source``.

The replay through ingest and dispatch is
``tests/orchestration/test_xid_cross_source_dedupe.py``; these cases pin the
rule itself: what pairs, what is already taken, what waits, what follows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import RecoveryAction
from gpu_fault.policy import (
    ActionDisposition,
    DynamicRecoveryAction,
    GpuFaultPolicyEngine,
    XidEvent,
)
from gpu_fault.policy.xid_cross_source import (
    FIELD_230_EVIDENCE,
    MAX_PAIRING_CANDIDATES,
    XID_REPORT_TWIN_WINDOW,
    candidate_window,
    join_gauge_twin,
    paired_reports,
    report_twin,
    report_twins,
    report_wait_window,
)

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def kernel(event_id: str = "kernel", seconds: float = 0, **values) -> XidEvent:
    return XidEvent(
        **{
            "event_id": event_id,
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "observed_at": NOW + timedelta(seconds=seconds),
            "event_source": "KERNEL_LOG",
            "source_boot_id": "boot-a",
            "xid": 48,
            "gpu_uuid": "GPU-a",
            "pci_bdf": "0000:b9:00",
            "product": "NVIDIA H200",
            "driver_branch": 580,
            "cuda_version": "13.0",
            **values,
        }
    )


def gauge(event_id: str = "gauge", seconds: float = 20, **values) -> XidEvent:
    return kernel(
        event_id,
        seconds,
        **{
            "event_source": "DCGM_EXPORTER",
            "source_boot_id": None,
            "pci_bdf": "00000000:B9:00.0",
            **values,
        },
    )


def test_a_kernel_line_and_a_field_230_change_of_one_xid_on_one_gpu_are_twins():
    assert report_twins(kernel(), gauge()), "one XID, one GPU, 20 s apart"
    assert report_twins(gauge(), kernel()), "the relation is symmetric"
    assert report_twins(kernel(gpu_uuid=None), gauge()), (
        "a line without a UUID names the device by its PCI address"
    )
    assert report_twins(kernel(), gauge(seconds=XID_REPORT_TWIN_WINDOW.seconds)), (
        "the window includes its edge"
    )


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        (kernel(), kernel("other", 20), "same source"),
        (gauge(), gauge("other"), "same source"),
        (kernel(), gauge(xid=63), "another XID"),
        (kernel(), gauge(gpu_uuid="GPU-b"), "another GPU"),
        (kernel(gpu_uuid=None), gauge(pci_bdf="00000000:BA:00.0"), "another device"),
        (kernel(), gauge(node_id="node-b"), "another node"),
        (kernel(), gauge(cluster_id="cluster-b"), "another cluster"),
        (kernel(synthetic=True), gauge(), "a synthetic line"),
        (kernel(), gauge(source_boot_id="boot-b"), "another boot"),
        (kernel(), gauge(seconds=XID_REPORT_TWIN_WINDOW.seconds + 1), "too far apart"),
        (kernel(), gauge(event_source="FABRIC_MANAGER_LOG"), "a Fabric Manager line"),
    ],
)
def test_only_the_same_occurrence_pairs(left: XidEvent, right: XidEvent, why: str):
    assert not report_twins(left, right), why


def test_a_change_prefers_the_kernel_line_carrying_the_decode_then_the_nearest():
    plain = kernel("plain", 15)
    decoded = kernel("decoded", 0, xid=154).model_copy(
        update={"xid_154_action": DynamicRecoveryAction.DRAIN_P2P}
    )
    change = gauge(xid=154)

    assert report_twin(change, [plain.model_copy(update={"xid": 154}), decoded]) == (
        decoded
    )
    near, far = kernel("near", 18), kernel("far", 0)
    assert report_twin(gauge(), [far, near]) == near
    assert report_twin(kernel(), [gauge("late", 60), gauge("soon", 10)]).event_id == (
        "soon"
    )


def test_a_paired_report_is_taken_and_a_storm_pairs_nothing():
    line, change, again = kernel(), gauge(), kernel("again", 15)
    decisions = {
        change.event_id: GpuFaultPolicyEngine()
        .evaluate_xid(change, companion_events=[line, change])
        .model_copy(update={"correlated_event_id": line.event_id})
    }

    taken = paired_reports(again, [line, change, again], decisions.get)
    assert taken == frozenset({line.event_id, change.event_id})
    assert report_twin(again, [line, change], paired=taken) is None
    assert paired_reports(line, [line, change], decisions.get) == frozenset(), (
        "a pair with the event in it is its own"
    )
    storm = [
        kernel(f"line-{index}", index) for index in range(MAX_PAIRING_CANDIDATES + 1)
    ]
    assert paired_reports(change, [change, *storm], {}.get) is None


def test_a_change_takes_the_decode_of_its_kernel_line():
    engine = GpuFaultPolicyEngine()
    line = kernel(
        xid=154, raw_message="NVRM: Xid (PCI:0000:b9:00): 154, ..."
    ).model_copy(update={"xid_154_action": DynamicRecoveryAction.DRAIN_P2P})
    change = gauge(xid=154)

    decision = engine.evaluate_xid(change, companion_events=[line, change])

    assert decision.official_action == "DRAIN_P2P"
    assert decision.action is RecoveryAction.STOP_WORKLOAD
    assert decision.correlated_event_id == line.event_id
    assert decision.reasons[0].startswith(FIELD_230_EVIDENCE), decision.reasons


@pytest.mark.parametrize("xid", [154, 74, 145])
def test_a_change_whose_decode_reads_the_line_waits_for_it(xid: int):
    product = "NVIDIA B200" if xid == 145 else "NVIDIA H200"
    change = gauge(xid=xid, product=product)
    engine = GpuFaultPolicyEngine()

    waiting = engine.evaluate_xid(
        change, companion_events=[change], report_twin_window_closed=False
    )
    closed = GpuFaultPolicyEngine().evaluate_xid(change, companion_events=[change])

    assert waiting.disposition is ActionDisposition.PENDING_CORRELATION
    assert FIELD_230_EVIDENCE in waiting.reasons[0]
    assert closed.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert report_wait_window(engine, change) == XID_REPORT_TWIN_WINDOW
    assert report_wait_window(engine, change.model_copy(update={"xid": 48})) == (
        timedelta(seconds=30)
    )
    assert report_wait_window(
        engine, change.model_copy(update={"event_source": "KERNEL_LOG"})
    ) == timedelta(seconds=30)


def test_a_change_decides_alone_without_a_free_twin():
    engine = GpuFaultPolicyEngine()
    line, change = kernel(), gauge()

    taken = engine.evaluate_xid(
        change, companion_events=[line, change], paired_report_ids=frozenset({"kernel"})
    )
    storm = GpuFaultPolicyEngine().evaluate_xid(
        change, companion_events=[line, change], paired_report_ids=None
    )

    for decision in (taken, storm):
        assert decision.correlated_event_id is None
        assert decision.action is RecoveryAction.RESET_GPU


def test_a_kernel_line_joins_a_change_decided_the_same_way_first():
    engine = GpuFaultPolicyEngine()
    change = gauge(seconds=0)
    line = kernel(seconds=20)
    stored = engine.evaluate_xid(change, companion_events=[change]).model_copy(
        update={"incident_id": "inc-gauge"}
    )
    decision = engine.evaluate_xid(line, companion_events=[line, change])

    joined = join_gauge_twin(
        line, decision, [line, change], frozenset(), {change.event_id: stored}.get
    )

    assert joined.correlated_event_id == change.event_id
    assert FIELD_230_EVIDENCE in joined.reasons[-1]
    for refused in (
        stored.model_copy(update={"incident_id": None}),
        stored.model_copy(
            update={"disposition": ActionDisposition.PENDING_CORRELATION}
        ),
        stored.model_copy(update={"official_action": "DRAIN_AND_RESET"}),
        stored.model_copy(update={"action": RecoveryAction.REBOOT_NODE}),
    ):
        assert (
            join_gauge_twin(
                line,
                decision,
                [line, change],
                frozenset(),
                {change.event_id: refused}.get,
            ).correlated_event_id
            is None
        ), refused
    companion = decision.model_copy(update={"correlated_event_id": "xid-64"})
    assert (
        join_gauge_twin(
            line, companion, [line, change], frozenset(), {change.event_id: stored}.get
        ).correlated_event_id
        == "xid-64"
    ), "a companion chosen by the decode keeps its incident"
    assert (
        join_gauge_twin(
            line, decision, [line, change], None, {change.event_id: stored}.get
        )
        == decision
    ), "an ambiguous pairing joins nothing"


def test_a_report_reads_two_twin_windows_and_any_other_source_its_companion_window():
    companion = timedelta(seconds=30)

    assert candidate_window(kernel(), companion) == 2 * XID_REPORT_TWIN_WINDOW
    assert candidate_window(gauge(), companion) == 2 * XID_REPORT_TWIN_WINDOW
    assert (
        candidate_window(kernel(event_source="FABRIC_MANAGER_LOG"), companion)
        == companion
    )
