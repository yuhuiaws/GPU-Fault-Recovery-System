"""One XID occurrence, reported by the kernel line and by DCGM field 230, is one
incident and one remediation, in either arrival order (batch 237).

On the b236 head the replay below (``_xid_report_twins_support``) ran the
second report's own remediation wherever the store's merge groups did not catch
it: every actionable XID on a node whose job the faulted GPU cannot be bound to
(a second STOP_WORKLOADS + RESET_GPU / RESTART_NODE + RESTART_WORKLOAD, queued
behind the first by the per-node budget and the rule-A busy-node hold, or both
held against each other until the 240 s timeout stopped the job with no reset
at all); every RESET/REBOOT-class XID whose first remediation had already
finished; a solo XID 45's RESTART_FABRIC_MANAGER, XID 54's CHECK_MECHANICALS
and XID 151's RESTART_VM twice; and for XID 154, 74 and 144-150, whose decode
reads the kernel line, a field-230 change decided alone: a BLOCKED quarantine
of a node the line's own DRAIN_P2P / reset had just repaired. The second
report now joins the first one's incident as a duplicate observation
(``policy/xid_cross_source``, ``provider_correlation.report_twin_incident``).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation
from gpu_fault.policy import ActionDisposition
from gpu_fault.store import SqliteStore
from tests.orchestration._xid_report_twins_support import (
    CASES,
    GAUGE,
    KERNEL,
    Replay,
    gauge_batch,
    kernel_line,
    new_replay,
    replay_both,
)

ORDERS = ("kernel-first", "gauge-first")
STATES = ("idle", "uncovered", "active", "active-unbound")
#: The classes that ran a second remediation on the b236 head in the full grid.
GRID_CASES = (
    "48-reset-gpu",
    "79-restart-bm",
    "13-restart-app",
    "154-drain-p2p",
    "154-reset",
    "74-reset",
    "45-restart-fm",
)
EDGE_CASES = (
    "154-reboot",
    "54-check-mechanicals",
    "151-restart-vm",
    "78-update-swfw",
    "159-not-applicable",
    "145-b200-nvlink5",
)


def assert_one_occurrence(replay: Replay, label: str) -> None:
    reports = replay.reports()
    assert sorted(reports) == [GAUGE, KERNEL], (label, sorted(reports))
    (kernel, kernel_decision, kernel_incident) = reports[KERNEL]
    (gauge, gauge_decision, gauge_incident) = reports[GAUGE]
    assert replay.post_errors == [], (label, replay.post_errors)
    assert kernel_incident is not None and gauge_incident is not None, label
    assert kernel_incident.incident_id == gauge_incident.incident_id, (
        f"{label}: the two reports of one XID opened two incidents",
        kernel_decision.reasons,
        gauge_decision.reasons,
    )
    assert [item.incident_id for item in replay.incidents()] == [
        kernel_incident.incident_id
    ], label
    assert kernel_decision.workflow_request_id == gauge_decision.workflow_request_id, (
        label
    )
    assert len(replay.workflows()) <= 1, (
        label,
        [(item.status.value, item.incident_id) for item in replay.workflows()],
    )
    twice = {
        operation.value: count
        for operation, count in replay.adapter.executions().items()
        if count > 1
    }
    assert twice == {}, f"{label}: ran twice: {twice}"
    incident = replay.store.get_incident(kernel_incident.incident_id)
    if incident.state is IncidentState.RECOVERED:
        live = [
            marker.marker_id
            for marker in replay.store.list_markers_for_incident(incident.incident_id)
            if marker.active
        ]
        assert live == [], f"{label}: a recovered incident keeps live markers {live}"


@pytest.mark.parametrize("timing", ["in-flight", "after-completion"])
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name", GRID_CASES)
def test_one_xid_reported_twice_is_one_incident_and_one_remediation(
    name: str, order: str, state: str, timing: str
) -> None:
    replay = replay_both(CASES[name], order=order, state=state, timing=timing)

    assert_one_occurrence(replay, f"{name}/{order}/{state}/{timing}")


@pytest.mark.parametrize("timing", ["in-flight", "after-completion"])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("48-reset-gpu", "active-unbound"),
        ("13-restart-app", "idle"),
        ("154-drain-p2p", "idle"),
        ("79-restart-bm", "active"),
    ],
)
def test_a_sql_store_links_the_second_report_the_same_way(
    tmp_path, name: str, state: str, order: str, timing: str
) -> None:
    """The join is the store's own event link: SQLite answers as memory does."""

    store = SqliteStore(str(tmp_path / "control.db"))
    try:
        replay = replay_both(
            CASES[name], order=order, state=state, timing=timing, store=store
        )
        assert_one_occurrence(replay, f"sqlite {name}/{state}/{order}/{timing}")
    finally:
        store.close()


@pytest.mark.parametrize("state", ["idle", "active-unbound"])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name", EDGE_CASES)
def test_every_other_workflow_class_dedupes_too(
    name: str, order: str, state: str
) -> None:
    replay = replay_both(CASES[name], order=order, state=state, timing="in-flight")

    assert_one_occurrence(replay, f"{name}/{order}/{state}")


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name", ["48-reset-gpu", "13-restart-app", "79-restart-bm"])
def test_two_unstarted_job_workflows_no_longer_hold_each_other_into_a_stop(
    monkeypatch: pytest.MonkeyPatch, name: str, order: str
) -> None:
    """Both reports pending before any claim on a node whose job the GPU cannot
    be bound to: each job workflow waited for the other (rule A) until the
    busy-node timeout rewrote both to a bare STOP_WORKLOADS and failed them --
    the job stopped twice, the GPU never reset. The wait is made to have
    elapsed here."""

    monkeypatch.setattr(
        WorkflowDispatcher,
        "_node_busy_wait_start",
        staticmethod(lambda _workflow, now: now - timedelta(hours=1)),
    )
    replay = replay_both(
        CASES[name], order=order, state="active-unbound", timing="before-dispatch"
    )

    assert_one_occurrence(replay, f"{name}/{order}")
    (workflow,) = replay.workflows()
    assert workflow.status.value == "SUCCEEDED", workflow.terminal_failure_reason


@pytest.mark.parametrize("order", ORDERS)
def test_a_graded_node_records_one_observe_only_xid_45(order: str) -> None:
    """The site's observe-only XID 45 rewrite runs before the two reports
    meet, so they meet as the incident sees them: one RECOVERED incident."""

    replay = replay_both(
        CASES["45-restart-fm"],
        order=order,
        state="idle",
        timing="in-flight",
        graded=True,
    )

    assert_one_occurrence(replay, f"45 graded/{order}")
    (incident,) = replay.incidents()
    assert incident.state is IncidentState.RECOVERED
    assert incident.workflow_request_id is None and replay.workflows() == []
    for _event, decision, _incident in replay.reports().values():
        assert decision.disposition is ActionDisposition.MONITOR_ONLY, decision


def test_the_kernel_line_decides_an_xid_154_that_field_230_reported_first() -> None:
    """The change carries no recovery-action label: it waits for the line, the
    line's DRAIN_P2P runs once, and nothing quarantines the repaired node."""

    replay = replay_both(
        CASES["154-drain-p2p"], order="gauge-first", state="idle", timing="in-flight"
    )

    assert_one_occurrence(replay, "154 gauge-first")
    executions = replay.adapter.executions()
    assert executions[WorkflowOperation.RECOVER_DRAIN_P2P] == 1
    assert executions[WorkflowOperation.QUARANTINE] == 0
    _, gauge_decision, _ = replay.reports()[GAUGE]
    assert gauge_decision.official_action == "DRAIN_P2P", gauge_decision.reasons
    assert gauge_decision.duplicate and gauge_decision.correlated_event_id, (
        "the change is booked against the kernel line"
    )


@pytest.mark.parametrize(
    ("name", "blocked"),
    [("154-drain-p2p", True), ("74-reset", True), ("48-reset-gpu", False)],
)
def test_a_field_230_change_without_its_kernel_line(name: str, blocked: bool) -> None:
    """A lost kernel line: an XID whose decode reads it waits out the twin
    window and then fails closed as before; any other XID decides at once."""

    case = CASES[name]
    replay = new_replay("idle", case)
    replay.post(
        GPU_METRICS_PATH,
        gauge_batch("gauge", replay.base + timedelta(seconds=10), case.xid, case),
    )
    (gauge, decision, _incident) = replay.reports()[GAUGE]
    assert (decision.disposition is ActionDisposition.PENDING_CORRELATION) is blocked

    replay.close_windows()
    decision = replay.store.get_xid_policy_decision(gauge.event_id)

    if blocked:
        assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    else:
        assert decision.disposition is ActionDisposition.EXECUTABLE
    assert not decision.duplicate and decision.correlated_event_id is None
    assert decision.incident_id == f"inc-{gauge.event_id}"


def test_a_repeated_xid_after_the_window_is_a_new_occurrence() -> None:
    """The gauge stays 48 through the repeat, so the second XID 48 arrives as a
    kernel line only; beyond the twin window it is a new fault with its own
    remediation, not a report of the first."""

    case = CASES["48-reset-gpu"]
    replay = replay_both(case, order="kernel-first", state="idle", timing="in-flight")
    first_reset = replay.adapter.executions()[WorkflowOperation.RESET_GPU]
    replay.post(
        NVIDIA_KERNEL_PATH,
        kernel_line(case, replay.base + timedelta(seconds=180), record_id="kmsg-again"),
    )
    replay.dispatch(3)
    replay.finish()

    (again,) = [
        event
        for event in replay.store.list_xid_events("cluster-a", "node-a")
        if event.event_id.startswith("kernel-log-kmsg-again")
    ]
    decision = replay.store.get_xid_policy_decision(again.event_id)
    assert decision.correlated_event_id is None and not decision.duplicate
    assert first_reset == 1
    assert replay.adapter.executions()[WorkflowOperation.RESET_GPU] == 2, (
        "the repeated XID is remediated again"
    )


def test_a_field_230_change_pairs_with_one_kernel_line() -> None:
    """Gauge first, then two kernel lines of XID 13 on the GPU: the change is
    one occurrence, so only the first line joins it."""

    case = CASES["13-restart-app"]
    replay = new_replay("active", case)
    replay.post(
        GPU_METRICS_PATH,
        gauge_batch("gauge", replay.base + timedelta(seconds=25), case.xid, case),
    )
    for seconds, record in ((10, "kmsg-first"), (20, "kmsg-second")):
        replay.post(
            NVIDIA_KERNEL_PATH,
            kernel_line(
                case, replay.base + timedelta(seconds=seconds), record_id=record
            ),
        )
    events = {
        event.event_id: event
        for event in replay.store.list_xid_events("cluster-a", "node-a")
    }
    gauge = next(event for event in events.values() if event.event_source == GAUGE)
    joined = [
        event_id
        for event_id, event in events.items()
        if event.event_source == KERNEL
        and replay.store.get_xid_policy_decision(event_id).correlated_event_id
        == gauge.event_id
    ]

    assert [events[item].event_id.split("-xid-")[0] for item in joined] == [
        "kernel-log-kmsg-first"
    ]


def test_a_second_change_does_not_take_a_kernel_line_already_paired() -> None:
    """The value went 48 -> 63 -> 48: two changes mean two XID 48 occurrences,
    one of whose kernel line was lost; the second change decides alone."""

    case = CASES["48-reset-gpu"]
    replay = new_replay("idle", case)
    replay.post(
        NVIDIA_KERNEL_PATH, kernel_line(case, replay.base + timedelta(seconds=5))
    )
    replay.post(
        GPU_METRICS_PATH,
        gauge_batch("first", replay.base + timedelta(seconds=20), 48, case),
    )
    replay.post(
        GPU_METRICS_PATH,
        gauge_batch("bounce", replay.base + timedelta(seconds=35), 63, case),
    )
    replay.post(
        GPU_METRICS_PATH,
        gauge_batch("second", replay.base + timedelta(seconds=50), 48, case),
    )
    decisions = {
        event.event_id: replay.store.get_xid_policy_decision(event.event_id)
        for event in replay.store.list_xid_events("cluster-a", "node-a")
        if event.event_source == GAUGE and event.xid == 48
    }
    kernel = next(
        event
        for event in replay.store.list_xid_events("cluster-a", "node-a")
        if event.event_source == KERNEL
    )

    assert decisions["first-GPU-h200-0-xid-48"].correlated_event_id == kernel.event_id
    assert decisions["second-GPU-h200-0-xid-48"].correlated_event_id is None


def test_a_kernel_line_later_than_the_gauge_window_keeps_both_decisions() -> None:
    """The documented residual: an XID 154 line delivered after the change's
    twin window has closed meets a change that already failed closed; the line
    runs its own decode and the quarantine stands for an operator."""

    replay = replay_both(
        CASES["154-drain-p2p"], order="gauge-first", state="idle", timing="late-second"
    )
    reports = replay.reports()

    assert reports[GAUGE][1].disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert reports[KERNEL][1].official_action == "DRAIN_P2P"
    assert reports[KERNEL][2].incident_id != reports[GAUGE][2].incident_id
    executions = replay.adapter.executions()
    assert executions[WorkflowOperation.RECOVER_DRAIN_P2P] == 1
    assert executions[WorkflowOperation.QUARANTINE] == 1
