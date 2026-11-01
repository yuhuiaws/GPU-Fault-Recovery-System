"""One XID occurrence reported twice -- the kernel NVRM line and the DCGM field
230 change -- sends one FAULT_DETECTED notice (batch 238).

Since batch 237 the second report joins the first one's incident as a duplicate
observation (``policy/xid_cross_source``), but ``finalize_xid`` still built the
per-XID investigatory notice (``xid-investigatory-zh-v1``, category
FAULT_DETECTED) for it, keyed by the report's own event id: two mails for one
fault, in either arrival order. The second report's notice now resolves to the
first one's through the store's notification dedupe (the deduplication key of
the occurrence's first report), so the second report is folded into that notice
and its decision points at it. The incident keeps the twin as evidence (event
link and marker). What does not fold: a different XID (a companion or a later
escalation) and a repeat of the same XID after the twin window -- each is a
fault of its own and notifies.

The notice names the first report only: it is written before the second report
exists. The second source shows up in the incident: its event link, and its
marker, whose evidence reference is part of the incident's evidence that the
notices previewed from the incident later read.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from tests.orchestration._xid_report_twins_support import (
    CASES,
    CLUSTER,
    GAUGE,
    KERNEL,
    NODE,
    Replay,
    XidCase,
    gauge_batch,
    kernel_line,
    new_replay,
    replay_both,
)

ORDERS = ("kernel-first", "gauge-first")
#: (case, node state, timing of the second report) -- the classes and timings
#: batch 237 joins: a reset, a reboot, a job restart, the XID 154 change that
#: waits for its line, and a second report after the first one's remediation.
SHAPES = (
    ("48-reset-gpu", "idle", "in-flight"),
    ("79-restart-bm", "active", "in-flight"),
    ("13-restart-app", "active-unbound", "in-flight"),
    ("154-drain-p2p", "idle", "in-flight"),
    ("48-reset-gpu", "idle", "after-completion"),
)


def fault_detected(replay: Replay, xid: int) -> list:
    return [
        item
        for item in replay.store.list_notifications()
        if item.category == "FAULT_DETECTED"
        and item.subject.startswith(f"[GPU FAULT][XID {xid} investigatory action]")
    ]


def first_and_second(replay: Replay):
    """The report decided first and the one that joined it (``correlated_event_id``
    names the first) -- the arrival order, except for an XID 154 change that
    waited for its line."""

    reports = replay.reports()
    kernel, gauge = reports[KERNEL], reports[GAUGE]
    if gauge[1].correlated_event_id == kernel[0].event_id:
        return kernel, gauge
    assert kernel[1].correlated_event_id == gauge[0].event_id, (
        "the two reports were not paired",
        kernel[1].reasons,
        gauge[1].reasons,
    )
    return gauge, kernel


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(("name", "state", "timing"), SHAPES)
def test_the_second_report_sends_no_second_fault_detected(
    name: str, state: str, timing: str, order: str
) -> None:
    case = CASES[name]
    label = f"{name}/{state}/{timing}/{order}"
    replay = replay_both(case, order=order, state=state, timing=timing)
    (first, first_decision, incident), (second, second_decision, joined) = (
        first_and_second(replay)
    )

    assert replay.post_errors == [], (label, replay.post_errors)
    notices = fault_detected(replay, case.xid)
    assert [item.deduplication_key for item in notices] == [
        f"{first.event_id}/xid-investigatory/xid-investigatory-zh-v1"
    ], f"{label}: one FAULT_DETECTED notice for one occurrence"
    (notice,) = notices
    assert first_decision.investigatory_notification_id == notice.notification_id, label
    assert second_decision.investigatory_notification_id == notice.notification_id, (
        f"{label}: the second report is folded into the first one's notice"
    )
    assert joined is not None and joined.incident_id == incident.incident_id, (
        f"{label}: the incident keeps the twin",
        second_decision.reasons,
    )
    markers = {
        marker.marker_id
        for marker in replay.store.list_markers_for_incident(incident.incident_id)
    }
    assert f"marker-{second.event_id}" in markers, (label, sorted(markers))
    assert first.event_id in notice.body_text, (label, notice.body_text)


@pytest.mark.parametrize("order", ORDERS)
def test_a_not_applicable_twin_already_shared_its_incident_notice(order: str) -> None:
    """Control: the NOT_APPLICABLE notice is keyed by incident, XID and policy
    version, so the second report found it before batch 238 too."""

    replay = replay_both(
        CASES["159-not-applicable"], order=order, state="idle", timing="in-flight"
    )

    notices = replay.store.list_notifications()
    assert [item.subject for item in notices] == [
        f"[GPU FAULT][NOT_APPLICABLE] {CLUSTER} {NODE} XID 159"
    ], [item.subject for item in notices]
    for _event, decision, _incident in replay.reports().values():
        assert decision.investigatory_notification_id == notices[0].notification_id


def test_a_later_escalation_on_the_same_gpu_still_notifies() -> None:
    """After the XID 48 pair (one notice) the GPU falls off the bus: XID 79 is a
    fault of its own, and the escalation is mailed."""

    replay = replay_both(
        CASES["48-reset-gpu"], order="kernel-first", state="idle", timing="in-flight"
    )
    replay.post(
        NVIDIA_KERNEL_PATH,
        kernel_line(
            CASES["79-restart-bm"],
            replay.base + timedelta(seconds=60),
            record_id="kmsg-escalation",
        ),
    )

    assert replay.post_errors == [], replay.post_errors
    assert len(fault_detected(replay, 48)) == 1, "the XID 48 pair is one notice"
    (escalation,) = fault_detected(replay, 79)
    assert escalation.deduplication_key.startswith("kernel-log-kmsg-escalation"), (
        escalation.deduplication_key
    )


def test_a_repeated_xid_after_the_twin_window_still_notifies() -> None:
    """The gauge stays 48 through a repeat, so a second XID 48 arrives as a kernel
    line only; beyond the window it is a new occurrence with its own notice."""

    case = CASES["48-reset-gpu"]
    replay = replay_both(case, order="kernel-first", state="idle", timing="in-flight")
    replay.post(
        NVIDIA_KERNEL_PATH,
        kernel_line(case, replay.base + timedelta(seconds=180), record_id="kmsg-again"),
    )

    keys = sorted(item.deduplication_key for item in fault_detected(replay, 48))
    assert len(keys) == 2, keys
    assert any(key.startswith("kernel-log-kmsg-again") for key in keys), keys


def test_the_second_source_is_evidence_of_the_incident() -> None:
    """Gauge first, then the kernel line with its kmsg evidence: the one notice is
    the gauge's, the line's evidence is on the incident's markers."""

    case = CASES["48-reset-gpu"]
    replay = new_replay("idle", case)
    replay.post(
        GPU_METRICS_PATH,
        gauge_batch("gauge", replay.base + timedelta(seconds=10), case.xid, case),
    )
    line = kernel_line(case, replay.base + timedelta(seconds=30))
    line["evidence_ref"] = "kmsg://node-a/boot-a/4242"
    replay.post(NVIDIA_KERNEL_PATH, line)
    reports = replay.reports()
    _gauge, gauge_decision, incident = reports[GAUGE]
    kernel, kernel_decision, _joined = reports[KERNEL]

    assert replay.post_errors == [], replay.post_errors
    assert kernel_decision.duplicate and kernel_decision.correlated_event_id, (
        kernel_decision.reasons
    )
    (notice,) = fault_detected(replay, case.xid)
    assert kernel_decision.investigatory_notification_id == notice.notification_id
    assert gauge_decision.investigatory_notification_id == notice.notification_id
    evidence = {
        marker.raw_evidence_ref
        for marker in replay.store.list_markers_for_incident(incident.incident_id)
    }
    assert line["evidence_ref"] in evidence, sorted(item or "" for item in evidence)
    assert kernel.event_id not in notice.body_text, (
        "the notice predates the line; the incident carries it"
    )


def test_a_companion_xid_keeps_its_own_notice() -> None:
    """XID 64 beside XID 48 on the GPU is a companion -- another XID of the same
    fault, joined to its incident -- not a second report of XID 48: both XIDs
    are mailed, as before batch 238."""

    first = CASES["48-reset-gpu"]
    companion = XidCase(64, "pid=1234, name=python, row remapper recording failure")
    replay = new_replay("idle", first)
    replay.post(
        NVIDIA_KERNEL_PATH,
        kernel_line(first, replay.base + timedelta(seconds=10), record_id="kmsg-48"),
    )
    replay.post(
        NVIDIA_KERNEL_PATH,
        kernel_line(
            companion, replay.base + timedelta(seconds=20), record_id="kmsg-64"
        ),
    )
    replay.close_windows()
    events = {event.xid: event for event in replay.store.list_xid_events(CLUSTER, NODE)}
    decision = replay.store.get_xid_policy_decision(events[64].event_id)

    assert replay.post_errors == [], replay.post_errors
    assert decision.correlated_event_id == events[48].event_id, decision.reasons
    assert (
        replay.store.get_incident_by_event(events[64].event_id).incident_id
        == replay.store.get_incident_by_event(events[48].event_id).incident_id
    ), "the companion joined the XID 48 incident"
    assert len(fault_detected(replay, 48)) == 1
    (notice,) = fault_detected(replay, 64)
    assert decision.investigatory_notification_id == notice.notification_id
