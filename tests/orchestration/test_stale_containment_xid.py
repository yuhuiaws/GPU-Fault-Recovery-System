"""A containment XID older than the running attempt keeps its containment, and a
recorded-and-done XID is answered 200, never 409 (batch 238).

The attempt-generation fence without a recovery baseline (F-B8,
``IncidentOrchestrator._fence_stale_generation_without_baseline``) records a
stale candidate that does not out-rank RESTART_WORKLOAD as an ignored
RECOVERED incident with no workflow. It ranked every decision by its official
steps, but a containment decision -- NOT_APPLICABLE, SITE_SAFETY,
BLOCKED_WORKFLOW, BLOCKED_MISSING_EVIDENCE -- compiles a safety-only workflow
whose official steps never run: its plan is the safety quarantine (or evidence
freeze), which a workload restart does not answer. So on the b237 head a lone
H200 XID 159 line (catalog products B100/GB200: NOT_APPLICABLE) older than the
attempt was recorded RECOVERED with a live QUARANTINE marker, then
``preview_not_applicable`` raised "NOT_APPLICABLE incident has no safety
workflow" and the kernel route answered 409: no quarantine, no notification, no
persisted decision. An XID absent from the catalog (SITE_SAFETY), XID 151
(BLOCKED_WORKFLOW) and an XID 74 without its registers lost their containment
the same way, only without the 409; XID 154's quarantine survived because its
unexecuted official steps happen to out-rank the restart.

A stale-boot record (``reboot_window.cover_fault``: the line's boot is one the
node has left) is recorded and done by design, and the NOT_APPLICABLE notice now
says so instead of refusing it. A recorded incident keeps no live marker
(ARCH-I4).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration.reboot_window import STALE_BOOT_REASON
from gpu_fault.policy import ActionDisposition
from gpu_fault.store import SqliteStore
from tests.orchestration._support import _active_agent
from tests.orchestration._xid_report_twins_support import (
    CASES,
    CLUSTER,
    GPU,
    NODE,
    Replay,
    XidCase,
    attempt,
    gauge_batch,
    kernel_line,
    new_replay,
)

#: The line is 20 s older than the attempt that runs on its GPU now.
LINE_AT = timedelta(seconds=40)
ATTEMPT_STARTED_AT = timedelta(seconds=60)

#: A line inside that attempt, for the plan a stale one is compared with.
FRESH_AT = timedelta(seconds=70)

#: Decisions whose plan no workload restart answers: containment (the safety
#: quarantine or evidence freeze) and the NVIDIA workflow XID 151 runs for a VM.
NON_RECOVERY = {
    "159-not-applicable": (
        CASES["159-not-applicable"],
        ActionDisposition.NOT_APPLICABLE,
    ),
    "199-site-safety": (
        XidCase(199, "pid=1234, name=python, unknown"),
        ActionDisposition.SITE_SAFETY,
    ),
    "151-blocked-workflow": (
        XidCase(151, "Key rotation error"),
        ActionDisposition.BLOCKED_WORKFLOW,
    ),
    "74-missing-registers": (
        XidCase(74, "pid=1234, name=python, Link 3"),
        ActionDisposition.BLOCKED_MISSING_EVIDENCE,
    ),
    # Kept on the b237 head too: its unexecuted official steps out-rank the
    # restart, which is all the old fence read.
    "154-unknown-label": (
        XidCase(154, "GPU recovery action changed from 0x0 (None) to 0x9 (Bogus)"),
        ActionDisposition.BLOCKED_MISSING_EVIDENCE,
    ),
}


def restarted_replay(case: XidCase, *, store=None) -> Replay:
    """The job on the faulted GPU restarted 20 s after the line: attempt a002
    runs, and no recovery workflow exists for it anywhere."""

    replay = new_replay("idle", case, store=store)
    replay.store.save_attempt_observation(
        attempt(
            "train-a002",
            replay.base + ATTEMPT_STARTED_AT + timedelta(seconds=5),
            started_at=replay.base + ATTEMPT_STARTED_AT,
            gpu_uuids=[GPU],
        )
    )
    return replay


def post_line(
    replay: Replay,
    case: XidCase,
    *,
    at: timedelta = LINE_AT,
    record_id: str = "kmsg-stale",
):
    replay.post(
        NVIDIA_KERNEL_PATH, kernel_line(case, replay.base + at, record_id=record_id)
    )
    (event,) = [
        item
        for item in replay.store.list_xid_events(CLUSTER, NODE)
        if item.event_id.startswith(f"kernel-log-{record_id}")
    ]
    return (
        event,
        replay.store.get_xid_policy_decision(event.event_id),
        replay.store.get_incident_by_event(event.event_id),
    )


def plan(replay: Replay, incident) -> tuple:
    workflow = replay.store.get_workflow(incident.workflow_request_id)
    return (
        workflow.status,
        [step.operation for step in workflow.official_steps],
        [step.operation for step in workflow.safety_steps],
    )


def assert_recorded_without_live_marker(replay: Replay, incident) -> None:
    live = [
        marker.marker_id
        for marker in replay.store.list_markers_for_incident(incident.incident_id)
        if marker.active
    ]
    assert live == [], f"a RECOVERED incident keeps live markers {live} (ARCH-I4)"


@pytest.mark.parametrize("name", sorted(NON_RECOVERY))
def test_a_stale_non_recovery_decision_is_planned_like_a_fresh_one(name: str) -> None:
    case, disposition = NON_RECOVERY[name]
    fresh_replay = restarted_replay(case)
    _fresh, _decision, fresh_incident = post_line(fresh_replay, case, at=FRESH_AT)
    replay = restarted_replay(case)

    _event, decision, incident = post_line(replay, case)

    assert replay.post_errors == [], (name, replay.post_errors)
    assert decision.disposition is disposition, (name, decision.reasons)
    assert decision.incident_id == incident.incident_id, (
        f"{name}: the decision was never finalized",
        decision.incident_id,
    )
    assert incident.state is not IncidentState.RECOVERED, (name, incident.reasons)
    assert not any("Ignored stale" in reason for reason in incident.reasons), (
        f"{name}: a workload restart does not answer this decision",
        incident.reasons,
    )
    stale_plan = plan(replay, incident)
    assert stale_plan == plan(fresh_replay, fresh_incident), (
        f"{name}: the stale line is planned as the same line inside the attempt",
        stale_plan,
    )
    assert decision.workflow_request_id == incident.workflow_request_id, name
    assert (
        decision.investigatory_notification_id or decision.advisory_notification_id
    ), f"{name}: no notification was recorded for the decision"

    replay.dispatch(3)
    replay.finish()

    status, _official, safety = stale_plan
    if (
        status is WorkflowStatus.SAFETY_PENDING
        and WorkflowOperation.QUARANTINE in safety
    ):
        assert replay.adapter.executions()[WorkflowOperation.QUARANTINE] == 1, (
            f"{name}: the quarantine the policy asked for ran once"
        )


def test_the_stale_not_applicable_line_is_notified_with_its_safety_workflow() -> None:
    """The brief's reproduction: one H200 XID 159 line, older than the attempt."""

    case = CASES["159-not-applicable"]
    replay = restarted_replay(case)

    _event, decision, incident = post_line(replay, case)

    assert replay.post_errors == [], replay.post_errors
    notification = replay.store.get_notification(decision.advisory_notification_id)
    assert notification.subject == (
        f"[GPU FAULT][NOT_APPLICABLE] {CLUSTER} {NODE} XID 159"
    ), notification.subject
    assert decision.investigatory_notification_id == notification.notification_id
    assert f"- Workflow: {incident.workflow_request_id}" in notification.body_text
    assert "- Workflow status: SAFETY_PENDING" in notification.body_text
    assert "QUARANTINE" in notification.body_text


def test_a_stale_field_230_not_applicable_report_keeps_its_batch(tmp_path) -> None:
    """The same decision reported by DCGM field 230 alone: on a SQL store the
    409 rolled back the whole metrics batch with the incident."""

    case = CASES["159-not-applicable"]
    store = SqliteStore(str(tmp_path / "control.db"))
    try:
        replay = restarted_replay(case, store=store)
        replay.post(
            GPU_METRICS_PATH,
            gauge_batch("gauge-stale", replay.base + LINE_AT, case.xid, case),
        )

        assert replay.post_errors == [], replay.post_errors
        (gauge,) = [
            event
            for event in store.list_xid_events(CLUSTER, NODE)
            if event.event_source == "DCGM_EXPORTER"
        ]
        decision = store.get_xid_policy_decision(gauge.event_id)
        assert decision.disposition is ActionDisposition.NOT_APPLICABLE
        incident = store.get_incident_by_event(gauge.event_id)
        assert incident is not None and incident.workflow_request_id, (
            "the field-230 report keeps its quarantine"
        )
        latest = {
            item.sample.canonical_name: item.sample.value
            for item in store.list_gpu_metrics_latest(CLUSTER, NODE)
        }
        assert latest.get("xid_last_error") == case.xid, latest
    finally:
        store.close()


def test_a_stale_recovery_is_still_recorded_and_retires_its_marker() -> None:
    """F-B8 itself is unchanged for a recovery the restart answered: XID 13's
    RESTART_APP is recorded as ignored, and the record holds no live marker."""

    case = CASES["13-restart-app"]
    replay = restarted_replay(case)

    _event, decision, incident = post_line(replay, case)

    assert replay.post_errors == [], replay.post_errors
    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert incident.state is IncidentState.RECOVERED, incident.reasons
    assert incident.workflow_request_id is None and replay.workflows() == []
    assert any(
        "Ignored stale XID 13 action" in reason and "train-a002" in reason
        for reason in incident.reasons
    ), incident.reasons
    assert decision.incident_id == incident.incident_id
    assert_recorded_without_live_marker(replay, incident)


@pytest.mark.parametrize(
    ("name", "case"),
    [
        ("43-monitor-only", XidCase(43, "pid=1234, name=python, Ch 00000010")),
        ("48-reset-gpu", CASES["48-reset-gpu"]),
    ],
)
def test_observation_and_escalation_keep_their_answers(name: str, case) -> None:
    """Controls: an observe-only XID ends RECOVERED either way; a reset out-ranks
    the restart and runs."""

    replay = restarted_replay(case)

    _event, decision, incident = post_line(replay, case)

    assert replay.post_errors == [], (name, replay.post_errors)
    assert decision.incident_id == incident.incident_id, name
    if name == "43-monitor-only":
        assert incident.state is IncidentState.RECOVERED, incident.reasons
        assert replay.workflows() == [], name
        assert_recorded_without_live_marker(replay, incident)
    else:
        workflow = replay.store.get_workflow(incident.workflow_request_id)
        assert WorkflowOperation.RESET_GPU in {
            step.operation for step in workflow.official_steps
        }, name


def agent_on_a_new_boot():
    """The node's Agent heartbeats on a boot the kernel line does not name."""

    now = datetime.now(timezone.utc)
    return _active_agent(boot_id="boot-b").model_copy(
        update={
            "cluster_id": CLUSTER,
            "node_id": NODE,
            "first_seen_at": now - timedelta(minutes=1),
            "last_seen_at": now,
            "lease_expires_at": now + timedelta(minutes=1),
        }
    )


def test_a_stale_boot_not_applicable_line_is_recorded_and_notified() -> None:
    """A line from a boot the node has left (an outbox record replayed after the
    reboot) is recorded as evidence by design; its NOT_APPLICABLE notice says
    that nothing is planned instead of answering 409."""

    case = CASES["159-not-applicable"]
    replay = new_replay("idle", case)
    replay.store.save_agent(agent_on_a_new_boot())

    _event, decision, incident = post_line(replay, case, record_id="kmsg-old")

    assert replay.post_errors == [], replay.post_errors
    assert incident.state is IncidentState.RECOVERED, incident.reasons
    assert incident.workflow_request_id is None and replay.workflows() == []
    assert STALE_BOOT_REASON in incident.reasons, incident.reasons
    assert decision.incident_id == incident.incident_id
    notification = replay.store.get_notification(decision.advisory_notification_id)
    assert "[NOT_APPLICABLE]" in notification.subject, notification.subject
    assert "- Workflow: NONE" in notification.body_text, notification.body_text
    assert "- Safety action: NONE" in notification.body_text, notification.body_text
    assert STALE_BOOT_REASON in notification.body_text, notification.body_text
    assert_recorded_without_live_marker(replay, incident)


def test_a_repeated_stale_boot_line_is_answered_the_same_way() -> None:
    """A redelivery (outbox requeue, processor retry) of the recorded line meets
    its incident with no workflow: the same notice, 200 again."""

    case = CASES["159-not-applicable"]
    replay = new_replay("idle", case)
    replay.store.save_agent(agent_on_a_new_boot())
    _event, first, _incident = post_line(replay, case, record_id="kmsg-old")

    _event, again, _incident = post_line(replay, case, record_id="kmsg-old")

    assert replay.post_errors == [], replay.post_errors
    assert again.advisory_notification_id == first.advisory_notification_id
    assert len(replay.store.list_notifications()) == 1, [
        item.subject for item in replay.store.list_notifications()
    ]
