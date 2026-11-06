"""A remediation that ends holding its node tells an operator, once.

A BMC sensor or host signal graded QUARANTINE compiles the node-health
QUARANTINE plan FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> [STOP_WORKLOADS] ->
QUARANTINE: no RESTORE_SCHEDULING, no ESCALATE_SUPPORT. It ends SUCCEEDED with
the node cordoned and tainted and the incident QUARANTINED -- an end only an
operator's validated restore (after the repair) or a departure close leaves.
The isolation step saves no notification and the plan has no support step, so
the record used to end there with nobody told (GF-GENERAL-LIVE-015 / 004 /
016). The executor now sends one ``quarantined-node-notice-zh-v1`` notice per
incident at that terminal write; the state and the operator's exits do not
change, and no other ending sends it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.common import quarantine_taint_value
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.terminal_state import TerminalDecision
from gpu_fault.execution.workflow_notifications import after_terminal_write
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY
from tests._builders import (
    active_workflow_executor,
    build_store,
    contained_stop_receipt,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._raw_health_e2e_support import RawHealthRig
from tests.execution._support import FakeAdapter

CATEGORY = "OPERATOR_REVIEW"
TEMPLATE = "quarantined-node-notice-zh-v1"
SUBJECT = "[GPU FAULT][quarantined, operator action required] cluster-a: node-a"
ALERTS = "CPU1 Temp=96.000 degrees C [cr; ucr 95.000]; Fan1=0.000 RPM [cr; lcr 300.000]"
INCIDENT = "inc-host-node-a-bmc"
WORKFLOW = "workflow-bmc-quarantine"
WORKLOAD = "training/pytorchjob/train"
NOW = datetime(2026, 10, 26, 6, 25, tzinfo=timezone.utc)
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
STOP = WorkflowOperation.STOP_WORKLOADS
QUARANTINE = WorkflowOperation.QUARANTINE
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
VALIDATE = WorkflowOperation.VALIDATE_GPU


@pytest.fixture
def rig(tmp_path, monkeypatch):
    value = RawHealthRig(tmp_path, monkeypatch)
    try:
        yield value
    finally:
        value.store.close()


def _host(rig, sample: HostMetricSample):
    batch = HostTelemetryBatch(
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=rig.clock,
        received_at=rig.clock,
        runtime_profile_version="simulated-v1",
        samples=[sample],
    )
    return rig.post(
        "/v1/collector-events/host-telemetry", batch.model_dump(mode="json")
    )


def _bmc(rig):
    return _host(
        rig,
        HostMetricSample(
            name="bmc_critical_sensor",
            value=2,
            unit="sensors",
            labels={"sensors": "CPU1 Temp, Fan1", "sensor_alerts": ALERTS},
        ),
    )


def _filesystem(rig):
    return _host(
        rig,
        HostMetricSample(
            name="filesystem_used_percent", value=99, unit="percent", device="/"
        ),
    )


def _kernel(rig):
    return rig.log("mce: uncorrectable hardware error")


def _notices(store):
    return [
        item
        for item in store.list_notifications()
        if item.deduplication_key.endswith("/quarantined-node-notice")
    ]


@pytest.mark.parametrize(
    "inject,finding,steps",
    [
        (_bmc, f"BMC reports 2 critical sensors: {ALERTS}", [FREEZE, MARK, QUARANTINE]),
        (_filesystem, "filesystem capacity exhausted", [FREEZE, MARK, QUARANTINE]),
        (_kernel, "CPU or memory hardware error", [FREEZE, MARK, STOP, QUARANTINE]),
    ],
    ids=["bmc-sensor", "filesystem-full", "kernel-mce"],
)
def test_a_node_health_quarantine_that_completes_sends_one_operator_notice(
    rig, inject, finding, steps
):
    result = inject(rig)
    [workflow_id] = result["workflow_request_ids"]
    workflow = rig.store.get_workflow(workflow_id)
    assert [step.operation for step in workflow.official_steps] == steps, (
        "the node-health QUARANTINE plan: no restore, no support hand-off"
    )

    outcome = rig.execute(workflow)

    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    incident = rig.store.get_incident(workflow.incident_id)
    assert incident.state is IncidentState.QUARANTINED, "the state is unchanged"
    node = rig.core.node
    assert node["spec"]["unschedulable"] is True, node
    taint = ("gpu-fault.io/quarantined", quarantine_taint_value(incident.incident_id))
    assert taint in {(item["key"], item["value"]) for item in node["spec"]["taints"]}, (
        node
    )
    assert SUPPORT not in rig.boundary.calls, "no support case is opened"
    [notice] = rig.store.list_notifications()
    assert notice.incident_id == incident.incident_id
    assert notice.category == CATEGORY
    assert notice.deduplication_key == (
        f"cluster-a/{incident.incident_id}/quarantined-node-notice"
    )
    assert notice.subject == SUBJECT, notice.subject
    assert "contained" not in notice.subject.lower(), (
        "the contained hand-off's live oracle must not count this notice"
    )
    body = notice.body_text
    assert f"  - {finding}" in body, body
    assert "Steps run: " + " -> ".join(item.value for item in steps) in body, body
    assert "gpu-fault.io/quarantined" in body
    assert "workflow SUCCEEDED, incident QUARANTINED" in body
    assert f"--incident-id\n  {incident.incident_id} --disposition restore" in body, (
        body
    )
    assert "--close-quarantined" in body
    assert body.rstrip().endswith(f"Template: {TEMPLATE}"), body[-120:]
    assert notice.support_case_draft == ""


def _quarantine_plan(store, steps):
    store.save_incident(
        fault_incident(
            INCIDENT,
            "host-node-a-bmc",
            event_type="NODE_HEALTH",
            policy_source="SITE_NODE_HEALTH",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=WORKFLOW,
            fencing_token=3,
            reasons=[f"BMC reports 2 critical sensors: {ALERTS}"],
            created_at=NOW,
            updated_at=NOW,
        )
    )
    store.save_workflow(
        workflow_request(
            WORKFLOW,
            INCIDENT,
            status=WorkflowStatus.PENDING,
            runtime_profile_version="site-v1",
            official_steps=steps,
            created_at=NOW,
            updated_at=NOW,
        )
    )


def _terminal_quarantine():
    return workflow_step(
        QUARANTINE, parameters={"terminal_quarantine_node_ids": ["node-a"]}
    )


def _run(store, sent, outcomes=None):
    results = {
        operation: WorkflowStepOutcome.succeeded(operation_id=operation.value)
        for operation in (FREEZE, MARK, STOP, QUARANTINE, RESTORE, SUPPORT, VALIDATE)
    }
    results.update(outcomes or {})
    executor = active_workflow_executor(
        store, [FakeAdapter(results)], list(results), notification_sender=sent.append
    )
    return execute_workflow(executor, WORKFLOW)


def test_the_notice_names_a_job_the_quarantine_stopped_and_nothing_restarts():
    store = build_store()
    sent: list[str] = []
    _quarantine_plan(
        store,
        [
            workflow_step(FREEZE),
            workflow_step(MARK),
            workflow_step(STOP, workload_ids=[WORKLOAD]),
            _terminal_quarantine(),
        ],
    )
    receipt = contained_stop_receipt(
        workflow_id=WORKFLOW,
        workloads=[
            {
                "workload_id": WORKLOAD,
                "namespace": "training",
                "kind": "pytorchjob",
                "name": "train",
                "uid": "train-uid",
                "attempt_id": "attempt-1",
                "owners": [],
            }
        ],
    )
    stopped = WorkflowStepOutcome.succeeded(
        operation_id="stop", details={STOP_OWNERSHIP_RECEIPT_KEY: receipt}
    )

    result = _run(store, sent, {STOP: stopped})

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert store.get_incident(INCIDENT).state is IncidentState.QUARANTINED
    [notice] = store.list_notifications()
    assert (
        "  - workload training/pytorchjob/train stays suspended (stopped by "
        f"workflow {WORKFLOW}): the restore does not restart it; its owner must "
        "restart it"
    ) in notice.body_text, notice.body_text
    assert sent == [notice.notification_id]


def _held_decision(request_id: str = WORKFLOW) -> TerminalDecision:
    """The terminal records of a QUARANTINE plan that ended holding node-a."""

    steps = [workflow_step(FREEZE), workflow_step(MARK), _terminal_quarantine()]
    return TerminalDecision(
        workflow=workflow_request(
            request_id,
            INCIDENT,
            status=WorkflowStatus.SUCCEEDED,
            official_steps=steps,
            completed_operations=[FREEZE, MARK, QUARANTINE],
            completed_step_indexes=[0, 1, 2],
            created_at=NOW,
            updated_at=NOW,
        ),
        incident=fault_incident(
            INCIDENT,
            "host-node-a-bmc",
            event_type="NODE_HEALTH",
            policy_source="SITE_NODE_HEALTH",
            state=IncidentState.QUARANTINED,
            workflow_request_id=request_id,
            reasons=[f"BMC reports 2 critical sensors: {ALERTS}"],
        ),
        error=None,
        diagnostic_inconclusive=False,
    )


def _executor(store, sent):
    """What ``after_terminal_write`` reads of the executor."""

    return SimpleNamespace(
        store=store,
        notification_sender=sent.append,
        config=SimpleNamespace(executor_id="executor-a"),
    )


def test_a_second_terminal_write_of_the_incident_sends_nothing_new():
    """One notice per incident: a re-run step's second terminal write, or a
    second record of the incident ending on the same hold, finds it there."""

    store = build_store()
    sent: list[str] = []
    executor = _executor(store, sent)

    after_terminal_write(executor, _held_decision(), None, True)
    [first] = store.list_notifications()
    after_terminal_write(executor, _held_decision(), None, True)
    after_terminal_write(executor, _held_decision("workflow-rerun"), None, True)

    assert store.list_notifications() == [first]
    assert set(sent) == {first.notification_id}, sent
    assert first.subject == SUBJECT
    assert "Steps run: FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> QUARANTINE" in (
        first.body_text
    )


def test_an_incident_this_record_did_not_write_gets_no_notice():
    """A stronger successor took the incident's pointer: its end speaks."""

    store = build_store()
    sent: list[str] = []

    after_terminal_write(_executor(store, sent), _held_decision(), None, False)

    assert store.list_notifications() == []
    assert sent == []


@pytest.mark.parametrize(
    "steps,outcomes,state",
    [
        (
            [workflow_step(FREEZE), workflow_step(MARK), workflow_step(QUARANTINE)]
            + [workflow_step(RESTORE)],
            {},
            IncidentState.RECOVERED,
        ),
        (
            [workflow_step(FREEZE), workflow_step(MARK), _terminal_quarantine()]
            + [workflow_step(SUPPORT)],
            {},
            IncidentState.ESCALATED,
        ),
        (
            [workflow_step(FREEZE), workflow_step(MARK), _terminal_quarantine()],
            {QUARANTINE: WorkflowStepOutcome.failed("taint patch refused")},
            IncidentState.QUARANTINED,
        ),
    ],
    ids=["restored", "support-hand-off", "failed-quarantine"],
)
def test_every_other_ending_sends_no_quarantined_node_notice(steps, outcomes, state):
    store = build_store()
    sent: list[str] = []
    _quarantine_plan(store, steps)

    _run(store, sent, outcomes)

    assert store.get_incident(INCIDENT).state is state
    assert _notices(store) == [], store.list_notifications()


def test_a_hold_its_predecessor_left_is_not_this_records_quarantine():
    """``predecessor_holds_isolation`` keeps the incident QUARANTINED behind a
    successor that held nothing itself: the hold is the predecessor's."""

    store = build_store()
    sent: list[str] = []
    store.save_workflow(
        workflow_request(
            "workflow-predecessor",
            INCIDENT,
            status=WorkflowStatus.FAILED,
            official_steps=[workflow_step(MARK)],
            completed_operations=[MARK],
            completed_step_indexes=[0],
            created_at=NOW,
            updated_at=NOW,
        )
    )
    _quarantine_plan(store, [workflow_step(FREEZE), workflow_step(VALIDATE)])
    store.save_workflow(
        store.get_workflow(WORKFLOW).model_copy(
            update={"predecessor_workflow_id": "workflow-predecessor"}
        )
    )

    result = _run(store, sent)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert store.get_incident(INCIDENT).state is IncidentState.QUARANTINED
    assert _notices(store) == [], store.list_notifications()
    assert sent == []
