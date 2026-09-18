from __future__ import annotations

import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.compile_blocked import (
    compile_blocked_reasons,
    settled_incident_blocked_reasons,
)
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

PROBE = Path(__file__).resolve().parents[2] / (
    "deploy/control-plane/regional/probes/workflow_safety.py"
)


def probe_report(monkeypatch, capsys, workflow, incident, commands=(), successor=None):
    def get_workflow(request_id):
        if successor is not None and successor.request_id == request_id:
            return successor
        raise KeyError(request_id)

    store = SimpleNamespace(
        list_workflows=lambda **_kwargs: [workflow],
        get_incident=lambda _incident_id: incident,
        get_workflow=get_workflow,
        list_remote_commands=lambda **_kwargs: commands,
    )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        staticmethod(lambda: SimpleNamespace(store=store)),
    )
    runpy.run_path(str(PROBE))
    return json.loads(capsys.readouterr().out)


def settled_pair():
    workflow = workflow_request(
        "workflow-a",
        "incident-a",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.SAFETY_SETTLED,
        execution_epoch=1,
        official_steps=[workflow_step(WorkflowOperation.RESET_GPU)],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.RESET_GPU, WorkflowStepStatus.FAILED
            )
        ],
    )
    incident = fault_incident(
        "incident-a",
        "event-a",
        state=IncidentState.RECOVERED,
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
    )
    return workflow, incident


def test_unresolved_physical_work_is_not_hidden_by_an_empty_official_plan(
    monkeypatch, capsys
):
    workflow, incident = settled_pair()
    workflow.official_steps = []
    workflow.step_executions[0].details["outcome_unknown"] = True
    report = probe_report(monkeypatch, capsys, workflow, incident)
    assert report["blockers"] == [workflow.request_id]
    assert report["settled_incident_blocked_count"] == 0


@pytest.mark.parametrize(
    "condition",
    [
        "settled",
        "owner",
        "empty-owner",
        "live-lease",
        "expired-lease",
        "occupying",
        "operator-occupying",
        "waiting",
        "provider-waiting",
        "open-command",
        "source-plan",
        "outcome_unknown",
        "node_action_interrupted",
        "node_action_response_unknown",
        "ownership_permit_delivery_unknown",
        "pending",
        "manual-confirmation",
        "confirmed-not-started",
        "newer-success",
        "different-phase-success",
        "old-waiting-new-success",
        "withheld-restoration",
    ],
)
def test_inline_settled_guard_matches_runtime_behavior(monkeypatch, capsys, condition):
    workflow, incident = settled_pair()
    execution = workflow.step_executions[0]
    commands = []
    now = datetime.now(timezone.utc)
    if condition in {"owner", "empty-owner"}:
        workflow.execution_owner_id = "executor-a" if condition == "owner" else ""
    elif condition in {"live-lease", "expired-lease"}:
        workflow.execution_lease_expires_at = now + timedelta(
            minutes=1 if condition == "live-lease" else -1
        )
    elif condition in {"occupying", "operator-occupying"}:
        workflow.blocked_kind = (
            BlockedKind.INTERNAL_ERROR
            if condition == "occupying"
            else BlockedKind.NEEDS_OPERATOR
        )
    elif condition in {"waiting", "provider-waiting"}:
        execution.status = WorkflowStepStatus.WAITING
        if condition == "provider-waiting":
            execution.operation = WorkflowOperation.FREEZE_EVIDENCE
    elif condition == "open-command":
        commands = [
            SimpleNamespace(command_id="command-a", status=RemoteCommandStatus.WAITING)
        ]
    elif condition == "source-plan":
        workflow.source_plan_id = "plan-a"
    elif condition in {
        "outcome_unknown",
        "node_action_interrupted",
        "node_action_response_unknown",
        "ownership_permit_delivery_unknown",
    }:
        execution.details[condition] = True
        commands = [
            SimpleNamespace(command_id="command-a", status=RemoteCommandStatus.FAILED)
        ]
    elif condition == "pending":
        execution.details.update(
            node_action_state="PENDING", node_action_command_id="command-a"
        )
    elif condition in {"manual-confirmation", "confirmed-not-started"}:
        execution.details["manual_confirmation_required"] = True
        if condition == "confirmed-not-started":
            execution.details["node_action_not_started"] = True
    elif condition in {
        "newer-success",
        "different-phase-success",
        "old-waiting-new-success",
    }:
        execution.details["outcome_unknown"] = True
        if condition == "old-waiting-new-success":
            execution.status = WorkflowStepStatus.WAITING
        workflow.step_executions.append(
            workflow_step_execution(
                0, WorkflowOperation.RESET_GPU, WorkflowStepStatus.SUCCEEDED
            )
        )
        if condition == "different-phase-success":
            execution.phase = "official"
            workflow.step_executions[-1].phase = "safety"
    elif condition == "withheld-restoration":
        execution.operation = WorkflowOperation.RESTORE_GPU_SERVICES
        execution.details.update(
            restore_gpu_services_withheld=True, outcome_unknown=True
        )

    reasons = settled_incident_blocked_reasons(
        workflow,
        incident,
        [
            command.command_id
            for command in commands
            if command.status is RemoteCommandStatus.WAITING
        ],
        evaluated_at=now,
    )
    report = probe_report(monkeypatch, capsys, workflow, incident, commands)
    assert report["blocker_count"] == int(bool(reasons)), reasons
    assert report["settled_incident_blocked_count"] == int(not reasons), reasons


@pytest.mark.parametrize(
    "condition",
    ["owner", "live-lease", "occupying", "waiting", "unknown", "open-command"],
)
def test_restore_successor_cannot_hide_unsettled_execution(
    monkeypatch, capsys, condition
):
    workflow, incident = settled_pair()
    commands = []
    if condition == "owner":
        workflow.execution_owner_id = ""
    elif condition == "live-lease":
        workflow.execution_lease_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=1
        )
    elif condition == "occupying":
        workflow.blocked_kind = BlockedKind.NEEDS_OPERATOR
    elif condition == "waiting":
        workflow.step_executions[0].status = WorkflowStepStatus.WAITING
    elif condition == "unknown":
        workflow.step_executions[0].details["outcome_unknown"] = True
    else:
        commands = [SimpleNamespace(status=RemoteCommandStatus.PENDING)]
    successor = workflow_request(
        "successor",
        workflow.incident_id,
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=workflow.fencing_token,
        completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
    )
    incident.workflow_request_id = successor.request_id
    report = probe_report(monkeypatch, capsys, workflow, incident, commands, successor)
    assert report["blockers"] == [workflow.request_id]
    assert report["resolved_blocked_count"] == 0


@pytest.mark.parametrize(
    "condition", ["never-claimed", "empty-owner", "lease", "expired-lease"]
)
def test_compile_time_probe_requires_the_runtime_never_claimed_shape(
    monkeypatch, capsys, condition
):
    workflow, incident = settled_pair()
    workflow.execution_epoch = 0
    workflow.step_executions = []
    workflow.blocked_kind = BlockedKind.NEEDS_OPERATOR
    workflow.blocked_reasons = ["no executable owner"]
    incident.state = IncidentState.ESCALATED
    if condition == "empty-owner":
        workflow.execution_owner_id = ""
    elif condition in {"lease", "expired-lease"}:
        workflow.execution_lease_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=1 if condition == "lease" else -1
        )
    reasons = compile_blocked_reasons(workflow, incident, [])
    report = probe_report(monkeypatch, capsys, workflow, incident)
    assert report["blocker_count"] == int(bool(reasons)), reasons
    assert report["compile_blocked_count"] == int(not reasons), reasons
