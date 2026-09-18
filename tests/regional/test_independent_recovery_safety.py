"""Exercise CPU restore and DESTR-014 with the same uncertain persisted records."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation as OP
from gpu_fault.models import WorkflowStatus, WorkflowStepStatus
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from scripts.e2e.regional.collector_recovery_safety import (
    require_bound_refresh,
    require_settled_recovery,
)
from scripts.e2e.regional.destr014_verdicts import recovery_cleanup_hold
from scripts.e2e.regional.warm_spare_fixture import CREATE_RESTORE_WORKFLOW
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution.test_recovery_safety_evidence import cancelled_command
from tests.regional._cov95_destr_exhaustion import ExhaustionHarness
from tests.regional.test_acceptance_reboot_hold_alignment import happy_workflow

UNKNOWN_DETAILS = [
    {"reset_outcome_unknown": ["GPU-a"]},
    {"node_results": {"node-a": {"details": {"outcome_unknown": True}}}},
    {"node_failure_details": {"node-a": {"outcome_unknown": True}}},
    {
        BATCHED_RESULTS_KEY: {
            "0": {"status": "FAILED", "details": {"outcome_unknown": True}}
        }
    },
]


def test_recovery_refresh_cannot_exchange_the_injection_identity():
    with pytest.raises(RuntimeError, match="marker identity changed"):
        require_bound_refresh(
            {"seed_marker": "owned"}, {"seed_marker": "other"}, "owned"
        )


@pytest.mark.parametrize("details", UNKNOWN_DETAILS)
def test_cpu_and_both_cleanup_entries_refuse_the_same_uncertain_workflow(
    monkeypatch, details
):
    store = build_store()
    incident = fault_incident("incident", "event", workflow_request_id="workflow")
    workflow = workflow_request(
        "workflow",
        "incident",
        status=WorkflowStatus.FAILED,
        official_steps=[workflow_step(OP.RESET_GPU)],
        step_executions=[
            workflow_step_execution(
                0, OP.RESET_GPU, WorkflowStepStatus.FAILED, details=details
            )
        ],
    )
    store.save_incident_and_workflow(incident, workflow)
    state = {"workflows": [workflow.model_dump(mode="json")], "commands": []}
    with pytest.raises(RuntimeError, match="unresolved"):
        require_settled_recovery(state)
    assert recovery_cleanup_hold({"workflow": state["workflows"][0], "commands": []}), (
        "cleanup must retain the same unresolved workflow hold as the CPU"
    )
    wake = Mock()
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: SimpleNamespace(store=store, dispatcher=SimpleNamespace(wake=wake)),
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "incident", "node-a", "profile", "review"]
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="operator reconciliation"):
            exec(compile(CREATE_RESTORE_WORKFLOW, "<cpu-restore>", "exec"), {})
    assert len(store.list_workflows()) == 1
    wake.assert_not_called()


@pytest.mark.parametrize("proof", ["none", "never-leased", "complete", "no-start"])
def test_cpu_cancellation_refresh_never_substitutes_for_actual_completion(
    monkeypatch, proof
):
    store = build_store()
    incident = fault_incident(
        "incident", "event", workflow_request_id="workflow", fencing_token=3
    )
    step = workflow_step(OP.RESET_GPU)
    workflow = workflow_request(
        "workflow", "incident", status=WorkflowStatus.FAILED, official_steps=[step]
    )
    store.save_incident_and_workflow(incident, workflow)
    details = (
        {"post_cancellation_status": "FAILED"}
        if proof == "complete"
        else {"node_action_not_started": True}
        if proof == "no-start"
        else {}
    )
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id="command",
            cluster_id=incident.cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=3,
            idempotency_key="owned",
            step=step,
            workflow=workflow,
            incident=incident,
            status=RemoteCommandStatus.FAILED,
            status_source="workflow-timeout",
            last_lease_owner=None if proof == "never-leased" else "executor",
            result_details=details,
        )
    )
    wake = Mock()
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: SimpleNamespace(store=store, dispatcher=SimpleNamespace(wake=wake)),
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "incident", "node-a", "profile", "review"]
    )
    with redirect_stdout(io.StringIO()):
        if proof == "none":
            with pytest.raises(RuntimeError, match="operator reconciliation"):
                exec(compile(CREATE_RESTORE_WORKFLOW, "<cpu-restore>", "exec"), {})
        else:
            exec(compile(CREATE_RESTORE_WORKFLOW, "<cpu-restore>", "exec"), {})
    assert wake.called is (proof != "none")
    assert len(store.list_workflows()) == (1 if proof == "none" else 2)


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "SUPERSEDED"])
def test_destr014_sql_terminal_state_does_not_resolve_a_cancelled_command(status):
    workflow = happy_workflow()
    workflow["status"] = status
    assert recovery_cleanup_hold(
        {"workflow": workflow, "commands": [cancelled_command()]}
    ), "a terminal workflow label must not settle an unproven cancelled command"


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "SUPERSEDED"])
def test_destr014_retained_hold_survives_terminal_reads_and_cleanup_retry(
    tmp_path, monkeypatch, status
):
    harness = ExhaustionHarness(tmp_path, monkeypatch)
    harness.workflow["status"] = status
    harness.plan(tmp_path)
    _, report = harness.execute(tmp_path)
    assert report["cleanup"]["operator_hold_preserved"]
    assert report["cleanup"]["workload_cleanup_deferred"]
    names = {name for name, _ in harness.calls}
    assert not {"workload.delete", "executor.close", "control.close"} & names
    saved = json.loads(harness.journal_path.read_text())
    assert saved["run"]["physical_outcome_unknown"] is True
    _, repeated = harness.execute(tmp_path)
    assert repeated["cleanup"]["operator_hold_preserved"]
    assert repeated["cleanup"]["workload_cleanup_deferred"]
