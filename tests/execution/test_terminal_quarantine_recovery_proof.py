"""Readmission proof is phase-bound; spare recovery never releases its original."""

from __future__ import annotations

from copy import deepcopy

import pytest

from gpu_fault.execution.terminal_state import terminal_incident_state
from gpu_fault.models import IncidentState, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.workflow_quarantine import (
    TERMINAL_QUARANTINE_NODES,
    has_terminal_quarantine_hold,
    has_unrecovered_quarantine,
)
from tests._builders import fault_incident, workflow_step_execution
from tests.completion.test_restart_execution_premise import BatchApi, restart_context
from tests.completion.test_restart_execution_premise import adapter as restart_adapter
from tests.execution.test_terminal_quarantine_ownership_alignment import (
    ownership_report,
)
from tests.execution.test_terminal_quarantine_rebinding import (
    OP,
    OwnershipProvider,
    RecordingAdapter,
    RecordingCore,
    context_for,
    execute_plan,
    isolated_node,
    plan,
    recovery,
    scheduler,
    step,
)


def partial_plan(*, safety=False, indexes=()):
    steps = [
        step(OP.QUARANTINE, ("node-b",)),
        step(OP.RESTORE_SCHEDULING),
        step(OP.RESTORE_SCHEDULING, ("node-b",)),
    ]
    phase = "safety" if safety else "official"
    return plan(
        [] if safety else steps,
        safety_steps=steps if safety else [],
        safety_only=safety,
        status=WorkflowStatus.SUCCEEDED,
        completed_step_indexes=list(indexes),
        completed_operations=[OP.QUARANTINE, OP.RESTORE_SCHEDULING],
        step_executions=[
            workflow_step_execution(0, OP.QUARANTINE, phase=phase),
            workflow_step_execution(1, OP.RESTORE_SCHEDULING, phase=phase),
        ],
    )


@pytest.mark.parametrize("safety", [False, True])
@pytest.mark.parametrize("indexes", [(), (0,), (1,)])
def test_partial_indexes_cannot_erase_phase_bound_quarantine(safety, indexes):
    workflow = partial_plan(safety=safety, indexes=indexes)
    assert has_terminal_quarantine_hold(workflow), (
        "partial completion indexes must not release the quarantined node"
    )
    assert has_unrecovered_quarantine(workflow), (
        "an unrestored node must keep the incident unrecovered"
    )
    assert terminal_incident_state(workflow, WorkflowStatus.SUCCEEDED) is (
        IncidentState.QUARANTINED
    )


@pytest.mark.parametrize("local", [False, True], ids=["regional", "local"])
@pytest.mark.parametrize("preserving", [False, True])
def test_partial_legacy_ownership_refuses_readmission_but_allows_preserving_successor(
    local, preserving
):
    old = partial_plan()
    store = InMemoryStore()
    store.save_incident(
        fault_incident(
            old.incident_id,
            "event",
            workflow_request_id=old.request_id,
            node_ids=["node-a", "node-b"],
            state=IncidentState.RECOVERED,
        )
    )
    store.save_workflow(old)
    report = ownership_report(store, old.incident_id)
    assert report.quarantine_hold, "the API must not trust an old RECOVERED label"
    candidate = plan(
        [
            step(OP.MARK_UNSCHEDULABLE, ("node-b",)),
            step(OP.QUARANTINE if preserving else OP.RESTORE_SCHEDULING, ("node-b",)),
        ]
    ).model_copy(
        update={"request_id": "successor", "incident_id": "successor-incident"}
    )
    core = RecordingCore({"node-b": isolated_node("node-b")})
    before = deepcopy(core.nodes)
    subject = scheduler(
        core,
        store=store if local else None,
        provider=None if local else OwnershipProvider(report),
    )
    outcome = subject.execute(context_for(candidate, 0))
    assert (outcome.status is WorkflowStepStatus.SUCCEEDED) is preserving
    assert core.nodes["node-b"]["spec"]["unschedulable"] is True
    if not preserving:
        assert core.nodes == before and not core.patches


@pytest.mark.parametrize("safety", [False, True])
@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing",
        "unphased",
        "wrong-phase",
        "waiting",
        "failed",
        "unknown",
        "retired",
        "wrong-operation",
        "wrong-index",
        "old-success",
        "wrong-node",
    ],
)
def test_only_matching_completed_phase_can_release_the_held_node(safety, defect):
    workflow = partial_plan(safety=safety, indexes=(0, 1, 2))
    phase = "safety" if safety else "official"
    receipt = workflow_step_execution(2, OP.RESTORE_SCHEDULING, phase=phase)
    if defect != "missing":
        workflow.step_executions.append(receipt)
    if defect == "unphased":
        receipt.phase = None
    elif defect == "wrong-phase":
        receipt.phase = "official" if safety else "safety"
    elif defect in {"waiting", "failed"}:
        receipt.status = WorkflowStepStatus(defect.upper())
    elif defect == "unknown":
        receipt.details = {"outcome_unknown": True}
    elif defect == "retired":
        workflow.superseded_step_indexes = [2]
    elif defect == "wrong-operation":
        receipt.operation = OP.RESTORE_GPU_SERVICES
    elif defect == "wrong-index":
        receipt.step_index = 99
    elif defect == "old-success":
        workflow.step_executions.append(receipt.model_copy(update={"phase": None}))
    elif defect == "wrong-node":
        steps = workflow.safety_steps if safety else workflow.official_steps
        steps[2].node_ids = ["node-a"]
    assert has_terminal_quarantine_hold(workflow) is (defect != "none")


@pytest.mark.parametrize(
    "defect", ["no-steps", "bad-index", "wrong-phase", "empty-scope"]
)
def test_unbound_quarantine_evidence_remains_held(defect):
    workflow = partial_plan()
    if defect == "no-steps":
        workflow.official_steps = []
    elif defect == "bad-index":
        workflow.step_executions[0].step_index = 99
    elif defect == "wrong-phase":
        workflow.step_executions[0].phase = "safety"
    else:
        workflow.official_steps[0].node_ids = []
    assert has_terminal_quarantine_hold(workflow), (
        "unbound quarantine evidence must retain the node ownership hold"
    )
    assert has_unrecovered_quarantine(workflow), (
        "unbound quarantine evidence must not prove incident recovery"
    )


def test_unexecuted_safety_plan_does_not_override_proven_official_release():
    workflow = partial_plan()
    workflow.step_executions.append(
        workflow_step_execution(2, OP.RESTORE_SCHEDULING, phase="official")
    )
    workflow.safety_steps = [step(OP.QUARANTINE, ("other-node",))]
    assert not has_terminal_quarantine_hold(workflow), (
        "an unexecuted safety plan must not undo proven official readmission"
    )


def test_an_official_release_cannot_discharge_executed_safety_quarantine():
    workflow = partial_plan()
    workflow.safety_steps = [step(OP.QUARANTINE, ("other-node",))]
    workflow.step_executions.extend(
        [
            workflow_step_execution(2, OP.RESTORE_SCHEDULING, phase="official"),
            workflow_step_execution(0, OP.QUARANTINE, phase="safety"),
        ]
    )
    assert has_terminal_quarantine_hold(workflow), (
        "official readmission must not discharge executed safety quarantine"
    )


def replacement_run(*, dag=False, explicit=False, mixed=False):
    workflow = recovery(dag=dag)
    workflow.official_steps.pop()
    quarantine = next(
        s for s in workflow.official_steps if s.operation is OP.QUARANTINE
    )
    if explicit:
        quarantine.parameters[TERMINAL_QUARANTINE_NODES] = ["node-a"]
    if mixed:
        workflow.official_steps.append(step(OP.QUARANTINE, ("node-b",)))
    core = RecordingCore(
        {node: isolated_node(node) for node in ("node-a", "node-b", "spare-a")}
    )
    result, store = execute_plan(
        workflow,
        core,
        RecordingAdapter(
            {"action": "SPARE_FAILOVER", "node_rebindings": {"node-a": "spare-a"}}
        ),
    )
    assert result.status is WorkflowStatus.SUCCEEDED, result
    return store.get_workflow(workflow.request_id), store, core


@pytest.mark.parametrize("dag", [False, True])
@pytest.mark.parametrize("hold", ["none", "explicit", "mixed"])
def test_successful_spare_recovery_and_original_node_ownership_are_distinct(dag, hold):
    workflow, store, core = replacement_run(
        dag=dag, explicit=hold == "explicit", mixed=hold == "mixed"
    )
    incident = store.get_incident(workflow.incident_id)
    recovered = hold == "none"
    assert (incident.state is IncidentState.RECOVERED) is recovered
    assert core.nodes["node-a"]["spec"]["unschedulable"] is True
    assert core.nodes["spare-a"]["spec"]["unschedulable"] is False
    assert has_terminal_quarantine_hold(workflow), (
        "successful spare recovery must leave the original node quarantined"
    )
    assert ownership_report(store, incident.incident_id).quarantine_hold, (
        "the ownership API must retain the original node's quarantine hold"
    )
    batch = BatchApi()
    outcome = restart_adapter(batch, store).execute(
        restart_context(
            {
                "requires_incident_state": "RECOVERED",
                "incident_id": incident.incident_id,
                "avoid_node_ids": ["node-a", "node-b"],
            }
        )
    )
    assert (outcome.status is WorkflowStepStatus.SUCCEEDED) is recovered
    assert len(batch.created) == int(recovered)


@pytest.mark.parametrize(
    "defect",
    [
        "missing-receipt",
        "replacement-failed",
        "replacement-phase",
        "replacement-unknown",
        "missing-action",
        "missing-binding",
        "empty-binding",
        "wrong-source",
        "same-node",
        "empty-target",
        "invalid-target",
        "unvalidated",
        "validation-failed",
        "validation-phase",
        "unrestored",
        "restore-failed",
        "restore-phase",
        "retired-validation",
        "no-dependency",
        "pending",
        "superseded",
        "safety",
        "malformed-hold",
    ],
)
def test_spare_success_label_without_complete_rebound_readmission_is_insufficient(
    defect,
):
    workflow, _, _ = replacement_run(dag=True)
    replacement = next(
        e for e in workflow.step_executions if e.operation is OP.REPLACE_NODE
    )
    validation = next(
        e for e in workflow.step_executions if e.operation is OP.VALIDATE_GPU
    )
    restore = next(
        e for e in workflow.step_executions if e.operation is OP.RESTORE_SCHEDULING
    )
    if defect == "missing-receipt":
        workflow.step_executions.remove(replacement)
    elif defect in {"replacement-failed", "validation-failed", "restore-failed"}:
        {"replacement": replacement, "validation": validation, "restore": restore}[
            defect.split("-")[0]
        ].status = WorkflowStepStatus.FAILED
    elif defect in {"replacement-phase", "validation-phase", "restore-phase"}:
        {"replacement": replacement, "validation": validation, "restore": restore}[
            defect.split("-")[0]
        ].phase = None
    elif defect == "replacement-unknown":
        replacement.details["node_results"] = {"node-a": {"outcome_unknown": True}}
    elif defect == "missing-action":
        replacement.details.pop("action")
    elif defect in {
        "missing-binding",
        "empty-binding",
        "wrong-source",
        "same-node",
        "empty-target",
        "invalid-target",
    }:
        replacement.details["node_rebindings"] = {
            "missing-binding": None,
            "empty-binding": {},
            "wrong-source": {"other": "spare-a"},
            "same-node": {"node-a": "node-a"},
            "empty-target": {"node-a": ""},
            "invalid-target": {"node-a": []},
        }[defect]
    elif defect in {"unvalidated", "unrestored"}:
        workflow.step_executions.remove(
            validation if defect == "unvalidated" else restore
        )
    elif defect == "retired-validation":
        workflow.superseded_step_indexes = [validation.step_index]
    elif defect == "no-dependency":
        workflow.official_steps[restore.step_index].depends_on_step_indexes = []
    elif defect in {"pending", "superseded"}:
        workflow.status = WorkflowStatus(defect.upper())
    elif defect == "safety":
        workflow.safety_only = True
    else:
        next(
            s for s in workflow.official_steps if s.operation is OP.QUARANTINE
        ).parameters[TERMINAL_QUARANTINE_NODES] = "node-a"
    assert has_unrecovered_quarantine(workflow), (
        "incomplete rebound readmission must not prove incident recovery"
    )
    assert has_terminal_quarantine_hold(workflow), (
        "incomplete rebound readmission must retain the node ownership hold"
    )


@pytest.mark.parametrize("later_phase", ["official", "safety"])
def test_a_replacement_does_not_discharge_later_or_unordered_quarantine(later_phase):
    workflow, _, _ = replacement_run(dag=True)
    if later_phase == "official":
        index = len(workflow.official_steps)
        workflow.official_steps.append(step(OP.QUARANTINE, dependencies=(index - 1,)))
    else:
        index = 0
        workflow.safety_steps = [step(OP.QUARANTINE)]
    workflow.step_executions.append(
        workflow_step_execution(index, OP.QUARANTINE, phase=later_phase)
    )
    assert has_unrecovered_quarantine(workflow), (
        "later or unordered quarantine must prevent incident recovery"
    )
    assert has_terminal_quarantine_hold(workflow), (
        "replacement must not release a later or unordered quarantine hold"
    )


@pytest.mark.parametrize("defect", ["duplicate-target", "missing-second-readmission"])
def test_multi_node_binding_requires_every_unique_spare_to_be_readmitted(defect):
    workflow, _, _ = replacement_run()
    replacement = next(
        item for item in workflow.step_executions if item.operation is OP.REPLACE_NODE
    )
    workflow.official_steps[replacement.step_index].node_ids.append("node-b")
    replacement.details["node_rebindings"]["node-b"] = (
        "spare-a" if defect == "duplicate-target" else "spare-b"
    )
    assert has_unrecovered_quarantine(workflow), (
        "partial or aliased spare recovery must not recover the original incident"
    )


def test_terminal_derivation_uses_requested_status_without_mutating_running_input():
    workflow, _, _ = replacement_run()
    workflow.status = WorkflowStatus.RUNNING
    assert terminal_incident_state(workflow, WorkflowStatus.SUCCEEDED) is (
        IncidentState.RECOVERED
    )
    assert terminal_incident_state(workflow, WorkflowStatus.SUPERSEDED) is (
        IncidentState.QUARANTINED
    )
    assert workflow.status is WorkflowStatus.RUNNING
