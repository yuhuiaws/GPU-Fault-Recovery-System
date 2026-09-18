from __future__ import annotations

import pytest

from gpu_fault.models import RecoveryAction, RecoveryPlan, WorkflowOperation
from gpu_fault.passive import PassiveWorkflowCompiler
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident


@pytest.fixture
def plan(failed_event):
    return RecoveryPlan(
        plan_id="plan-passive-replay",
        incident_id="incident-passive-replay",
        attempt_id=failed_event.attempt_id,
        trigger="terminal-recovery",
        runtime_profile_version=failed_event.runtime_profile_version,
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a", "node-b"],
                "execution_owner": "simulated-runtime",
            }
        ],
    )


def test_linked_passive_plan_is_a_read_only_replay(plan, failed_event):
    store = InMemoryStore()
    compiler = PassiveWorkflowCompiler(store)
    linked = compiler.compile(plan, failed_event)
    original = store.get_workflow(linked.workflow_request_id)
    replayed = compiler.compile(linked, failed_event)
    assert replayed == linked, replayed
    assert store.list_workflows() == [original], store.list_workflows()


def test_existing_incident_gets_a_distinct_idempotent_completion_successor(
    plan, failed_event
):
    store = InMemoryStore()
    original = fault_incident(plan.incident_id, "original-fault")
    store.save_incident(original)
    compiler = PassiveWorkflowCompiler(store)
    linked = compiler.compile(
        plan, failed_event, predecessor_workflow_id="containment-workflow"
    )
    replay = compiler.compile(plan, failed_event)
    assert replay == linked, (linked, replay)
    assert linked.incident_id != original.incident_id, linked
    assert store.get_incident(original.incident_id) == original, (
        "completion must not overwrite the original fault"
    )
    workflow = store.get_workflow(linked.workflow_request_id)
    assert workflow.source_plan_id == plan.plan_id, workflow
    assert workflow.predecessor_workflow_id == "containment-workflow", workflow
    assert len(store.list_workflows()) == 1, store.list_workflows()
    derived = store.get_incident(linked.incident_id)
    assert derived.event_id == f"{failed_event.event_key}/{plan.plan_id}", derived
    assert (derived.cluster_id, derived.job_id, derived.attempt_id) == (
        failed_event.cluster_id,
        failed_event.job_id,
        failed_event.attempt_id,
    ), derived


@pytest.mark.parametrize("corruption", ["missing-pointer", "other-plan"])
def test_passive_replay_refuses_incomplete_or_conflicting_derived_identity(
    plan, failed_event, corruption
):
    store = InMemoryStore()
    store.save_incident(fault_incident(plan.incident_id, "original-fault"))
    compiler = PassiveWorkflowCompiler(store)
    linked = compiler.compile(plan, failed_event)
    if corruption == "missing-pointer":
        incident = store.get_incident(linked.incident_id)
        store.save_incident(
            incident.model_copy(update={"workflow_request_id": None}), expected=incident
        )
        message = "no workflow pointer"
    else:
        workflow = store.get_workflow(linked.workflow_request_id)
        store.save_workflow(
            workflow.model_copy(update={"source_plan_id": "a-different-plan"}),
            expected=workflow,
        )
        message = "belongs to another plan"
    before = store.list_workflows()
    with pytest.raises(RuntimeError, match=message):
        compiler.compile(plan, failed_event)
    assert store.list_workflows() == before, (
        "ambiguous derived identity must not create or amend recovery"
    )


def test_empty_passive_plan_does_not_invent_an_action(plan, failed_event):
    store = InMemoryStore()
    linked = PassiveWorkflowCompiler(store).compile(
        plan.model_copy(update={"steps": []}), failed_event
    )
    workflow = store.get_workflow(linked.workflow_request_id)
    incident = store.get_incident(linked.incident_id)
    assert workflow.official_steps == [] and workflow.official_action is None, workflow
    assert incident.official_action is None and incident.effective_action is None, (
        incident
    )


def test_passive_evidence_uses_its_explicit_owner_without_workload_mutation(
    plan, failed_event
):
    store = InMemoryStore()
    evidence = plan.steps[0].model_copy(
        update={"action": RecoveryAction.COLLECT_EVIDENCE}
    )
    linked = PassiveWorkflowCompiler(store, evidence_owner="unit-evidence").compile(
        plan.model_copy(update={"steps": [evidence]}), failed_event
    )
    workflow = store.get_workflow(linked.workflow_request_id)
    assert len(workflow.official_steps) == 1, workflow
    step = workflow.official_steps[0]
    assert step.operation is WorkflowOperation.FREEZE_EVIDENCE, step
    assert step.execution_owner == "unit-evidence" and step.workload_ids == [], step
