"""A read-only hung diagnostic always reaches its support hand-off.

Live (HyperPod, GF-GENERAL-LIVE-011.active_to_zero, release bf1bb0037847): the
grouped hung incident's VALIDATE_FABRIC stayed WAITING until the 600 s per-step
cap and FAILED; the DAG loop then terminalized the workflow and step 7
ESCALATE_SUPPORT never executed. The incident read ESCALATED but no ticket was
drafted while the training job stayed hung. A failed evidence step now retires
only what needs its output; independent evidence and the hand-off still run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.hung_handoff import (
    HUNG_SUPPORT_HANDOFF,
    deadline_failure,
    support_handoff,
    support_handoff_follows,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.telemetry_models import TelemetryMetricLatest
from tests._builders import (
    active_workflow_executor,
    build_store,
    execute_workflow,
    workflow_step_execution,
)
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture
from tests.execution._support import FakeAdapter, workflow_state
from tests.execution.test_nccl_continuous_e2e import FORBIDDEN

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
TRIAGE = WorkflowOperation.COLLECT_HUNG_TRIAGE
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
FABRIC = WorkflowOperation.VALIDATE_FABRIC
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
#: FREEZE(0) -> TRIAGE(1) -> BUNDLE(2); FABRIC(3) after FREEZE; SUPPORT(4) last,
#: the dependency shape of ``orchestration/hung_dag.py`` in miniature.
OPERATIONS = [FREEZE, TRIAGE, BUNDLE, FABRIC, SUPPORT]
DEPENDENCIES = [[], [0], [1], [0], [2, 3]]


def contract(mode="HUNG_DIAGNOSTIC"):
    return {
        "health_recovery_contract": {
            "mode": mode,
            "metric": "efa_traffic_bytes_per_second",
            "detected_at": "2026-10-24T13:50:17+00:00",
        }
    }


def hung_dag(store, mode="HUNG_DIAGNOSTIC", operations=OPERATIONS):
    incident, workflow = workflow_state(store, operations)
    steps = [
        step.model_copy(
            update={"depends_on_step_indexes": needs, "parameters": contract(mode)}
        )
        for step, needs in zip(workflow.official_steps, DEPENDENCIES)
    ]
    current = store.get_workflow(workflow.request_id)
    store.save_workflow(
        current.model_copy(
            update={"official_steps": steps, "dag_enabled": True, "dag_revision": 1}
        ),
        expected=current,
    )
    return incident, store.get_workflow(workflow.request_id)


def run(store, outcomes, mode="HUNG_DIAGNOSTIC"):
    incident, workflow = hung_dag(store, mode)
    adapter = FakeAdapter(outcomes)
    executor = active_workflow_executor(store, [adapter], OPERATIONS)
    result = execute_workflow(executor, workflow.request_id)
    return result, store.get_workflow(workflow.request_id), adapter


def called(adapter):
    return [key.split("/")[-1] for key in adapter.calls]


def succeeded(operation):
    return WorkflowStepOutcome.succeeded(operation_id=operation.value)


def test_a_failed_evidence_step_retires_its_dependents_and_still_hands_off():
    store = build_store()
    outcomes = {operation: succeeded(operation) for operation in OPERATIONS}
    outcomes[TRIAGE] = WorkflowStepOutcome.failed("agent unavailable")
    result, saved, adapter = run(store, outcomes)
    assert result.status is WorkflowStatus.FAILED, result
    assert called(adapter) == ["FREEZE_EVIDENCE", "COLLECT_HUNG_TRIAGE"] + [
        "VALIDATE_FABRIC",
        "ESCALATE_SUPPORT",
    ], "the bundle needs the triage; fabric evidence and the hand-off do not"
    assert set(saved.superseded_step_indexes) == {1, 2}, saved
    assert saved.completed_operations == [FREEZE, FABRIC, SUPPORT], saved
    assert saved.terminal_failure_reason == "agent unavailable", saved
    incident = store.get_incident(saved.incident_id)
    assert incident.state is IncidentState.ESCALATED, incident
    assert not any("diagnostic inconclusive" in item for item in incident.reasons), (
        incident.reasons
    )
    [event] = [item for item in saved.events if item.code == HUNG_SUPPORT_HANDOFF]
    assert event.kind is WorkflowEventKind.PLAN_REWRITE, event
    assert event.step_index == 1 and event.operation is TRIAGE, event
    assert event.details["retired_step_indexes"] == [2], event
    assert event.details["handoff_step_indexes"] == [4], event


def test_a_failed_fabric_validation_keeps_the_bundle_and_hands_off():
    store = build_store()
    outcomes = {operation: succeeded(operation) for operation in OPERATIONS}
    outcomes[FABRIC] = WorkflowStepOutcome.failed(
        "validation failed on nodes: node-a (rdma_link_down:efa0/1)"
    )
    result, saved, adapter = run(store, outcomes)
    assert result.status is WorkflowStatus.FAILED, result
    assert called(adapter)[-1] == "ESCALATE_SUPPORT", called(adapter)
    assert "COLLECT_DIAGNOSTIC_BUNDLE" in called(adapter), called(adapter)
    assert saved.superseded_step_indexes == [3], saved
    assert saved.terminal_failure_reason.startswith("validation failed"), saved
    assert store.get_incident(saved.incident_id).state is IncidentState.ESCALATED, saved


@pytest.mark.parametrize("mode", ["TRAINING", None])
def test_other_diagnostics_keep_the_established_failure_path(mode):
    """Only the read-only hung DAG goes on: a TRAINING or uncontracted plan of
    the same shape ends at its first failure, as before."""

    store = build_store()
    outcomes = {operation: succeeded(operation) for operation in OPERATIONS}
    outcomes[TRIAGE] = WorkflowStepOutcome.failed("agent unavailable")
    result, saved, adapter = run(store, outcomes, mode=mode)
    assert result.status is WorkflowStatus.FAILED, result
    assert "ESCALATE_SUPPORT" not in called(adapter), called(adapter)
    assert not saved.superseded_step_indexes, saved


def test_a_failed_hand_off_ends_the_workflow():
    store = build_store()
    outcomes = {operation: succeeded(operation) for operation in OPERATIONS}
    outcomes[SUPPORT] = WorkflowStepOutcome.failed("ticket store unavailable")
    result, saved, _ = run(store, outcomes)
    assert result.status is WorkflowStatus.FAILED, result
    assert saved.terminal_failure_reason == "ticket store unavailable", saved
    assert store.get_incident(saved.incident_id).state is IncidentState.ESCALATED, saved


def test_a_mutating_step_takes_the_plan_out_of_the_hand_off_path():
    store = build_store()
    _, workflow = hung_dag(store)
    steps = list(workflow.official_steps)
    steps[2] = steps[2].model_copy(update={"operation": WorkflowOperation.RESET_GPU})
    mixed = workflow.model_copy(update={"official_steps": steps})
    outcome = WorkflowStepOutcome.failed("agent unavailable")
    assert support_handoff(mixed, 1, outcome) is None
    assert support_handoff(workflow, 1, outcome) is not None


def test_a_spent_lifetime_or_a_parked_failure_keeps_the_failure_path():
    store = build_store()
    _, workflow = hung_dag(store)
    lifetime = WorkflowStepOutcome.failed(
        "workflow lifetime exceeded", details={"workflow_lifetime_exceeded": True}
    )
    assert support_handoff(workflow, 1, lifetime) is None
    parked = workflow.model_copy(
        update={"pending_failure_step_index": 3, "pending_failure_error": "x"}
    )
    assert support_handoff(parked, 1, WorkflowStepOutcome.failed("x")) is None
    assert support_handoff(workflow, 4, WorkflowStepOutcome.failed("x")) is None


def test_past_the_deadline_nothing_new_starts_but_the_hand_off():
    store = build_store()
    _, workflow = hung_dag(store)
    waiting = workflow.model_copy(
        update={
            "completed_step_indexes": [0],
            "step_executions": [
                workflow_step_execution(1, TRIAGE, WorkflowStepStatus.WAITING),
                workflow_step_execution(3, FABRIC, WorkflowStepStatus.FAILED),
            ],
        }
    )
    expired = WorkflowStepOutcome.failed(
        "workflow execution deadline exceeded",
        details={"workflow_execution_deadline": "2026-10-24T14:00:00+00:00"},
    )
    handoff = support_handoff(waiting, 3, expired)
    assert handoff is not None
    # The in-flight triage fails on its own next attempt; the bundle never starts.
    assert handoff.workflow.superseded_step_indexes == [2, 3], handoff.workflow


def test_the_hand_off_is_owed_past_the_execution_deadline_but_not_the_lifetime():
    store = build_store()
    _, workflow = hung_dag(store)
    executor = active_workflow_executor(store, [], OPERATIONS)
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    expired = workflow.model_copy(update={"execution_deadline": past})
    steps = expired.official_steps
    assert deadline_failure(executor, expired, steps[4], 4) is None
    assert deadline_failure(executor, expired, steps[3], 3) is not None, (
        "every other step still fails on the deadline"
    )
    spent = expired.model_copy(update={"lifetime_deadline_at": past})
    failure = deadline_failure(executor, spent, steps[4], 4)
    assert failure is not None and failure.details["workflow_lifetime_exceeded"]
    ordinary = expired.model_copy(
        update={
            "official_steps": [
                step.model_copy(update={"parameters": {}}) for step in steps
            ]
        }
    )
    assert deadline_failure(executor, ordinary, steps[4], 4) is not None, (
        "an ordinary support escalation keeps the deadline"
    )


def test_a_hand_off_follows_only_steps_it_waits_on():
    store = build_store()
    _, workflow = hung_dag(store)
    assert all(support_handoff_follows(workflow, index) for index in range(4)), (
        "every evidence step reaches the hand-off"
    )
    assert not support_handoff_follows(workflow, 4), "nothing follows the hand-off"
    done = workflow.model_copy(update={"completed_step_indexes": [4]})
    assert not support_handoff_follows(done, 3), "a resolved hand-off runs no more"
    flat = workflow.model_copy(update={"dag_enabled": False})
    assert support_handoff_follows(flat, 3) and not support_handoff_follows(flat, 4)


def test_failed_task_fabric_validation_still_reaches_the_support_handoff(chain):
    """The real hung DAG, node agents and validation: node-b's task EFA port is
    down, so VALIDATE_FABRIC fails; the bundle and the hand-off still run."""

    rig = chain()
    workflow = rig.trigger()
    rig.healthy()
    # Written to the latest-metric view, not posted: a posted batch would raise
    # an RDMA finding of its own and a second workflow on node-b.
    rig.store.observe_telemetry_metrics(
        [
            TelemetryMetricLatest(
                cluster_id="cluster-a",
                node_id="node-b",
                observed_at=rig.clock + timedelta(seconds=1),
                name="rdma_link_down",
                value=1,
                device="efa0/1",
            )
        ]
    )
    result = rig.run(workflow)
    assert result.status is WorkflowStatus.FAILED, result
    saved = rig.store.get_workflow(workflow.request_id)
    final = {item.step_index: item for item in saved.step_executions}
    statuses = {
        index: (step.operation, final[index].status)
        for index, step in enumerate(saved.official_steps)
        if index in final
    }
    assert statuses[6] == (FABRIC, WorkflowStepStatus.FAILED), statuses
    assert "rdma_link_down:efa0/1" in final[6].error, final[6]
    assert statuses[5] == (BUNDLE, WorkflowStepStatus.SUCCEEDED), statuses
    assert statuses[7] == (SUPPORT, WorkflowStepStatus.SUCCEEDED), statuses
    assert SUPPORT in rig.boundary.calls, rig.boundary.calls
    assert saved.terminal_failure_reason.startswith("validation failed on nodes"), saved
    incident = rig.store.get_incident(saved.incident_id)
    assert incident.state is IncidentState.ESCALATED, incident
    assert not any("diagnostic inconclusive" in item for item in incident.reasons), (
        incident.reasons
    )
    assert not FORBIDDEN & {step.operation for step in saved.official_steps}, saved
