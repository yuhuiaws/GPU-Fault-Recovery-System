from __future__ import annotations

import pytest

from gpu_fault.models import CapabilityName, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import (
    container_observation,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)
from tests.orchestration._cov95_orch_extra_support import (
    NOW,
    memory_store,
    observation,
    stored_workflow,
)

REBOOT = WorkflowOperation.RESTART_NODE
RESTART = WorkflowOperation.RESTART_WORKLOAD
STOP = WorkflowOperation.STOP_WORKLOADS
FREEZE = WorkflowOperation.FREEZE_EVIDENCE


@pytest.mark.parametrize("nodes", ["empty", "terminated", "unknown"])
def test_an_attempt_without_live_node_allocation_does_not_open_a_hold(nodes):
    store = memory_store()
    stored_workflow(store, [REBOOT])
    containers = []
    if nodes == "terminated":
        containers = [
            container_observation(
                "pod",
                "worker",
                0,
                "node-0",
                terminated=True,
                exit_code=0,
                finished_at=NOW,
            )
        ]
    elif nodes == "unknown":
        containers = [container_observation("pod", "worker", 0, None)]
    orchestrator = IncidentOrchestrator(store)
    assert (
        orchestrator.hold_attempt_on_repairing_nodes(observation(containers=containers))
        is None
    )
    assert orchestrator.placement_holds_opened_total == 0
    assert len(store.list_workflows()) == 1


@pytest.mark.parametrize("profile", ["missing", "no-stop-owner"])
def test_placement_hold_refuses_an_uncompilable_stop_instead_of_minting_a_workflow(
    profile, caplog
):
    store = memory_store()
    stored_workflow(store, [REBOOT])
    if profile == "no-stop-owner":
        resolved = store.get_profile("simulated-v1")
        store.save_profile(
            resolved.model_copy(
                update={
                    "capabilities": [
                        item
                        for item in resolved.capabilities
                        if item.capability is not CapabilityName.WORKLOAD_STOP
                    ]
                }
            )
        )
    orchestrator = IncidentOrchestrator(store)
    held = orchestrator.hold_attempt_on_repairing_nodes(
        observation(
            runtime_profile_version="missing"
            if profile == "missing"
            else "simulated-v1"
        )
    )
    assert held is None
    assert len(store.list_workflows()) == 1
    assert store.list_active_workflow_incidents("cluster-a", job_id="unit-job") == []
    assert any("not opened" in record.getMessage() for record in caplog.records), (
        "the rejected hold lost its operator diagnostic"
    )


@pytest.mark.parametrize("restart_marker", ["unit-attempt", "another-attempt", None])
def test_existing_job_workflow_is_matched_to_its_observed_restart_attempt(
    restart_marker,
):
    store = memory_store()
    stored_workflow(store, [REBOOT])
    history = [
        workflow_step_execution(
            0,
            RESTART,
            details={"restart_attempt_id": restart_marker} if restart_marker else {},
        ),
        workflow_step_execution(1, FREEZE),
    ]
    stored_workflow(
        store,
        [RESTART, FREEZE],
        identity="job-recovery",
        incident_updates={"job_id": "unit-job", "attempt_id": "old-attempt"},
        workflow_updates={"step_executions": history},
    )
    orchestrator = IncidentOrchestrator(store)
    held = orchestrator.hold_attempt_on_repairing_nodes(observation())
    if restart_marker == "unit-attempt":
        assert held is None
        assert len(store.list_workflows()) == 2
    else:
        assert held is not None, (
            "a different attempt's recovery suppressed the new hold"
        )
        assert held[0].attempt_id == "unit-attempt"
        assert held[1].official_steps[0].operation is STOP
        assert len(store.list_workflows()) == 3


def test_atomic_create_race_keeps_the_winning_hold_without_reporting_a_second_open(
    monkeypatch,
):
    store = memory_store()
    stored_workflow(store, [REBOOT])
    original = store.create_incident_workflow_if_absent
    results = []

    def raced(event_id, builder, **kwargs):
        winner = original(event_id, builder, **kwargs)
        loser = original(event_id, builder, **kwargs)
        results.extend([winner, loser])
        return loser

    monkeypatch.setattr(store, "create_incident_workflow_if_absent", raced)
    orchestrator = IncidentOrchestrator(store)
    assert orchestrator.hold_attempt_on_repairing_nodes(observation()) is None
    assert [result[2] for result in results] == [True, False]
    assert results[0][:2] == results[1][:2]
    assert orchestrator.placement_holds_opened_total == 0
    assert (
        len(store.list_active_workflow_incidents("cluster-a", job_id="unit-job")) == 1
    )
    assert len(store.list_workflows()) == 2


def test_safety_plan_repairs_hold_the_whole_attempt_and_keep_a_timezone_bound_clock():
    store = memory_store()
    stored_workflow(
        store,
        [FREEZE],
        workflow_updates={
            "status": WorkflowStatus.SAFETY_PENDING,
            "safety_steps": [workflow_step(REBOOT, node_ids=["node-0"])],
        },
    )
    observed = observation(observed_at=NOW.replace(tzinfo=None), workload_ids=[])
    held = IncidentOrchestrator(store).hold_attempt_on_repairing_nodes(observed)
    assert held is not None, "an unresolved safety-plan repair was ignored"
    incident, workflow = held
    assert workflow.created_at == NOW
    assert incident.created_at == NOW
    assert workflow.official_steps[0].workload_ids == ["unit-job"]
    assert workflow.official_steps[0].node_ids == ["node-0", "node-1"]
    assert workflow.official_steps[0].parameters == {
        "termination_initiator_incident_id": incident.incident_id
    }


def test_hold_identity_cannot_collide_when_cluster_and_attempt_separators_are_ambiguous():
    store = memory_store()
    orchestrator = IncidentOrchestrator(store)
    ids = []
    for index, (cluster, attempt) in enumerate(
        [("cluster-a", "unit-attempt"), ("cluster", "a-unit-attempt")]
    ):
        stored_workflow(
            store,
            [REBOOT],
            identity=f"repair-{index}",
            incident_updates={"cluster_id": cluster},
        )
        held = orchestrator.hold_attempt_on_repairing_nodes(
            observation(cluster_id=cluster, attempt_id=attempt)
        )
        assert held is not None, (
            "a different cluster/attempt consumed this placement hold's deterministic ID"
        )
        assert held[0].cluster_id == cluster
        assert held[0].attempt_id == attempt
        ids.append(held[0].incident_id)
    assert len(set(ids)) == 2, (
        "separate cluster-scoped attempts shared a placement hold"
    )
    assert orchestrator.placement_holds_opened_total == 2
