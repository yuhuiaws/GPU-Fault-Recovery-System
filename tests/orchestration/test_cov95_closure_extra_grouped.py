from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import RecoveryAction, WorkflowOperation
from tests._builders import container_observation
from tests.orchestration._cov95_closure_extra_grouped import (
    QUIESCE,
    RESET,
    RESTART,
    RESTORE,
    STOP,
    WORKLOAD,
    candidate_pair,
    finding,
    grouped_service,
    observed_attempt,
    seed_group,
)
from tests.orchestration._cov95_closure_extra_safety import (
    closure_extra_isolation as closure_extra_isolation,
)
from tests.orchestration._cov95_closure_extra_support import NOW
from tests.orchestration._cov95_closure_extra_support import (
    closure_store_fixture as closure_store_fixture,
)


@pytest.mark.parametrize("node", ["outside-allocation", "node-retired"])
def test_reporter_outside_the_live_allocation_cannot_create_an_attempt_recovery(
    closure_store, node
):
    observation = observed_attempt()
    observation = observation.model_copy(
        update={
            "containers": [
                *observation.containers,
                container_observation(
                    "retired-pod",
                    "retired-trainer",
                    2,
                    "node-retired",
                    terminated=True,
                    exit_code=0,
                ),
            ]
        }
    )
    harness = grouped_service(closure_store, observation, candidate_pair())
    event = finding(node_id=node)

    assert harness.service.ingest(event) is None, (
        f"reporter {node} outside the live allocation was grouped into its attempt"
    )
    assert harness.candidates == [], (
        "an ineligible reporter reached candidate compilation"
    )
    assert closure_store.get_incident_by_event(event.event_id) is None, (
        "declining an out-of-scope reporter still consumed its event identity"
    )
    assert closure_store.list_workflows() == [], (
        "an out-of-allocation recovery was persisted"
    )


@pytest.mark.parametrize("active_kind", ["missing", "node-only", "restart"])
def test_read_only_health_joins_only_a_recovery_with_an_active_job_restart(
    closure_store, active_kind
):
    observation = observed_attempt()
    active = None
    if active_kind != "missing":
        operations = (RESET,) if active_kind == "node-only" else (STOP, RESET, RESTART)
        active = seed_group(
            closure_store, observation, *candidate_pair("active", operations=operations)
        )
    harness = grouped_service(
        closure_store,
        observation,
        candidate_pair("diagnostic", operations=(WorkflowOperation.VALIDATE_GPU,)),
        active=active,
        groupable=False,
    )
    event = finding(recommended_action=RecoveryAction.RUN_DIAGNOSTICS)

    result = harness.service.ingest(event)

    if active_kind == "restart":
        assert result is not None, (
            "read-only evidence did not join an active restart recovery"
        )
        incident, workflow = result
        assert (incident.job_id, incident.attempt_id) == (
            observation.job_id,
            observation.attempt_id,
        ), "grouped diagnostics lost the live attempt identity"
        assert (
            closure_store.get_incident_by_event(event.event_id).incident_id
            == incident.incident_id
        ), "the grouped diagnostic event was not committed with its incident"
        assert workflow.request_id == incident.workflow_request_id, (
            "the grouped diagnostic's incident points at a different workflow"
        )
    else:
        assert result is None, (
            f"{active_kind} recovery unexpectedly admitted job-scoped diagnostics"
        )
        assert harness.candidates == [], (
            "an ineligible diagnostic was compiled before rejection"
        )
        assert closure_store.get_incident_by_event(event.event_id) is None, (
            "ineligible diagnostics consumed an event before normal health ingestion"
        )


def test_missing_trial_workflow_aborts_without_partial_incident_or_event_link(
    closure_store,
):
    incident, _ = candidate_pair()
    harness = grouped_service(closure_store, observed_attempt(), (incident, None))
    event = finding()
    with pytest.raises(RuntimeError, match="did not produce a workflow"):
        harness.service.ingest(event)

    assert closure_store.list_workflows() == [], (
        "a failed candidate left a workflow behind"
    )
    assert closure_store.get_incident_by_event(event.event_id) is None, (
        "failed candidate compilation consumed the incoming event"
    )
    assert closure_store.get_incident_by_event(incident.event_id) is None, (
        "failed candidate compilation persisted a partial incident"
    )


@pytest.mark.parametrize("profile", [None, "explicit-unit-profile"])
def test_grouping_scopes_workload_stop_restart_and_quiesce_proof_without_widening_the_repair(
    closure_store, profile
):
    observation = observed_attempt()
    candidate = candidate_pair()
    before = candidate[1].model_dump(mode="json")
    harness = grouped_service(closure_store, observation, candidate)
    event = finding(
        runtime_profile_version=profile,
        affected_workload_ids=[WORKLOAD, "training/job/auxiliary"],
    )

    incident, workflow = harness.service.ingest(event)

    assert len(harness.candidates) == 1, "one event compiled multiple nested candidates"
    trial, controls = harness.candidates[0]
    assert controls == {
        "_skip_attempt_grouping": True,
        "_skip_node_resource_merge": True,
        "_skip_terminal_quarantine_merge": True,
        "_persist": False,
    }, "candidate compilation could persist or recursively acquire another group lock"
    assert trial.runtime_profile_version == (
        profile or observation.runtime_profile_version
    ), "candidate compilation did not use the selected runtime profile"
    assert (incident.event_type, incident.job_id, incident.attempt_id) == (
        "GPU_FAULT_GROUP",
        observation.job_id,
        observation.attempt_id,
    ), "grouping did not bind the incident to the observed attempt"
    expected_workloads = sorted({WORKLOAD, "training/job/auxiliary"})
    for step in workflow.official_steps:
        assert step.workload_ids == expected_workloads, (
            f"{step.operation} did not receive the complete workload allocation"
        )
        assert step.node_ids == (
            ["node-a", "node-b"] if step.operation in {STOP, RESTART} else ["node-a"]
        ), f"{step.operation} confused workload-wide and node-repair scope"
    by_operation = {step.operation: step for step in workflow.official_steps}
    assert (
        by_operation[STOP].parameters["termination_initiator_incident_id"]
        == incident.incident_id
    ), "workload termination was attributed to the temporary candidate"
    assert by_operation[RESTART].parameters == {
        "cluster_id": event.cluster_id,
        "job_id": observation.job_id,
        "source_attempt_id": observation.attempt_id,
        "source_gpu_count": observation.gpu_count,
        "restart_budget": observation.restart_budget,
    }, "restart lost the original attempt's allocation or budget"
    assert by_operation[QUIESCE].parameters["workload_cgroup_paths_by_node"] == {
        "node-a": ["/unit-workload/a"],
        "node-b": ["/unit-workload/b"],
    }, "quiesce did not retain real allocation-derived cgroup evidence"
    assert (
        workflow.not_before is not None
        and workflow.aggregation_max_deadline > workflow.not_before
    ), "a new grouped recovery has no bounded aggregation window"
    assert (
        closure_store.get_incident_by_event(event.event_id).incident_id
        == incident.incident_id
    ), "the event link was not committed atomically with the grouped incident"
    assert (
        closure_store.get_workflow(workflow.request_id).official_steps
        == workflow.official_steps
    ), "the persisted recovery plan differs from the returned grouped plan"
    assert candidate[1].model_dump(mode="json") == before, (
        "grouping modified the node-health family's trial workflow in place"
    )


@pytest.mark.parametrize("candidate_operation", [RESET, WorkflowOperation.RESTART_NODE])
def test_stale_weaker_or_equivalent_health_records_evidence_without_rearming_recovery(
    closure_store, candidate_operation
):
    observation = observed_attempt(started_at=NOW)
    existing = seed_group(
        closure_store,
        observation,
        *candidate_pair(
            "active",
            operations=(STOP, WorkflowOperation.RESTART_NODE, RESTORE, RESTART),
            action=RecoveryAction.REBOOT_NODE,
        ),
    )
    before_incident, before_workflow = existing
    harness = grouped_service(
        closure_store,
        observation,
        candidate_pair(
            "stale", operations=(STOP, candidate_operation, RESTORE, RESTART)
        ),
    )
    event = finding("stale-generation", observed_at=NOW - timedelta(seconds=1))

    incident, workflow = harness.service.ingest(event)

    assert workflow.request_id == before_workflow.request_id, (
        "stale evidence queued another recovery"
    )
    assert (
        workflow.official_steps,
        workflow.fencing_token,
        workflow.events,
        workflow.not_before,
        workflow.aggregation_max_deadline,
    ) == (
        before_workflow.official_steps,
        before_workflow.fencing_token,
        before_workflow.events,
        before_workflow.not_before,
        before_workflow.aggregation_max_deadline,
    ), (
        "stale non-escalating evidence changed the executable plan or its aggregation window"
    )
    assert incident.effective_action is before_incident.effective_action, (
        "stale weaker evidence replaced the winning action"
    )
    assert incident.reasons[:-1] == before_incident.reasons, (
        "stale evidence removed existing audit reasons"
    )
    assert "Ignored stale health action" in incident.reasons[-1], (
        "stale-generation classification was not recorded for the operator"
    )
    assert (
        closure_store.get_incident_by_event(event.event_id).incident_id
        == incident.incident_id
    ), "ignored stale evidence was not atomically linked for duplicate suppression"
    replay = harness.service.ingest(event)
    assert replay == (incident, workflow), (
        "a duplicate stale event rewrote its committed decision"
    )
    assert len(harness.candidates) == 1, (
        "duplicate suppression recompiled the stale action"
    )
    assert harness.widenings == [], (
        "a stale action entered the active-plan widening path"
    )


@pytest.mark.parametrize(
    ("existing_operation", "new_operation"),
    [
        (RESET, WorkflowOperation.RESTART_NODE),
        (
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
            WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        ),
    ],
    ids=["higher-rank", "same-rank-new-intent"],
)
def test_old_timestamp_cannot_hide_a_stronger_recovery_or_a_new_operation_intent(
    closure_store, existing_operation, new_operation
):
    observation = observed_attempt(started_at=NOW)
    previous_incident, previous_workflow = seed_group(
        closure_store,
        observation,
        *candidate_pair("active", operations=(STOP, existing_operation, RESTART)),
    )
    harness = grouped_service(
        closure_store,
        observation,
        candidate_pair("escalation", operations=(STOP, new_operation, RESTART)),
    )
    event = finding("older-but-new-action", observed_at=NOW - timedelta(seconds=1))

    incident, workflow = harness.service.ingest(event)

    assert any(step.operation is new_operation for step in workflow.official_steps), (
        f"stale-time filtering silently discarded the required {new_operation} action"
    )
    assert incident.incident_id == previous_incident.incident_id, (
        "escalation abandoned the existing grouped incident"
    )
    assert incident.reasons[-1].endswith(event.reason), (
        "new actionable evidence has no incident audit"
    )
    assert not any(
        "Ignored stale health action" in reason for reason in incident.reasons
    ), "a stronger or newly required operation was classified as stale-only evidence"
    assert workflow.official_steps != previous_workflow.official_steps, (
        "new operation intent did not change the executable recovery"
    )
    assert (
        closure_store.get_workflow(workflow.request_id).official_steps
        == workflow.official_steps
    ), "the escalation was returned but not persisted"


def test_an_active_recovery_from_another_attempt_is_not_borrowed_by_a_new_group(
    closure_store,
):
    observation = observed_attempt()
    old_observation = observed_attempt(attempt_id="attempt-previous")
    active = seed_group(closure_store, old_observation, *candidate_pair("previous"))
    harness = grouped_service(
        closure_store, observation, candidate_pair("current"), active=active
    )

    incident, workflow = harness.service.ingest(finding("new-attempt"))

    assert incident.incident_id != active[0].incident_id, (
        "the previous attempt's incident was reused"
    )
    assert workflow.request_id != active[1].request_id, (
        "the previous attempt's recovery was rewritten"
    )
    assert incident.attempt_id == observation.attempt_id, (
        "new recovery lost the current attempt binding"
    )
    assert (
        closure_store.get_workflow(active[1].request_id).official_steps
        == active[1].official_steps
    ), "joining a new attempt changed the previous attempt's plan"


def test_same_attempt_recovery_from_another_ingest_family_is_reused_without_a_second_plan(
    closure_store,
):
    observation = observed_attempt()
    incident, workflow = candidate_pair("other-family")
    incident = incident.model_copy(
        update={"job_id": observation.job_id, "attempt_id": observation.attempt_id}
    )
    closure_store.save_incident_and_workflow(incident, workflow)
    active = (
        closure_store.get_incident(incident.incident_id),
        closure_store.get_workflow(workflow.request_id),
    )
    harness = grouped_service(
        closure_store, observation, candidate_pair("health-family"), active=active
    )
    event = finding("first-grouped-health")

    merged_incident, merged_workflow = harness.service.ingest(event)

    assert merged_incident.incident_id == active[0].incident_id, (
        "the health family forked a second incident for the same active attempt"
    )
    assert merged_workflow.request_id == active[1].request_id, (
        "an existing same-attempt recovery did not retain plan ownership"
    )
    assert merged_workflow.fencing_token == active[1].fencing_token, (
        "joining an existing same-scope recovery changed its generation"
    )
    assert len(closure_store.list_workflows()) == 1, (
        "cross-family reuse persisted an additional recovery plan"
    )
    assert (
        closure_store.get_incident_by_event(event.event_id).incident_id
        == active[0].incident_id
    ), "the newly indexed health event did not point at the original incident"
    assert harness.widenings == [{"node-a": ["GPU-a"]}], (
        "cross-family reuse skipped flat-plan scope reconciliation"
    )


def test_absorbing_into_a_flat_group_invokes_scope_widening_without_replacing_the_plan(
    closure_store,
):
    observation = observed_attempt()
    active_incident, active_workflow = seed_group(
        closure_store, observation, *candidate_pair("active")
    )
    harness = grouped_service(closure_store, observation, candidate_pair("second"))

    incident, workflow = harness.service.ingest(finding("same-scope"))

    assert harness.widenings == [{"node-a": ["GPU-a"]}], (
        "flat-group absorption skipped the node/GPU scope reconciliation boundary"
    )
    assert workflow.request_id == active_workflow.request_id, (
        "same-scope evidence queued a new plan"
    )
    assert workflow.fencing_token == active_workflow.fencing_token, (
        "an absorbed same-scope fault changed the recovery generation"
    )
    assert incident.incident_id == active_incident.incident_id, (
        "same-scope evidence replaced its incident"
    )
    assert [step.operation for step in workflow.official_steps] == [
        step.operation for step in active_workflow.official_steps
    ], "same-scope evidence changed the operation sequence"
    assert len(incident.reasons) == len(active_incident.reasons) + 1, (
        "the absorbed health event was not recorded"
    )
