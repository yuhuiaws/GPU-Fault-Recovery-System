from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.store import NotFoundError
from tests._builders import workflow_step
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)
from tests.store._cov95_compat_workflows import save_pair, workflow_pair


def test_incident_state_reads_filter_cluster_nodes_and_states_before_limit(
    compat_store,
):
    originals = {}
    for name, fields in (
        ("a-match", {}),
        ("b-match", {}),
        ("other-node", {"node_ids": ["node-other"]}),
        ("foreign", {"cluster_id": "cluster-foreign"}),
        ("recovered", {"incident_state": IncidentState.RECOVERED}),
    ):
        incident, _ = workflow_pair(
            name,
            at=NOW if name.endswith("match") else NOW + timedelta(minutes=1),
            **fields,
        )
        compat_store.save_incident(incident)
        originals[name] = incident

    assert compat_store.list_incidents_by_state(
        "cluster-local",
        [IncidentState.ACTION_PENDING],
        node_ids={"node-local"},
        limit=1,
    ) == [originals["b-match"]], (
        "foreign, wrong-state and wrong-node rows must not consume the limit"
    )
    assert compat_store.list_incidents_by_state(
        "cluster-local", [IncidentState.ACTION_PENDING]
    ) == [originals[name] for name in ("other-node", "b-match", "a-match")], (
        "unbounded node selection still needs stable newest-first incident order"
    )
    assert compat_store.list_incidents_by_state("cluster-local", []) == [], (
        "an empty state filter must match nothing"
    )
    assert (
        compat_store.list_incidents_by_state(
            "cluster-local", [IncidentState.ACTION_PENDING], node_ids=set()
        )
        == []
    ), "an empty node allowlist must not expand to every node"
    assert (
        compat_store.list_incidents_by_state(
            "cluster-local", [IncidentState.ACTION_PENDING], node_ids={"absent-node"}
        )
        == []
    ), "unknown nodes must not fall back to cluster-wide results"
    for incident in originals.values():
        assert compat_store.get_incident(incident.incident_id) == incident, (
            "read-only incident scans must not repair or rewrite missing workflow pointers"
        )


def test_active_workflow_reads_bind_job_and_node_without_leaking_foreign_pairs(
    compat_store,
):
    expected = save_pair(compat_store, "match")
    for name, fields in (
        ("other-job", {"job_id": "other-job"}),
        ("other-node", {"node_ids": ["other-node"]}),
        ("foreign", {"cluster_id": "foreign-cluster"}),
        ("finished", {"status": WorkflowStatus.SUCCEEDED}),
    ):
        save_pair(compat_store, name, at=NOW + timedelta(minutes=1), **fields)
    _, dangling = workflow_pair("dangling", at=NOW + timedelta(minutes=2))
    compat_store.save_workflow(dangling)

    assert compat_store.list_active_workflow_incidents(
        "cluster-local", job_id="job-local", node_ids={"node-local"}, limit=1
    ) == [expected], (
        "the result must pair only current matching incident/workflow records"
    )
    assert (
        compat_store.list_active_workflow_incidents(
            "cluster-local", job_id="job-local", node_ids=set()
        )
        == []
    ), "an empty node selection must return no active ownership records"
    assert (
        compat_store.list_active_workflow_incidents(
            "cluster-local", job_id="missing-job"
        )
        == []
    ), "a missing job must not inherit another job's active workflow"
    assert compat_store.list_active_workflow_incidents("missing-cluster") == [], (
        "the active-workflow reader must keep the cluster boundary"
    )


def test_job_recovery_reads_find_live_attempts_and_completed_restart_evidence(
    compat_store,
):
    for index, status in enumerate(
        (WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING)
    ):
        save_pair(
            compat_store, status.value, status=status, at=NOW + timedelta(seconds=index)
        )
    restart_steps = [
        workflow_step(WorkflowOperation.RESTART_WORKLOAD),
        workflow_step(WorkflowOperation.VALIDATE_GPU),
    ]
    restart_executions = [
        WorkflowStepExecution(
            step_index=0,
            operation=WorkflowOperation.RESTART_WORKLOAD,
            status=WorkflowStepStatus.SUCCEEDED,
            details={"restart_attempt_id": "attempt-local"},
        ),
        WorkflowStepExecution(
            step_index=1,
            operation=WorkflowOperation.VALIDATE_GPU,
            status=WorkflowStepStatus.SUCCEEDED,
            details={"restart_attempt_id": "unrelated-detail"},
        ),
    ]
    save_pair(
        compat_store,
        "restart-proof",
        status=WorkflowStatus.SUCCEEDED,
        attempt_id="previous-attempt",
        official_steps=restart_steps,
        step_executions=restart_executions,
        at=NOW + timedelta(seconds=3),
    )
    for name, fields in (
        ("foreign", {"cluster_id": "other-cluster"}),
        ("other-job", {"job_id": "other-job"}),
        ("other-attempt", {"attempt_id": "other-attempt"}),
        ("finished-without-restart", {"status": WorkflowStatus.SUCCEEDED}),
    ):
        save_pair(compat_store, name, at=NOW + timedelta(hours=1), **fields)
    _, dangling = workflow_pair("missing-incident")
    compat_store.save_workflow(dangling)

    rows = compat_store.list_job_recovery_workflow_incidents(
        "cluster-local", "job-local", "attempt-local"
    )
    assert [workflow.request_id for _, workflow in rows] == [
        "workflow/restart-proof",
        "workflow/SAFETY_PENDING",
        "workflow/RUNNING",
        "workflow/PENDING",
    ], (
        "recovery lookup must retain completed restart provenance and exclude unrelated work"
    )
    assert all(
        incident.incident_id == workflow.incident_id for incident, workflow in rows
    ), "each recovery row must pair the workflow with its own incident"
    assert (
        compat_store.list_job_recovery_workflow_incidents(
            "cluster-local", "job-local", "attempt-local", limit=2
        )
        == rows[:2]
    ), "the recovery limit must apply after cluster, job and attempt filtering"
    assert (
        compat_store.list_job_recovery_workflow_incidents(
            "cluster-local", "job-local", "unrelated-detail"
        )
        == []
    ), "non-restart step details must never claim provenance for an attempt"


def test_unhandled_failure_reads_are_bounded_and_do_not_reopen_handled_work(
    compat_store,
):
    saved = {}
    for name in ("b-failed", "a-failed"):
        saved[name] = save_pair(compat_store, name, status=WorkflowStatus.FAILED)[1]
    save_pair(
        compat_store,
        "already-handled",
        status=WorkflowStatus.FAILED,
        failure_handled_at=NOW,
        at=NOW - timedelta(days=1),
    )
    save_pair(compat_store, "pending", at=NOW - timedelta(days=1))

    assert compat_store.list_unhandled_failed_workflows(limit=1) == [
        saved["a-failed"]
    ], "old handled and pending work must not consume the failure scan budget"
    assert compat_store.list_unhandled_failed_workflows() == [
        saved["a-failed"],
        saved["b-failed"],
    ], "unhandled failures must retain stable update-time and request-ID order"


def test_paired_saves_preserve_all_event_aliases_across_followup_updates(compat_store):
    incident, workflow = workflow_pair("aliases")
    compat_store.save_incident_and_workflow(
        incident, workflow, extra_event_ids=[incident.event_id, "merged-event"]
    )
    compat_store.save_incident(
        incident,
        expected=incident,
        extra_event_ids=[incident.event_id, "followup-event"],
    )

    for event_id in (incident.event_id, "merged-event", "followup-event"):
        assert compat_store.get_incident_by_event(event_id) == incident, (
            f"the persisted event alias {event_id} must still resolve to the incident"
        )
    assert compat_store.get_workflow(workflow.request_id) == workflow, (
        "writing an event alias must not mutate the workflow generation"
    )


@pytest.mark.parametrize("wrong_pointer", ["incident", "workflow"])
def test_paired_write_rejects_mismatched_identity_without_publishing_half_a_pair(
    compat_store, wrong_pointer
):
    incident, workflow = workflow_pair("refused-pair")
    if wrong_pointer == "incident":
        incident = incident.model_copy(update={"workflow_request_id": "wrong-workflow"})
    else:
        workflow = workflow.model_copy(update={"incident_id": "wrong-incident"})

    with pytest.raises(ValueError, match="pointer"):
        compat_store.save_incident_and_workflow(incident, workflow)
    with pytest.raises(NotFoundError):
        compat_store.get_incident(incident.incident_id)
    with pytest.raises(NotFoundError):
        compat_store.get_workflow(workflow.request_id)
    assert compat_store.get_incident_by_event(incident.event_id) is None, (
        "a refused pair must not publish an event-to-incident link"
    )


def test_missing_workflow_amend_and_event_link_do_not_create_records(compat_store):
    with pytest.raises(NotFoundError, match="missing-workflow"):
        compat_store.amend_workflow(
            "missing-workflow", {"status": WorkflowStatus.FAILED}
        )
    with pytest.raises(NotFoundError, match="missing-incident"):
        compat_store.link_event_to_incident("new-event", "missing-incident")
    assert compat_store.list_workflows(None) == [], (
        "an amend must not create an audit-only row"
    )
    assert compat_store.get_incident_by_event("new-event") is None, (
        "an invalid event link must not be published"
    )
