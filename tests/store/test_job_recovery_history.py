"""The optional terminal-history read keeps the existing three-backend contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.store.contracts import WorkflowStore
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store.test_workflow_scan_whole_second import store as store

NOW = datetime(2030, 1, 1, tzinfo=UTC)


def save_pair(
    backend: WorkflowStore,
    name: str,
    *,
    status: WorkflowStatus,
    cluster_id: str = "history-cluster",
    job_id: str = "history-job",
    attempt_id: str = "history-attempt",
    offset: int = 0,
    restart_attempt_id: str | None = None,
    operation: WorkflowOperation = WorkflowOperation.RESTART_WORKLOAD,
) -> tuple[FaultIncident, WorkflowRequest]:
    at = NOW + timedelta(seconds=offset)
    incident = fault_incident(
        f"inc-{name}",
        f"event-{name}",
        cluster_id=cluster_id,
        job_id=job_id,
        attempt_id=attempt_id,
        fencing_token=1,
        workflow_request_id=f"wf-{name}",
        created_at=at,
        updated_at=at,
    )
    executions = (
        []
        if restart_attempt_id is None
        else [
            WorkflowStepExecution(
                step_index=0,
                operation=operation,
                status=WorkflowStepStatus.SUCCEEDED,
                details={"restart_attempt_id": restart_attempt_id},
                started_at=at,
                updated_at=at,
            )
        ]
    )
    workflow = workflow_request(
        f"wf-{name}",
        incident.incident_id,
        status=status,
        fencing_token=1,
        official_steps=[workflow_step(operation)],
        step_executions=executions,
        created_at=at,
        updated_at=at,
    )
    backend.save_incident_and_workflow(incident, workflow)
    return incident, workflow


@pytest.mark.usefixtures("store")
def test_terminal_history_is_opt_in_and_default_restart_semantics_are_unchanged(
    request: pytest.FixtureRequest,
) -> None:
    backend: WorkflowStore = request.getfixturevalue("store")
    originals = [
        save_pair(backend, status.value, status=status, offset=index)
        for index, status in enumerate(WorkflowStatus)
    ]
    restarted = save_pair(
        backend,
        "restarted",
        status=WorkflowStatus.SUCCEEDED,
        offset=20,
        attempt_id="source-attempt",
        restart_attempt_id="history-attempt",
    )
    ignored = save_pair(
        backend,
        "non-restart-detail",
        status=WorkflowStatus.SUCCEEDED,
        offset=30,
        attempt_id="other-attempt",
        restart_attempt_id="history-attempt",
        operation=WorkflowOperation.VALIDATE_GPU,
    )
    default = backend.list_job_recovery_workflow_incidents(
        "history-cluster", "history-job", "history-attempt"
    )
    explicit = backend.list_job_recovery_workflow_incidents(
        "history-cluster", "history-job", "history-attempt", include_terminal=False
    )
    expected_live = {
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.SAFETY_PENDING,
    }
    assert (
        default
        == explicit
        == [
            restarted,
            *[pair for pair in reversed(originals) if pair[1].status in expected_live],
        ]
    )
    history = backend.list_job_recovery_workflow_incidents(
        "history-cluster", "history-job", "history-attempt", include_terminal=True
    )
    assert history == [restarted, *reversed(originals)]
    assert ignored not in history, (
        "ordinary step details must not manufacture restart provenance"
    )
    for incident, workflow in [*originals, restarted, ignored]:
        assert backend.get_incident(incident.incident_id) == incident
        assert backend.get_workflow(workflow.request_id) == workflow


@pytest.mark.usefixtures("store")
def test_history_scope_and_sorting_precede_limit_on_every_backend(
    request: pytest.FixtureRequest,
) -> None:
    backend: WorkflowStore = request.getfixturevalue("store")
    first = save_pair(backend, "a-terminal", status=WorkflowStatus.FAILED)
    second = save_pair(backend, "z-terminal", status=WorkflowStatus.BLOCKED)
    for name, fields in [
        ("foreign-cluster", {"cluster_id": "foreign"}),
        ("foreign-job", {"job_id": "foreign"}),
        ("foreign-attempt", {"attempt_id": "foreign"}),
    ]:
        save_pair(backend, name, status=WorkflowStatus.FAILED, offset=100, **fields)
    orphan = workflow_request(
        "wf-orphan", "missing-incident", status=WorkflowStatus.FAILED
    )
    backend.save_workflow(orphan)
    assert backend.list_job_recovery_workflow_incidents(
        "history-cluster", "history-job", "history-attempt", include_terminal=True
    ) == [second, first]
    assert backend.list_job_recovery_workflow_incidents(
        "history-cluster",
        "history-job",
        "history-attempt",
        include_terminal=True,
        limit=1,
    ) == [second]
    assert (
        backend.list_job_recovery_workflow_incidents(
            "history-cluster",
            "history-job",
            "history-attempt",
            include_terminal=True,
            limit=0,
        )
        == []
    )
    assert (
        backend.list_job_recovery_workflow_incidents(
            "history-cluster", "history-job", "absent", include_terminal=True
        )
        == []
    )
