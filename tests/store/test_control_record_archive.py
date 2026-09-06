"""The archiver only removes what nobody will ever act on again, and says
why it kept the rest.

F-I1 (docs/review/FINAL-建议汇总.md). "Active" was a blacklist of three
statuses, so a BLOCKED workflow -- held for reconciliation -- counted as
inactive and its incident could be archived away unreconciled. An incident
that never got a workflow was never a candidate at all. Every safety
refusal was `continue`: a retention that was permanently stuck was
invisible. Now archivable is a whitelist of terminal statuses, orphan
incidents age out like the others, and each refusal is logged and counted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.control_record_archive import (
    ARCHIVABLE_WORKFLOW_STATUSES,
    ArchiveSafetyError,
    ControlRecordArchiver,
)
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import NotFoundError
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

OLD = datetime.now(timezone.utc) - timedelta(days=3)


def test_archivable_means_terminal_not_merely_not_running():
    assert ARCHIVABLE_WORKFLOW_STATUSES == frozenset(
        {WorkflowStatus.SUCCEEDED, WorkflowStatus.SUPERSEDED}
    )
    # FAILED is archivable only once its failure handler ran (see below).
    assert WorkflowStatus.FAILED not in ARCHIVABLE_WORKFLOW_STATUSES
    assert WorkflowStatus.BLOCKED not in ARCHIVABLE_WORKFLOW_STATUSES


@pytest.fixture
def store():
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, **kwargs) -> None:
        self.puts.append(kwargs)


def _archiver(s3: _FakeS3) -> ControlRecordArchiver:
    assert POSTGRES_URL is not None
    return ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
    )


def _old_incident(
    store, incident_id: str, status: WorkflowStatus | None, **workflow_values
):
    workflow_id = f"wf-{incident_id}" if status is not None else None
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.RECOVERED,
        workflow_request_id=workflow_id,
        created_at=OLD,
        updated_at=OLD,
    )
    if status is None:
        store.save_incident(incident)
        return
    workflow = workflow_request(
        workflow_id,
        incident_id,
        status=status,
        official_steps=[workflow_step(WorkflowOperation.RESET_GPU)],
        created_at=OLD,
        updated_at=OLD,
        **workflow_values,
    )
    store.save_incident_and_workflow(incident, workflow)


def test_a_blocked_workflow_keeps_its_incident_out_of_the_archive(store):
    _old_incident(
        store, "inc-blocked", WorkflowStatus.BLOCKED, blocked_reasons=["needs operator"]
    )
    s3 = _FakeS3()
    archiver = _archiver(s3)

    archived = archiver.run_once()

    assert archived == []
    assert s3.puts == []
    with pytest.raises(ArchiveSafetyError):
        archiver.archive_one("inc-blocked")
    assert store.get_workflow("wf-inc-blocked").status is WorkflowStatus.BLOCKED


def test_terminal_and_orphan_incidents_age_out(store):
    _old_incident(store, "inc-done", WorkflowStatus.SUCCEEDED)
    _old_incident(store, "inc-orphan", None)
    s3 = _FakeS3()

    archived = _archiver(s3).run_once()

    assert sorted(archived) == sorted(
        [key for key in archived if "inc-done" in key]
        + [key for key in archived if "inc-orphan" in key]
    )
    assert len(archived) == 2 and len(s3.puts) == 2
    for incident_id in ("inc-done", "inc-orphan"):
        with pytest.raises(NotFoundError):
            store.get_incident(incident_id)
    with pytest.raises(NotFoundError):
        store.get_workflow("wf-inc-done")


def test_a_withheld_incident_is_logged_and_counted(store, caplog):
    _old_incident(store, "inc-pred", WorkflowStatus.SUCCEEDED)
    # A later incident's workflow still points back at inc-pred's workflow.
    successor_incident = fault_incident(
        "inc-succ",
        "event-succ",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-succ",
    )
    store.save_incident_and_workflow(
        successor_incident,
        workflow_request(
            "wf-succ",
            "inc-succ",
            official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
            predecessor_workflow_id="wf-inc-pred",
        ),
    )
    s3 = _FakeS3()
    archiver = _archiver(s3)

    with caplog.at_level("WARNING", logger="gpu_fault.control_record_archive"):
        archived = archiver.run_once()

    assert archived == [] and s3.puts == []
    assert archiver.withheld_total == {"incident has external successor": 1}
    assert any("inc-pred" in record.getMessage() for record in caplog.records), (
        'expected any("inc-pred" in record.getMessage() for record in caplog.records) to be true'
    )
    assert store.get_incident("inc-pred").incident_id == "inc-pred"


def test_a_failed_workflow_is_archived_only_after_its_failure_was_handled(store):
    _old_incident(store, "inc-failed-open", WorkflowStatus.FAILED)
    _old_incident(
        store, "inc-failed-handled", WorkflowStatus.FAILED, failure_handled_at=OLD
    )
    s3 = _FakeS3()

    archived = _archiver(s3).run_once()

    assert len(archived) == 1 and "inc-failed-handled" in archived[0]
    assert store.get_incident("inc-failed-open").incident_id == "inc-failed-open"
    with pytest.raises(NotFoundError):
        store.get_incident("inc-failed-handled")
