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


# --------------------------------------------------------------------------
# Control-plane review 2026-09-08, F-8: throughput and connections. ``run_once``
# took 25 incidents an hour (~600/day) and opened two bare ``psycopg.connect``
# per incident outside the store's pool; a year of backlog could never drain.


def test_the_archiver_borrows_the_store_pool_and_never_connects_bare(
    store, monkeypatch
):
    import psycopg

    def no_bare_connections(*args, **kwargs):
        raise AssertionError("archiver opened a connection outside the pool")

    _old_incident(store, "inc-pooled", WorkflowStatus.SUCCEEDED)
    s3 = _FakeS3()
    archiver = ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
        store=store,
    )

    # Scoped: the store fixture's own teardown truncates over a bare connection.
    with monkeypatch.context() as scoped:
        scoped.setattr(psycopg, "connect", no_bare_connections)
        archived = archiver.run_once()

    assert len(archived) == 1 and len(s3.puts) == 1
    assert archiver.archived_total == 1
    with pytest.raises(NotFoundError):
        store.get_incident("inc-pooled")


def test_run_once_takes_batch_size_candidates_oldest_first(store):
    for index in range(3):
        incident = fault_incident(
            f"inc-batch-{index}",
            f"event-batch-{index}",
            state=IncidentState.RECOVERED,
            created_at=OLD - timedelta(hours=3 - index),
            updated_at=OLD - timedelta(hours=3 - index),
        )
        store.save_incident(incident)
    s3 = _FakeS3()
    archiver = ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
        store=store,
        batch_size=2,
    )

    first = archiver.run_once()
    assert len(first) == 2
    assert "inc-batch-0" in first[0] and "inc-batch-1" in first[1]
    assert store.get_incident("inc-batch-2").incident_id == "inc-batch-2"

    assert len(archiver.run_once(limit=1)) == 1
    assert archiver.run_once() == []
    assert archiver.archived_total == 3


def test_a_failing_upload_is_counted_and_does_not_end_the_round(store, caplog):
    _old_incident(store, "inc-s3-broken", WorkflowStatus.SUCCEEDED)
    _old_incident(store, "inc-s3-fine", WorkflowStatus.SUCCEEDED)

    class _FlakyS3(_FakeS3):
        def put_object(self, **kwargs) -> None:
            if "inc-s3-broken" in kwargs["Key"]:
                raise ConnectionError("s3 endpoint unreachable")
            super().put_object(**kwargs)

    s3 = _FlakyS3()
    archiver = ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
        store=store,
    )

    with caplog.at_level("ERROR", logger="gpu_fault.control_record_archive"):
        archived = archiver.run_once()

    assert len(archived) == 1 and "inc-s3-fine" in archived[0]
    assert archiver.errors_total == {"ConnectionError": 1}
    assert archiver.archived_total == 1
    assert store.get_incident("inc-s3-broken").incident_id == "inc-s3-broken"
    assert any("inc-s3-broken" in record.getMessage() for record in caplog.records)


def test_the_candidate_predicate_matches_the_declared_index_shape():
    """Agent 5 declares ``gpu_fault_incident_archive_candidate`` on
    ``((payload->>'updated_at')) WHERE kind='incident'``; the query must
    compare the same text expression, not a ``::timestamptz`` cast."""

    from gpu_fault.control_record_archive import ARCHIVE_CANDIDATE_SQL

    assert "i.kind='incident'" in ARCHIVE_CANDIDATE_SQL
    assert "i.payload->>'updated_at' <= %s" in ARCHIVE_CANDIDATE_SQL
    assert "ORDER BY i.payload->>'updated_at', i.key" in ARCHIVE_CANDIDATE_SQL
    assert "::timestamptz" not in ARCHIVE_CANDIDATE_SQL


# --------------------------------------------------------------------------
# Final review, minor 9 / 10 (passive-terminal-simplify).


def test_pre_cutover_diagnostic_and_triage_rows_are_still_swept_with_their_incident():
    """Nothing decodes or cleans ``diagnostic``/``triage`` rows any more; the
    archive is the only path that removes the ones written before the cutover."""

    from gpu_fault.control_record_archive import RELATED_RECORDS_SQL

    assert "kind IN ('diagnostic','triage')" in RELATED_RECORDS_SQL, (
        "the archive bundle must keep selecting the pre-cutover kinds"
    )


def test_a_closed_successor_no_longer_holds_its_predecessor_out_of_the_archive(store):
    """Every passive recovery names its containment workflow as predecessor;
    a recovery that ended ESCALATED (FAILED, handled) must not pin the
    containment incident for ever -- only an open successor still reads it."""

    _old_incident(store, "inc-pred", WorkflowStatus.SUCCEEDED)
    successor_incident = fault_incident(
        "inc-succ",
        "event-succ",
        state=IncidentState.ESCALATED,
        workflow_request_id="wf-succ",
    )
    store.save_incident_and_workflow(
        successor_incident,
        workflow_request(
            "wf-succ",
            "inc-succ",
            status=WorkflowStatus.FAILED,
            official_steps=[workflow_step(WorkflowOperation.RESTART_WORKLOAD)],
            predecessor_workflow_id="wf-inc-pred",
            failure_handled_at=OLD,
        ),
    )
    s3 = _FakeS3()
    archiver = _archiver(s3)

    archived = archiver.run_once()

    assert len(archived) == 1 and "inc-pred" in archived[0], (
        "the containment behind a closed recovery is archived"
    )
    assert archiver.withheld_total == {}, "nothing was withheld"
    with pytest.raises(NotFoundError):
        store.get_incident("inc-pred")
    assert store.get_incident("inc-succ").incident_id == "inc-succ", (
        "the successor itself is recent and stays"
    )
