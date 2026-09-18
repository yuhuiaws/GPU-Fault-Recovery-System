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

import gzip
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault import control_record_archive
from gpu_fault.control_record_archive import (
    ARCHIVABLE_WORKFLOW_STATUSES,
    ArchiveSafetyError,
    ControlRecordArchiver,
)
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.store import NotFoundError
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)
from tests.store.test_postgres_workflow_state_tables import select_mode

OLD = datetime.now(timezone.utc) - timedelta(days=3)


def test_archivable_means_terminal_not_merely_not_running():
    assert ARCHIVABLE_WORKFLOW_STATUSES == frozenset(
        {WorkflowStatus.SUCCEEDED, WorkflowStatus.SUPERSEDED}
    )
    # FAILED is archivable only once its failure handler ran (see below).
    assert WorkflowStatus.FAILED not in ARCHIVABLE_WORKFLOW_STATUSES
    assert WorkflowStatus.BLOCKED not in ARCHIVABLE_WORKFLOW_STATUSES


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def store(request):
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    import psycopg

    with contextmanager(postgres_store_instance)() as instance:
        try:
            with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
                for kind in ("remote_command", "workflow"):
                    select_mode(connection, kind, request.param)
            yield instance
        finally:
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


def _remote_command(
    store,
    incident_id: str,
    *,
    command_id: str | None = None,
    status: RemoteCommandStatus = RemoteCommandStatus.SUCCEEDED,
    result_details: dict | None = None,
):
    workflow = store.get_workflow(f"wf-{incident_id}")
    return store.ensure_remote_command(
        RemoteActionCommand(
            command_id=command_id or f"command-{incident_id}",
            cluster_id="cluster-a",
            workflow_request_id=workflow.request_id,
            incident_id=incident_id,
            step_index=0,
            fencing_token=workflow.fencing_token,
            idempotency_key=f"{workflow.request_id}/0",
            step=workflow.official_steps[0],
            workflow=workflow,
            incident=store.get_incident(incident_id),
            status=status,
            result_details=result_details or {},
            created_at=OLD,
            updated_at=OLD,
        )
    )


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


def test_a_withheld_incident_is_logged_and_counted(store, caplog, monkeypatch):
    _old_incident(store, "inc-pred", WorkflowStatus.SUCCEEDED)
    s3 = _FakeS3()
    archiver = _archiver(s3)
    selected = archiver.candidates()
    assert selected == ["inc-pred"]
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
    # The hold arrived after candidate selection; attempted refusals still
    # need accounting even though stable holds are now filtered before LIMIT.
    monkeypatch.setattr(archiver, "candidates", lambda **_kwargs: selected)

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


@pytest.mark.parametrize("store", ["legacy"], indirect=True)
def test_empty_failure_handled_at_does_not_prove_the_failure_was_handled(store):
    import psycopg

    incident_id = "inc-failed-empty-handled"
    _old_incident(store, incident_id, WorkflowStatus.FAILED, failure_handled_at=OLD)
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_objects "
            "SET payload=jsonb_set(payload, '{failure_handled_at}', %s::jsonb) "
            "WHERE kind='workflow' AND key=%s",
            (json.dumps(""), f"wf-{incident_id}"),
        )
        s3 = _FakeS3()
        archiver = _archiver(s3)
        with pytest.raises(ArchiveSafetyError, match="non-terminal workflow"):
            archiver.archive_one(incident_id)
        assert archiver.candidates() == []
        assert s3.puts == [], "an empty failure-handled timestamp allowed an upload"
        assert store.get_incident(incident_id).incident_id == incident_id
        assert connection.execute(
            "SELECT payload->>'failure_handled_at' FROM gpu_fault_objects "
            "WHERE kind='workflow' AND key=%s",
            (f"wf-{incident_id}",),
        ).fetchone() == ("",), "archive removed or rewrote the unhandled failure"


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


@pytest.mark.parametrize(
    ("successor_status", "failure_handled_at"),
    [
        (WorkflowStatus.SUCCEEDED, None),
        (WorkflowStatus.FAILED, None),
        (WorkflowStatus.FAILED, OLD),
        (WorkflowStatus.SUPERSEDED, None),
    ],
)
def test_a_closed_successor_no_longer_holds_its_predecessor_out_of_the_archive(
    store, successor_status, failure_handled_at
):
    """Known closed successors release their predecessor, even before failure
    handling; their own incident still needs handling before it can be archived."""

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
            status=successor_status,
            official_steps=[workflow_step(WorkflowOperation.RESTART_WORKLOAD)],
            predecessor_workflow_id="wf-inc-pred",
            failure_handled_at=failure_handled_at,
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


@pytest.mark.parametrize("store", ["legacy"], indirect=True)
@pytest.mark.parametrize("status", [None, "UNKNOWN"])
def test_an_unknown_successor_status_keeps_the_predecessor_out_of_archive(
    store, status
) -> None:
    import psycopg

    _old_incident(store, "inc-pred", WorkflowStatus.SUCCEEDED)
    successor = workflow_request(
        "wf-unknown-successor",
        "inc-unknown-successor",
        predecessor_workflow_id="wf-inc-pred",
    )
    store.save_incident_and_workflow(
        fault_incident(
            successor.incident_id,
            "event-unknown-successor",
            workflow_request_id=successor.request_id,
        ),
        successor,
    )
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=jsonb_set(payload, '{status}', %s::jsonb) "
            "WHERE kind='workflow' AND key=%s",
            (json.dumps(status), successor.request_id),
        )
        s3 = _FakeS3()
        archiver = _archiver(s3)
        with pytest.raises(ArchiveSafetyError, match="external successor"):
            archiver.archive_one("inc-pred")
        assert archiver.run_once() == []
        assert archiver.candidates() == []
        assert archiver.withheld_total == {}, (
            "a known hold must be filtered before spending the attempt quota"
        )
        assert s3.puts == [], "an unknown successor status allowed an archive upload"
        assert store.get_incident("inc-pred").incident_id == "inc-pred"
        assert store.get_workflow("wf-inc-pred").status is WorkflowStatus.SUCCEEDED
        assert connection.execute(
            "SELECT payload->>'status' FROM gpu_fault_objects "
            "WHERE kind='workflow' AND key=%s",
            (successor.request_id,),
        ).fetchone() == (status,), "archive changed the unknown successor"


@pytest.mark.parametrize(
    "status",
    [
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.WAITING,
        RemoteCommandStatus.LEASED,
    ],
)
def test_open_remote_commands_block_archive_before_upload(store, status) -> None:
    incident_id = "inc-open-command"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    command = _remote_command(store, incident_id, status=status)
    s3 = _FakeS3()
    archiver = _archiver(s3)
    assert archiver.candidates() == [], "an open command spent the candidate quota"
    with pytest.raises(ArchiveSafetyError, match="open remote command"):
        archiver.archive_one(incident_id)
    assert s3.puts == [], "archive uploaded an incident with an open command"
    assert store.get_remote_command(command.command_id) == command
    assert store.get_incident(incident_id).incident_id == incident_id


@pytest.mark.parametrize("store", ["legacy"], indirect=True)
@pytest.mark.parametrize("status", [None, "UNRECOGNIZED"])
def test_unknown_remote_command_status_blocks_archive(store, status) -> None:
    import psycopg

    incident_id = "inc-unknown-command"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    command = _remote_command(store, incident_id)
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=jsonb_set(payload, '{status}', %s::jsonb) "
            "WHERE kind='remote_command' AND key=%s",
            (json.dumps(status), command.command_id),
        )
        s3 = _FakeS3()
        archiver = _archiver(s3)
        assert archiver.candidates() == [], (
            "an unknown command spent the candidate quota"
        )
        with pytest.raises(ArchiveSafetyError, match="open remote command"):
            archiver.archive_one(incident_id)
        assert s3.puts == [], "an unknown command status was accepted as terminal"
        assert store.get_incident(incident_id).incident_id == incident_id
        assert connection.execute(
            "SELECT payload->>'status' FROM gpu_fault_objects "
            "WHERE kind='remote_command' AND key=%s",
            (command.command_id,),
        ).fetchone() == (status,)


@pytest.mark.parametrize("store", ["legacy"], indirect=True)
def test_unknown_workflow_status_blocks_candidates_and_archive(store) -> None:
    import psycopg

    incident_id = "inc-unknown-workflow"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=payload - 'status' "
            "WHERE kind='workflow' AND key=%s",
            (f"wf-{incident_id}",),
        )
        s3 = _FakeS3()
        archiver = _archiver(s3)
        assert archiver.candidates() == [], (
            "a workflow with unknown status was eligible"
        )
        with pytest.raises(ArchiveSafetyError, match="non-terminal workflow"):
            archiver.archive_one(incident_id)
        assert s3.puts == [], "archive uploaded a workflow with unknown status"
        assert store.get_incident(incident_id).incident_id == incident_id


def test_archive_contains_the_complete_terminal_remote_command(store) -> None:
    incident_id = "inc-command-evidence"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    command = _remote_command(
        store, incident_id, result_details={"steps": {"0": {"status": "SUCCEEDED"}}}
    )
    s3 = _FakeS3()
    _archiver(s3).archive_one(incident_id)
    records = [
        json.loads(line) for line in gzip.decompress(s3.puts[0]["Body"]).splitlines()
    ]
    assert [item for item in records if item["kind"] == "remote_command"] == [
        {
            "table": "gpu_fault_objects",
            "kind": "remote_command",
            "key": command.command_id,
            "payload": command.model_dump(mode="json"),
        }
    ]
    with pytest.raises(NotFoundError):
        store.get_remote_command(command.command_id)


@pytest.mark.parametrize("store", ["dual"], indirect=True)
def test_archive_deletes_legacy_commands_with_reconstructed_payloads(store) -> None:
    import psycopg

    incident_id = "inc-legacy-command-projection"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    command = _remote_command(store, incident_id)
    payload = command.model_dump(mode="json")
    payload.pop("last_lease_owner")
    payload["created_at"] = command.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=%s::jsonb "
            "WHERE kind='remote_command' AND key=%s",
            (json.dumps(payload), command.command_id),
        )
        projected = connection.execute(
            "SELECT payload FROM gpu_fault_remote_command_records WHERE key=%s",
            (command.command_id,),
        ).fetchone()[0]
        assert projected != payload, "the fixture did not exercise a reconstructed row"
        s3 = _FakeS3()
        _archiver(s3).archive_one(incident_id)
        records = [
            json.loads(line)
            for line in gzip.decompress(s3.puts[0]["Body"]).splitlines()
        ]
        archived_commands = [
            item["payload"] for item in records if item["kind"] == "remote_command"
        ]
        assert archived_commands == [command.model_dump(mode="json")]
        assert store.list_remote_commands() == []
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_objects WHERE kind='remote_command' AND key=%s",
            (command.command_id,),
        ).fetchone() == (0,), "archive left the authoritative legacy row behind"


@pytest.mark.parametrize("store", ["legacy"], indirect=True)
@pytest.mark.parametrize("workflow_mode", ["legacy", "dedicated"])
def test_dedicated_archive_ignores_retired_legacy_commands(
    store, workflow_mode
) -> None:
    import psycopg

    incident_id = "inc-retired-command"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    retired = _remote_command(store, incident_id, command_id="command-retired")
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        select_mode(connection, "remote_command", "dedicated")
        select_mode(connection, "workflow", workflow_mode)
        assert (
            store.cleanup_terminal_remote_commands(
                older_than=OLD + timedelta(hours=1), limit=10
            )
            == 1
        )
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_objects "
            "WHERE kind='remote_command' AND key=%s",
            (retired.command_id,),
        ).fetchone() == (1,)
        current = _remote_command(
            store,
            incident_id,
            command_id="command-current",
            result_details={"source": "dedicated"},
        )
        s3 = _FakeS3()
        _archiver(s3).archive_one(incident_id)
        records = [
            json.loads(line)
            for line in gzip.decompress(s3.puts[0]["Body"]).splitlines()
        ]
        commands = [item for item in records if item["kind"] == "remote_command"]
        assert [item["key"] for item in commands] == [current.command_id]
        assert commands[0]["payload"] == current.model_dump(mode="json")
        assert store.list_remote_commands() == []
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_objects "
            "WHERE kind='remote_command' AND key=%s",
            (retired.command_id,),
        ).fetchone() == (1,), "archive bypassed explicit legacy retirement"


def test_a_command_created_during_upload_prevents_archive_deletion(store) -> None:
    incident_id = "inc-command-arrived"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    terminal = _remote_command(store, incident_id)

    class _ArrivingCommandS3(_FakeS3):
        def put_object(self, **kwargs) -> None:
            super().put_object(**kwargs)
            _remote_command(
                store,
                incident_id,
                command_id="command-after-upload",
                status=RemoteCommandStatus.PENDING,
            )

    s3 = _ArrivingCommandS3()
    with pytest.raises(ArchiveSafetyError, match="open remote command"):
        _archiver(s3).archive_one(incident_id)
    assert len(s3.puts) == 1
    assert store.get_remote_command(terminal.command_id) == terminal
    assert (
        store.get_remote_command("command-after-upload").status
        is RemoteCommandStatus.PENDING
    )
    assert store.get_incident(incident_id).incident_id == incident_id


@pytest.mark.parametrize("hold", ["open-command", "external-successor"])
def test_one_incident_batch_progresses_past_an_older_stable_hold(store, hold) -> None:
    _old_incident(store, "inc-held", WorkflowStatus.SUCCEEDED)
    for index in range(2):
        key = f"inc-eligible-{index}"
        _old_incident(store, key, WorkflowStatus.SUCCEEDED)
        incident = store.get_incident(key)
        store.save_incident(
            incident.model_copy(
                update={"updated_at": OLD + timedelta(hours=index + 1)}
            ),
            expected=incident,
        )
    if hold == "open-command":
        _remote_command(store, "inc-held", status=RemoteCommandStatus.WAITING)
    else:
        successor = workflow_request(
            "wf-holding-successor",
            "inc-successor",
            predecessor_workflow_id="wf-inc-held",
        )
        store.save_incident_and_workflow(
            fault_incident(
                successor.incident_id,
                "event-successor",
                workflow_request_id=successor.request_id,
            ),
            successor,
        )
    s3 = _FakeS3()
    archiver = ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
        store=store,
        batch_size=1,
    )
    assert archiver.candidates() == ["inc-eligible-0"]
    first = archiver.run_once()
    assert len(first) == 1 and "inc-eligible-0" in first[0], (
        "an older held incident starved an eligible incident behind LIMIT"
    )
    assert store.get_incident("inc-eligible-1").incident_id == "inc-eligible-1"
    second = archiver.run_once()
    assert len(second) == 1 and "inc-eligible-1" in second[0]
    assert archiver.run_once() == []
    assert len(s3.puts) == 2
    assert archiver.archived_total == 2
    assert archiver.withheld_total == {}
    assert store.get_incident("inc-held").incident_id == "inc-held"
    assert store.get_workflow("wf-inc-held").status is WorkflowStatus.SUCCEEDED


def test_a_successor_created_during_upload_still_prevents_archive_deletion(
    store,
) -> None:
    incident_id = "inc-successor-arrived"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)

    class _ArrivingSuccessorS3(_FakeS3):
        def put_object(self, **kwargs) -> None:
            super().put_object(**kwargs)
            successor = workflow_request(
                "wf-after-upload",
                "inc-after-upload",
                predecessor_workflow_id=f"wf-{incident_id}",
            )
            store.save_incident_and_workflow(
                fault_incident(
                    successor.incident_id,
                    "event-after-upload",
                    workflow_request_id=successor.request_id,
                ),
                successor,
            )

    s3 = _ArrivingSuccessorS3()
    archiver = _archiver(s3)
    assert archiver.candidates() == [incident_id]
    assert archiver.run_once() == []
    assert archiver.withheld_total == {"incident has external successor": 1}
    assert len(s3.puts) == 1, "the successor race must occur after the archive upload"
    assert store.get_incident(incident_id).incident_id == incident_id
    assert store.get_workflow(f"wf-{incident_id}").status is WorkflowStatus.SUCCEEDED


@pytest.mark.parametrize("pooled", [False, True])
def test_a_candidate_refreshed_before_archive_keeps_its_records(
    store, monkeypatch, pooled: bool
) -> None:
    incident_id = "inc-refreshed-candidate"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    command = _remote_command(store, incident_id)
    workflow = store.get_workflow(f"wf-{incident_id}")
    s3 = _FakeS3()
    archiver = ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
        store=store if pooled else None,
    )
    selected = archiver.candidates()
    assert selected == [incident_id], "the race must begin with an expired candidate"
    previous = store.get_incident(incident_id)
    refreshed = previous.model_copy(update={"updated_at": datetime.now(timezone.utc)})
    store.save_incident(refreshed, expected=previous)

    with pytest.raises(ArchiveSafetyError, match="within retention"):
        archiver.archive_one(incident_id)
    assert archiver.candidates() == [], "fresh data must no longer be selected"
    monkeypatch.setattr(archiver, "candidates", lambda **_kwargs: selected)
    assert archiver.run_once() == [], (
        "cached candidates do not authorize fresh deletion"
    )
    assert archiver.withheld_total == {"incident is within retention": 1}
    assert archiver.errors_total == {}
    assert s3.puts == [], "retention must be rechecked before uploading fresh records"
    assert store.get_incident(incident_id) == refreshed
    assert store.get_workflow(workflow.request_id) == workflow
    assert store.get_remote_command(command.command_id) == command
    assert store.get_incident_by_event(previous.event_id) == refreshed


@pytest.mark.parametrize(
    ("delta", "archived"),
    [
        (timedelta(microseconds=-1), True),
        (timedelta(0), True),
        (timedelta(microseconds=1), False),
        (timedelta(days=2), False),
    ],
)
def test_direct_archive_calls_honor_the_retention_boundary(
    store, monkeypatch, delta: timedelta, archived: bool
) -> None:
    now = datetime.now(timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(control_record_archive, "datetime", Clock)
    incident_id = "inc-direct-retention"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    previous = store.get_incident(incident_id)
    current = previous.model_copy(
        update={"updated_at": now - timedelta(days=1) + delta}
    )
    store.save_incident(current, expected=previous)
    s3 = _FakeS3()
    archiver = _archiver(s3)

    if archived:
        assert incident_id in archiver.archive_one(incident_id)
        assert len(s3.puts) == 1
        with pytest.raises(NotFoundError):
            store.get_incident(incident_id)
    else:
        with pytest.raises(ArchiveSafetyError, match="within retention"):
            archiver.archive_one(incident_id)
        assert s3.puts == [], "direct calls must not bypass the retention policy"
        assert store.get_incident(incident_id) == current


@pytest.mark.parametrize("pooled", [False, True])
def test_an_incident_refreshed_during_upload_is_retained(store, pooled: bool) -> None:
    incident_id = "inc-refreshed-during-upload"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    previous = store.get_incident(incident_id)
    refreshed = previous.model_copy(update={"updated_at": datetime.now(timezone.utc)})
    workflow = store.get_workflow(f"wf-{incident_id}")
    command = _remote_command(store, incident_id)

    class RefreshingS3(_FakeS3):
        def put_object(self, **kwargs) -> None:
            super().put_object(**kwargs)
            store.save_incident(refreshed, expected=previous)

    s3 = RefreshingS3()
    archiver = ControlRecordArchiver(
        POSTGRES_URL,
        "s3://audit-bucket/control",
        retention=timedelta(days=1),
        s3_client=s3,
        store=store if pooled else None,
    )
    assert archiver.run_once() == [], "the final transaction must recheck retention"
    assert len(s3.puts) == 1, "this race occurs after uploading the old audit bundle"
    assert archiver.withheld_total == {"incident is within retention": 1}
    assert archiver.archived_total == 0
    assert store.get_incident(incident_id) == refreshed
    assert store.get_workflow(workflow.request_id) == workflow
    assert store.get_remote_command(command.command_id) == command
    assert store.get_incident_by_event(previous.event_id) == refreshed


@pytest.mark.parametrize(
    "timestamp",
    [
        OLD.replace(tzinfo=None).isoformat(),
        OLD.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        OLD.isoformat(timespec="seconds"),
    ],
    ids=("naive", "offset", "seconds"),
)
def test_legacy_timestamp_encodings_keep_retention_and_audit_semantics(
    store, timestamp: str
) -> None:
    import psycopg

    incident_id = "inc-legacy-retention"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=jsonb_set(payload, '{updated_at}', %s::jsonb) "
            "WHERE kind='incident' AND key=%s",
            (json.dumps(timestamp), incident_id),
        )
    s3 = _FakeS3()
    _archiver(s3).archive_one(incident_id)
    records = [
        json.loads(line) for line in gzip.decompress(s3.puts[0]["Body"]).splitlines()
    ]
    archived = next(row["payload"] for row in records if row["kind"] == "incident")
    assert archived["updated_at"] == timestamp, "validation must not rewrite audit data"
    with pytest.raises(NotFoundError):
        store.get_incident(incident_id)


@pytest.mark.parametrize("during_upload", [False, True])
@pytest.mark.parametrize(
    "timestamp",
    [
        pytest.param(Ellipsis, id="missing"),
        None,
        "",
        "invalid-datetime",
        "1900-01-01",
        "1900-99-99T00:00:00Z",
        0,
        {},
        [],
    ],
)
def test_invalid_incident_timestamps_never_authorize_archive(
    store, timestamp: object, during_upload: bool
) -> None:
    import psycopg

    incident_id = "inc-invalid-retention"
    _old_incident(store, incident_id, WorkflowStatus.SUCCEEDED)
    incident = store.get_incident(incident_id)
    workflow = store.get_workflow(f"wf-{incident_id}")
    payload = incident.model_dump(mode="json")
    if timestamp is Ellipsis:
        payload.pop("updated_at")
    else:
        payload["updated_at"] = timestamp
    assert POSTGRES_URL is not None, "the archive fixture requires PostgreSQL"
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:

        def invalidate_timestamp() -> None:
            connection.execute(
                "UPDATE gpu_fault_objects SET payload=%s::jsonb "
                "WHERE kind='incident' AND key=%s",
                (json.dumps(payload), incident_id),
            )

        class InvalidatingS3(_FakeS3):
            def put_object(self, **kwargs) -> None:
                super().put_object(**kwargs)
                invalidate_timestamp()

        s3 = InvalidatingS3() if during_upload else _FakeS3()
        if not during_upload:
            invalidate_timestamp()
        with pytest.raises(ArchiveSafetyError, match="invalid updated_at"):
            _archiver(s3).archive_one(incident_id)
        assert len(s3.puts) == int(during_upload)
        assert store.get_workflow(workflow.request_id) == workflow
        assert connection.execute(
            "SELECT payload FROM gpu_fault_objects WHERE kind='incident' AND key=%s",
            (incident_id,),
        ).fetchone() == (payload,), "invalid retention evidence must remain untouched"
