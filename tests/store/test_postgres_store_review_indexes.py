"""The v12 indexes serve the queries they were declared for.

Store review 2026-09-07, items G, H1 and H2. Each test seeds a few hundred
realistic rows (the same JSON the store writes), runs ANALYZE, prices out
sequential scans and asks EXPLAIN which index the query can use -- the same
planner-neutral method as ``test_postgres_workflow_indexes.py``. The SQL under
test is copied from the store methods named in each test; those methods live in
files this review item does not touch.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import PostgresStore
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

WORKFLOW_STATUSES = list(WorkflowStatus)
INCIDENT_STATES = list(IncidentState)
COMMAND_STATUSES = list(RemoteCommandStatus)

# Item G. Postgres never plans an Index Only Scan over an expression index
# (indxpath.c ``check_index_only`` ignores expression columns), so the
# ``GROUP BY payload->>'status'`` text in ``workflow_status_counts`` /
# ``incident_state_counts`` / ``decision_status_counts`` cannot use the count
# indexes at all. This is the per-status shape those methods have to take: one
# index range scan per enum value, ``count(*)`` needs no column, so the payload
# is never evaluated and never detoasted.
PER_STATUS_COUNT_SQL = """
    SELECT s.value, (
        SELECT count(*) FROM gpu_fault_objects
        WHERE kind=%s AND payload->>%s = s.value
    )
    FROM unnest(%s::text[]) AS s(value)
"""
LIST_REMOTE_COMMANDS_SQL = """
    SELECT payload
    FROM gpu_fault_objects
    WHERE kind='remote_command'
      AND payload->>'workflow_request_id' = ANY(%s)
    ORDER BY payload->>'created_at', key
"""
CANCEL_CANDIDATES_SQL = """
    SELECT key FROM gpu_fault_objects
    WHERE kind='remote_command'
      AND payload->>'workflow_request_id'=%s
      AND payload->>'status' NOT IN (
          'SUCCEEDED', 'FAILED'
      )
    ORDER BY key
"""
JOB_RECOVERY_SQL = """
    SELECT i.payload, w.payload
    FROM gpu_fault_objects w
    JOIN gpu_fault_objects i
      ON i.kind='incident'
     AND i.key=w.payload->>'incident_id'
    WHERE w.kind='workflow'
      AND i.payload->>'cluster_id'=%s
      AND i.payload->>'job_id'=%s
      AND (
            (
                w.payload->>'status' IN (
                    'PENDING', 'RUNNING', 'SAFETY_PENDING'
                )
                AND i.payload->>'attempt_id'=%s
            )
            OR EXISTS (
                SELECT 1
                FROM jsonb_array_elements(
                    coalesce(
                        w.payload->'step_executions',
                        '[]'::jsonb
                    )
                ) execution
                WHERE execution->>'operation'
                      ='RESTART_WORKLOAD'
                  AND execution->'details'
                      ->>'restart_attempt_id'=%s
            )
      )
    ORDER BY w.payload->>'updated_at' DESC,
             w.key DESC
    LIMIT %s
"""


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _insert_objects(rows: list[tuple[str, str, str]]) -> None:
    """Bulk-load ``(kind, key, payload_json)`` the way ``_put`` writes them,
    then ANALYZE so the planner sees the real distribution."""

    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT(kind, key) DO UPDATE SET payload=excluded.payload
                """,
                rows,
            )
            cursor.execute("ANALYZE gpu_fault_objects")


# Sequential scans and every sort strategy are priced out so the plan answers
# "can the index serve this query?" rather than "does the planner prefer it on
# this table size?".
_PLAN_ONLY_KNOBS = ("enable_seqscan", "enable_sort", "enable_incremental_sort")


def _plan(sql: str, params, knobs: tuple[str, ...] = _PLAN_ONLY_KNOBS) -> str:
    with _connect() as connection:
        with connection.cursor() as cursor:
            for knob in knobs:
                cursor.execute(f"SET {knob}=off")
            cursor.execute("EXPLAIN " + sql, params)
            return "\n".join(row[0] for row in cursor.fetchall())


def _seed_incidents_and_workflows(count: int = 400) -> None:
    rows: list[tuple[str, str, str]] = []
    for index in range(count):
        incident = fault_incident(
            f"inc-{index:04d}",
            f"event-{index:04d}",
            cluster_id=f"cluster-{index % 4}",
            state=INCIDENT_STATES[index % len(INCIDENT_STATES)],
            job_id=f"job-{index % 40}",
            attempt_id=f"attempt-{index % 80}",
            workflow_request_id=f"wf-{index:04d}",
            fencing_token=1,
        )
        workflow = workflow_request(
            f"wf-{index:04d}",
            incident.incident_id,
            status=WORKFLOW_STATUSES[index % len(WORKFLOW_STATUSES)],
            fencing_token=1,
            official_steps=[workflow_step(WorkflowOperation.RESTART_WORKLOAD)],
            step_executions=[
                workflow_step_execution(
                    0,
                    WorkflowOperation.RESTART_WORKLOAD,
                    WorkflowStepStatus.SUCCEEDED,
                    details={"restart_attempt_id": f"attempt-{(index + 1) % 80}"},
                )
            ]
            if index % 3 == 0
            else [],
            created_at=NOW + timedelta(seconds=index),
            updated_at=NOW + timedelta(seconds=2 * index),
        )
        rows.append(("incident", incident.incident_id, incident.model_dump_json()))
        rows.append(("workflow", workflow.request_id, workflow.model_dump_json()))
    _insert_objects(rows)


def _seed_decisions(count: int = 300) -> None:
    statuses = list(DecisionStatus)
    rows = []
    for index in range(count):
        decision = CompletionDecision(
            cluster_id=f"cluster-{index % 4}",
            attempt_id=f"attempt-{index}",
            event_key=f"event-{index:04d}",
            status=statuses[index % len(statuses)],
            reason="seeded",
        )
        rows.append(("decision", decision.event_key, decision.model_dump_json()))
    _insert_objects(rows)


def _seed_remote_commands(workflows: int = 60, per_workflow: int = 6) -> None:
    rows: list[tuple[str, str, str]] = []
    for w in range(workflows):
        incident = fault_incident(
            f"inc-{w:03d}",
            f"event-{w:03d}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=f"wf-{w:03d}",
            fencing_token=1,
        )
        workflow = workflow_request(
            f"wf-{w:03d}",
            incident.incident_id,
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
        )
        for step in range(per_workflow):
            command = RemoteActionCommand(
                command_id=f"cmd-{w:03d}-{step}",
                cluster_id=incident.cluster_id,
                workflow_request_id=workflow.request_id,
                incident_id=incident.incident_id,
                step_index=0,
                fencing_token=1,
                idempotency_key=f"{workflow.request_id}/{step}/RESTART_NODE",
                step=workflow.official_steps[0],
                workflow=workflow,
                incident=incident,
                status=COMMAND_STATUSES[(w + step) % len(COMMAND_STATUSES)],
                created_at=NOW + timedelta(seconds=step),
                updated_at=NOW + timedelta(seconds=step),
            )
            rows.append(
                ("remote_command", command.command_id, command.model_dump_json())
            )
    _insert_objects(rows)


# --- item G: the per-status counts are index range scans -------------------


def _per_status_plan(kind: str, field: str, values: list[str]) -> str:
    # ``kind`` and ``field`` are literals here, as they are in the store.
    sql = PER_STATUS_COUNT_SQL.replace("kind=%s", f"kind='{kind}'").replace(
        "payload->>%s", f"payload->>'{field}'"
    )
    return _plan(sql, (values,))


def test_workflow_status_counts_per_status_use_the_count_index() -> None:
    """``PostgresWorkflowMixin.workflow_status_counts`` (postgres/workflows.py)."""

    _seed_incidents_and_workflows()

    plan = _per_status_plan(
        "workflow", "status", [status.value for status in WorkflowStatus]
    )

    assert "gpu_fault_workflow_status_count" in plan, plan
    assert "Index Cond: ((payload ->> 'status'::text) = s.value)" in plan, plan
    assert "Seq Scan on gpu_fault_objects" not in plan, plan


def test_incident_state_counts_per_state_use_the_count_index() -> None:
    """``PostgresWorkflowMixin.incident_state_counts`` (postgres/workflows.py)."""

    _seed_incidents_and_workflows()

    plan = _per_status_plan(
        "incident", "state", [state.value for state in IncidentState]
    )

    assert "gpu_fault_incident_state_count" in plan, plan
    assert "Index Cond: ((payload ->> 'state'::text) = s.value)" in plan, plan
    assert "Seq Scan on gpu_fault_objects" not in plan, plan


def test_decision_status_counts_per_status_use_the_count_index() -> None:
    """``PostgresControlRecordMixin.decision_status_counts``
    (postgres/control_records.py)."""

    _seed_decisions()

    plan = _per_status_plan(
        "decision", "status", [status.value for status in DecisionStatus]
    )

    assert "gpu_fault_decision_status_count" in plan, plan
    assert "Index Cond: ((payload ->> 'status'::text) = s.value)" in plan, plan
    assert "Seq Scan on gpu_fault_objects" not in plan, plan


# --- item H1: remote commands by workflow in every status --------------------


def test_list_remote_commands_by_workflow_uses_the_all_status_index() -> None:
    """``PostgresRemoteCommandMixin.list_remote_commands(workflow_request_ids=)``
    (postgres/remote_commands.py) runs inside reconcile transactions that hold
    the workflow's advisory and row locks; it must not scan the kind."""

    _seed_remote_commands()

    plan = _plan(LIST_REMOTE_COMMANDS_SQL, (["wf-003", "wf-017", "wf-042"],))

    assert "gpu_fault_remote_command_workflow_all" in plan, plan
    assert "Seq Scan on gpu_fault_objects" not in plan, plan


def test_cancel_candidates_still_get_an_index_scan_after_the_partial_is_gone() -> None:
    """``cancel_remote_commands_for_workflow`` (postgres/remote_commands.py)
    used the open-only partial that v12 drops; the all-status index serves the
    same prefix with a status filter."""

    _seed_remote_commands()

    # Sorting stays enabled here: the query orders by ``key`` and the index
    # orders by ``created_at, key`` within one workflow, so the handful of
    # rows it returns are sorted afterwards. With sorts priced out the planner
    # would walk the primary key over the whole kind just to avoid that.
    plan = _plan(CANCEL_CANDIDATES_SQL, ("wf-017",), knobs=("enable_seqscan",))

    assert "gpu_fault_remote_command_workflow_all" in plan, plan
    assert "Seq Scan on gpu_fault_objects" not in plan, plan


def test_the_open_only_partial_index_is_no_longer_declared() -> None:
    from gpu_fault.store.postgres.ddl import declared_index_names

    names = declared_index_names()

    assert "gpu_fault_remote_command_workflow" not in names
    assert "gpu_fault_remote_command_workflow_all" in names
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('gpu_fault_remote_command_workflow')")
            assert cursor.fetchone()[0] is None, "v12 apply should have dropped it"


# --- item H2: the job-recovery join ------------------------------------------


def test_job_recovery_join_uses_the_incident_indexes_on_both_sides() -> None:
    """``PostgresWorkflowMixin.list_job_recovery_workflow_incidents``
    (postgres/workflows.py) runs once per attempt observation. The incident
    side is found through ``gpu_fault_incident_scope`` and the workflows of
    each incident through the new ``gpu_fault_workflow_incident``; nothing
    scans ``kind='workflow'``."""

    _seed_incidents_and_workflows()

    # Sorting stays enabled here: the join result is a handful of rows and its
    # ORDER BY is meant to be a sort. With sorts priced out the planner would
    # instead walk ``gpu_fault_workflow_updated_all`` (v13) backwards for the
    # order, which is the right answer to the wrong question.
    plan = _plan(
        JOB_RECOVERY_SQL,
        ("cluster-1", "job-13", "attempt-53", "attempt-53", 100),
        knobs=("enable_seqscan",),
    )

    assert "gpu_fault_workflow_incident" in plan, plan
    assert "gpu_fault_incident_scope" in plan, plan
    assert "Seq Scan on gpu_fault_objects" not in plan, plan


def test_job_recovery_reader_orders_by_text_updated_at() -> None:
    """Text order equals time order for rows written after item E, so the
    reader must not cast; the store returns the newest workflow first."""

    assert POSTGRES_URL is not None
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        for index, offset in enumerate((5, 1, 9, 3)):
            incident = fault_incident(
                f"inc-order-{index}",
                f"event-order-{index}",
                state=IncidentState.ACTION_PENDING,
                job_id="job-order",
                attempt_id="attempt-order",
                workflow_request_id=f"wf-order-{index}",
                fencing_token=1,
            )
            workflow = workflow_request(
                f"wf-order-{index}",
                incident.incident_id,
                status=WorkflowStatus.RUNNING,
                fencing_token=1,
                official_steps=[workflow_step(WorkflowOperation.RESTART_WORKLOAD)],
                created_at=NOW,
                updated_at=NOW + timedelta(minutes=offset),
            )
            store.save_incident_and_workflow(incident, workflow)
        rows = store.list_job_recovery_workflow_incidents(
            "cluster-a", "job-order", "attempt-order"
        )
    finally:
        store.close()

    assert [workflow.request_id for _, workflow in rows] == [
        "wf-order-2",
        "wf-order-0",
        "wf-order-3",
        "wf-order-1",
    ]


# ---------------------------------------------------------------------------
# Control-plane review 2026-09-08, G-2 / G-9 (schema v13). Same method: seed a
# realistic distribution, ANALYZE, price out sequential scans and sorts, and
# ask EXPLAIN whether the declared index can serve the query the store runs.
# ---------------------------------------------------------------------------

# ``list_incidents_with_missing_workflow`` with its pointer predicate spelled
# as ``> ''`` (a btree can serve ``>``; it cannot serve ``COALESCE(..) <> ''``).
# The store method (postgres/workflows.py, another owner) is to adopt this
# spelling; the semantics are identical for text pointers.
MISSING_WORKFLOW_SQL = """
    SELECT i.payload
    FROM gpu_fault_objects i
    WHERE i.kind='incident'
      AND i.payload->>'workflow_request_id' > ''
      AND NOT EXISTS (
          SELECT 1 FROM gpu_fault_objects w
          WHERE w.kind='workflow'
            AND w.key=i.payload->>'workflow_request_id'
      )
    ORDER BY i.payload->>'created_at', i.key
    LIMIT %s
"""
HAS_SUCCESSOR_SQL = """
    SELECT EXISTS (
        SELECT 1 FROM gpu_fault_objects
        WHERE kind='workflow'
          AND payload->>'predecessor_workflow_id'=%s
    )
"""
FLEET_TERMINAL_SWEEP_SQL = """
    SELECT key
    FROM gpu_fault_objects
    WHERE kind='fleet_deployment'
      AND payload->>'status' IN ('SUCCEEDED', 'FAILED')
      AND payload->>'updated_at' <= %s
    ORDER BY payload->>'updated_at', key
    LIMIT %s
"""
# The archiver's candidate scan in text order. Its current ``::timestamptz``
# form cannot be indexed (the cast is STABLE, not IMMUTABLE) nor use an index;
# ``control_record_archive.py`` (another owner) is to compare the isoformat
# text, as the fleet sweep already does.
ARCHIVE_CANDIDATE_SQL = """
    SELECT i.key
    FROM gpu_fault_objects i
    WHERE i.kind='incident'
      AND i.payload->>'updated_at' < %s
    ORDER BY i.payload->>'updated_at', i.key
    LIMIT %s
"""


def _seed_fleet_deployments(count: int = 300) -> None:
    rows = []
    for index in range(count):
        status = ("SUCCEEDED", "FAILED", "PLANNED", "RUNNING")[index % 4]
        payload = {
            "deployment_id": f"dep-{index:04d}",
            "cluster_id": f"cluster-{index % 4}",
            "status": status,
            "created_at": (NOW + timedelta(seconds=index)).isoformat(),
            "updated_at": (NOW + timedelta(seconds=2 * index)).isoformat(),
        }
        rows.append(
            (
                "fleet_deployment",
                payload["deployment_id"],
                __import__("json").dumps(payload),
            )
        )
    _insert_objects(rows)


def test_terminal_workflow_newest_first_scan_walks_the_all_status_updated_index():
    """The /metrics detail scan's terminal slice (G-2). Before v13 no index
    covered ``updated_at`` across statuses, so the newest-first read sorted the
    whole workflow kind on disk on every scrape."""
    _seed_incidents_and_workflows()
    sql, params = PostgresStore.workflow_scan_query(
        {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.SUPERSEDED},
        limit=101,
        newest_first=True,
    )

    plan = _plan(sql, params)

    assert "gpu_fault_workflow_updated_all" in plan, plan
    assert "Sort" not in plan, plan


def test_missing_workflow_inspection_reads_incident_pointers_from_the_index():
    _seed_incidents_and_workflows()

    # Sorting stays enabled: the anti-join's result is ordered by created_at,
    # which no pointer index can provide.
    plan = _plan(MISSING_WORKFLOW_SQL, (1000,), knobs=("enable_seqscan",))

    assert "gpu_fault_incident_workflow_pointer" in plan, plan


def test_successor_lookups_use_the_predecessor_index():
    _seed_incidents_and_workflows()

    plan = _plan(HAS_SUCCESSOR_SQL, ("wf-0001",))

    assert "gpu_fault_workflow_predecessor" in plan, plan


def test_fleet_deployment_retention_sweep_uses_the_terminal_index():
    """F-9: the terminal sweep ran every minute against a kind whose only
    status index is the open-rows complement."""
    _seed_fleet_deployments()

    plan = _plan(
        FLEET_TERMINAL_SWEEP_SQL, ((NOW + timedelta(seconds=100)).isoformat(), 100)
    )

    assert "gpu_fault_fleet_deployment_terminal" in plan, plan
    assert "Sort" not in plan, plan


def test_archiver_candidate_scan_uses_the_incident_archive_index():
    """F-I1 / F-8: the archiver orders incidents by a ``::timestamptz`` cast no
    text index can serve; the candidate index carries that exact expression."""
    _seed_incidents_and_workflows()

    plan = _plan(
        ARCHIVE_CANDIDATE_SQL, ((NOW + timedelta(seconds=100)).isoformat(), 25)
    )

    assert "gpu_fault_incident_archive_candidate" in plan, plan
    assert "Sort" not in plan, plan


def test_objects_table_autovacuum_is_tuned_for_a_hot_jsonb_table():
    """G-9: ``gpu_fault_objects`` is rewritten whole-row on every lease renewal
    and carries ~50 expression indexes, so the default 20 % dead-tuple
    threshold let a 300k-row table accumulate 60k dead tuples before a
    vacuum. The DDL sets the per-table factors, catalog-checked first."""
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT reloptions FROM pg_class WHERE relname='gpu_fault_objects'"
            )
            options = set(cursor.fetchone()[0] or [])

    assert "autovacuum_vacuum_scale_factor=0.02" in options, options
    assert "autovacuum_analyze_scale_factor=0.01" in options, options


def test_database_diagnostics_report_dead_tuples_of_the_objects_table():
    from gpu_fault.store.postgres.index_builder import database_diagnostics

    with _connect() as connection:
        with connection.cursor() as cursor:
            report = database_diagnostics(cursor)

    assert isinstance(report["gpu_fault_objects_n_dead_tup"], int), report
    assert isinstance(report["gpu_fault_objects_n_live_tup"], int), report
    assert set(report["gpu_fault_objects_autovacuum_options"]) >= {
        "autovacuum_vacuum_scale_factor=0.02",
        "autovacuum_analyze_scale_factor=0.01",
    }, report


# --- retention sweeps switched on by the review (F-8) and the marker OR ------

INACTIVE_MARKER_SWEEP_SQL = """
    SELECT marker.key
    FROM gpu_fault_objects AS marker
    WHERE marker.kind='marker'
      AND marker.payload->>'active'='false'
      AND marker.payload->>'observed_at' <= %s
    ORDER BY marker.payload->>'observed_at', marker.key
    LIMIT %s
"""
NOTIFICATION_SWEEP_SQL = """
    SELECT notification.key
    FROM gpu_fault_objects AS notification
    WHERE notification.kind='notification'
      AND notification.payload->>'created_at' <= %s
    ORDER BY notification.payload->>'created_at', notification.key
    LIMIT %s
"""
EVENT_ENDED_SQL = """
    SELECT event.key
    FROM gpu_fault_objects AS event
    WHERE event.kind='event'
      AND event.payload->>'ended_at' <= %s
    ORDER BY event.payload->>'ended_at', event.key
    LIMIT %s
"""
INCIDENT_BY_EVENT_SQL = """
    SELECT 1 FROM gpu_fault_objects AS incident
    WHERE incident.kind='incident'
      AND incident.payload->>'event_id'=%s
"""


def _seed_markers_and_events(count: int = 300) -> None:
    import json

    rows = []
    for index in range(count):
        observed = (NOW + timedelta(seconds=index)).isoformat()
        rows.append(
            (
                "marker",
                f"marker-{index:04d}",
                json.dumps(
                    {
                        "marker_id": f"marker-{index:04d}",
                        "incident_id": f"inc-{index:04d}",
                        "active": "true" if index % 3 else "false",
                        "trusted": "true" if index % 4 else "false",
                        "recommended_action": "RESET_GPU" if index % 2 else None,
                        "observed_at": observed,
                        "scope": {
                            "node_ids": [f"node-{index % 50}"],
                            "gpu_uuids": [f"GPU-{index % 200}"],
                            "fabric_partitions": [f"fabric-{index % 8}"],
                        },
                    }
                ),
            )
        )
        rows.append(
            (
                "event",
                f"event-{index:04d}",
                json.dumps({"event_id": f"event-{index:04d}", "ended_at": observed}),
            )
        )
        rows.append(
            (
                "notification",
                f"notif-{index:04d}",
                json.dumps(
                    {
                        "notification_id": f"notif-{index:04d}",
                        "priority": index % 3,
                        "created_at": observed,
                        "incident_id": f"inc-{index:04d}",
                    }
                ),
            )
        )
    _insert_objects(rows)


def test_inactive_marker_sweep_uses_its_partial_index():
    _seed_markers_and_events()
    plan = _plan(
        INACTIVE_MARKER_SWEEP_SQL, ((NOW + timedelta(seconds=100)).isoformat(), 100)
    )
    assert "gpu_fault_marker_inactive_observed" in plan, plan
    assert "Sort" not in plan, plan


def test_notification_sweep_uses_the_created_at_index():
    _seed_markers_and_events()
    plan = _plan(
        NOTIFICATION_SWEEP_SQL, ((NOW + timedelta(seconds=100)).isoformat(), 100)
    )
    assert "gpu_fault_notification_created_at" in plan, plan
    assert "Sort" not in plan, plan


def test_completion_record_sweep_reads_events_and_incident_pointers_from_indexes():
    _seed_incidents_and_workflows()
    _seed_markers_and_events()
    ended = _plan(EVENT_ENDED_SQL, ((NOW + timedelta(seconds=100)).isoformat(), 100))
    assert "gpu_fault_event_ended" in ended, ended
    pointer = _plan(INCIDENT_BY_EVENT_SQL, ("event-0007",))
    assert "gpu_fault_incident_event" in pointer, pointer
