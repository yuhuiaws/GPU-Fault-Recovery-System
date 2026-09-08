"""The dispatcher's hot queries must be index scans, and the schema check
must notice a missing index or a disabled trigger.

FINAL-建议汇总 F-A9 / F-J3 / F-D10 (P0-73C, P1-73E, P1-73F, P2-73H, P0-74B,
P1-75D). The repository's DDL runs in one transaction, so a new index cannot
be built ``CONCURRENTLY`` here -- the three-step method is: operators build
it concurrently, the DDL declares it ``IF NOT EXISTS``, and the schema check
refuses to start when it is missing, so a forgotten step is loud instead of a
silent full-table scan on every tick.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.ddl import declared_index_names
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
NOW = datetime(2026, 9, 5, 21, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _store(*, initialize_schema: bool = False) -> PostgresStore:
    assert POSTGRES_URL is not None
    return PostgresStore(POSTGRES_URL, initialize_schema=initialize_schema)


def _seed(store: PostgresStore, count: int = 30) -> None:
    statuses = [
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
    ]
    for index in range(count):
        status = statuses[index % len(statuses)]
        incident = fault_incident(
            f"inc-{index}",
            f"event-{index}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=f"wf-{index}",
            fencing_token=1,
        )
        workflow = workflow_request(
            f"wf-{index}",
            f"inc-{index}",
            status=status,
            fencing_token=1,
            official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
            created_at=NOW,
            updated_at=NOW,
        )
        store.save_incident_and_workflow(incident, workflow)


# The plan assertions ask "can the index serve this query?", not "does this
# planner, on this table size and cost model, prefer it?" - so sequential scans
# and every sort strategy are priced out first. A Sort node can then only
# appear when no index yields the order (see test_postgres_claim_window.py).
_PLAN_ONLY_KNOBS = ("enable_seqscan", "enable_sort", "enable_incremental_sort")


def _plan(store: PostgresStore, sql: str, params) -> str:
    with store._db.cursor() as cursor:
        for knob in _PLAN_ONLY_KNOBS:
            cursor.execute(f"SET {knob}=off")
        try:
            cursor.execute("EXPLAIN " + sql, params)
            return "\n".join(row[0] for row in cursor.fetchall())
        finally:
            for knob in _PLAN_ONLY_KNOBS:
                cursor.execute(f"SET {knob}=on")


def test_postgres_dispatch_candidate_scan_uses_the_executable_order_index():
    store = _store()
    try:
        _seed(store)
        sql, params = store.workflow_scan_query(
            {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            },
            limit=100,
            newest_first=False,
        )
        plan = _plan(store, sql, params)

        assert "gpu_fault_executable_workflow_order" in plan, plan
        assert "Seq Scan on gpu_fault_objects" not in plan, plan
        # The literal IN list is what lets the planner prove the partial index.
        assert "= ANY" not in sql
    finally:
        store.close()


def test_postgres_pushed_down_dispatch_scan_still_walks_the_executable_index():
    """F-A2(b): the not_before / predecessor / retired filters ride along as
    residual filters on the open-workflow partial index.

    Since F-A2(a) the dispatch scan orders by eligibility rather than
    ``updated_at``; the first page has no cursor, so the planner may take the
    rows off either executable-status partial index -- never a sequential
    scan. The paged form below must walk
    ``gpu_fault_executable_workflow_dispatch_order`` without a sort.
    """

    store = _store()
    try:
        _seed(store)
        sql, params = store.workflow_scan_query(
            {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            },
            limit=100,
            newest_first=False,
            dispatchable_at=NOW,
            exclude_request_ids={"wf-retired"},
        )
        plan = _plan(store, sql, params)

        assert "gpu_fault_executable_workflow_" in plan, plan
        assert "Seq Scan on gpu_fault_objects" not in plan, plan
        assert "::timestamptz" not in sql
    finally:
        store.close()


def test_postgres_unhandled_failed_scan_uses_the_unhandled_failed_index():
    store = _store()
    try:
        _seed(store)
        sql, params = store.unhandled_failed_workflows_query(limit=100)
        plan = _plan(store, sql, params)

        assert "gpu_fault_unhandled_failed_workflow_updated" in plan, plan
        assert "Seq Scan on gpu_fault_objects" not in plan, plan
    finally:
        store.close()


def test_postgres_blocked_scan_uses_the_blocked_index():
    store = _store()
    try:
        _seed(store)
        sql, params = store.workflow_scan_query(
            {WorkflowStatus.BLOCKED}, limit=1001, newest_first=False
        )
        plan = _plan(store, sql, params)

        assert "gpu_fault_blocked_workflow_updated" in plan, plan
        assert "Seq Scan on gpu_fault_objects" not in plan, plan
    finally:
        store.close()


def test_declared_indexes_include_the_dispatcher_indexes():
    names = declared_index_names()

    assert {
        "gpu_fault_executable_workflow_order",
        "gpu_fault_executable_workflow_dispatch_order",
        "gpu_fault_unhandled_failed_workflow_updated",
        "gpu_fault_blocked_workflow_updated",
        "gpu_fault_active_workflow_scope",
        "gpu_fault_workflow_incident",
    } <= names


def test_postgres_schema_validation_fails_closed_when_a_declared_index_is_missing():
    import psycopg

    _store().close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP INDEX IF EXISTS gpu_fault_executable_workflow_order")
    try:
        with pytest.raises(RuntimeError, match="gpu_fault_executable_workflow_order"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        _store(initialize_schema=True).close()


def test_postgres_schema_validation_fails_closed_when_a_counter_trigger_is_disabled():
    import psycopg

    _store().close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT tgname FROM pg_trigger
                WHERE tgrelid='gpu_fault_processor_queue'::regclass
                  AND NOT tgisinternal
                ORDER BY tgname
                """
            )
            names = [row[0] for row in cursor.fetchall()]
            assert names, "expected processor queue triggers"
            cursor.execute(
                f"ALTER TABLE gpu_fault_processor_queue DISABLE TRIGGER {names[0]}"
            )
    try:
        with pytest.raises(RuntimeError, match="disabled"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"ALTER TABLE gpu_fault_processor_queue ENABLE TRIGGER {names[0]}"
                )
        _store(initialize_schema=True).close()


def test_postgres_paged_dispatch_scan_walks_the_dispatch_order_index_without_a_sort():
    """F-A2(a)(c): ordered by dispatch_eligible_at with a row-value cursor, the
    page is one index range scan -- no Sort node, no sequential scan.

    ``gpu_fault_executable_workflow_dispatch_order`` is declared by the DDL
    since v12 (store review 2026-09-07, item H3); until then this test built it
    itself and production sorted every tick.
    """

    store = _store()
    try:
        _seed(store, count=400)
        anchor = store.list_workflows(
            {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            },
            limit=1,
            dispatchable_at=NOW,
        )[0]
        sql, params = store.workflow_scan_query(
            {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            },
            limit=100,
            dispatchable_at=NOW,
            exclude_request_ids={"wf-retired"},
            after=anchor,
        )
        plan = _plan(store, sql, params)

        assert (
            "Index Scan using gpu_fault_executable_workflow_dispatch_order "
            "on gpu_fault_objects w"
        ) in plan, plan
        assert "Sort" not in plan, plan
        assert "Seq Scan on gpu_fault_objects" not in plan, plan
        assert "::timestamptz" not in sql
    finally:
        store.close()


# Control-plane review 2026-09-08, G-6: presence by name was the whole check.
# A CREATE INDEX CONCURRENTLY that failed leaves an INVALID index the planner
# never uses, and an index recreated by hand with another definition keeps its
# name; both passed the startup check and degraded the hot queries silently.


def _set_index_validity(name: str, valid: bool) -> None:
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE pg_index SET indisvalid=%s WHERE indexrelid=to_regclass(%s)",
                (valid, name),
            )


def test_postgres_schema_validation_fails_closed_when_a_declared_index_is_invalid():
    _store().close()
    _set_index_validity("gpu_fault_executable_workflow_order", False)
    try:
        with pytest.raises(
            RuntimeError, match="invalid.*gpu_fault_executable_workflow_order"
        ):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        _set_index_validity("gpu_fault_executable_workflow_order", True)
        _store(initialize_schema=True).close()


def test_postgres_schema_validation_fails_closed_when_an_index_definition_drifted():
    import psycopg

    _store().close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP INDEX IF EXISTS gpu_fault_executable_workflow_order")
            cursor.execute(
                """
                CREATE INDEX gpu_fault_executable_workflow_order
                ON gpu_fault_objects (key)
                WHERE kind='workflow'
                """
            )
    try:
        with pytest.raises(
            RuntimeError, match="definition.*gpu_fault_executable_workflow_order"
        ):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
        # The idempotent DDL sees the name present and leaves the drifted
        # definition alone; the preflight names it and the online builder is
        # the sanctioned repair (drop, then --build-indexes-concurrently).
        with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
            from gpu_fault.store.postgres.index_builder import schema_preflight

            report = schema_preflight(connection)
        assert report["indexes"]["drifted"] == ["gpu_fault_executable_workflow_order"]
        assert report["ok"] is False
    finally:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DROP INDEX IF EXISTS gpu_fault_executable_workflow_order"
                )
        _store(initialize_schema=True).close()
