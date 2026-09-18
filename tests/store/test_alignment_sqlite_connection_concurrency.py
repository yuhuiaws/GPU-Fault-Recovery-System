"""SQLite readers share the writer's connection, not its uncommitted transaction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event

import pytest

from gpu_fault.models import IncidentState, NotificationResult
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import fault_incident, workflow_request

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


class Cancelled(BaseException):
    pass


@pytest.fixture
def store(tmp_path):
    value = SqliteStore(str(tmp_path / "isolated-concurrency.db"))
    try:
        yield value
    finally:
        value.close()


def held_records(store: SqliteStore) -> None:
    store.save_incident(
        fault_incident(
            "held-incident",
            "held-event",
            fencing_token=3,
            workflow_request_id="held-workflow",
        )
    )
    store.save_workflow(workflow_request("held-workflow", "held-incident"))
    store.save_notification_result(
        NotificationResult(notification_id="held-result", status="SENT")
    )


def read_present(store: SqliteStore, operation: str) -> bool:
    queries = {
        "record": lambda: store.get_incident("held-incident"),
        "optional-record": lambda: store.get_notification_result("held-result"),
        "link": lambda: store.get_incident_by_event("held-event"),
        "list": store.list_workflows,
        "workflow-count": lambda: any(store.workflow_status_counts().values()),
        "incident-count": lambda: any(store.incident_state_counts().values()),
        "blocked-count": store.blocked_workflows_without_verified_restore,
        "orphan-query": lambda: store.list_orphan_workflows(
            created_before=NOW + timedelta(days=365)
        ),
        "missing-workflow-query": store.list_incidents_with_missing_workflow,
        "incident-query": lambda: store.list_incidents_by_state(
            "cluster-a", list(IncidentState)
        ),
        "active-query": lambda: store.list_active_workflow_incidents("cluster-a"),
        "job-query": lambda: store.list_job_recovery_workflow_incidents(
            "cluster-a", "job-a", "attempt-a"
        ),
        "fleet-query": lambda: store.list_active_fleet_deployments("cluster-a"),
        "cluster-query": store.list_regional_cluster_ids,
        "command-query": store.list_remote_commands,
        "open-command-query": lambda: store.find_open_remote_command(
            "held-workflow", 0, "official"
        ),
        "compound-command-query": lambda: store.find_remote_command_covering_step(
            "held-workflow", 0, "official", fencing_token=3
        ),
        "decision-count": lambda: any(store.decision_status_counts().values()),
        "completion-count": store.count_completion_events_without_decision,
        "marker-query": lambda: store.list_markers_for_incident("held-incident"),
    }
    try:
        return bool(queries[operation]())
    except NotFoundError:
        return False


@pytest.mark.parametrize(
    "operation",
    [
        "record",
        "optional-record",
        "link",
        "list",
        "workflow-count",
        "incident-count",
        "blocked-count",
        "orphan-query",
        "missing-workflow-query",
        "incident-query",
        "active-query",
        "job-query",
        "fleet-query",
        "cluster-query",
        "command-query",
        "open-command-query",
        "compound-command-query",
        "decision-count",
        "completion-count",
        "marker-query",
    ],
)
def test_readers_wait_for_a_foreign_transaction_and_never_observe_its_rollback(
    store: SqliteStore, operation: str
) -> None:
    entered, release, attempted, finished = Event(), Event(), Event(), Event()

    def owner() -> None:
        with pytest.raises(Cancelled):
            with store.collector_ingestion_transaction("cluster-a", "node-a", "held"):
                held_records(store)
                entered.set()
                assert release.wait(5), "test transaction release was not signalled"
                raise Cancelled

    def reader() -> bool:
        attempted.set()
        try:
            return read_present(store, operation)
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        holding = workers.submit(owner)
        try:
            assert entered.wait(5), "test transaction never acquired its connection"
            reading = workers.submit(reader)
            assert attempted.wait(5), "test reader never started"
            assert not finished.wait(0.05), (
                f"{operation} bypassed the foreign transaction's connection lock"
            )
        finally:
            release.set()
        holding.result(timeout=5)
        assert reading.result(timeout=5) is False, (
            f"{operation} observed state from an aborted ingestion"
        )


def test_concurrent_cached_statement_reads_cannot_mix_kinds_keys_or_missing_rows(
    store: SqliteStore,
) -> None:
    for index in range(4):
        store.save_incident(fault_incident(f"incident-{index}", f"event-{index}"))
        store.save_notification_result(
            NotificationResult(notification_id=f"result-{index}", status="SENT")
        )
    start = Barrier(4)

    def reader(index: int) -> None:
        start.wait(timeout=5)
        for _ in range(250):
            incident = store.get_incident(f"incident-{index}")
            notification = store.get_notification_result(f"result-{index}")
            missing = store.get_notification_result(f"missing-{index}")
            assert incident.incident_id == f"incident-{index}"
            assert incident.event_id == f"event-{index}"
            assert notification is not None
            assert notification.notification_id == f"result-{index}"
            assert notification.status == "SENT"
            assert missing is None, "a cache hit must not borrow another query's row"

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(reader, index) for index in range(4)]
        for future in futures:
            future.result(timeout=15)


def test_base_exception_rolls_back_nested_savepoint_without_aborting_outer_write(
    store: SqliteStore,
) -> None:
    with store.collector_ingestion_transaction("cluster-a", "node-a", "outer"):
        store.save_incident(fault_incident("before-inner", "before-event"))
        with pytest.raises(Cancelled):
            with store.collector_ingestion_transaction("cluster-a", "node-a", "inner"):
                held_records(store)
                raise Cancelled
        assert store.get_incident_by_event("held-event") is None, (
            "cancelled nested work must disappear before the outer transaction resumes"
        )
        store.save_incident(fault_incident("after-inner", "after-event"))
    assert store.get_incident("before-inner").event_id == "before-event"
    assert store.get_incident("after-inner").event_id == "after-event"
    assert store.get_incident_by_event("held-event") is None
    assert store.get_notification_result("held-result") is None
    assert store.list_workflows() == []


def test_cancelled_outer_transaction_does_not_capture_the_next_independent_write(
    store: SqliteStore, tmp_path
) -> None:
    with pytest.raises(Cancelled):
        with store.collector_ingestion_transaction("cluster-a", "node-a", "cancelled"):
            held_records(store)
            raise Cancelled
    store.save_incident(fault_incident("committed-later", "later-event"))
    reopened = SqliteStore(store.path)
    try:
        assert reopened.get_incident("committed-later").event_id == "later-event"
        assert reopened.get_incident_by_event("held-event") is None
        assert reopened.list_workflows() == []
    finally:
        reopened.close()
