from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.workflow_resolution import verified_restore_successor
from tests._builders import build_store, fault_incident, workflow_request
from tests.store._blocked_backlog_support import (
    FLAWS,
    blocked,
    break_one_clause,
    restore,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """A backend with an empty object table.

    Postgres is deliberately not a parameter here: the gauge is a whole-table
    aggregate, and this file runs in the parallel phase, which clears
    ``GPU_FAULT_TEST_POSTGRES_URL``. The ``jsonb`` rewrite is covered by
    ``tests/store/test_postgres_store.py``, which the Makefile runs serially with
    the variable set.
    """

    if request.param == "memory":
        yield build_store()
        return
    sqlite = SqliteStore(str(tmp_path / "backlog.db"))
    try:
        yield sqlite
    finally:
        sqlite.close()


def expected(store) -> int:
    """The same count computed through the normative predicate.

    ``blocked_workflows_without_verified_restore`` is a hand-written server-side
    aggregate in each backend, so nothing in the implementations shares code with
    ``verified_restore_successor``. Deriving the expectation from that function is
    what keeps the three copies from drifting away from it.
    """
    return sum(
        1
        for workflow in store.list_workflows(
            statuses={WorkflowStatus.BLOCKED}, limit=50
        )
        if verified_restore_successor(store, workflow) is None
    )


def test_a_blocked_workflow_holding_a_node_is_counted(store) -> None:
    blocked(store, "a")
    blocked(store, "b")

    assert store.blocked_workflows_without_verified_restore() == 2
    assert expected(store) == 2


def test_a_restored_node_leaves_the_count_without_an_operator_action(store) -> None:
    """The defect this gauge exists for.

    ``gpu_fault_workflow_total{status="BLOCKED"}`` counts the whole workflow
    history and BLOCKED is terminal, so it stays at 2 here forever: only
    ``workflow-reconcile`` or archival moves it. A threshold on that number
    stayed true after the node was back in the training pool and re-notified
    every ``repeat_interval``.
    """

    blocked(store, "a")
    blocked(store, "b")
    restore(store, "a")

    assert store.blocked_workflows_without_verified_restore() == 1
    assert expected(store) == 1
    assert store.workflow_status_counts()[WorkflowStatus.BLOCKED] == 2


@pytest.mark.parametrize("flaw", FLAWS)
def test_an_unverified_restore_still_counts(store, flaw: str) -> None:
    """Fail closed: anything short of a verified restore keeps the node held.

    Each case breaks exactly one clause of ``verified_restore_successor``. A
    backend answering from a cheaper predicate -- "the incident says RECOVERED",
    say -- would clear the alert while the node was still cordoned.
    """

    blocked(store, "a")
    restore(store, "a")
    break_one_clause(store, flaw, "a")

    assert store.blocked_workflows_without_verified_restore() == 1
    assert expected(store) == 1


def test_workflows_in_other_statuses_are_not_counted(store) -> None:
    blocked(store, "a")
    store.save_incident(fault_incident("incident-live", "event-live"))
    for status in (
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.FAILED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.SUPERSEDED,
    ):
        store.save_workflow(
            workflow_request(
                f"workflow-{status.value.lower()}", "incident-live", status=status
            )
        )

    assert store.blocked_workflows_without_verified_restore() == 1


def test_an_empty_table_reports_no_backlog(store) -> None:
    assert store.blocked_workflows_without_verified_restore() == 0


def test_the_backends_agree_on_the_same_records(tmp_path) -> None:
    """Two hand-written aggregates, one answer.

    The memory backend walks its dicts while SQLite runs a correlated
    ``NOT EXISTS``, so agreement is not structural.
    """

    memory = InMemoryStore()
    sqlite = SqliteStore(str(tmp_path / "agree.db"))
    try:
        for target in (memory, sqlite):
            blocked(target, "a")
            blocked(target, "b")
            blocked(target, "c")
            restore(target, "a")
            restore(target, "b")
            break_one_clause(target, "incident_fencing_token", "b")

        assert (
            memory.blocked_workflows_without_verified_restore()
            == sqlite.blocked_workflows_without_verified_restore()
            == expected(memory)
            == 2
        )
    finally:
        sqlite.close()
