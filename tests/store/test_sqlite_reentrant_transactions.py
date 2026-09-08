"""SQLite transactions nest, so the two collector-side contexts are real (F-M1 / F-G2(3)).

``_state_transaction`` used to issue ``BEGIN IMMEDIATE`` unconditionally and
therefore could not be entered while a transaction was open; the SQLite store
inherited the in-memory no-op ``collector_ingestion_transaction`` and offered a
lock-only ``completion_transaction``. An incident and the finding (or a
completion event and its decision) committed separately, and a failure between
the two left the first half behind. Nested entries are now savepoints of the
outer transaction, and both contexts open that outer transaction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.store import NotFoundError, SqliteStore
from gpu_fault.store.shared.errors import StaleFencingTokenError
from tests._builders import fault_incident, workflow_request

NOW = datetime(2026, 9, 6, 13, 0, tzinfo=timezone.utc)
KEY = "cluster-a/node-a/network_link_down/eth0"


@pytest.fixture
def store(tmp_path):
    sqlite = SqliteStore(str(tmp_path / "reentrant.db"))
    try:
        yield sqlite
    finally:
        sqlite.close()


def _incident(incident_id: str = "incident-a"):
    return fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-a",
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )


def _workflow(request_id: str = "workflow-a"):
    return workflow_request(
        request_id,
        "incident-a",
        WorkflowStatus.PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )


def test_a_failure_between_two_ingestion_writes_leaves_neither_row(store) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with store.collector_ingestion_transaction("cluster-a", "node-a", "batch-1"):
            store.save_incident(_incident())
            store.save_workflow(_workflow())
            assert store.get_workflow("workflow-a").request_id == "workflow-a"
            raise RuntimeError("boom")

    with pytest.raises(NotFoundError):
        store.get_incident("incident-a")
    with pytest.raises(NotFoundError):
        store.get_workflow("workflow-a")
    assert store.get_incident_by_event("event-incident-a") is None


def test_a_health_signal_claim_rolls_back_with_its_ingestion_transaction(store) -> None:
    """P0-38B on SQLite: the claim inside a failed ingestion is undone, so the
    replay re-claims the signal instead of finding it already spent."""

    with pytest.raises(RuntimeError, match="incident write failed"):
        with store.collector_ingestion_transaction("cluster-a", "node-a", "batch-1"):
            emitted = store.claim_health_signal_transitions(
                [(KEY, True, NOW, 0.0)], received_at=NOW
            )
            assert emitted == [True]
            raise RuntimeError("incident write failed")

    assert store.get_health_signal_state(KEY) is None
    assert store.claim_health_signal_transitions(
        [(KEY, True, NOW, 0.0)], received_at=NOW
    ) == [True]


def test_a_nested_failure_rolls_back_only_its_savepoint(store) -> None:
    """The outer transaction survives a nested store call that raised: the
    stale claim's own writes are undone, the incident written before it stays."""

    store.save_workflow(_workflow())

    with store.collector_ingestion_transaction("cluster-a", "node-a", "batch-2"):
        store.save_incident(_incident())
        with pytest.raises(StaleFencingTokenError):
            store.claim_workflow("workflow-a", "executor-a", fencing_token=99, now=NOW)
        store.save_workflow(
            _workflow().model_copy(update={"updated_at": NOW + timedelta(seconds=1)})
        )

    assert store.get_incident("incident-a").incident_id == "incident-a"
    assert store.get_workflow("workflow-a").updated_at == NOW + timedelta(seconds=1)
    assert store.get_workflow("workflow-a").execution_owner_id is None


def test_completion_transaction_commits_event_and_decision_together(store) -> None:
    with pytest.raises(RuntimeError, match="decision failed"):
        with store.completion_transaction("event-key-1"):
            store.save_incident(_incident("incident-completion"))
            store.save_workflow(_workflow("workflow-completion"))
            raise RuntimeError("decision failed")

    with pytest.raises(NotFoundError):
        store.get_incident("incident-completion")
    with pytest.raises(NotFoundError):
        store.get_workflow("workflow-completion")


def test_a_successful_outer_transaction_commits_every_nested_write(store) -> None:
    with store.collector_ingestion_transaction("cluster-a", "node-a", "batch-3"):
        store.save_incident(_incident())
        store.save_workflow(_workflow())
        assert store.claim_health_signal_transitions(
            [(KEY, True, NOW, 0.0)], received_at=NOW
        ) == [True]

    assert store.get_incident("incident-a").incident_id == "incident-a"
    assert store.get_workflow("workflow-a").request_id == "workflow-a"
    state = store.get_health_signal_state(KEY)
    assert state is not None, "the claim inside the transaction must be committed"
    assert state.active is True


def test_sqlite_db_file_is_owner_only(tmp_path) -> None:
    """M-7: the on-disk state file holds tenant fault data and tokens.

    It must never be created with the process umask (which is commonly
    world/group readable); every persisted file must be 0600 and the
    containing directory 0700.
    """
    import os
    import stat

    db_dir = tmp_path / "secure-state"
    db_path = db_dir / "state.db"
    store = SqliteStore(str(db_path))
    try:
        # Force the WAL sidecar files into existence with a real write.
        store.save_incident(_incident("perm-check"))
        for suffix in ("", "-wal", "-shm"):
            candidate = db_dir / f"state.db{suffix}"
            if candidate.exists():
                mode = stat.S_IMODE(os.stat(candidate).st_mode)
                assert mode == 0o600, (
                    f"{candidate.name} mode is {oct(mode)}, expected 0o600"
                )
        dir_mode = stat.S_IMODE(os.stat(db_dir).st_mode)
        assert dir_mode == 0o700, f"dir mode is {oct(dir_mode)}, expected 0o700"
    finally:
        store.close()
