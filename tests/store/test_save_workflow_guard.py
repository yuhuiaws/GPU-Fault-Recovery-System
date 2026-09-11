"""``save_workflow`` never overwrites a row that moved since the caller read it.

Store review 2026-09-07, item B. PostgreSQL inherited the SQLite blind upsert
under a process lock that ``PostgresStore`` neutralizes, so a copy read before
a merge (``merge_revision``), a re-lease (``execution_epoch``) or a generation
change (``fencing_token``) was written straight over them. Without ``expected``
the write is now guarded on those three fields and names the one that moved;
with ``expected`` it is a compare-and-set on the whole payload.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "guard.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _seed(store):
    incident = fault_incident(
        "inc-g",
        "event-g",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-g",
        fencing_token=3,
    )
    workflow = workflow_request(
        "wf-g",
        "inc-g",
        fencing_token=3,
        official_steps=[workflow_step(WorkflowOperation.RESET_GPU)],
    )
    store.save_incident_and_workflow(incident, workflow)
    return store.get_workflow("wf-g")


def test_a_new_row_is_inserted(store) -> None:
    workflow = workflow_request("wf-new", "inc-new")

    store.save_workflow(workflow)

    assert store.get_workflow("wf-new") == workflow


def test_unchanged_versions_land(store) -> None:
    stored = _seed(store)

    store.save_workflow(copy_model(stored, status=WorkflowStatus.FAILED))

    assert store.get_workflow("wf-g").status is WorkflowStatus.FAILED


def test_a_copy_read_before_a_merge_is_refused_and_names_the_field(store) -> None:
    stored = _seed(store)
    store.amend_workflow("wf-g", {"blocked_reasons": ["widened"]})

    with pytest.raises(StaleWriteError, match="merge_revision") as raised:
        store.save_workflow(copy_model(stored, status=WorkflowStatus.FAILED))

    assert "re-read" in str(raised.value)
    assert "execution_epoch" not in str(raised.value), (
        "only the field that moved is named"
    )
    current = store.get_workflow("wf-g")
    assert current.status is WorkflowStatus.PENDING
    assert current.blocked_reasons == ["widened"]


def test_a_copy_read_before_a_lease_is_refused(store) -> None:
    stored = _seed(store)
    store.claim_workflow("wf-g", "executor-a", 3)

    with pytest.raises(StaleWriteError, match="execution_epoch"):
        store.save_workflow(copy_model(stored, status=WorkflowStatus.FAILED))

    assert store.get_workflow("wf-g").execution_owner_id == "executor-a"


def test_a_copy_read_before_a_generation_change_is_refused(store) -> None:
    stored = _seed(store)
    store.save_incident_and_workflow(
        copy_model(store.get_incident("inc-g"), fencing_token=4),
        copy_model(stored, fencing_token=4),
    )

    with pytest.raises(StaleWriteError, match="fencing_token"):
        store.save_workflow(copy_model(stored, status=WorkflowStatus.FAILED))

    assert store.get_workflow("wf-g").fencing_token == 4


def test_expected_mismatch_is_refused(store) -> None:
    stored = _seed(store)
    # A write that keeps every version field still changes the payload.
    store.save_workflow(copy_model(stored, blocked_reasons=["moved"]))

    with pytest.raises(StaleWriteError):
        store.save_workflow(
            copy_model(stored, status=WorkflowStatus.FAILED), expected=stored
        )

    current = store.get_workflow("wf-g")
    assert current.status is WorkflowStatus.PENDING
    assert current.blocked_reasons == ["moved"]


def test_expected_match_lands(store) -> None:
    stored = _seed(store)

    store.save_workflow(
        copy_model(stored, status=WorkflowStatus.FAILED), expected=stored
    )

    assert store.get_workflow("wf-g").status is WorkflowStatus.FAILED


def test_expected_on_a_missing_row_is_refused(store) -> None:
    workflow = workflow_request("wf-missing", "inc-missing")

    with pytest.raises(StaleWriteError):
        store.save_workflow(workflow, expected=workflow)

    assert store.list_workflows() == []


def test_save_then_read_round_trips_a_fresh_copy(store) -> None:
    """The ordinary test idiom -- read, modify, save -- keeps working."""

    _seed(store)
    for status in (WorkflowStatus.RUNNING, WorkflowStatus.SUCCEEDED):
        store.save_workflow(copy_model(store.get_workflow("wf-g"), status=status))
        assert store.get_workflow("wf-g").status is status


def test_compare_and_set_matches_a_row_written_before_a_field_existed() -> None:
    """A stored row from before a model field existed lacks that key; the
    model decodes it with the default and re-encodes with it, so a literal
    payload compare never matches even though nothing moved. Live 2026-09-11:
    18 BLOCKED records from a week earlier failed the dispatcher's settled
    sweep on every tick with StaleWriteError. The CAS compares the decoded row
    instead, and still refuses a row that really changed."""

    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    import psycopg

    for postgres in postgres_store_instance():
        read = _seed(postgres)
        with psycopg.connect(
            os.environ["GPU_FAULT_TEST_POSTGRES_URL"], autocommit=True
        ) as conn:
            conn.execute(
                """
                UPDATE gpu_fault_objects
                SET payload = payload - 'placement_hold' - 'superseded_step_indexes'
                WHERE kind='workflow' AND key='wf-g'
                """
            )
        assert postgres.get_workflow("wf-g") == read, "defaults decode alike"

        postgres.save_workflow(
            copy_model(read, status=WorkflowStatus.SUPERSEDED), expected=read
        )
        assert postgres.get_workflow("wf-g").status is WorkflowStatus.SUPERSEDED

        moved = postgres.get_workflow("wf-g")
        with psycopg.connect(
            os.environ["GPU_FAULT_TEST_POSTGRES_URL"], autocommit=True
        ) as conn:
            # A concurrent writer moved the row after ``moved`` was read.
            conn.execute(
                """
                UPDATE gpu_fault_objects
                SET payload = jsonb_set(payload, '{fencing_token}', '9')
                WHERE kind='workflow' AND key='wf-g'
                """
            )
        with pytest.raises(StaleWriteError):
            postgres.save_workflow(
                copy_model(moved, status=WorkflowStatus.FAILED), expected=moved
            )
    _truncate()
