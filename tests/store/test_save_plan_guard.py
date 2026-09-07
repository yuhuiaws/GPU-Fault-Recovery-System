"""``save_plan(expected=)`` is a compare-and-set on the plan the caller read.

Architecture review 2026-09-07, item D2. Three writers rewrite plan rows: the
dispatcher's ``_sync_plan`` (get_plan -> save_plan, no lock), the reconcile
transaction (row lock + ``_put``) and the workflow-reconcile admin path. On
PostgreSQL the inherited ``save_plan`` ran under a process lock the store
neutralizes, so the dispatcher's mirror of a workflow status could land over a
reconcile that had just resolved the plan. ``RecoveryPlan`` has no version
field, so the only guard is the whole payload the caller read.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.models import PlanStatus, RecoveryPlan
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import build_store, copy_model
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
        sqlite = SqliteStore(str(tmp_path / "plan-guard.db"))
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


def _plan(plan_id: str = "plan-g") -> RecoveryPlan:
    return RecoveryPlan(
        plan_id=plan_id,
        incident_id="inc-g",
        attempt_id="attempt-g",
        trigger="terminal-event",
        runtime_profile_version="profile-v1",
        steps=[],
    )


def test_a_new_plan_is_inserted(store) -> None:
    plan = _plan()

    store.save_plan(plan)

    assert store.get_plan("plan-g") == plan


def test_an_unconditional_save_still_lands(store) -> None:
    store.save_plan(_plan())

    store.save_plan(copy_model(store.get_plan("plan-g"), status=PlanStatus.RUNNING))

    assert store.get_plan("plan-g").status is PlanStatus.RUNNING


def test_expected_mismatch_is_refused(store) -> None:
    store.save_plan(_plan())
    stored = store.get_plan("plan-g")
    # The reconcile resolved the plan while the dispatcher held its copy.
    store.save_plan(
        copy_model(
            stored,
            status=PlanStatus.SUPERSEDED,
            resolved_by_restore_workflow_id="wf-restore",
        )
    )

    with pytest.raises(StaleWriteError):
        store.save_plan(copy_model(stored, status=PlanStatus.RUNNING), expected=stored)

    current = store.get_plan("plan-g")
    assert current.status is PlanStatus.SUPERSEDED
    assert current.resolved_by_restore_workflow_id == "wf-restore"


def test_expected_match_lands(store) -> None:
    store.save_plan(_plan())
    stored = store.get_plan("plan-g")

    store.save_plan(copy_model(stored, status=PlanStatus.SUCCEEDED), expected=stored)

    assert store.get_plan("plan-g").status is PlanStatus.SUCCEEDED


def test_expected_on_a_missing_row_is_refused(store) -> None:
    plan = _plan("plan-missing")

    with pytest.raises(StaleWriteError):
        store.save_plan(plan, expected=plan)
