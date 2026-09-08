"""Store side of F-G2 (6): the two completion gauges, and -- on PostgreSQL --
the atomicity of the completion transaction.

The PostgreSQL cases need ``GPU_FAULT_TEST_POSTGRES_URL``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from gpu_fault.app.context import default_simulated_profile
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    EffectiveRuntimeProfile,
    Environment,
    TerminalEvent,
    TerminalStatus,
)
from gpu_fault.service import CompletionService
from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStore()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "decisions.db"))
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


def _event(attempt_id: str, *, ended_at: datetime = NOW) -> TerminalEvent:
    return TerminalEvent(
        cluster_id="cluster-a",
        environment=Environment.KUBERNETES,
        job_id="job-a",
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.FAILED,
        ended_at=ended_at,
        runtime_profile_version="simulated-v1",
    )


def _decision(event: TerminalEvent, status: DecisionStatus) -> CompletionDecision:
    return CompletionDecision(
        cluster_id=event.cluster_id,
        attempt_id=event.attempt_id,
        event_key=event.event_key,
        status=status,
        reason="test",
    )


def test_decision_status_counts_and_orphan_events(store) -> None:
    decided = _event("attempt-decided")
    planned = _event("attempt-planned")
    undecided = _event("attempt-undecided")
    for event in (decided, planned, undecided):
        assert store.save_event_if_absent(event), f"{event.attempt_id} not inserted"
    store.save_decision(_decision(decided, DecisionStatus.NO_ACTION))
    store.save_decision(_decision(planned, DecisionStatus.PLAN_CREATED))

    assert store.decision_status_counts() == {
        DecisionStatus.NO_ACTION: 1,
        DecisionStatus.PLAN_CREATED: 1,
    }
    assert store.count_completion_events_without_decision() == 1

    store.save_decision(_decision(undecided, DecisionStatus.NO_ACTION))
    assert store.count_completion_events_without_decision() == 0


def test_completion_transaction_is_reentrant_and_serializes_by_key(store) -> None:
    """The service nests store writes (each with their own transaction) inside
    it, so it must be re-enterable on every backend."""

    event = _event("attempt-nested")
    with store.completion_transaction(event.event_key):
        with store.completion_transaction(event.event_key):
            assert store.save_event_if_absent(event), "nested write was rejected"
        store.save_decision(_decision(event, DecisionStatus.NO_ACTION))
    assert store.get_decision_by_event(event.event_key) is not None, (
        "the nested completion transaction did not commit"
    )


class _RaisingPlanner:
    def from_missing_allocation(
        self, event: TerminalEvent, profile: EffectiveRuntimeProfile
    ):
        raise RuntimeError("planner exploded after the event row was written")


def test_postgres_terminal_event_and_decision_are_one_transaction() -> None:
    """P0-48B: a crash between the event write and the decision write used to
    leave a poisoned event row. Inside one transaction both roll back."""

    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for store in postgres_store_instance():
        store.save_profile(default_simulated_profile())
        service = CompletionService(store, planner=_RaisingPlanner())
        event = _event("attempt-atomic")

        with pytest.raises(RuntimeError, match="planner exploded"):
            service.handle_terminal(event)

        assert store.get_decision_by_event(event.event_key) is None
        with pytest.raises(NotFoundError):
            store.get_event_by_attempt(event.cluster_id, event.attempt_id)
        assert store.count_completion_events_without_decision() == 0
    _truncate()
