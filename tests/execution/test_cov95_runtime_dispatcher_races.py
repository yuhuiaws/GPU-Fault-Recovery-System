from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution import WorkflowExecutionError
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowStructureError
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store import NotFoundError
from tests.execution._cov95_runtime_workflows import FlowHarness, RecordingAdapter

FREEZE = WorkflowOperation.FREEZE_EVIDENCE


class RacingAdapter(RecordingAdapter):
    def __init__(self, race: Callable[[], None], error: Exception) -> None:
        super().__init__()
        self.race = race
        self.error = error

    def supports(self, step: Any) -> bool:
        self.race()
        raise self.error


@pytest.mark.parametrize(
    "field", ["poll_interval_seconds", "batch_size", "max_workers"]
)
def test_invalid_dispatch_capacity_is_rejected_before_starting_workers(
    field: str,
) -> None:
    h = FlowHarness([FREEZE])
    with pytest.raises(WorkflowExecutionError, match="must be positive"):
        WorkflowDispatcher(
            h.store, h.executor, WorkflowDispatcherConfig(enabled=True, **{field: 0})
        )
    assert h.adapter.calls == [], (
        "invalid dispatcher configuration must not execute work"
    )


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("peer_change", ["lease", "terminal"])
def test_internal_error_cannot_overwrite_a_peer_lease_or_terminal_result(
    blocking: bool, peer_change: str
) -> None:
    h = FlowHarness([FREEZE])
    snapshots = []

    def race() -> None:
        current = h.store.get_workflow(h.workflow.request_id)
        updates = (
            {"execution_owner_id": "peer-executor"}
            if peer_change == "lease"
            else {
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "completed_step_indexes": [0],
                "completed_operations": [FREEZE],
            }
        )
        h.store.save_workflow(current.model_copy(update=updates), expected=current)
        snapshots.append(h.store.get_workflow(current.request_id))

    error = (
        WorkflowStructureError("invalid graph")
        if blocking
        else RuntimeError("replica wiring")
    )
    h.executor.adapters = [RacingAdapter(race, error)]
    dispatcher = WorkflowDispatcher(
        h.store, h.executor, WorkflowDispatcherConfig(enabled=True, max_workers=1)
    )
    try:
        report = dispatcher.run_once()
    finally:
        dispatcher.stop()
    assert len(snapshots) == 1, snapshots
    assert h.store.get_workflow(h.workflow.request_id) == snapshots[0], (
        "an error from this replica must not undo the peer's durable state",
        report,
    )
    assert report.failed == int(blocking) and report.internal_errors == int(
        not blocking
    ), report
    assert dispatcher.sweep_errors_total.get("internal_error_block", 0) == int(
        blocking and peer_change == "lease"
    ), dispatcher.sweep_errors_total
    assert (
        h.store.get_incident(h.incident.incident_id).state
        is IncidentState.ACTION_PENDING
    ), report


def test_error_before_claim_is_released_with_backoff_and_succeeds_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = FlowHarness([FREEZE])
    original = h.store.claim_workflow
    attempts = 0

    def claim(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("local claim decoding failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(h.store, "claim_workflow", claim)
    dispatcher = WorkflowDispatcher(
        h.store,
        h.executor,
        WorkflowDispatcherConfig(
            enabled=True, max_workers=1, internal_error_backoff_seconds=60
        ),
    )
    try:
        first = dispatcher.run_once()
        held = h.store.get_workflow(h.workflow.request_id)
        assert first.internal_errors == 1 and first.executed == 0, first
        assert (
            held.execution_owner_id is None and held.status is WorkflowStatus.PENDING
        ), held
        assert held.not_before is not None and held.not_before > datetime.now(
            timezone.utc
        ), held
        second = dispatcher.run_once()
        assert second.executed == 0 and not h.adapter.calls, second
        h.amend(not_before=datetime.now(timezone.utc) - timedelta(seconds=1))
        third = dispatcher.run_once()
        assert third.completed == 1, third
        assert len(h.adapter.calls) == 1, h.adapter.calls
    finally:
        dispatcher.stop()


def test_disappearing_row_during_internal_error_release_does_not_abort_the_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = FlowHarness([FREEZE])
    missing = False
    original = h.store.get_workflow

    def read(request_id: str) -> Any:
        if missing:
            raise NotFoundError("row archived by another writer")
        return original(request_id)

    def race() -> None:
        nonlocal missing
        missing = True

    monkeypatch.setattr(h.store, "get_workflow", read)
    h.executor.adapters = [RacingAdapter(race, RuntimeError("adapter unavailable"))]
    dispatcher = WorkflowDispatcher(
        h.store, h.executor, WorkflowDispatcherConfig(enabled=True, max_workers=1)
    )
    try:
        report = dispatcher.run_once()
    finally:
        dispatcher.stop()
    assert report.internal_errors == 1 and report.executed == 0, report
    assert dispatcher.sweep_errors_total == {}, dispatcher.sweep_errors_total
    assert h.adapter.calls == [], "no workflow adapter executed after the failure"


def test_structural_error_can_block_its_row_without_fabricating_incident_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = FlowHarness([FREEZE])
    missing = False
    original = h.store.get_incident

    def read(incident_id: str) -> Any:
        if missing:
            raise NotFoundError("incident temporarily unavailable")
        return original(incident_id)

    def race() -> None:
        nonlocal missing
        missing = True

    monkeypatch.setattr(h.store, "get_incident", read)
    h.executor.adapters = [
        RacingAdapter(race, WorkflowStructureError("cyclic dependency"))
    ]
    dispatcher = WorkflowDispatcher(
        h.store, h.executor, WorkflowDispatcherConfig(enabled=True, max_workers=1)
    )
    try:
        report = dispatcher.run_once()
    finally:
        dispatcher.stop()
    saved = h.store.get_workflow(h.workflow.request_id)
    assert report.failed == 1 and report.internal_errors == 0, report
    assert (
        saved.status is WorkflowStatus.BLOCKED
        and saved.blocked_kind is BlockedKind.INTERNAL_ERROR
    ), saved
    assert saved.execution_owner_id is None, saved
    assert original(h.incident.incident_id).state is IncidentState.ACTION_PENDING, (
        "missing incident evidence must not be replaced with an invented incident"
    )


def test_disabled_dispatch_loop_starts_no_scan_and_can_be_stopped() -> None:
    h = FlowHarness([FREEZE])
    dispatcher = WorkflowDispatcher(
        h.store, h.executor, WorkflowDispatcherConfig(enabled=False)
    )
    try:
        dispatcher.run_forever()
        assert dispatcher.last_cycle_timestamp_seconds == 0, (
            "disabled loop must not scan"
        )
        assert (
            h.store.get_workflow(h.workflow.request_id).status is WorkflowStatus.PENDING
        ), h.workflow
    finally:
        dispatcher.stop()
