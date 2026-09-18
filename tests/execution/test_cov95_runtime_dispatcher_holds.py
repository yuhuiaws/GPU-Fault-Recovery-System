from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.execution import WorkflowDispatcher, WorkflowDispatcherConfig
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import WorkflowLeaseError
from tests.execution._cov95_runtime_workflows import FlowHarness


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("fake dissolution claim persistence failed"),
        WorkflowLeaseError("fake concurrent claim won the lease"),
    ],
)
def test_failed_placement_hold_dissolution_never_dispatches_the_stop(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FlowHarness([WorkflowOperation.STOP_WORKLOADS])
    stop = h.workflow.official_steps[0].model_copy(
        update={"workload_ids": ["training/job/train-1"]}
    )
    h.amend(placement_hold=True, official_steps=[stop])
    original = h.store.claim_workflow
    claims = 0

    def claim(*args: Any, **kwargs: Any) -> Any:
        nonlocal claims
        claims += 1
        if claims == 1:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(h.store, "claim_workflow", claim)
    dispatcher = WorkflowDispatcher(
        h.store,
        h.executor,
        WorkflowDispatcherConfig(enabled=True, max_workers=1, batch_size=10),
    )
    try:
        first = dispatcher.run_once()
        assert h.adapter.calls == [], h.adapter.calls
        assert first.executed == 0 and first.filtered.get("placement_hold") == 1, first
        assert (
            h.store.get_workflow(h.workflow.request_id).status is WorkflowStatus.PENDING
        ), first
        assert dispatcher.placement_holds_dissolved_total == 0, first
        second = dispatcher.run_once()
        assert second.filtered.get("placement_hold_dissolved") == 1, second
        assert (
            h.store.get_workflow(h.workflow.request_id).status
            is WorkflowStatus.SUPERSEDED
        ), second
        assert (
            h.store.get_incident(h.incident.incident_id).state
            is IncidentState.RECOVERED
        ), second
        assert h.adapter.calls == [] and claims == 2, (h.adapter.calls, claims)
    finally:
        dispatcher.stop()


def test_transient_store_failure_during_dissolution_aborts_the_tick_without_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from psycopg import OperationalError

    h = FlowHarness([WorkflowOperation.STOP_WORKLOADS])
    h.amend(placement_hold=True)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise OperationalError("fake database connection unavailable")

    monkeypatch.setattr(h.store, "claim_workflow", unavailable)
    dispatcher = WorkflowDispatcher(
        h.store, h.executor, WorkflowDispatcherConfig(enabled=True, max_workers=1)
    )
    try:
        with pytest.raises(OperationalError, match="fake database"):
            dispatcher.run_once()
        assert h.adapter.calls == [], h.adapter.calls
        assert (
            h.store.get_workflow(h.workflow.request_id).status is WorkflowStatus.PENDING
        ), h.workflow
        assert dispatcher.sweep_errors_total == {}, dispatcher.sweep_errors_total
    finally:
        dispatcher.stop()
