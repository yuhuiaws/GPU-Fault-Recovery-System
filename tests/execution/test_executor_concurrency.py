"""Distinct workflows execute concurrently; transient store errors from an
adapter are retried, not written FAILED; a dispatch cycle has a deadline.

F-C7 (docs/review/FINAL-建议汇总.md). One executor instance is shared by the
dispatcher's worker pool, and ``execute()`` held a single re-entrant lock
for its whole duration, so eight workers ran one workflow at a time. Any
exception an adapter raised -- an Aurora failover included -- became a
FAILED step and a hardware escalation. A dispatch cycle waited for the
whole batch (100 rows over 8 workers) before it would scan again.
"""

from __future__ import annotations

import threading
import time

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)

FREEZE = WorkflowOperation.FREEZE_EVIDENCE


def _seed(store, request_id: str) -> None:
    incident = fault_incident(
        f"inc-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        fencing_token=1,
    )
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        fencing_token=1,
        official_steps=[workflow_step(FREEZE)],
    )
    store.save_incident_and_workflow(incident, workflow)


class _RendezvousAdapter:
    """Succeeds only if two workflows are inside execute() at the same time."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=2.0)
        self.broken = False

    def supports(self, step) -> bool:
        return step.operation is FREEZE

    def execute(self, context) -> WorkflowStepOutcome:
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            self.broken = True
            return WorkflowStepOutcome.failed(
                "serialized: the other workflow never arrived"
            )
        return WorkflowStepOutcome.succeeded()


def test_two_workflows_execute_at_the_same_time():
    store = build_store()
    _seed(store, "wf-1")
    _seed(store, "wf-2")
    adapter = _RendezvousAdapter()
    executor = active_workflow_executor(store, [adapter], {FREEZE})
    results: dict[str, WorkflowStatus] = {}

    def run(request_id: str) -> None:
        results[request_id] = executor.execute(
            request_id, WorkflowExecutionRequest(expected_fencing_token=1)
        ).status

    threads = [threading.Thread(target=run, args=(rid,)) for rid in ("wf-1", "wf-2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not adapter.broken, "expected adapter.broken to be false"
    assert results == {
        "wf-1": WorkflowStatus.SUCCEEDED,
        "wf-2": WorkflowStatus.SUCCEEDED,
    }


def test_the_same_workflow_is_still_serialized():
    store = build_store()
    _seed(store, "wf-1")
    entered = threading.Event()
    release = threading.Event()
    concurrent = []

    class Slow:
        inside = 0

        def supports(self, step) -> bool:
            return True

        def execute(self, context) -> WorkflowStepOutcome:
            Slow.inside += 1
            concurrent.append(Slow.inside)
            entered.set()
            release.wait(timeout=2)
            Slow.inside -= 1
            return WorkflowStepOutcome.succeeded()

    executor = active_workflow_executor(store, [Slow()], {FREEZE})
    request = WorkflowExecutionRequest(expected_fencing_token=1)
    first = threading.Thread(target=executor.execute, args=("wf-1", request))
    first.start()
    assert entered.wait(timeout=2), "expected entered.wait(timeout=2) to be true"
    second = threading.Thread(target=executor.execute, args=("wf-1", request))
    second.start()
    time.sleep(0.2)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert max(concurrent) == 1


def test_a_transient_store_error_from_an_adapter_is_retried_not_failed():
    class OperationalError(Exception):
        pass

    OperationalError.__module__ = "psycopg"

    class Flaky:
        def supports(self, step) -> bool:
            return True

        def execute(self, context) -> WorkflowStepOutcome:
            raise OperationalError("SSL connection has been closed unexpectedly")

    store = build_store()
    _seed(store, "wf-1")
    executor = active_workflow_executor(store, [Flaky()], {FREEZE})

    with pytest.raises(OperationalError):
        executor.execute("wf-1", WorkflowExecutionRequest(expected_fencing_token=1))

    saved = store.get_workflow("wf-1")
    assert saved.status is WorkflowStatus.RUNNING
    assert not any(
        e.status is WorkflowStepStatus.FAILED for e in saved.step_executions
    ), (
        "expected any(e.status is WorkflowStepStatus.FAILED for e in saved.step_executions) to be false"
    )
    assert store.get_incident("inc-wf-1").state is IncidentState.ACTION_PENDING


def test_a_programming_error_in_an_adapter_still_fails_the_step():
    class Broken:
        def supports(self, step) -> bool:
            return True

        def execute(self, context) -> WorkflowStepOutcome:
            raise KeyError("missing parameter")

    store = build_store()
    _seed(store, "wf-1")
    executor = active_workflow_executor(store, [Broken()], {FREEZE})

    result = executor.execute(
        "wf-1", WorkflowExecutionRequest(expected_fencing_token=1)
    )

    assert result.status is WorkflowStatus.FAILED
    assert "KeyError" in (result.error or "")


def test_a_dispatch_cycle_stops_waiting_at_its_deadline_and_leaves_the_rest_pending():
    class Slow:
        def supports(self, step) -> bool:
            return True

        def execute(self, context) -> WorkflowStepOutcome:
            time.sleep(0.4)
            return WorkflowStepOutcome.succeeded()

    store = build_store()
    for rid in ("wf-1", "wf-2", "wf-3"):
        _seed(store, rid)
    executor = active_workflow_executor(store, [Slow()], {FREEZE})
    dispatcher = WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, cycle_deadline_seconds=0.25
        ),
    )

    started = time.monotonic()
    report = dispatcher.run_once()
    elapsed = time.monotonic() - started

    assert report.executed == 1
    assert report.deferred == 2
    assert elapsed < 0.9
    statuses = sorted(
        store.get_workflow(rid).status for rid in ("wf-1", "wf-2", "wf-3")
    )
    assert statuses == [
        WorkflowStatus.PENDING,
        WorkflowStatus.PENDING,
        WorkflowStatus.SUCCEEDED,
    ]
