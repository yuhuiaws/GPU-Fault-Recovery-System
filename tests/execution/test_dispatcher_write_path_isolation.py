"""One row's failing write never aborts the fleet's dispatch tick.

Control-plane review 2026-09-08, D-4 (F-A3 remainder). Three write paths of
``run_once`` still ran outside any per-row guard:

* the internal-error handlers in the future collection
  (``_block_after_internal_error`` / ``_release_after_internal_error``) can
  themselves raise (``WorkflowLeaseError``, ``WorkflowMergedError``,
  ``NotFoundError``) -- from inside ``except Exception`` -- which aborted the
  loop over the remaining futures and lost the whole tick's report;
* ``_supersede_abandoned_generations`` guarded only its ``claim_workflow``; a
  merge that bumped ``merge_revision`` between the claim and the save raised
  ``WorkflowMergedError`` out of the sweep, and the tick ended before its
  scan -- every 5 s, for as long as the row matched;
* ``_eligible`` wrote through ``_dissolve_placement_hold``,
  ``_record_node_busy_hold`` and ``_fail_node_busy`` with no guard, so one row
  aborted the scan of every row behind it.

Each is now isolated, counted in ``sweep_errors_total`` (by path) and logged;
a transient store error still propagates, because the whole tick is what
retries it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store.shared.errors import WorkflowMergedError
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution import test_abandoned_generation as abandoned
from tests.execution import test_node_busy_wait as node_busy
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

OP = WorkflowOperation.FREEZE_EVIDENCE
VALIDATE = WorkflowOperation.VALIDATE_GPU


class _Raising:
    owner = "simulated-runtime"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def supports(self, step) -> bool:
        if step.execution_owner == "owner-broken":
            raise self.error
        return False

    def execute(self, _context):
        raise AssertionError("execute must not be reached")


class _Store:
    """Pass-through store with one method replaced."""

    def __init__(self, store, **overrides) -> None:
        self._store = store
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._store, name)


def _other_workflow(store, request_id: str = "wf-other") -> None:
    incident = fault_incident(
        f"inc-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        fencing_token=3,
    )
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        status=WorkflowStatus.PENDING,
        official_steps=[workflow_step(OP)],
    )
    store.save_incident_and_workflow(incident, workflow)


def test_a_release_that_raises_does_not_lose_the_rest_of_the_tick() -> None:
    store = build_store()
    _other_workflow(store, "wf-other")
    broken_incident = fault_incident(
        "inc-broken",
        "event-broken",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-broken",
        fencing_token=3,
    )
    store.save_incident_and_workflow(
        broken_incident,
        workflow_request(
            "wf-broken",
            "inc-broken",
            status=WorkflowStatus.PENDING,
            official_steps=[workflow_step(OP, "owner-broken")],
        ),
    )

    def save_workflow_if_leased(workflow, executor_id, execution_epoch, **kwargs):
        if workflow.request_id == "wf-broken" and workflow.execution_owner_id is None:
            # The release write: another family merged the row meanwhile.
            raise WorkflowMergedError("workflow was merged since it was read")
        return store.save_workflow_if_leased(
            workflow, executor_id, execution_epoch, **kwargs
        )

    wrapped = _Store(store, save_workflow_if_leased=save_workflow_if_leased)
    adapters = [
        FakeAdapter({OP: WorkflowStepOutcome.succeeded()}),
        _Raising(AttributeError("adapter wiring is broken")),
    ]
    dispatcher = WorkflowDispatcher(
        wrapped,
        active_workflow_executor(wrapped, adapters, {OP}),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    report = dispatcher.run_once()

    assert store.get_workflow("wf-other").status is WorkflowStatus.SUCCEEDED
    assert report.executed == 1
    assert report.completed == 1
    assert report.internal_errors == 1
    assert dispatcher.internal_errors_total == 1
    assert dispatcher.sweep_errors_total.get("internal_error_release") == 1
    assert [failure.workflow_request_id for failure in report.failures] == ["wf-broken"]


def test_an_abandoned_generation_whose_save_is_refused_does_not_abort_the_tick() -> (
    None
):
    store = build_store()
    abandoned.scenario(store)
    _other_workflow(store, "wf-other")

    def save_workflow_and_incident_if_leased(workflow, *args, **kwargs):
        if workflow.request_id == abandoned.ABANDONED:
            raise WorkflowMergedError("workflow was merged since it was read")
        return store.save_workflow_and_incident_if_leased(workflow, *args, **kwargs)

    wrapped = _Store(
        store, save_workflow_and_incident_if_leased=save_workflow_and_incident_if_leased
    )
    adapter = FakeAdapter(
        {OP: WorkflowStepOutcome.succeeded(), VALIDATE: WorkflowStepOutcome.succeeded()}
    )
    dispatcher = WorkflowDispatcher(
        wrapped,
        active_workflow_executor(wrapped, [adapter], {OP, VALIDATE}),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    report = dispatcher.run_once()

    assert store.get_workflow("wf-other").status is WorkflowStatus.SUCCEEDED
    assert store.get_workflow(abandoned.CURRENT).status is WorkflowStatus.SUCCEEDED
    assert report.executed == 2
    assert dispatcher.sweep_errors_total.get("abandoned_generation") == 1


def test_a_node_busy_hold_whose_write_fails_still_holds_the_row_and_scans_on() -> None:
    store = build_store()
    node_busy._busy_node(store)
    node_busy._job_workflow(store, created_at=datetime.now(timezone.utc))
    _other_workflow(store, "wf-other")

    def amend_workflow(request_id, updates, **kwargs):
        raise RuntimeError("events column rejected the write")

    wrapped = _Store(store, amend_workflow=amend_workflow)
    adapter = FakeAdapter(
        {
            OP: WorkflowStepOutcome.succeeded(),
            node_busy.STOP: WorkflowStepOutcome.succeeded(),
            node_busy.RESET: WorkflowStepOutcome.succeeded(),
            node_busy.RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )
    dispatcher = WorkflowDispatcher(
        wrapped,
        active_workflow_executor(
            wrapped,
            [adapter],
            {OP, node_busy.STOP, node_busy.RESET, node_busy.RESTART_JOB},
        ),
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, node_busy_wait_seconds=300
        ),
    )

    report = dispatcher.run_once()

    assert store.get_workflow("wf-other").status is WorkflowStatus.SUCCEEDED
    assert store.get_workflow("wf-job").status is WorkflowStatus.PENDING
    assert report.filtered.get("node_busy") == 1
    assert dispatcher.sweep_errors_total.get("node_busy_hold") == 1
    assert adapter.calls == ["wf-other/0/FREEZE_EVIDENCE"]


def test_a_failed_node_busy_timeout_write_still_scans_on() -> None:
    store = build_store()
    node_busy._busy_node(store)
    long_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    node_busy._job_workflow(store, created_at=long_ago, held_since=long_ago)
    _other_workflow(store, "wf-other")

    def amend_workflow(request_id, updates, **kwargs):
        raise RuntimeError("events column rejected the write")

    wrapped = _Store(store, amend_workflow=amend_workflow)
    adapter = FakeAdapter(
        {
            OP: WorkflowStepOutcome.succeeded(),
            node_busy.STOP: WorkflowStepOutcome.succeeded(),
            node_busy.RESET: WorkflowStepOutcome.succeeded(),
            node_busy.RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )
    dispatcher = WorkflowDispatcher(
        wrapped,
        active_workflow_executor(
            wrapped,
            [adapter],
            {OP, node_busy.STOP, node_busy.RESET, node_busy.RESTART_JOB},
        ),
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, node_busy_wait_seconds=300
        ),
    )

    report = dispatcher.run_once()

    assert store.get_workflow("wf-other").status is WorkflowStatus.SUCCEEDED
    assert report.filtered.get("node_busy_timeout") == 1
    assert dispatcher.sweep_errors_total.get("node_busy_timeout") == 1
    assert dispatcher.node_busy_timeouts_total == 0, (
        "a refused rewrite is not a timeout"
    )
