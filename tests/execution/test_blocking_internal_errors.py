"""Only proof that *this record* cannot execute may write it BLOCKED.

Control-plane review 2026-09-08, D-5 and D-6.

D-5: ``BLOCKING_INTERNAL_ERRORS`` was ``(ValidationError,)`` -- any pydantic
decode failure anywhere under ``execute()`` (a plan row, a remote command's
embedded snapshot, an adapter's own model) was read as "the record is
undecodable" and the workflow ended BLOCKED / INTERNAL_ERROR with its incident
ESCALATED. During a rolling release that is exactly what an old process
reading a new row raises. Now only the decode of the workflow row itself
(``WorkflowRecordInvalidError``) blocks; every other ValidationError releases
the row with a backoff like any other internal error, and ``_sync_plan`` runs
after the outcome is judged and swallows its own decode errors.

D-6: a structurally invalid DAG (cycle, dependency on a missing step, over the
step cap) is deterministic, so releasing it with a 60 s backoff retried it for
ever and it never reached an operator. ``WorkflowStructureError`` now blocks.
"""

from __future__ import annotations

from pydantic import ValidationError

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowStructureError
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    PlanStatus,
    RecoveryAction,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from tests._builders import build_store, copy_model, workflow_step
from tests.execution._support import (
    FakeAdapter,
    WorkflowStepOutcome,
    active_workflow_executor,
    workflow_state,
)

OP = WorkflowOperation.QUARANTINE
FREEZE = WorkflowOperation.FREEZE_EVIDENCE


def _validation_error() -> ValidationError:
    try:
        WorkflowRequest.model_validate({"incident_id": "x"})
    except ValidationError as error:
        return error
    raise AssertionError("expected a ValidationError")


class _Raising:
    owner = "simulated-runtime"

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def supports(self, _step) -> bool:
        self.calls += 1
        raise self.error

    def execute(self, _context):
        raise AssertionError("execute must not be reached")


class _DecodeFailsOnce:
    """The store, except that one ``get_workflow`` read of ``request_id`` fails
    to decode -- the shape of a row a newer release wrote."""

    def __init__(self, store, request_id: str) -> None:
        self._store = store
        self._request_id = request_id
        self.armed = True

    def get_workflow(self, request_id: str):
        if self.armed and request_id == self._request_id:
            self.armed = False
            raise _validation_error()
        return self._store.get_workflow(request_id)

    def __getattr__(self, name):
        return getattr(self._store, name)


class _PlanDecodeFails:
    def __init__(self, store) -> None:
        self._store = store
        self.reads = 0

    def get_plan(self, plan_id: str):
        self.reads += 1
        raise _validation_error()

    def __getattr__(self, name):
        return getattr(self._store, name)


def _dispatcher(store, adapter, operations):
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], operations),
        WorkflowDispatcherConfig(enabled=True),
    )


# ---------------------------------------------------------------- D-5


def test_a_validation_error_from_an_adapter_leaves_the_workflow_executable() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [OP])
    adapter = _Raising(_validation_error())
    dispatcher = _dispatcher(store, adapter, {OP})

    report = dispatcher.run_once()

    current = store.get_workflow(workflow.request_id)
    assert current.status is WorkflowStatus.PENDING, current.status
    assert current.blocked_kind is None
    assert current.not_before is not None, "released with a backoff, not blocked"
    assert report.internal_errors == 1
    assert report.failed == 0
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.ACTION_PENDING
    )


def test_a_workflow_row_that_cannot_be_decoded_is_blocked_as_internal_error() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [OP])
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    dispatcher = _dispatcher(
        _DecodeFailsOnce(store, workflow.request_id), adapter, {OP}
    )

    report = dispatcher.run_once()

    blocked = store.get_workflow(workflow.request_id)
    assert blocked.status is WorkflowStatus.BLOCKED
    assert blocked.blocked_kind is BlockedKind.INTERNAL_ERROR
    assert "WorkflowRecordInvalidError" in blocked.blocked_reasons[-1]
    assert report.failed == 1
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


def test_a_plan_that_cannot_be_decoded_does_not_block_a_waiting_workflow() -> None:
    store = build_store()
    plan = RecoveryPlan(
        incident_id="inc-plan",
        attempt_id="attempt-plan",
        trigger="quick-triage:PASS",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a"],
                "execution_owner": "simulated-runtime",
            }
        ],
    )
    store.save_plan(plan)
    incident, workflow = workflow_state(store, [OP])
    store.save_workflow(copy_model(workflow, source_plan_id=plan.plan_id))
    adapter = FakeAdapter({OP: WorkflowStepOutcome.waiting(operation_id="op-1")})
    wrapped = _PlanDecodeFails(store)
    dispatcher = _dispatcher(wrapped, adapter, {OP})

    report = dispatcher.run_once()

    current = store.get_workflow(workflow.request_id)
    assert current.status is WorkflowStatus.RUNNING, current.status
    assert current.blocked_kind is None
    assert report.waiting == 1
    assert report.failed == 0
    assert wrapped.reads == 1, "the plan mirror was attempted once and gave up"
    assert dispatcher.plan_sync_misses_total == 1
    assert store.get_plan(plan.plan_id).status is PlanStatus.PENDING


# ---------------------------------------------------------------- D-6


def test_a_dag_with_a_cycle_is_blocked_for_an_operator_not_retried_for_ever() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE, FREEZE])
    store.save_workflow(
        copy_model(
            workflow,
            dag_enabled=True,
            official_steps=[
                workflow_step(FREEZE, depends_on_step_indexes=[1]),
                workflow_step(FREEZE, depends_on_step_indexes=[0]),
            ],
        )
    )
    adapter = FakeAdapter({FREEZE: WorkflowStepOutcome.succeeded()})
    dispatcher = _dispatcher(store, adapter, {FREEZE})

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    blocked = store.get_workflow(workflow.request_id)
    assert blocked.status is WorkflowStatus.BLOCKED, blocked.status
    assert blocked.blocked_kind is BlockedKind.INTERNAL_ERROR
    assert "cycle" in blocked.blocked_reasons[-1]
    assert first.failed == 1
    assert first.internal_errors == 0
    assert second.scanned == 0
    assert adapter.calls == []
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


def test_structure_errors_are_the_blocking_kind() -> None:
    assert issubclass(WorkflowStructureError, ValueError), (
        "callers that catch ValueError must still see structural errors"
    )
    assert WorkflowStructureError in WorkflowDispatcher.BLOCKING_INTERNAL_ERRORS
    assert ValidationError not in WorkflowDispatcher.BLOCKING_INTERNAL_ERRORS
