"""Every writer of BLOCKED names its cause; internal errors no longer block.

FINAL-建议汇总 F-B4 (1) and (3) (P0-59A, P0-58A, P0-72A). The executor's
settled safety plan and the dispatcher's internal error both produced an
indistinguishable BLOCKED row; the operator tooling then treated a plan that
ended as designed like a broken record. And ``_block_after_internal_error``
turned *any* unrecognised exception into a permanent BLOCK -- an adapter
wiring bug on one replica took the workflow out of every replica's hands.
"""

from __future__ import annotations

from pydantic import ValidationError

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
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


def test_a_settled_safety_plan_is_blocked_as_safety_settled() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [OP])
    safety = copy_model(
        workflow,
        status=WorkflowStatus.SAFETY_PENDING,
        safety_steps=[workflow_step(OP)],
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
        blocked_reasons=["policy: restart not allowed on this node"],
    )
    store.save_workflow(safety)
    executor = active_workflow_executor(
        store, [FakeAdapter({OP: WorkflowStepOutcome.succeeded()})], {OP}
    )

    result = executor.execute(
        safety.request_id,
        WorkflowExecutionRequest(expected_fencing_token=safety.fencing_token),
    )

    assert result.status is WorkflowStatus.BLOCKED
    settled = store.get_workflow(safety.request_id)
    assert settled.blocked_kind is BlockedKind.SAFETY_SETTLED
    assert store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED


def test_an_unrecognised_internal_error_leaves_the_workflow_executable() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [OP])
    adapter = _Raising(AttributeError("adapter wiring is broken"))
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], {OP}),
        WorkflowDispatcherConfig(enabled=True),
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    current = store.get_workflow(workflow.request_id)
    assert current.status is WorkflowStatus.PENDING
    assert current.execution_owner_id is None
    assert current.blocked_reasons == []
    assert current.blocked_kind is None
    assert first.internal_errors == 1
    assert first.failed == 0
    assert [failure.workflow_request_id for failure in first.failures] == [
        workflow.request_id
    ]
    # Still executable, but behind a backoff: no hot loop on a broken replica.
    assert current.not_before is not None
    assert second.scanned == 0
    assert second.filtered.get("not_before") == 1
    assert adapter.calls == 1
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.ACTION_PENDING
    )


def test_a_record_that_cannot_be_validated_is_blocked_as_internal_error() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [OP])
    try:
        WorkflowRequest.model_validate({"incident_id": "x"})
    except ValidationError as error:
        invalid = error
    adapter = _Raising(invalid)
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], {OP}),
        WorkflowDispatcherConfig(enabled=True),
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    blocked = store.get_workflow(workflow.request_id)
    assert blocked.status is WorkflowStatus.BLOCKED
    assert blocked.blocked_kind is BlockedKind.INTERNAL_ERROR
    assert "dispatcher internal error: ValidationError" in blocked.blocked_reasons[-1]
    assert first.failed == 1
    assert second.scanned == 0
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


# ---------------------------------------------------------------- F-A4


def test_dispatcher_holds_a_successor_behind_an_operator_blocked_predecessor() -> None:
    """The predecessor gate used to treat every BLOCKED row as terminal, so a
    node-exclusive workflow parked for an operator released its successor
    immediately (P1-39C, P0-42E)."""

    from datetime import timedelta

    from tests._builders import fault_incident, workflow_request

    store = build_store()
    for name, kind in (
        ("settled", BlockedKind.SAFETY_SETTLED),
        ("operator", BlockedKind.NEEDS_OPERATOR),
    ):
        blocked = workflow_request(
            f"wf-{name}",
            f"inc-{name}",
            status=WorkflowStatus.BLOCKED,
            blocked_kind=kind,
            fencing_token=1,
            official_steps=[workflow_step(OP)],
        )
        store.save_incident_and_workflow(
            fault_incident(
                f"inc-{name}",
                f"event-{name}",
                state=IncidentState.QUARANTINED,
                workflow_request_id=blocked.request_id,
                fencing_token=1,
            ),
            blocked,
        )
        successor = workflow_request(
            f"wf-behind-{name}",
            f"inc-behind-{name}",
            status=WorkflowStatus.PENDING,
            predecessor_workflow_id=blocked.request_id,
            fencing_token=1,
            official_steps=[workflow_step(OP)],
        )
        store.save_incident_and_workflow(
            fault_incident(
                f"inc-behind-{name}",
                f"event-behind-{name}",
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=successor.request_id,
                fencing_token=1,
            ),
            successor,
        )
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], {OP}),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    report = dispatcher.run_once()

    assert store.get_workflow("wf-behind-settled").status is WorkflowStatus.SUCCEEDED
    assert store.get_workflow("wf-behind-operator").status is WorkflowStatus.PENDING
    assert report.filtered.get("predecessor") == 1
    del timedelta
