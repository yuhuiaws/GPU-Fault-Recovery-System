"""Rule A has one clock: ``GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS``.

A training job whose node is under another remediation waits one bounded
window. The dispatcher applies it to a job workflow that has not started
(``_nodes_under_other_remediation``); the executor applies it to the
``after_incident`` restart premise once the step is running
(``step_bounds.bounded_waiting_outcome`` on a WAITING outcome whose
``reason`` is ``NODE_UNDER_REMEDIATION``). Both read the same variable, both
default to 240s, and the timing validator refuses a pair that disagrees.

One exception (逻辑 6, last section): while the remediation the premise names
is still open, the executor bounds the running restart by that remediation's
own lifetime instead, so a repair that legitimately outlasts the window can
finish and the restart it was waiting for still happens.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.execution import step_bounds
from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
    validate_timing_relationships,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_store,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

KNOB = "GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS"
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
HOLDING = WorkflowEventCode.NODE_UNDER_REMEDIATION.value
TIMEOUT = WorkflowEventCode.NODE_REMEDIATION_TIMEOUT.value
OTHER_REASON = "PROVIDER_PENDING"


# --- one variable, two readers ------------------------------------------------


def test_executor_and_dispatcher_read_the_same_variable_and_share_its_default():
    executor = ProductionExecutorConfig.from_mapping({})
    dispatcher = WorkflowDispatcherConfig.from_mapping({}, executor_enabled=True)

    assert executor.node_busy_wait_seconds == 240.0, executor
    assert dispatcher.node_busy_wait_seconds == 240.0, dispatcher

    values = {KNOB: "90"}
    assert ProductionExecutorConfig.from_mapping(values).node_busy_wait_seconds == 90.0
    assert (
        WorkflowDispatcherConfig.from_mapping(
            values, executor_enabled=True
        ).node_busy_wait_seconds
        == 90.0
    ), "the dispatcher must keep reading the same variable"


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_a_non_positive_window_is_refused_by_both_readers(raw: str):
    with pytest.raises(WorkflowExecutionError, match="node-busy wait"):
        ProductionExecutorConfig.from_mapping({KNOB: raw})
    with pytest.raises(WorkflowExecutionError, match="node-busy wait"):
        WorkflowDispatcherConfig.from_mapping({KNOB: raw}, executor_enabled=True)


def test_the_validator_refuses_a_pair_whose_windows_differ():
    executor = ProductionExecutorConfig.from_mapping({})
    dispatcher = WorkflowDispatcherConfig.from_mapping({}, executor_enabled=True)

    assert (
        validate_timing_relationships(
            executor, dispatcher, verify_max_attempts=60, branch_max_rungs=2
        )
        == []
    ), "one variable read twice must agree"

    violations = validate_timing_relationships(
        replace(executor, node_busy_wait_seconds=120.0),
        dispatcher,
        verify_max_attempts=60,
        branch_max_rungs=2,
    )

    assert len(violations) == 1, violations
    assert violations[0].startswith(
        "the executor's job workflow node-busy wait (120s) differs from the "
        "dispatcher's (240s)"
    ), violations[0]


# --- the premise backstop's cap --------------------------------------------------


def _executor(*, step_cap: int, busy_wait: float, store=None) -> SimpleNamespace:
    # No ``store`` by default: the cap must then fall back to today's window
    # even when the WAITING details name a remediation it cannot look up.
    return SimpleNamespace(
        store=store,
        config=ProductionExecutorConfig(
            enabled=True,
            executor_id="executor-rule-a",
            allowed_operations=frozenset({RESTART_JOB}),
            step_waiting_timeout_seconds=step_cap,
            step_waiting_warning_seconds=step_cap // 2,
            step_waiting_timeout_overrides={},
            node_busy_wait_seconds=busy_wait,
        ),
    )


def _waiting_for(seconds: int):
    workflow = workflow_request(
        "wf-restart",
        "inc-node-completion-train-1-a1-0123456789ab",
        official_steps=[workflow_step(RESTART_JOB, node_ids=["node-a", "node-b"])],
        step_executions=[
            workflow_step_execution(
                0,
                RESTART_JOB,
                WorkflowStepStatus.WAITING,
                started_at=datetime.now(timezone.utc) - timedelta(seconds=seconds),
            )
        ],
    )
    return workflow, workflow.official_steps[0]


# (case, reason on the WAITING outcome, seconds waited, step cap, busy wait,
#  expected status, expected cap recorded, expected reason after bounding)
CASES: list[tuple[str, str, int, int, float, WorkflowStepStatus, int | None, str]] = [
    (
        "premise hold inside the window keeps waiting",
        HOLDING,
        100,
        600,
        240.0,
        WorkflowStepStatus.WAITING,
        None,
        HOLDING,
    ),
    (
        "premise hold past the window fails before the generic cap",
        HOLDING,
        250,
        600,
        240.0,
        WorkflowStepStatus.FAILED,
        240,
        TIMEOUT,
    ),
    (
        "a window longer than the step cap is cut to the step cap",
        HOLDING,
        650,
        600,
        900.0,
        WorkflowStepStatus.FAILED,
        600,
        TIMEOUT,
    ),
    (
        "a window longer than the step cap still waits inside the step cap",
        HOLDING,
        500,
        600,
        900.0,
        WorkflowStepStatus.WAITING,
        None,
        HOLDING,
    ),
    (
        "another wait reason is not shortened by the window",
        OTHER_REASON,
        250,
        600,
        240.0,
        WorkflowStepStatus.WAITING,
        None,
        OTHER_REASON,
    ),
    (
        "another wait reason past the step cap keeps its own reason",
        OTHER_REASON,
        650,
        600,
        240.0,
        WorkflowStepStatus.FAILED,
        600,
        OTHER_REASON,
    ),
]


@pytest.mark.parametrize(
    ("case", "reason", "waited", "step_cap", "busy_wait", "status", "cap", "expected"),
    CASES,
    ids=[case for case, *_ in CASES],
)
def test_the_window_caps_only_the_remediation_premise(
    case: str,
    reason: str,
    waited: int,
    step_cap: int,
    busy_wait: float,
    status: WorkflowStepStatus,
    cap: int | None,
    expected: str,
):
    workflow, step = _waiting_for(waited)
    outcome = WorkflowStepOutcome.waiting(
        operation_id="restart-1",
        details={"reason": reason, "remediation_workflow_id": "wf-node"},
    )

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=step_cap, busy_wait=busy_wait), workflow, step, 0, outcome
    )

    assert bounded.status is status, f"{case}: {bounded}"
    assert bounded.details.get("reason") == expected, f"{case}: {bounded.details}"
    assert bounded.details.get("step_waiting_timeout_seconds") == cap, (
        f"{case}: {bounded.details}"
    )
    assert bounded.details["step_waiting_seconds"] >= waited, f"{case}: {bounded}"
    assert bounded.details["remediation_workflow_id"] == "wf-node", (
        f"{case}: the hold's target must survive the bounding"
    )


def test_a_premise_timeout_is_recorded_under_its_own_event_code():
    workflow, step = _waiting_for(250)
    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0),
        workflow,
        step,
        0,
        WorkflowStepOutcome.waiting(
            operation_id="restart-1", details={"reason": HOLDING}
        ),
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert step_bounds.attempt_event_code(bounded) is (
        WorkflowEventCode.NODE_REMEDIATION_TIMEOUT
    ), "the dispatcher's give-up and the executor's must spell the same code"
    generic = WorkflowStepOutcome.failed(
        "past the cap",
        details={"step_waiting_seconds": 650, "step_waiting_timeout_seconds": 600},
    )
    assert step_bounds.attempt_event_code(generic) is (
        WorkflowEventCode.STEP_WAITING_TIMEOUT
    ), "a generic cap keeps its generic code"


# --- an open remediation lifts the window to its own lifetime ------------------
#
# 逻辑 6: a GPU reset chain (quiesce window up to 420 s, fresh telemetry for
# VALIDATE_GPU, RESTORE_SCHEDULING) or a reboot chain (HyperPod, up to 2700 s)
# legitimately outlasts 240 s. While the remediation the premise names is still
# open, the restart wait is bounded by that remediation's lifetime, not by the
# node-busy window -- otherwise the restart fails just before the repair lands.

REMEDIATION = "wf-node"


def _store_with_remediation(
    *, status: WorkflowStatus, lifetime_deadline_at: datetime | None
):
    store = build_store()
    store.save_workflow(
        workflow_request(
            REMEDIATION,
            "inc-node",
            status=status,
            official_steps=[
                workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])
            ],
            lifetime_deadline_at=lifetime_deadline_at,
        )
    )
    return store


def _premise_hold(*, remediation_workflow_id: str | None = REMEDIATION):
    details: dict[str, object] = {"reason": HOLDING}
    if remediation_workflow_id is not None:
        details["remediation_workflow_id"] = remediation_workflow_id
    return WorkflowStepOutcome.waiting(operation_id="restart-1", details=details)


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING, WorkflowStatus.RUNNING],
    ids=lambda status: status.value,
)
def test_an_open_remediation_with_lifetime_left_keeps_the_restart_waiting(
    status: WorkflowStatus,
):
    now = datetime.now(timezone.utc)
    store = _store_with_remediation(
        status=status, lifetime_deadline_at=now + timedelta(hours=1)
    )
    workflow, step = _waiting_for(250)  # past the 240 s window

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=store),
        workflow,
        step,
        0,
        _premise_hold(),
    )

    assert bounded.status is WorkflowStepStatus.WAITING, bounded
    assert bounded.details["reason"] == HOLDING, bounded.details
    assert "step_waiting_timeout_seconds" not in bounded.details, bounded.details
    assert bounded.details["remediation_workflow_id"] == REMEDIATION


def test_an_open_remediation_outlasting_the_generic_step_cap_still_waits():
    # A reboot chain: 2700 s of HyperPod time, far past the 600 s generic cap.
    now = datetime.now(timezone.utc)
    store = _store_with_remediation(
        status=WorkflowStatus.RUNNING,
        lifetime_deadline_at=now + timedelta(seconds=2700),
    )
    workflow, step = _waiting_for(1500)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=store),
        workflow,
        step,
        0,
        _premise_hold(),
    )

    assert bounded.status is WorkflowStepStatus.WAITING, bounded


def test_a_remediation_past_its_own_lifetime_fails_the_restart_wait():
    now = datetime.now(timezone.utc)
    lifetime = now - timedelta(seconds=30)
    store = _store_with_remediation(
        status=WorkflowStatus.RUNNING, lifetime_deadline_at=lifetime
    )
    workflow, step = _waiting_for(300)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=store),
        workflow,
        step,
        0,
        _premise_hold(),
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["reason"] == TIMEOUT, bounded.details
    # The bound that fired is the remediation's lifetime, measured from when
    # this step started waiting: 300 s ago, lifetime 30 s ago -> 270 s (269
    # once the sub-second gap between the two clock reads is truncated).
    assert bounded.details["step_waiting_timeout_seconds"] in (269, 270), (
        bounded.details
    )
    assert bounded.details["step_waiting_seconds"] >= 300, bounded.details
    assert bounded.details["remediation_lifetime_deadline_at"] == (
        lifetime.isoformat()
    ), bounded.details
    assert bounded.details["remediation_workflow_id"] == REMEDIATION
    assert bounded.error is not None and bounded.error.startswith(TIMEOUT), (
        bounded.error
    )
    assert step_bounds.attempt_event_code(bounded) is (
        WorkflowEventCode.NODE_REMEDIATION_TIMEOUT
    )


def test_a_remediation_whose_lifetime_ends_inside_the_window_keeps_the_window():
    # The lifetime bound only ever extends the wait; it never shortens it below
    # the node-busy window the dispatcher already granted.
    now = datetime.now(timezone.utc)
    store = _store_with_remediation(
        status=WorkflowStatus.RUNNING, lifetime_deadline_at=now - timedelta(seconds=10)
    )
    workflow, step = _waiting_for(100)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=store),
        workflow,
        step,
        0,
        _premise_hold(),
    )

    assert bounded.status is WorkflowStepStatus.WAITING, bounded


def test_an_open_remediation_without_a_lifetime_falls_back_to_the_step_cap():
    # Not yet claimed, so ``claim_deadlines`` has not stamped a lifetime: the
    # restart's own per-operation cap is the fallback, above the window.
    store = _store_with_remediation(
        status=WorkflowStatus.PENDING, lifetime_deadline_at=None
    )
    executor = _executor(step_cap=600, busy_wait=240.0, store=store)

    inside, step = _waiting_for(500)
    assert (
        step_bounds.bounded_waiting_outcome(
            executor, inside, step, 0, _premise_hold()
        ).status
        is WorkflowStepStatus.WAITING
    )

    past, step = _waiting_for(650)
    bounded = step_bounds.bounded_waiting_outcome(
        executor, past, step, 0, _premise_hold()
    )
    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["reason"] == TIMEOUT, bounded.details
    assert bounded.details["step_waiting_timeout_seconds"] == 600, bounded.details


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    ],
    ids=lambda status: status.value,
)
def test_a_terminal_remediation_leaves_the_window_in_force(status: WorkflowStatus):
    # Terminal: SUCCEEDED means the incident turns RECOVERED and the adapter
    # proceeds; FAILED/BLOCKED the adapter refuses via INCIDENT_NOT_RECOVERABLE.
    # Either way a WAITING premise that still names it gets today's window.
    now = datetime.now(timezone.utc)
    store = _store_with_remediation(
        status=status, lifetime_deadline_at=now + timedelta(hours=1)
    )
    workflow, step = _waiting_for(250)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=store),
        workflow,
        step,
        0,
        _premise_hold(),
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["reason"] == TIMEOUT, bounded.details
    assert bounded.details["step_waiting_timeout_seconds"] == 240, bounded.details


def test_a_hold_without_a_remediation_id_keeps_the_window():
    # Regression guard: no ``remediation_workflow_id`` in the details -> the
    # store is never consulted and the window fails the wait as before.
    store = _store_with_remediation(
        status=WorkflowStatus.RUNNING,
        lifetime_deadline_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    workflow, step = _waiting_for(250)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=store),
        workflow,
        step,
        0,
        _premise_hold(remediation_workflow_id=None),
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["reason"] == TIMEOUT, bounded.details
    assert bounded.details["step_waiting_timeout_seconds"] == 240, bounded.details


def test_an_unreadable_remediation_fails_closed_to_the_window(caplog):
    # The id names a workflow the store does not have (NotFoundError), or the
    # store itself is down: both fall back to today's window.
    missing = build_store()  # nothing saved
    workflow, step = _waiting_for(250)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(step_cap=600, busy_wait=240.0, store=missing),
        workflow,
        step,
        0,
        _premise_hold(),
    )
    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["step_waiting_timeout_seconds"] == 240, bounded.details

    class _Broken:
        def get_workflow(self, request_id: str):
            raise RuntimeError("store unavailable")

    with caplog.at_level("WARNING", logger=step_bounds.LOGGER.name):
        bounded = step_bounds.bounded_waiting_outcome(
            _executor(step_cap=600, busy_wait=240.0, store=_Broken()),
            workflow,
            step,
            0,
            _premise_hold(),
        )
    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["step_waiting_timeout_seconds"] == 240, bounded.details
    assert any("remediation" in record.getMessage() for record in caplog.records), (
        caplog.text
    )
