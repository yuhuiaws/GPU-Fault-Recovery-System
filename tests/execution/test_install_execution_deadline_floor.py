"""The workflow execution deadline against a full-length node install (F2).

R4 fix round 2 raised the per-step WAITING cap for REMEDIATE_DRIVER,
UPDATE_SOFTWARE_FIRMWARE and REMEDIATE_EFA_DRIVER to
``GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS`` (1900 s), but
``claim_deadlines`` still stamped ``execution_deadline = first claim + 1800 s``
on the whole workflow, so the install had ``1800 - delta`` seconds where delta is
the quiesce and verify that precede it -- the raised cap could never be the
bound that fired. A workflow that contains one of the three installs now gets
its execution deadline floored at ``claim + install ceiling + containment
allowance`` (``GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS``, 600 s: one
default per-step cap, which holds the quiesce and the 60 x 5 s verify wait),
still capped by the lifetime. A lifetime that holds the ceiling but not the
floor is refused at boot.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.execution import (
    ProductionWorkflowExecutor,
    WorkflowStepOutcome,
    step_bounds,
)
from gpu_fault.execution.config import (
    NODE_INSTALL_OPERATIONS,
    ProductionExecutorConfig,
    TimingConfigurationError,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
    validate_timing_from_environment,
    validate_timing_relationships,
)
from gpu_fault.execution.restart_budget_preflight import claim_deadlines
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, workflow_state

NOW = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS
INSTALL = WorkflowOperation.REMEDIATE_DRIVER
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
INSTALL_CEILING = 1900
CONTAINMENT = 600
FLOOR = INSTALL_CEILING + CONTAINMENT
EXECUTION_TIMEOUT = 1800
LIFETIME = 3600
INSTALL_KNOB = "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS"
CONTAINMENT_KNOB = "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS"
LIFETIME_KNOB = "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS"
KWARGS = dict(
    timeout_seconds=EXECUTION_TIMEOUT,
    job_lifetime_seconds=LIFETIME,
    node_lifetime_seconds=LIFETIME,
)


def _claimed(
    store: InMemoryStore, operations: list[WorkflowOperation]
) -> tuple[ProductionWorkflowExecutor, datetime, object]:
    """Run ``operations`` once with the install left WAITING; return the claim."""

    _, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {
            operation: (
                WorkflowStepOutcome.waiting(operation_id=f"{operation.value}/op")
                if operation in NODE_INSTALL_OPERATIONS or operation is RESET
                else WorkflowStepOutcome.succeeded()
            )
            for operation in operations
        }
    )
    executor = active_workflow_executor(store, [adapter], operations)
    claimed_at = datetime.now(timezone.utc)
    result = execute_workflow(executor, workflow.request_id)
    assert result.status is WorkflowStatus.RUNNING, result
    return executor, claimed_at, store.get_workflow(workflow.request_id)


# --- the deadline the claim stamps ------------------------------------------------------


def test_a_workflow_holding_an_install_gets_the_install_floor_as_its_budget() -> None:
    store = build_store()
    _, claimed_at, workflow = _claimed(store, [QUIESCE, VERIFY, INSTALL, RESTORE])

    budget = (workflow.execution_deadline - claimed_at).total_seconds()

    assert FLOOR - 5 <= budget <= FLOOR + 5, (
        "the execution deadline gives a "
        f"{int(budget)} s budget to a workflow whose install step alone may wait "
        f"{INSTALL_CEILING} s; the floor is {FLOOR} s (ceiling + containment)"
    )
    assert workflow.lifetime_deadline_at - claimed_at <= timedelta(
        seconds=LIFETIME + 5
    ), "the floor lifts the execution deadline, never the lifetime"


def test_a_twenty_five_minute_install_after_a_five_minute_containment_completes() -> (
    None
):
    """delta = 300 s of quiesce + verify, then a 1500 s install: 1800 s in all.

    Before the floor the execution deadline fired at exactly this moment, with
    ``workflow execution deadline exceeded ... before step 2/REMEDIATE_DRIVER``.
    """

    store = build_store()
    executor, _, workflow = _claimed(store, [QUIESCE, VERIFY, INSTALL, RESTORE])
    elapsed = timedelta(seconds=300 + 1500 + 5)
    later = copy_model(
        workflow,
        execution_deadline=workflow.execution_deadline - elapsed,
        lifetime_deadline_at=workflow.lifetime_deadline_at - elapsed,
    )

    outcome = step_bounds.workflow_deadline_failure(
        executor, later, later.official_steps[2], 2
    )

    assert outcome is None, (
        f"the install was failed by the workflow deadline: {outcome.error}"
    )


def test_a_reset_workflow_keeps_the_ordinary_execution_budget() -> None:
    store = build_store()
    _, claimed_at, workflow = _claimed(store, [RESET])

    budget = (workflow.execution_deadline - claimed_at).total_seconds()

    assert EXECUTION_TIMEOUT - 5 <= budget <= EXECUTION_TIMEOUT + 5, (
        f"a RESET_GPU workflow must keep the {EXECUTION_TIMEOUT} s budget, got "
        f"{int(budget)} s"
    )


@pytest.mark.parametrize("operation", NODE_INSTALL_OPERATIONS, ids=lambda op: op.value)
def test_claim_deadlines_floors_the_execution_deadline_for_each_install(
    operation: WorkflowOperation,
) -> None:
    installing = workflow_request(
        "wf-install",
        "inc",
        official_steps=[
            workflow_step(QUIESCE, node_ids=["node-a"]),
            workflow_step(operation, node_ids=["node-a"]),
        ],
    )

    execution, lifetime = claim_deadlines(
        installing, NOW, install_floor_seconds=FLOOR, **KWARGS
    )

    assert execution == NOW + timedelta(seconds=FLOOR)
    assert lifetime == NOW + timedelta(seconds=LIFETIME)


def test_the_floor_is_capped_by_the_lifetime_and_stamped_once() -> None:
    installing = workflow_request(
        "wf-install", "inc", official_steps=[workflow_step(INSTALL)]
    )

    capped, lifetime = claim_deadlines(
        installing,
        NOW,
        timeout_seconds=EXECUTION_TIMEOUT,
        job_lifetime_seconds=LIFETIME,
        node_lifetime_seconds=2000,
        install_floor_seconds=FLOOR,
    )
    assert capped == lifetime == NOW + timedelta(seconds=2000), (
        "a lifetime shorter than the floor still wins; the validator reports it"
    )

    first_claim = NOW - timedelta(minutes=10)
    reclaimed = copy_model(
        installing,
        execution_deadline=first_claim + timedelta(seconds=FLOOR),
        lifetime_deadline_at=first_claim + timedelta(seconds=LIFETIME),
    )
    execution, _ = claim_deadlines(
        reclaimed, NOW, install_floor_seconds=FLOOR, **KWARGS
    )
    assert execution == first_claim + timedelta(seconds=FLOOR), (
        "a re-claim must not slide the install floor forward"
    )

    plain = workflow_request("wf-reset", "inc", official_steps=[workflow_step(RESET)])
    execution, _ = claim_deadlines(plain, NOW, install_floor_seconds=FLOOR, **KWARGS)
    assert execution == NOW + timedelta(seconds=EXECUTION_TIMEOUT)


# --- the step's own clock under the wider window --------------------------------------


def test_an_install_steps_wait_is_measured_from_the_claim_not_the_old_budget() -> None:
    """``step_elapsed_since`` recovers the window start from the deadline minus
    the budget the claim used. Subtracting only the 1800 s timeout from a
    floored deadline puts the window start 700 s in the future, is clamped to
    now, and the install's ``step_waiting_seconds`` reads 0 for 700 s."""

    waited = 900
    now = datetime.now(timezone.utc)
    workflow = workflow_request(
        "wf-install",
        "inc",
        official_steps=[workflow_step(INSTALL, node_ids=["node-a"])],
        execution_deadline=now + timedelta(seconds=FLOOR - waited),
        step_executions=[
            workflow_step_execution(
                0,
                INSTALL,
                WorkflowStepStatus.WAITING,
                started_at=now - timedelta(seconds=waited),
            )
        ],
    )
    executor = SimpleNamespace(config=ProductionExecutorConfig.from_mapping({}))

    bounded = step_bounds.bounded_waiting_outcome(
        executor,
        workflow,
        workflow.official_steps[0],
        0,
        WorkflowStepOutcome.waiting(operation_id="install/op"),
    )

    assert bounded.status is WorkflowStepStatus.WAITING, bounded
    assert waited - 5 <= bounded.details["step_waiting_seconds"] <= waited + 5, (
        bounded.details
    )


# --- the knob and the boot-time rule --------------------------------------------------


def test_the_containment_allowance_is_one_default_step_cap_and_tunable() -> None:
    config = ProductionExecutorConfig.from_mapping({})

    assert config.node_install_containment_seconds == CONTAINMENT
    assert config.install_execution_floor_seconds() == FLOOR
    assert (
        ProductionExecutorConfig.from_mapping(
            {CONTAINMENT_KNOB: "0"}
        ).install_execution_floor_seconds()
        == INSTALL_CEILING
    ), "0 is legal: the floor is then exactly the install ceiling"
    assert (
        ProductionExecutorConfig.from_mapping(
            {INSTALL_KNOB: "2400", CONTAINMENT_KNOB: "300"}
        ).install_execution_floor_seconds()
        == 2700
    )
    with pytest.raises(WorkflowExecutionError, match="must not be negative"):
        ProductionExecutorConfig.from_mapping({CONTAINMENT_KNOB: "-1"})


def _violations(executor: ProductionExecutorConfig) -> list[str]:
    return validate_timing_relationships(
        executor,
        WorkflowDispatcherConfig.from_mapping({}, executor_enabled=True),
        verify_max_attempts=60,
        branch_max_rungs=2,
    )


def test_a_lifetime_that_holds_the_ceiling_but_not_the_floor_is_refused() -> None:
    executor = ProductionExecutorConfig.from_mapping({LIFETIME_KNOB: "2400"})

    violations = _violations(executor)

    assert len(violations) == 1, violations
    sentence = violations[0]
    for knob in (INSTALL_KNOB, CONTAINMENT_KNOB, LIFETIME_KNOB):
        assert knob in sentence, f"the sentence must name {knob}: {sentence}"
    assert "2500s" in sentence and "2400s" in sentence, sentence
    with pytest.raises(TimingConfigurationError, match=CONTAINMENT_KNOB):
        validate_timing_from_environment({LIFETIME_KNOB: "2400"})


def test_a_lifetime_below_the_ceiling_is_reported_once_not_twice() -> None:
    """The per-step ceiling rule already says the lifetime fails the install;
    the floor rule is the one relationship the ceiling rule cannot see."""

    executor = ProductionExecutorConfig.from_mapping(
        {LIFETIME_KNOB: "1700", "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS": "1700"}
    )

    violations = _violations(executor)

    assert len(violations) == 1, violations
    assert violations[0].startswith("step waiting ceilings exceed"), violations[0]


def test_the_shipped_defaults_hold_the_floor() -> None:
    executor = ProductionExecutorConfig.from_mapping({})

    assert _violations(executor) == []
    assert _violations(replace(executor, node_workflow_lifetime_seconds=FLOOR)) == [], (
        "a lifetime exactly at the floor is enough"
    )
