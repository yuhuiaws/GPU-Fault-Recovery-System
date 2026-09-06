"""The timing knobs have an order nobody validated (review item 3).

A step's waiting cap, the managed recovery window, the execution timeout, the
lifetimes, the lease, the dispatcher's poll/lease/cycle and the node-busy
wait each bound the others. Each relationship broken alone is one sentence;
a clean set is silent apart from ``warning:`` strings for what could not be
checked.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    TimingConfigurationError,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
    validate_timing_from_environment,
    validate_timing_or_raise,
    validate_timing_relationships,
)

EXECUTOR = ProductionExecutorConfig.from_mapping({})
DISPATCHER = WorkflowDispatcherConfig.from_mapping({}, executor_enabled=True)
# The shipped defaults sit exactly on one boundary (see the finding below), so
# the healthy baseline for the table lowers the node-busy wait by one minute.
HEALTHY = replace(DISPATCHER, node_busy_wait_seconds=240.0)
VERIFY_ATTEMPTS = 60
RUNGS = 2


def _check(
    executor: ProductionExecutorConfig = EXECUTOR,
    dispatcher: WorkflowDispatcherConfig = HEALTHY,
    *,
    verify_max_attempts: int | None = VERIFY_ATTEMPTS,
    branch_max_rungs: int = RUNGS,
) -> list[str]:
    return validate_timing_relationships(
        executor,
        dispatcher,
        verify_max_attempts=verify_max_attempts,
        branch_max_rungs=branch_max_rungs,
    )


# (case, executor overrides, dispatcher overrides, keyword overrides, phrase)
BROKEN: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str]] = [
    (
        "managed ceilings above the node lifetime",
        {
            "node_workflow_lifetime_seconds": 1700,
            "workflow_execution_timeout_seconds": 1700,
        },
        {},
        {},
        "step waiting ceilings exceed the node workflow lifetime (1700s): "
        "REPLACE_NODE 1800s, RESTART_NODE 1800s",
    ),
    (
        "default ceiling above the node lifetime",
        {
            "node_workflow_lifetime_seconds": 500,
            "workflow_execution_timeout_seconds": 500,
            "step_waiting_timeout_seconds": 590,
            "step_waiting_warning_seconds": 300,
            "step_waiting_timeout_overrides": {},
        },
        {},
        {"branch_max_rungs": 1},
        "step waiting ceilings exceed the node workflow lifetime (500s): default 590s",
    ),
    (
        "managed window plus a reboot above the job lifetime",
        {"job_workflow_lifetime_seconds": 3000},
        {},
        {"branch_max_rungs": 1},
        "the managed recovery window (1800s) plus one reboot allowance (1800s) "
        "exceeds the job workflow lifetime (3000s)",
    ),
    (
        "execution timeout above both lifetimes",
        {"workflow_execution_timeout_seconds": 4000},
        {},
        {},
        "the workflow execution timeout (4000s) exceeds the job workflow lifetime "
        "(3600s) and the node workflow lifetime (3600s)",
    ),
    (
        "execution timeout above the job lifetime only",
        {
            "workflow_execution_timeout_seconds": 4000,
            "node_workflow_lifetime_seconds": 5000,
        },
        {},
        {},
        "the workflow execution timeout (4000s) exceeds the job workflow lifetime "
        "(3600s)",
    ),
    (
        "lease not below the execution timeout",
        {"lease_duration_seconds": 1800},
        {},
        {},
        "the workflow lease duration (1800s) is not below the workflow execution "
        "timeout (1800s)",
    ),
    (
        "node-busy wait not below the VERIFY total",
        # Rule A: the executor reads the same knob, so the pair moves together.
        {"node_busy_wait_seconds": 300.0},
        {"node_busy_wait_seconds": 300.0},
        {},
        "the job workflow node-busy wait (300s) is not below the "
        "VERIFY_NO_GPU_CLIENTS total wait (60 attempts x 5s poll = 300s)",
    ),
    (
        "too many rungs for the job lifetime",
        {},
        {},
        {"branch_max_rungs": 3},
        "the job workflow lifetime (3600s) cannot hold 3 escalation rungs of the "
        "managed recovery window (1800s each, 5400s in all)",
    ),
    (
        "dispatch lease below three polls",
        {},
        {"dispatch_lease_seconds": 10.0},
        {},
        "the dispatch lease (10s) is below three poll intervals (15s)",
    ),
    (
        "cycle deadline below one poll",
        {},
        {"cycle_deadline_seconds": 2.0},
        {},
        "the dispatch cycle deadline (2s) is below the poll interval (5s)",
    ),
]


@pytest.mark.parametrize(
    ("case", "executor_updates", "dispatcher_updates", "keywords", "phrase"),
    BROKEN,
    ids=[case for case, *_ in BROKEN],
)
def test_each_relationship_broken_alone_is_exactly_its_sentence(
    case: str,
    executor_updates: dict[str, Any],
    dispatcher_updates: dict[str, Any],
    keywords: dict[str, Any],
    phrase: str,
) -> None:
    violations = _check(
        replace(EXECUTOR, **executor_updates),
        replace(HEALTHY, **dispatcher_updates),
        **keywords,
    )

    assert len(violations) == 1, f"{case}: expected one sentence, got {violations}"
    assert violations[0].startswith(phrase), f"{case}: {violations[0]!r}"


def test_a_healthy_set_is_silent_and_a_disabled_dispatch_lease_is_not_checked() -> None:
    assert _check() == [], "the healthy baseline must produce no sentence"
    assert _check(dispatcher=replace(HEALTHY, dispatch_lease_seconds=0.0)) == [], (
        "0 disables the fleet-wide dispatch lease; there is nothing to compare"
    )


def test_the_operator_acknowledgement_ceiling_is_exempt_from_the_lifetimes() -> None:
    # CHECK_MECHANICALS waits a working day; claim_deadlines floors both
    # deadlines at that window, so the ceiling never exceeds the lifetime it
    # actually gets.
    inspecting = ProductionExecutorConfig.from_mapping(
        {"GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS": "172800"}
    )

    assert _check(inspecting) == [], "a longer acknowledgement window is not a fault"


def test_an_unknown_verify_total_is_a_warning_not_a_violation() -> None:
    reported = _check(verify_max_attempts=None)

    assert len(reported) == 1, reported
    assert reported[0].startswith("warning: "), reported[0]
    assert "VERIFY_NO_GPU_CLIENTS total wait is unknown" in reported[0], reported[0]
    assert (
        validate_timing_or_raise(
            EXECUTOR, HEALTHY, verify_max_attempts=None, branch_max_rungs=RUNGS
        )
        == reported
    ), "warnings are returned to the caller to log, never raised"


def test_or_raise_fails_closed_with_every_violation_in_the_message() -> None:
    with pytest.raises(TimingConfigurationError) as raised:
        validate_timing_or_raise(
            replace(EXECUTOR, lease_duration_seconds=1800),
            replace(HEALTHY, cycle_deadline_seconds=2.0),
            verify_max_attempts=VERIFY_ATTEMPTS,
            branch_max_rungs=RUNGS,
        )

    message = str(raised.value)
    assert "workflow lease duration" in message, message
    assert "dispatch cycle deadline" in message, message
    assert isinstance(raised.value, WorkflowExecutionError), (
        "startup treats it like any other configuration error"
    )


def test_from_environment_reads_both_configs_and_the_two_foreign_knobs() -> None:
    clean = validate_timing_from_environment(
        {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "90"}
    )
    assert clean == [], clean

    with pytest.raises(TimingConfigurationError, match="3 escalation rungs"):
        validate_timing_from_environment(
            {
                "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "90",
                "GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS": "3",
            }
        )
    with pytest.raises(TimingConfigurationError, match="dispatch lease"):
        validate_timing_from_environment(
            {
                "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "90",
                "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS": "10",
                "GPU_FAULT_WORKFLOW_DISPATCH_LEASE_SECONDS": "20",
            }
        )


def test_dispatcher_from_mapping_mirrors_from_environment() -> None:
    config = WorkflowDispatcherConfig.from_mapping(
        {
            "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS": "7",
            "GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS": "120",
            "GPU_FAULT_WORKFLOW_DISPATCH_CYCLE_SECONDS": "30",
        },
        executor_enabled=True,
    )

    assert config.poll_interval_seconds == 7.0, config
    assert config.dispatch_lease_seconds == 21.0, "default lease is max(15, 3 x poll)"
    assert config.node_busy_wait_seconds == 120.0, config
    assert config.cycle_deadline_seconds == 30.0, config
    assert WorkflowDispatcherConfig.from_mapping({}, executor_enabled=False) == replace(
        DISPATCHER, enabled=False
    ), "the executor being off turns the dispatcher off"


def test_the_shipped_defaults_satisfy_every_timing_relationship() -> None:
    """The review found the defaults sitting on one boundary: a 300s job
    busy wait equalled 60 VERIFY attempts x 5s poll, so a job workflow's STOP
    could never land before the node remediation's VERIFY gave up. The busy
    wait default moved to 240s; this pins that the shipped defaults now pass
    and that the old value is refused."""

    assert _check(EXECUTOR, DISPATCHER) == [], _check(EXECUTOR, DISPATCHER)

    old_default = replace(DISPATCHER, node_busy_wait_seconds=300.0)
    violations = _check(replace(EXECUTOR, node_busy_wait_seconds=300.0), old_default)

    assert len(violations) == 1, violations
    assert violations[0].startswith(
        "the job workflow node-busy wait (300s) is not below the "
        "VERIFY_NO_GPU_CLIENTS total wait (60 attempts x 5s poll = 300s)"
    ), violations[0]
