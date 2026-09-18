"""Fail-closed compound barrier evidence in the fixed-main integration."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import destr016_verdicts as preempt
from scripts.e2e.regional import destr018_verdicts as lifetime
from tests.regional import test_destr018_lifetime_deadline as lifetime_data
from tests.regional._destr016_builders import (
    REBOOT_ID,
    barrier_commands,
    compound_hold,
    parked_reset_workflow,
)


def pinned() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = parked_reset_workflow()
    command = compound_hold()
    workflow["step_executions"][3]["details"]["remote_command_id"] = command[
        "command_id"
    ]
    workflow["fencing_token"] = command["fencing_token"] = 3
    return workflow, command


@pytest.mark.parametrize(
    "field", ["command_id", "workflow_request_id", "incident_id", "fencing_token"]
)
def test_a_pinned_barrier_cannot_fall_back_to_unbound_hold_metadata(field: str) -> None:
    workflow, command = pinned()
    command[field] = "unrelated"
    assert preempt.barrier_commands([command], workflow) == []
    assert preempt.barrier_reason_errors([command], workflow), (
        "identity drift must invalidate the workflow's pinned barrier"
    )


def test_a_duplicate_id_is_ambiguous_even_when_the_workflow_pins_it() -> None:
    workflow, command = pinned()
    assert len(preempt.barrier_commands([command, dict(command)], workflow)) == 2
    assert preempt.barrier_reason_errors([command, dict(command)], workflow), (
        "a duplicate command ID cannot uniquely prove the barrier"
    )


@pytest.mark.parametrize("value", [None, "", 3])
def test_missing_or_invalid_pointer_never_selects_a_different_command(
    value: Any,
) -> None:
    workflow, command = pinned()
    workflow["step_executions"][3]["details"]["remote_command_id"] = value
    assert preempt.barrier_commands([command], workflow) == []


@pytest.mark.parametrize(
    "operation", ["MARK_UNSCHEDULABLE", "FREEZE_EVIDENCE", "QUIESCE_GPU_SERVICES"]
)
def test_an_attempt_key_on_an_unrelated_step_is_not_a_barrier(operation: str) -> None:
    command = barrier_commands()[0]
    command["step"]["operation"] = operation
    assert preempt.barrier_commands([command]) == []
    assert preempt.barrier_reason_errors([command]), (
        "an unrelated operation cannot supply client-verification proof"
    )


@pytest.mark.parametrize("attempt", [None, 0, -1, True, 1.5, "4", float("inf")])
def test_a_hold_requires_a_positive_integer_attempt(attempt: Any) -> None:
    command = barrier_commands()[0]
    command["result_details"]["gpu_client_quiesce_attempt"] = attempt
    assert "barrier command recorded no client-verification attempt" in (
        preempt.barrier_reason_errors([command])
    )


@pytest.mark.parametrize("drift", ["wrong-index", "wrong-operation", "duplicate-index"])
def test_compound_progress_must_point_to_the_actual_verify_step(drift: str) -> None:
    workflow, command = pinned()
    if drift == "wrong-index":
        command["result_details"]["batched_step_index"] = 4
    elif drift == "wrong-operation":
        command["result_details"]["batched_operation"] = "RESET_GPU"
    else:
        command["batched_steps"].append(dict(command["batched_steps"][0]))
    assert preempt.barrier_reason_errors([command], workflow), (
        "compound progress must identify exactly the planned verification step"
    )


@pytest.mark.parametrize(
    "drift",
    [
        "reset-waiting",
        "reset-succeeded",
        "verify-succeeded",
        "missing-quiesce",
        "unknown-step",
        "contradictory-hold",
    ],
)
def test_a_compound_tail_is_unstarted_only_with_complete_waiting_progress(
    drift: str,
) -> None:
    command = compound_hold("FAILED")
    progress = command["result_details"]["batched_results"]
    if drift.startswith("reset-"):
        progress["4"] = {"status": drift.removeprefix("reset-").upper()}
    elif drift == "verify-succeeded":
        progress["3"]["status"] = "SUCCEEDED"
    elif drift == "missing-quiesce":
        del progress["2"]
    elif drift == "unknown-step":
        progress["99"] = {"status": "SUCCEEDED"}
    else:
        progress["3"]["details"]["reason"] = "a different failure"
    errors = preempt.cancelled_command_errors([command], successor_request_id=REBOOT_ID)
    assert "the superseded reset issued a RESET_GPU command" in errors, errors
    command["status"] = "WAITING"
    assert preempt.barrier_reason_errors([command]), (
        "WAITING alone cannot prove the compound reset tail remains unstarted"
    )


def test_a_proven_unstarted_tail_does_not_hide_a_separate_reset_command() -> None:
    command = compound_hold("FAILED")
    assert (
        preempt.cancelled_command_errors([command], successor_request_id=REBOOT_ID)
        == []
    )
    reset = {"step": {"operation": "RESET_GPU"}, "status": "FAILED"}
    errors = preempt.cancelled_command_errors(
        [command, reset], successor_request_id=REBOOT_ID
    )
    assert "the superseded reset issued a RESET_GPU command" in errors


def test_lifetime_cancellation_of_an_unrelated_command_cannot_prove_the_hold() -> None:
    commands = lifetime_data.happy_commands()
    unrelated = dict(commands[1])
    unrelated["step"] = {"operation": "FREEZE_EVIDENCE"}
    commands[1].update(status="SUCCEEDED", status_source="executor-result")
    commands.append(unrelated)
    errors = lifetime.remote_command_errors(commands, t_cancel=lifetime_data.T_CANCEL)
    assert "the waiting barrier command was not cancelled by the deadline" in errors


@pytest.mark.parametrize("status", ["PENDING", "LEASED", "WAITING", None])
def test_lifetime_completion_cannot_leave_another_command_nonterminal(
    status: Any,
) -> None:
    commands = lifetime_data.happy_commands()
    commands.append({"step": {"operation": "FREEZE_EVIDENCE"}, "status": status})
    errors = lifetime.remote_command_errors(commands, t_cancel=lifetime_data.T_CANCEL)
    assert any("still nonterminal" in error for error in errors), errors


@pytest.mark.parametrize("stamp", [None, "invalid", "2026-09-06T10:00:00"])
def test_lifetime_cannot_place_a_success_without_an_aware_timestamp(stamp: Any) -> None:
    commands = lifetime_data.happy_commands()
    commands[0]["updated_at"] = stamp
    errors = lifetime.remote_command_errors(commands, t_cancel=lifetime_data.T_CANCEL)
    assert any("no valid update time" in error for error in errors), errors
