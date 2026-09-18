"""CPU and runner restoration share one fail-closed physical outcome contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from gpu_fault.recovery_safety import (
    command_recovery_error,
    recovery_safety_errors,
    unresolved_details,
    workflow_recovery_error,
)
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "unknown",
        {"outcome_unknown": True},
        {"outcome_unknown": "false"},
        {"manual_confirmation_required": True, "node_action_not_started": True},
        {"node_action_interrupted": True},
        {"node_action_response_unknown": True},
        {"ownership_permit_delivery_unknown": True},
        {"reset_outcome_unknown": ["GPU-a"]},
        {"reset_outcome_unknown": None},
        {"node_action_state": "PENDING"},
        {"details": None},
        {"details": {"outcome_unknown": True}},
        {"node_results": []},
        {"node_results": {"node": None}},
        {"node_results": {"node": {"outcome_unknown": True, "details": {}}}},
        {"node_failure_details": {"node": {"details": {"outcome_unknown": True}}}},
        {BATCHED_RESULTS_KEY: {"0": {"status": "WAITING", "details": {}}}},
        {
            BATCHED_RESULTS_KEY: {
                "0": {
                    "status": "FAILED",
                    "status_source": "workflow-timeout",
                    "details": {},
                }
            }
        },
    ],
)
def test_unknown_or_malformed_receipts_cannot_be_restoration_authority(value):
    assert unresolved_details(value), (
        "unknown or malformed action details must retain the recovery hold"
    )


def test_known_nested_receipts_and_bounded_recursion():
    assert not unresolved_details(
        {
            "outcome_unknown": False,
            "reset_outcome_unknown": [],
            "node_results": {"node": {"details": {}}},
            "node_failure_details": {},
            BATCHED_RESULTS_KEY: {
                "0": {"status": "SUCCEEDED", "details": {}},
                "1": {"status": "FAILED", "details": {"node_action_not_started": True}},
            },
        }
    ), "known nested completion receipts must not require reconciliation"
    value = {}
    for _ in range(26):
        value = {"details": value}
    assert unresolved_details(value), (
        "receipt nesting beyond the bound must fail closed"
    )


def cancelled_command():
    return {
        "status": "FAILED",
        "status_source": "workflow-timeout",
        "lease_owner": None,
        "last_lease_owner": "executor",
        "result_details": {},
    }


@pytest.mark.parametrize(
    "proof,accepted",
    [
        ("none", False),
        ("never-leased", True),
        ("missing-lease-key", False),
        ("success", True),
        ("failure", True),
        ("waiting", False),
        ("no-start", True),
        ("accepted", False),
        ("provider", False),
        ("unknown", False),
        ("bad-source", False),
        ("ordinary", True),
    ],
)
def test_cancellation_requires_a_complete_known_outcome_or_proven_no_submission(
    proof, accepted
):
    command = cancelled_command()
    details = command["result_details"]
    if proof in {"never-leased", "missing-lease-key"}:
        command["last_lease_owner"] = None
        if proof == "missing-lease-key":
            command.pop("lease_owner")
    elif proof in {"success", "failure", "waiting"}:
        details["post_cancellation_status"] = {
            "success": "SUCCEEDED",
            "failure": "FAILED",
            "waiting": "WAITING",
        }[proof]
    elif proof in {"no-start", "accepted", "provider"}:
        details["node_action_not_started"] = True
        if proof == "accepted":
            details["node_action_accepted_nodes"] = ["node"]
        elif proof == "provider":
            details["provider_operation_id"] = "provider-command"
    elif proof == "unknown":
        details.update(post_cancellation_status="SUCCEEDED", outcome_unknown=True)
    elif proof == "bad-source":
        command["status_source"] = []
    elif proof == "ordinary":
        command["status_source"] = "executor-result"
    assert (command_recovery_error(command) is None) is accepted


@pytest.mark.parametrize(
    "source", ["workflow-preempted", "completed-after-cancellation"]
)
def test_every_cancellation_source_retains_the_same_hold(source):
    command = cancelled_command()
    command["status_source"] = source
    assert (
        command_recovery_error(command)
        == "remote cancellation is not physical completion"
    )


@pytest.mark.parametrize(
    "command",
    [
        None,
        [],
        {},
        {"status": "LEASED"},
        {"status": "SUCCEEDED", "result_details": None},
    ],
)
def test_unknown_remote_inventory_is_not_empty_or_terminal(command):
    assert command_recovery_error(command) is not None


@pytest.mark.parametrize(
    "workflow",
    [
        None,
        [],
        {},
        {"status": "invalid"},
        {"status": []},
        {"status": "FAILED", "blocked_kind": "invalid"},
        {"status": "PENDING"},
        {"status": "BLOCKED"},
        {"status": "BLOCKED", "blocked_kind": "NEEDS_OPERATOR"},
        {"status": "FAILED", "blocked_kind": "NEEDS_OPERATOR"},
        {"status": "SUCCEEDED", "blocked_kind": "NEEDS_OPERATOR"},
        {"status": "SUPERSEDED", "blocked_kind": "NEEDS_OPERATOR"},
        {"status": "FAILED", "step_executions": {}},
        {"status": "FAILED", "step_executions": [None]},
        {
            "status": "FAILED",
            "step_executions": [{"details": {"outcome_unknown": True}}],
        },
    ],
)
def test_occupied_or_unknown_workflow_inventory_retains_recovery(workflow):
    assert workflow_recovery_error(workflow) is not None


@pytest.mark.parametrize(
    "workflow",
    [
        {"status": "SUCCEEDED"},
        {"status": "FAILED", "step_executions": []},
        {"status": "SUPERSEDED"},
        {"status": "BLOCKED", "blocked_kind": "SAFETY_SETTLED"},
    ],
)
def test_confirmed_settled_workflow_is_not_an_operator_hold(workflow):
    assert workflow_recovery_error(workflow) is None


def test_inventory_validation_is_pure_complete_and_reports_both_sources():
    assert recovery_safety_errors(None, []) == [
        "collector recovery snapshot is incomplete"
    ]
    assert recovery_safety_errors([], None) == [
        "collector recovery snapshot is incomplete"
    ]
    workflows, commands = [{"status": "RUNNING"}], [cancelled_command()]
    before = deepcopy((workflows, commands))
    assert len(recovery_safety_errors(workflows, commands)) == 2
    assert (workflows, commands) == before
    assert recovery_safety_errors([], []) == []
