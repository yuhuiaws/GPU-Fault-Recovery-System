"""The preemption may freeze the barrier command mid-poll; it is still the barrier.

Live (eks-first site): the predecessor's compound command was cancelled while
its VERIFY_NO_GPU_CLIENTS attempt was pending at the agent. The record the
preemption froze carried only the poll's own keys, so the verdict could prove
neither that the batched RESET_GPU never started nor that the command was the
client-verification barrier -- "the superseded reset issued a RESET_GPU
command" and "cancelled command does not carry the client-verification
barrier". With the adapter carrying the last verdict under the pending poll,
the same freeze is the barrier, and the contract is unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.e2e.regional import destr016_verdicts as verdicts
from tests.regional._destr016_builders import (
    NODE,
    REBOOT_ID,
    RESET_ID,
    compound_hold,
    parked_reset_workflow,
)

POLL_KEYS = {
    "node_action_command_id": (
        f"{RESET_ID}/3/VERIFY_NO_GPU_CLIENTS/{NODE}/attempt-20/agent-14"
    ),
    "node_action_state": "PENDING",
    "node_action_accepted_nodes": [NODE],
    "node_action_endpoint": "https://node-b.example:9099",
}
LIVE_ERRORS = (
    "the superseded reset issued a RESET_GPU command",
    "cancelled command does not carry the client-verification barrier",
)


def _frozen_mid_poll(*, carried: bool) -> dict[str, Any]:
    """The cancelled compound command as a pending poll left it: with the last
    verdict carried under the poll's keys (the adapter now), or bare (before)."""

    command = compound_hold("FAILED")
    verdict = copy.deepcopy(
        command["result_details"]["batched_results"]["3"]["details"]
    )
    held = {**(verdict if carried else {}), **POLL_KEYS}
    compound = {
        key: command["result_details"][key]
        for key in ("batched_step_index", "batched_operation", "batched_results")
    }
    compound["batched_results"]["3"] = {"status": "WAITING", "details": held}
    command["result_details"] = {**held, **compound}
    return command


def _workflow() -> dict[str, Any]:
    workflow = parked_reset_workflow()
    for execution in workflow["step_executions"]:
        if execution["step_index"] == 3:
            execution["details"]["remote_command_id"] = "remote-compound"
    return workflow


@pytest.mark.parametrize(
    "with_workflow", [True, False], ids=["by-workflow", "by-shape"]
)
def test_a_barrier_frozen_mid_poll_with_its_continuation_is_the_barrier(
    with_workflow: bool,
) -> None:
    commands = [_frozen_mid_poll(carried=True)]

    errors = verdicts.cancelled_command_errors(
        commands,
        successor_request_id=REBOOT_ID,
        workflow=_workflow() if with_workflow else None,
    )

    assert errors == [], errors
    assert verdicts.unexecuted_barrier_tail(commands[0]) == 2, (
        "RESET_GPU and RESTORE_GPU_SERVICES are proved unstarted behind the WAITING prefix"
    )


def test_a_barrier_frozen_mid_poll_without_its_continuation_is_the_live_failure() -> (
    None
):
    """What the record looked like before the adapter carried the verdict: the
    poll's keys alone. Both live errors, in order, from the one record."""

    errors = verdicts.cancelled_command_errors(
        [_frozen_mid_poll(carried=False)],
        successor_request_id=REBOOT_ID,
        workflow=_workflow(),
    )

    assert errors == list(LIVE_ERRORS), errors
