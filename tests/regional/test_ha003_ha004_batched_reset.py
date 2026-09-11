"""HA-003 and HA-004 must recognise RESET_GPU inside a round-trip batch.

Since protocol 3 the node command that resets a GPU heads with
QUIESCE_GPU_SERVICES and carries RESET_GPU in ``batched_steps``; HA-003
attempt 1 filtered on ``step.operation`` alone, saw no reset command at all
and failed a workflow that had SUCCEEDED.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004

_BATCHED = {
    "command_id": "remote-batch",
    "status": "WAITING",
    "step": {"operation": "QUIESCE_GPU_SERVICES"},
    "batched_steps": [
        {"step": {"operation": "VERIFY_NO_GPU_CLIENTS"}},
        {"step": {"operation": "RESET_GPU"}},
        {"step": {"operation": "RESTORE_GPU_SERVICES"}},
    ],
}
_MARK = {
    "command_id": "remote-mark",
    "status": "SUCCEEDED",
    "step": {"operation": "MARK_UNSCHEDULABLE"},
}


def test_the_reset_status_is_read_off_the_command_that_batches_it() -> None:
    state = {"commands": [_MARK, _BATCHED]}

    assert ha003.reset_command_status(state) == "WAITING", (
        "RESET_GPU inside batched_steps was not recognised"
    )
    assert ha003.reset_command_status({"commands": [_MARK]}) is None, (
        "a command without RESET_GPU counted as the reset"
    )


def test_neither_runner_filters_on_the_head_step_alone() -> None:
    pattern = re.compile(
        r"""step["']?\s*,?\s*\{?\}?\)?\.get\(["']operation["']\)\s*==\s*["']RESET_GPU["']"""
    )
    for module in (ha003, ha004):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not pattern.search(source), (
            f"{module.__name__} still reads only step.operation"
        )
        assert "command_operations(item)" in source, (
            f"{module.__name__} does not use command_operations"
        )


def test_the_env_window_only_assigns_what_the_live_executor_does_not_already_carry() -> (
    None
):
    """The shipped poll interval is 2 s, the case's own value; assigning it made
    executor_env_window read the live env as an unrecorded open and refuse."""

    live = {
        ha004.LEASE_ENV: {"present": True, "value": "120"},
        ha004.POLL_ENV: {"present": True, "value": "2"},
    }
    assert ha004.window_assignments(live) == {ha004.LEASE_ENV: "10"}, (
        "poll=2 was re-assigned"
    )
    assert ha004.window_assignments({}) == {
        ha004.LEASE_ENV: "10",
        ha004.POLL_ENV: "2",
    }, "absent variables must be assigned"
    assert ha004.window_assignments(
        {ha004.LEASE_ENV: {"present": True, "value": "10"}}
    ) == {ha004.POLL_ENV: "2"}, (
        "a lease already at the test value must be left to the window's own refusal"
    )
