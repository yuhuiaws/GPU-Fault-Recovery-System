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
