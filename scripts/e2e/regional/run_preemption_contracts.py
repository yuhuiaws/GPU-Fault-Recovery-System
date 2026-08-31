from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
CASE_NODEIDS = {
    "GF-REGIONAL-PREEMPT-001": (
        "tests/execution/test_executor.py::"
        "test_preemption_config_disabled_keeps_predecessor_running",
    ),
    "GF-REGIONAL-PREEMPT-002": (
        "tests/orchestration/test_cross_fault_arbitration.py::"
        "test_preemption_marks_stronger_successor_and_reuses_containment",
        "tests/execution/test_executor.py::"
        "test_executor_supersedes_at_clean_step_boundary",
    ),
    "GF-REGIONAL-PREEMPT-003": (
        "tests/orchestration/test_merge.py::"
        "test_idle_node_stronger_action_requests_safe_preemption",
        "tests/orchestration/test_merge.py::test_preemption_scope_boundaries",
    ),
    "GF-REGIONAL-PREEMPT-004": (
        "tests/orchestration/test_merge.py::"
        "test_preemption_requires_strictly_higher_recovery_rank",
    ),
    "GF-REGIONAL-PREEMPT-005": (
        "tests/execution/test_node_action.py::"
        "test_unclaimed_remote_command_is_cancelled_before_preemption",
    ),
    "GF-REGIONAL-PREEMPT-006": (
        "tests/execution/test_misc.py::"
        "test_submitted_reset_restores_before_reboot_preemption",
    ),
    "GF-REGIONAL-PREEMPT-007": (
        "tests/execution/test_misc.py::"
        "test_submitted_reboot_finishes_before_replace_successor",
    ),
    "GF-REGIONAL-PREEMPT-008": (
        "tests/execution/test_misc.py::"
        "test_reset_to_reboot_hands_off_quiesce_and_skips_low_reset",
        "tests/execution/test_executor.py::"
        "test_executor_restores_when_successor_cannot_take_quiesce_handoff",
    ),
    "GF-REGIONAL-PREEMPT-009": (
        "tests/execution/test_validation.py::"
        "test_local_validation_waiting_is_safe_to_preempt",
        "tests/execution/test_node_action.py::"
        "test_safe_remote_waiting_is_cancelled_before_preemption",
        "tests/execution/test_node_action.py::"
        "test_remote_waiting_step_is_not_preempted",
        "tests/execution/test_node_action.py::"
        "test_restart_workload_remote_waiting_is_not_preempted",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one deterministic PREEMPT-001..009 contract fixture."
    )
    parser.add_argument("--case", choices=tuple(CASE_NODEIDS), required=True)
    arguments = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *CASE_NODEIDS[arguments.case],
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    print(
        json.dumps(
            {
                "case_id": arguments.case,
                "verdict": "PASS" if completed.returncode == 0 else "FAIL",
                "pytest_nodeids": list(CASE_NODEIDS[arguments.case]),
            },
            sort_keys=True,
        )
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
