"""Run one deterministic PREEMPT-001..009 contract fixture.

Each case is a fixed set of pytest selectors with complete, source-bound
execution receipts. Given ``--run-dir`` it writes ``cases/<id>/<id>.json`` so the
PREEMPT chain (PREEMPT-012's predecessor resolution in particular) can find a
PASS where the formal order expects one. Stdout contains pytest output
followed by one source-bound JSON verdict line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pytest_result_identity import PytestReceipt  # noqa: E402
from tools.run_fault_test_cases import (  # noqa: E402
    build_isolated_environment,
    run_reported_pytest,
)

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

PytestRunner = Callable[..., "subprocess.CompletedProcess[str]"]


def _run_pytest(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)


def case_evidence(
    case_id: str,
    completed: subprocess.CompletedProcess[str],
    *,
    release_id: str = "",
    receipt: PytestReceipt | None = None,
    receipt_error: str | None = None,
) -> dict[str, Any]:
    """The evidence document for one PREEMPT contract case."""

    errors = (
        ["pytest execution receipt is missing"]
        if receipt is None
        else receipt.complete_selection_errors(list(CASE_NODEIDS[case_id]), root=ROOT)
    )
    if receipt_error:
        errors.append(receipt_error)
    if completed.returncode:
        errors.append("pytest did not finish successfully")
    session = receipt.session if receipt is not None else None
    document: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "fault-acceptance",
        "case_id": case_id,
        "verdict": "PASS" if not errors else "FAIL",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "fixture": "deterministic pytest contract; no cluster touched",
        "pytest_nodeids": list(CASE_NODEIDS[case_id]),
        "pytest_returncode": completed.returncode,
        "source_identity": (session or {}).get("source_identity"),
        "pytest_session": session,
        "pytest_records": receipt.records if receipt is not None else {},
        "receipt_errors": errors,
        "proof_scope": "local-contract",
        "pytest_output_sha256": hashlib.sha256(
            (completed.stdout or "").encode("utf-8")
        ).hexdigest(),
    }
    if release_id:
        document["release_id"] = release_id
    return document


def run_case(
    case_id: str,
    *,
    run_dir: Path | None = None,
    release_id: str = "",
    runner: PytestRunner = _run_pytest,
) -> int:
    command = [sys.executable, "-m", "pytest", "-q", *CASE_NODEIDS[case_id], "-n", "0"]
    completed, receipt, receipt_error = run_reported_pytest(
        command, environment=build_isolated_environment(), runner=runner
    )
    document = case_evidence(
        case_id,
        completed,
        release_id=release_id,
        receipt=receipt,
        receipt_error=receipt_error,
    )
    print(completed.stdout, end="")
    if run_dir is not None:
        from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
        from scripts.e2e.regional.regional_case_contract import case_evidence_path

        write_json_atomic(
            case_evidence_path(run_dir, case_id),
            document,
        )
    print(
        json.dumps(
            {
                "case_id": case_id,
                "verdict": document["verdict"],
                "pytest_nodeids": list(CASE_NODEIDS[case_id]),
                "source_identity": document["source_identity"],
                "receipt_errors": document["receipt_errors"],
                "proof_scope": "local-contract",
            },
            sort_keys=True,
        )
    )
    return 0 if document["verdict"] == "PASS" else (completed.returncode or 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", choices=tuple(CASE_NODEIDS), required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="write cases/<id>/<id>.json for the case under this directory",
    )
    parser.add_argument(
        "--release-id",
        default="",
        help="release annotation only; this local contract does not verify a deployment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return run_case(
        arguments.case,
        run_dir=arguments.run_dir,
        release_id=arguments.release_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
