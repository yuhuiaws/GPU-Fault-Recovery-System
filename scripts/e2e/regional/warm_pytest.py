"""Supervised receipt collection for the warm-spare audit's shared pytest batch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.pytest_result_identity import PytestReceipt


def run_pytest(
    case_dir: Path, nodeids: list[str], *, root: Path
) -> PytestReceipt | None:
    """Run one supervised batch; only a complete source-bound receipt is evidence."""

    from gpu_fault.admin import execution
    from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
    from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
    from scripts.e2e.regional import regional_commands
    from scripts.e2e.regional.acceptance_supervision import record_supervision_loss
    from scripts.e2e.regional.focused_pytest import prepare_focused_pytest
    from tools import pytest_result_identity as evidence

    if not nodeids:
        raise ValueError("focused pytest requires an explicit selector")
    command = [sys.executable, "-m", "pytest", "-q", *nodeids]
    with prepare_focused_pytest(command, cwd=root, environment=None) as prepared:
        if prepared is None:
            raise RuntimeError("focused pytest command was not recognized")
        try:
            completed = execution.run_command(
                prepared.command,
                cwd=prepared.root,
                environment=prepared.environment,
                input_text=None,
                timeout_seconds=300,
            )
        except ProcessSupervisionLost:
            try:
                record_supervision_loss()
            except BaseException:
                raise ProcessSupervisionLost(
                    "pytest supervision was lost and its recovery marker could not be persisted"
                ) from None
            raise
        except (subprocess.TimeoutExpired, DeploymentDeadlineExceeded):
            raise regional_commands.RegionalCommandTimeout(command, 300) from None
        except OSError as exc:
            raise regional_commands.RegionalFixtureError(
                f"pytest could not start ({type(exc).__name__}, errno={exc.errno})"
            ) from None
        output = (completed.stdout or "") + (completed.stderr or "")
        receipt = None
        try:
            if completed.returncode not in {0, 1}:
                raise ValueError("pytest did not complete normal test execution")
            if evidence.source_identity(prepared.root) != prepared.source_identity:
                raise ValueError("source changed during focused pytest")
            candidate = evidence.load_pytest_receipt(
                prepared.report_path,
                root=prepared.root,
                expected_identity=prepared.source_identity,
                require_session=True,
                expected_exitstatus=completed.returncode,
            )
            if candidate.collection_skips:
                raise ValueError("pytest skipped a collector")
            if completed.returncode and not any(
                isinstance(record, dict)
                and record.get("status") == "FAIL"
                and isinstance(record.get("phases"), dict)
                and "failed" in record["phases"].values()
                for record in candidate.records.values()
            ):
                raise ValueError("pytest failure has no corresponding failed phase")
            receipt = candidate
        except (OSError, RuntimeError, ValueError) as exc:
            output += f"\nfocused pytest evidence rejected: {exc}\n"
        path = case_dir / "pytest.log"
        path.write_text(output, encoding="utf-8")
        path.chmod(0o600)
        return receipt
