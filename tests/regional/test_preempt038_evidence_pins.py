"""Contract tests for GF-REGIONAL-PREEMPT-038 (evidence pinned to incidents)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import preempt038_verdicts as verdicts
from scripts.e2e.regional import run_preempt038_evidence_pins as preempt038

ROOT = Path(__file__).resolve().parents[2]


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def _report(**overrides: Any) -> dict[str, Any]:
    report = {
        "unrelated_key": "audit-expired-evidence-abc",
        "pinned_key": "audit-pinned-evidence-abc",
        "unrelated_deleted_after_seconds": 42.0,
        "pinned_present_after_unrelated_deleted": True,
        "pinned_present_after_extra_wait": True,
        "pinned_deleted_after_recovered_seconds": 55.0,
        "residual_rows": 0,
    }
    report.update(overrides)
    return report


def test_the_audit_contract_passes_when_only_the_pinned_row_survives() -> None:
    assert verdicts.audit_errors(_report()) == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"unrelated_deleted_after_seconds": None}, "never deleted"),
        ({"pinned_present_after_unrelated_deleted": False}, "did not survive"),
        (
            {"pinned_present_after_extra_wait": False},
            "while its incident was still open",
        ),
        (
            {"pinned_deleted_after_recovered_seconds": None},
            "not deleted after its incident recovered",
        ),
        ({"residual_rows": 2}, "2 audit rows remain"),
    ],
)
def test_each_audit_deviation_fails(overrides: dict[str, Any], fragment: str) -> None:
    assert fragment in _text(verdicts.audit_errors(_report(**overrides)))


def test_the_cleanup_log_contract_names_both_deleted_keys() -> None:
    logs = (
        "INFO gpu_fault.store.cleanup cleanup raw_evidence deleted 1 rows; "
        "keys[:20]=audit-expired-evidence-abc\n"
        "INFO gpu_fault.store.cleanup cleanup raw_evidence deleted 3 rows; "
        "keys[:20]=other-1, audit-pinned-evidence-abc, other-2\n"
    )
    assert (
        verdicts.log_errors(
            logs,
            unrelated_key="audit-expired-evidence-abc",
            pinned_key="audit-pinned-evidence-abc",
        )
        == []
    )
    assert len(verdicts.cleanup_lines(logs)) == 2
    assert "no 'cleanup raw_evidence deleted'" in _text(
        verdicts.log_errors("nothing", unrelated_key="a", pinned_key="b")
    )
    assert "no cleanup line names the pinned key" in _text(
        verdicts.log_errors(
            logs.splitlines()[0],
            unrelated_key="audit-expired-evidence-abc",
            pinned_key="missing",
        )
    )
    assert "zero rows" in _text(
        verdicts.log_errors(
            "cleanup raw_evidence deleted 0 rows; keys[:20]=",
            unrelated_key="a",
            pinned_key="b",
        )
    )


def test_the_runner_is_plan_by_default_and_ships_the_audit(tmp_path: Path) -> None:
    parser = preempt038.parser()
    arguments = parser.parse_args(["--run-dir", str(tmp_path)])
    assert arguments.execute is False and arguments.confirm == ""
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in parser.format_help(), flag
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", str(tmp_path), "--plan", "--execute"])
    assert preempt038.CASE_ID == "GF-REGIONAL-PREEMPT-038"
    assert preempt038.CONFIRMATION == "PREEMPT038_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-PREEMPT-037"
    assert preempt038.AUDIT.is_file(), preempt038.AUDIT
    assert preempt038.AUDIT.name == "audit_raw_evidence_periodic_cleanup.py"
