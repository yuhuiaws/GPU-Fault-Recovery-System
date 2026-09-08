"""The BOOT-001..010 shell runner's verdict contract.

The shell cannot be run here (it needs a cluster), so the contract is pinned
by syntax-checking every script and reading the runner for the structures the
2026-09-07 review asked for: one ``VERDICT`` line per case, an ERR trap that
writes ``VERDICT FAIL``, a resume gate on the last line, no silent ``[[ ]]``
checks, and a Deployment wait before any Pod name is read.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/e2e/regional/run_regional_boot_guard_cases.sh"
GUARD_DIR = ROOT / "scripts/e2e/regional/boot_guard"


@pytest.mark.parametrize(
    "script", [RUNNER, *sorted(GUARD_DIR.glob("*.sh"))], ids=lambda path: path.name
)
def test_shell_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def _runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_every_case_begins_and_ends_with_one_verdict() -> None:
    runner = _runner()

    begins = re.findall(r"^\s*begin_case (\d+)$", runner, flags=re.MULTILINE)
    # BOOT-006 was deleted (quick diagnostics feature removed), so 1-5 and 7-10.
    assert [int(item) for item in begins] == list(range(1, 6)) + list(range(7, 11))
    assert runner.count("\n  pass_case\n") + runner.count("\npass_case\n") == 9
    assert "trap on_error ERR" in runner
    assert "VERDICT FAIL" in runner and 'echo "VERDICT PASS"' in runner


def test_resume_gate_reads_the_last_line_only() -> None:
    runner = _runner()

    assert '"$(tail -n 1 "$1")" == "VERDICT PASS"' in runner
    assert "if ! case_passed" in runner
    # The old gate matched assert.sh's own PASS line, written before the later
    # checks of the same case ran.
    assert 'grep -qx "PASS"' not in runner


def test_no_silent_test_expressions_remain() -> None:
    silent = [
        line
        for line in _runner().splitlines()
        if re.fullmatch(r"\s*\[\[ .* \]\]\s*", line)
    ]
    assert silent == [], silent
    assert _runner().count("fail_case ") >= 8


def test_pod_names_are_read_after_the_deployment_is_available() -> None:
    runner = _runner()

    assert 'wait deployment "${PROBE}" --for=condition=Available' in runner
    assert runner.count('probe_pod="$(probe_pod_name)"') == 3
    # BOOT-001 re-reads Ready one readiness period after the guard fired; that is
    # the single remaining .items[0] read and it happens after assert.sh matched.
    assert runner.count(".items[0].metadata.name") == 1
    assert 'sleep "${readiness_period}"' in runner


def test_case_specific_additions_are_present() -> None:
    runner = _runner()

    assert "cloudtrail_provisional=true" in runner
    assert "CONTROL_PLANE_ROLE_NAME" in runner
    assert "/healthz" in runner
    assert "/v1/collector-status/" in runner
    assert "GPU_FAULT_EXECUTION_TOKEN" in runner
