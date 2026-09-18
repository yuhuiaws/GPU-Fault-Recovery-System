"""The BOOT-001..010 shell runner's verdict contract.

The complete live entry is never invoked. Syntax and verdict structure checks
are supplemented by an extracted Pod lookup function with a mocked transport.
"""

from __future__ import annotations

import json
import os
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


POD_LOOKUP_TRANSPORT = r"""
kubectl() {
  printf '%s\n' "$*" >>"${TEST_CALLS}"
  case "$*" in
    "--kubeconfig /dev/null -n unit-ns rollout status deployment probe --timeout=600s")
      return "${TEST_ROLLOUT_STATUS}" ;;
    "--kubeconfig /dev/null -n unit-ns get pod -l app=probe -o json")
      printf '%s\n' "${TEST_PODS}"
      return "${TEST_INVENTORY_STATUS}" ;;
    *) printf 'unexpected mocked kubectl invocation\n' >&2; return 90 ;;
  esac
}
fail_case() { printf '%s\n' "$*" >&2; exit 1; }
"""


def pod_lookup(
    tmp_path: Path,
    pods: list[dict],
    *,
    rollout_status: int = 0,
    inventory_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    source = _runner()
    start = source.index("\nprobe_pod_name() {") + 1
    end = source.index("\n}\n", start) + 3
    calls_file = tmp_path / "kubectl-calls.txt"
    harness = "\n".join(
        [
            "set -Eeuo pipefail",
            POD_LOOKUP_TRANSPORT,
            source[start:end],
            'probe_pod="$(probe_pod_name)"',
            'printf "POD=%s\\n" "${probe_pod}"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={
            **os.environ,
            "CPU_KUBECONFIG": "/dev/null",
            "NAMESPACE": "unit-ns",
            "PROBE": "probe",
            "TEST_CALLS": str(calls_file),
            "TEST_PODS": json.dumps({"items": pods}),
            "TEST_ROLLOUT_STATUS": str(rollout_status),
            "TEST_INVENTORY_STATUS": str(inventory_status),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    calls = calls_file.read_text(encoding="utf-8").splitlines()
    return result, calls


def test_pod_lookup_waits_for_current_rollout_and_ignores_terminating_pods(
    tmp_path: Path,
) -> None:
    result, calls = pod_lookup(
        tmp_path,
        [
            {
                "metadata": {
                    "name": "previous",
                    "deletionTimestamp": "2026-09-01T00:00:00Z",
                }
            },
            {"metadata": {"name": "current"}},
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "POD=current\n", (
        "a terminating Pod from the prior rollout must not be selected"
    )
    assert calls == [
        "--kubeconfig /dev/null -n unit-ns rollout status deployment probe --timeout=600s",
        "--kubeconfig /dev/null -n unit-ns get pod -l app=probe -o json",
    ], "the bound current rollout must finish before Pod inventory is read"


@pytest.mark.parametrize("failure", ["rollout", "inventory"])
def test_failed_probe_read_cannot_return_a_pod_from_command_substitution(
    failure: str, tmp_path: Path
) -> None:
    result, calls = pod_lookup(
        tmp_path,
        [{"metadata": {"name": "stale-pod"}}],
        rollout_status=1 if failure == "rollout" else 0,
        inventory_status=1 if failure == "inventory" else 0,
    )
    assert result.returncode != 0, (
        f"{failure} failure must propagate through the caller's command substitution"
    )
    assert "POD=" not in result.stdout, "failed reads must not yield a usable Pod name"
    assert len(calls) == (1 if failure == "rollout" else 2), (
        "rollout failure must stop before inventory; inventory must be read only once"
    )


@pytest.mark.parametrize(
    "pods",
    [
        [],
        [{"metadata": {"name": "a"}}, {"metadata": {"name": "b"}}],
        [{"metadata": {"name": "old", "deletionTimestamp": "2026-09-01T00:00:00Z"}}],
    ],
    ids=["empty", "multiple", "terminating-only"],
)
def test_probe_lookup_requires_one_nonterminating_pod(
    pods: list[dict], tmp_path: Path
) -> None:
    result, _calls = pod_lookup(tmp_path, pods)
    assert result.returncode != 0, (
        "ambiguous or absent probe inventory must fail closed"
    )
    assert "expected exactly one probe Pod" in result.stderr, (
        "invalid inventory must include a useful refusal diagnostic"
    )
    assert "POD=" not in result.stdout, "invalid inventory must not yield a Pod name"


def test_case_specific_additions_are_present() -> None:
    runner = _runner()

    assert "cloudtrail_provisional=true" in runner
    assert "CONTROL_PLANE_ROLE_NAME" in runner
    assert "/healthz" in runner
    assert "/v1/collector-status/" in runner
    assert "GPU_FAULT_EXECUTION_TOKEN" in runner
