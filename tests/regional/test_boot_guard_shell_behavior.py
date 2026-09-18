from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

GUARDS = Path(__file__).resolve().parents[2] / "scripts/e2e/regional/boot_guard"
SHELL = r"""
kubectl() {
  local args="$*"
  if [[ "${args}" == *"delete deployment"* ]]; then
    [[ "${TEST_FAIL}" != "delete" ]] || return 1
  elif [[ "${args}" == *"get pod -l"* ]]; then
    [[ "${TEST_FAIL}" != "list" ]] || return 1
    printf '%s\n' "${TEST_PODS}"
  elif [[ "${args}" == *"get pod "* ]]; then
    [[ "${TEST_FAIL}" != "read" ]] || return 1
    printf '%s\n' "${TEST_POD}"
  elif [[ "${args}" == *"logs "* ]]; then
    [[ "${TEST_FAIL}" != "logs" ]] || return 1
    printf '%s\n' "expected startup refusal"
  elif [[ "${args}" == *"get endpoints "* ]]; then
    printf '%s\n' "${TEST_ENDPOINTS}"
  elif [[ "${args}" == *"delete secret"* || "${args}" == *"get secret"* ]]; then
    [[ "${TEST_FAIL}" != "secret" ]] || return 1
  else
    echo "unexpected mocked kubectl command" >&2
    return 90
  fi
}
sleep() { :; }
export -f kubectl sleep
bash "$@"
"""


def run_guard(
    script: str,
    *,
    failure: str = "",
    ready: str | None = "False",
    changed_uid: bool = False,
    endpoint: bool = False,
) -> subprocess.CompletedProcess[str]:
    pod = {
        "metadata": {"name": "probe", "uid": "original"},
        "spec": {"containers": [{"name": "api"}]},
        "status": {
            "phase": "Running",
            "conditions": [] if ready is None else [{"type": "Ready", "status": ready}],
            "containerStatuses": [{"name": "api", "ready": False, "restartCount": 0}],
        },
    }
    inventory = json.dumps({"items": [pod] if script == "assert.sh" else []})
    if changed_uid:
        pod["metadata"]["uid"] = "replacement"
    environment = {
        **os.environ,
        "CPU_KUBECONFIG": "/dev/null",
        "NAMESPACE": "synthetic-namespace",
        "TEST_FAIL": failure,
        "TEST_PODS": inventory,
        "TEST_POD": json.dumps(pod),
        "TEST_ENDPOINTS": json.dumps(
            {
                "subsets": [
                    {
                        "notReadyAddresses": [
                            {"targetRef": {"name": "gpu-fault-api-guard-probe-example"}}
                        ]
                    }
                ]
                if endpoint
                else []
            }
        ),
    }
    arguments = ["expected startup refusal", "1"] if script == "assert.sh" else []
    return subprocess.run(
        ["bash", "-c", SHELL, "mocked-boot-guard", str(GUARDS / script), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
        check=False,
    )


@pytest.mark.parametrize("failure", ["delete", "list"])
def test_reset_does_not_turn_a_kubernetes_error_into_absence(failure: str) -> None:
    result = run_guard("reset.sh", failure=failure)

    assert result.returncode != 0, "failed deletion or inventory must fail reset"
    assert "probe reset: 0 pods" not in result.stdout, "unknown is not verified absence"


@pytest.mark.parametrize("failure", ["list", "read", "logs"])
def test_assertion_requires_readable_evidence(failure: str) -> None:
    result = run_guard("assert.sh", failure=failure)

    assert result.returncode != 0, "an unreadable probe cannot prove refusal"
    assert "PASS" not in result.stdout, "a command error is not a guard verdict"


@pytest.mark.parametrize("ready", ["True", "Unknown", None])
def test_assertion_requires_an_explicit_false_ready_condition(
    ready: str | None,
) -> None:
    result = run_guard("assert.sh", ready=ready)

    assert result.returncode != 0, "Ready or missing readiness must not pass"
    assert "PASS" not in result.stdout, "absence of Ready=True alone proves nothing"


def test_assertion_rejects_a_replaced_pod() -> None:
    result = run_guard("assert.sh", changed_uid=True)

    assert result.returncode != 0, "log and readiness must describe the same Pod UID"


def test_assertion_accepts_matching_logs_and_explicit_refusal() -> None:
    result = run_guard("assert.sh")

    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("PASS"), "the observed refusal should pass"


@pytest.mark.parametrize("failure", ["delete", "list", "secret"])
def test_cleanup_failures_propagate(failure: str) -> None:
    result = run_guard("cleanup.sh", failure=failure)

    assert result.returncode != 0, "unverified resource deletion must fail cleanup"


def test_cleanup_checks_not_ready_service_endpoints_too() -> None:
    result = run_guard("cleanup.sh", endpoint=True)

    assert result.returncode != 0, "a not-Ready production endpoint is still exposure"
