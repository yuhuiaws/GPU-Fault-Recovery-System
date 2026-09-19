from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "deploy/control-plane/tools/wait-for-kubernetes-job.sh"
)
FAKE_KUBECTL = """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"$CALLS"
if [[ "$*" == *"get ${FAIL_ON:-never}"* && (-n "${FAIL_ALWAYS:-}" || ! -f "$HOME/failed-once") ]]; then
    touch "$HOME/failed-once"
    printf '%s\\n' "$API_ERROR" >&2
    exit 1
fi
case "$*" in
  *"get job/"*) cat "$JOB_RESPONSE" ;;
  *"get pods"*) cat "$POD_RESPONSE" ;;
  *" wait "*) sleep "${WATCH_DELAY:-0}"; cp "$NEXT_RESPONSE" "$JOB_RESPONSE"; cat "$JOB_RESPONSE" ;;
  *) exit 99 ;;
esac
"""


def run_wait(
    tmp_path: Path,
    first: dict,
    *,
    after: dict | None = None,
    pods: dict | None = None,
    seconds: int = 3600,
    error: str = "",
    fail_on: str = "job/",
    always: bool = False,
    watch_delay: str = "0",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    binary = tmp_path / "fake-kubernetes-api"
    binary.write_text(FAKE_KUBECTL, encoding="utf-8")
    binary.chmod(0o755)
    files = {
        "JOB_RESPONSE": tmp_path / "job.json",
        "NEXT_RESPONSE": tmp_path / "next.json",
        "POD_RESPONSE": tmp_path / "pods.json",
    }
    for name, value in (
        ("JOB_RESPONSE", first),
        ("NEXT_RESPONSE", after or first),
        ("POD_RESPONSE", pods or {"items": []}),
    ):
        files[name].write_text(json.dumps(value), encoding="utf-8")
    calls = tmp_path / "calls"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(seconds),
            "schema",
            str(binary),
            "--kubeconfig",
            str(tmp_path / "cpu"),
            "-n",
            "system",
        ],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            **{key: str(path) for key, path in files.items()},
            "CALLS": str(calls),
            "FAIL_ON": fail_on if error else "never",
            "API_ERROR": error,
            "FAIL_ALWAYS": "yes" if always else "",
            "WATCH_DELAY": watch_delay,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result, calls.read_text(
        encoding="utf-8"
    ).splitlines() if calls.exists() else []


def job(condition: str | None = None, *, uid: str = "job-uid") -> dict:
    return {
        "metadata": {"uid": uid},
        "status": {
            "conditions": (
                [
                    {
                        "type": condition,
                        "status": "True",
                        "reason": "BackoffLimitExceeded",
                    }
                ]
                if condition
                else []
            )
        },
    }


@pytest.mark.parametrize("condition", ["Failed", "FailureTarget"])
def test_failed_job_does_not_spend_the_hour_timeout(
    tmp_path: Path, condition: str
) -> None:
    result, calls = run_wait(tmp_path, job(condition))

    assert result.returncode == 1, result.stderr
    assert "BackoffLimitExceeded" in result.stderr
    assert len(calls) == 1, "failed Job entered a watch or a second poll"


def test_completed_job_returns_without_wait(tmp_path: Path) -> None:
    result, calls = run_wait(tmp_path, job("Complete"))

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1


def test_wait_wakes_and_rechecks_the_completed_job(tmp_path: Path) -> None:
    result, calls = run_wait(tmp_path, job(), after=job("Complete"))

    assert result.returncode == 0, result.stderr
    assert sum("get job/" in call for call in calls) == 1
    assert sum(" wait " in call for call in calls) == 1


def test_job_replacement_is_not_accepted_as_completion(tmp_path: Path) -> None:
    result, _calls = run_wait(tmp_path, job(), after=job("Complete", uid="replacement"))

    assert result.returncode == 1
    assert "replaced" in result.stderr


def test_deterministic_pod_configuration_failure_is_reported(tmp_path: Path) -> None:
    result, calls = run_wait(
        tmp_path,
        job(),
        pods={
            "items": [
                {
                    "metadata": {"ownerReferences": [{"uid": "job-uid"}]},
                    "status": {
                        "initContainerStatuses": [
                            {
                                "state": {
                                    "waiting": {"reason": "CreateContainerConfigError"}
                                }
                            }
                        ]
                    },
                }
            ]
        },
    )

    assert result.returncode == 1
    assert "CreateContainerConfigError" in result.stderr
    assert not any(" wait " in call for call in calls), (
        "a fatal container configuration error entered the long Job wait"
    )


@pytest.mark.parametrize("seconds", [0, -1])
def test_invalid_timeout_never_starts_kubectl(tmp_path: Path, seconds: int) -> None:
    result, calls = run_wait(tmp_path, job("Complete"), seconds=seconds)

    assert result.returncode == 2
    assert calls == []


@pytest.mark.parametrize("fail_on", ["job/", "pods"])
def test_transient_get_failure_is_retried_within_the_shared_deadline(
    tmp_path: Path, fail_on: str
) -> None:
    result, calls = run_wait(
        tmp_path,
        job(),
        after=job("Complete"),
        seconds=4,
        error="Error from server (ServiceUnavailable)",
        fail_on=fail_on,
    )
    assert result.returncode == 0, result.stderr
    assert sum(f"get {fail_on}" in call for call in calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        "Forbidden",
        "Unauthorized",
        "NotFound",
        "x509: unknown authority",
        "unexpected response",
    ],
)
def test_permanent_api_errors_do_not_enter_a_long_retry(
    tmp_path: Path, error: str
) -> None:
    result, calls = run_wait(tmp_path, job(), error=error)
    assert result.returncode == 1
    assert len(calls) == 1
    assert "API failure" in result.stderr


def test_persistent_transient_failure_exhausts_the_total_timeout(
    tmp_path: Path,
) -> None:
    result, calls = run_wait(
        tmp_path, job(), seconds=2, error="connection reset", always=True
    )
    assert result.returncode == 1
    assert "within 2s" in result.stderr
    assert len(calls) <= 2


def test_successful_watch_supplies_completion_without_a_late_get(
    tmp_path: Path,
) -> None:
    # Bash ``SECONDS`` has whole-second granularity, so a one-second budget can
    # expire almost at once and the 0.9 s fake watch then loses to ``timeout``
    # on a loaded test host (deploy gate #21, 2026-09-19). Three seconds keep
    # the watch inside the deadline while the assertion stays the same.
    result, calls = run_wait(
        tmp_path, job(), after=job("Complete"), seconds=3, watch_delay="0.9"
    )
    assert result.returncode == 0, result.stderr
    assert sum("get job/" in call for call in calls) == 1
