"""Deferred batch failures retain their explicit, sanitized original cause."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from tests.admin._cov95_join_support import JoinScenario


@pytest.mark.parametrize("boundary", ["executor-role:gpu-b", "join-cluster:hp-gpu-b"])
@pytest.mark.parametrize("ambient", [False, True])
def test_deferred_batch_failure_records_its_own_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, ambient: bool
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch)
    marker = "example-sensitive-value"
    error = BootstrapError(
        json.dumps(
            {
                "message": "deferred worker failure",
                "kind": "Secret",
                "data": {"payload": marker},
            }
        )
    )
    scenario.failure = boundary
    scenario.failure_error = error
    unrelated = LookupError("unrelated ambient failure") if ambient else None

    def run_batch() -> None:
        assert sys.exception() is unrelated, (
            "the regression must exercise the requested ambient exception context"
        )
        with pytest.raises(BootstrapError, match="batch join completed") as failure:
            scenario.batch()
        assert marker not in str(failure.value), (
            "the batch summary exposed the worker's credential"
        )

    if unrelated is None:
        run_batch()
    else:
        try:
            raise unrelated
        except LookupError:
            run_batch()

    state = scenario.state()
    assert state["phase"] == "FAILED", "an ordinary worker failure changed lifecycle"
    assert state["failure"]["error"] == (
        "BootstrapError: " + diagnostic_text(str(error))
    ), "deferred handling lost its cause or substituted an ambient exception"
    assert marker not in state["failure"]["error"], (
        "the whole error message must be redacted before adding its exception class"
    )
    assert state["failure"]["after_step"] in state["completed_steps"], (
        "the diagnostic must name a barrier this failed attempt actually completed"
    )
    if boundary == "join-cluster:hp-gpu-b":
        assert state["failure"]["after_step"] == "JOIN_STARTED", (
            "a failed rollout must retain its last persisted start barrier"
        )
    assert scenario.state("c")["phase"] == "COMPLETED", (
        "recording a worker failure must preserve the successful sibling"
    )
    assert "failure" not in scenario.state("c"), (
        "the failed worker's diagnostic leaked into its successful sibling"
    )


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_deferred_batch_failure_survives_rollback_and_cleanup_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_failure: bool
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",), rollback=True)
    error = BootstrapError("kubeconfig preparation refused; token=example-sensitive")
    cleanup_error = BootstrapError("cleanup unavailable; password=example-sensitive")
    scenario.failure = "kubeconfig"
    scenario.failure_error = error
    cleanup_blocked = cleanup_failure

    def command(
        arguments: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        if cleanup_blocked and "config" in arguments and "get-contexts" in arguments:
            raise cleanup_error
        return scenario.command(arguments, **options)

    monkeypatch.setattr(join, "run_command", command)
    with pytest.raises(BootstrapError, match="batch join completed"):
        scenario.batch(("b",))

    state = scenario.state()
    original = state["failure"]
    assert original["error"] == "BootstrapError: " + diagnostic_text(str(error)), (
        "rollback replaced the deferred worker's original cause"
    )
    assert original["after_step"] == "LOCAL_INPUTS_STARTED", (
        "the diagnostic must retain the barrier preceding kubeconfig preparation"
    )
    assert state["phase"] == (
        "ROLLBACK_FAILED" if cleanup_failure else "ROLLED_BACK"
    ), "failure recording changed compensation completion semantics"

    scenario.failure = None
    if cleanup_failure:
        assert state["rollback_errors"] == [diagnostic_text(str(cleanup_error))], (
            "the cleanup failure must be recorded separately and sanitized"
        )
        with pytest.raises(BootstrapError, match="batch join completed"):
            scenario.batch(("b",))
        assert scenario.state()["failure"] == original, (
            "a repeated rollback error overwrote the original cause or timestamp"
        )
        assert scenario.state()["phase"] == "ROLLBACK_FAILED", (
            "a failed cleanup retry was declared terminal"
        )

    cleanup_blocked = False
    assert scenario.batch(("b",))["phase"] == "COMPLETED", (
        "a completed rollback must still allow a fresh batch attempt"
    )
    archive = scenario.state_path().parent / "state.attempt-001.json"
    archived = json.loads(archive.read_text(encoding="utf-8"))
    assert archived["failure"] == original, (
        "a fresh attempt discarded the archived original failure"
    )
    assert "failure" not in scenario.state(), "a fresh attempt retained the old cause"


def test_batch_activation_failure_keeps_the_irreversible_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",), rollback=True)
    scenario.failure = "activate-cluster:hp-gpu-b"
    error = BootstrapError("activation unavailable; token=example-sensitive")
    scenario.failure_error = error

    with pytest.raises(BootstrapError, match="batch join completed"):
        scenario.batch(("b",))

    state = scenario.state()
    assert state["phase"] == "FAILED_AFTER_ACTIVATION", (
        "explicit failure reporting crossed the irreversible activation boundary"
    )
    assert state["failure"]["error"] == "BootstrapError: " + diagnostic_text(
        str(error)
    ), "the activation failure lost its sanitized original cause"
    assert state["failure"]["after_step"] == "ACTIVATION_STARTED", (
        "the activation intent must remain the last completed barrier"
    )
    assert not any(
        mode in {"fail-cluster", "rollback-cluster"}
        for mode, _cluster in scenario.driver_calls
    ), "recording a failure must not compensate after activation starts"
