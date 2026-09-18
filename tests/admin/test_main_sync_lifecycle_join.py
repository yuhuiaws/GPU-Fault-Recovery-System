"""Failure reporting stays bounded and uses the supervised join lifecycle."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_state as state_io
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from tests.admin._cov95_join_support import JoinScenario


def test_noted_failure_retains_cause_and_the_last_completed_step() -> None:
    state: dict[str, Any] = {
        "completed_steps": ["JOINED", "PRECHECKED", "DISCOVERED"],
        "step_completed_at": {
            "PRECHECKED": "2026-09-15T05:47:47+00:00",
            "DISCOVERED": "2026-09-15T05:47:54+00:00",
            "JOINED": "2026-09-15T05:54:10+00:00",
        },
    }
    before = datetime.now().astimezone()

    state_io.note_join_failure(
        state, RuntimeError("release component wheels and bundle must exist")
    )

    failure = state["failure"]
    assert failure["error"] == (
        "RuntimeError: release component wheels and bundle must exist"
    ), "the failure record lost the error class or safe cause"
    assert failure["after_step"] == "JOINED", (
        "the last completed step must come from its timestamp, not sorted step names"
    )
    assert before <= datetime.fromisoformat(failure["recorded_at"]), (
        "the failure timestamp must describe this observation"
    )


@pytest.mark.parametrize("kind", ["assignment", "url", "json", "private-key", "long"])
def test_failure_diagnostics_sanitize_the_whole_message_before_prefixing(
    kind: str,
) -> None:
    marker = "example-sensitive-value"
    private_key_label = "PRIVATE KEY"
    messages = {
        "assignment": "candidate unavailable; token=" + marker,
        "url": "cannot connect to postgresql://operator:" + marker + "@example.invalid",
        "json": json.dumps({"kind": "Secret", "data": {"payload": marker}}),
        "private-key": (
            f"-----BEGIN {private_key_label}-----\n"
            f"{marker}\n-----END {private_key_label}-----"
        ),
        "long": "x" * 9000 + " token=" + marker,
    }
    state: dict[str, Any] = {}

    state_io.note_join_failure(state, RuntimeError(messages[kind]))

    failure = state["failure"]
    assert failure["error"] == "RuntimeError: " + diagnostic_text(messages[kind]), (
        "join diagnostics must use the existing whole-document sanitizer"
    )
    assert marker not in failure["error"], "failure diagnostics exposed a credential"
    assert len(failure["error"]) < 4200, "failure diagnostics are not bounded"
    assert failure["after_step"] is None, "no step evidence must remain unknown"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"JOINED": None},
        {"JOINED": "not-a-timestamp"},
        {"JOINED": "2026-09-15T05:54:10"},
    ],
)
def test_incomplete_step_timestamps_do_not_mask_the_original_failure(
    metadata: object,
) -> None:
    state: dict[str, Any] = {
        "completed_steps": ["JOINED"],
        "step_completed_at": metadata,
    }

    state_io.note_join_failure(state, RuntimeError("verification refused"))

    assert state["failure"]["error"] == "RuntimeError: verification refused", (
        "diagnostic metadata must not replace the original failure"
    )
    assert state["failure"]["after_step"] is None, (
        "invalid timestamps cannot establish the last completed step"
    )


@pytest.mark.parametrize("rollback", [False, True])
def test_public_join_persists_sanitized_failure_through_compensation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback: bool
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch, rollback=rollback)
    scenario.failure = "discover:gpu-b"
    error = BootstrapError("GPU discovery refused; token=example-sensitive-value")
    scenario.discovery_error = error
    scenario.failure_error = error

    with pytest.raises(BootstrapError, match="GPU discovery refused"):
        scenario.join()

    state = scenario.state()
    assert state["phase"] == ("ROLLED_BACK" if rollback else "FAILED"), (
        "failure reporting changed the configured compensation policy"
    )
    assert state["failure"]["error"] == (
        "BootstrapError: " + diagnostic_text(str(error))
    ), "compensation must preserve the sanitized original failure"
    assert state["failure"]["after_step"] == "PRECHECKED", (
        "discovery failed after the local precheck but before target preparation"
    )
    assert scenario.state_path().stat().st_mode & 0o077 == 0, (
        "failure diagnostics lost the state file's private permissions"
    )

    if rollback:
        original_failure = state["failure"]
        scenario.failure = None
        scenario.discovery_error = None
        assert scenario.join()["phase"] == "COMPLETED", (
            "a clean rollback must still permit a fresh attempt"
        )
        current = scenario.state()
        archive = scenario.state_path().parent / "state.attempt-001.json"
        archived = json.loads(archive.read_text(encoding="utf-8"))
        assert archived["failure"] == original_failure, (
            "the archived attempt lost its original diagnostic"
        )
        assert "failure" not in current, "a new attempt retained the old failure"


def test_failed_supervised_verification_becomes_a_reportable_join_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch)
    commands: list[str] = []

    def driver(arguments: list[str], **options: Any) -> Any:
        commands.append(arguments[1])
        if arguments[1] == "verify":
            return scenario.result(
                arguments,
                code=2,
                error="candidate release invalid; token=example-sensitive-value",
            )
        return scenario.driver(arguments, **options)

    monkeypatch.setattr(join, "run_driver", driver)
    with pytest.raises(BootstrapError, match="regional verify failed"):
        scenario.join()

    state = scenario.state()
    assert commands == ["preflight", "join-cluster", "verify"], (
        "verification failure must stop before membership activation"
    )
    assert state["failure"]["after_step"] == "COLLECTORS_READY", (
        "a failed candidate verify must retain its last successful barrier"
    )
    assert state["failure"]["error"] == "BootstrapError: regional verify failed", (
        "the supervised driver's raw stderr must not become state diagnostics"
    )
    assert scenario.cluster_states["hp-gpu-b"] == "PENDING", (
        "an unloadable candidate was activated"
    )


def test_failure_after_activation_intent_preserves_the_fail_forward_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch, rollback=True)
    scenario.failure = "activate-cluster:hp-gpu-b"
    scenario.failure_error = BootstrapError(
        "activation unavailable; token=example-sensitive-value"
    )

    with pytest.raises(BootstrapError, match="activation unavailable"):
        scenario.join()

    state = scenario.state()
    assert state["phase"] == "FAILED_AFTER_ACTIVATION", (
        "failure diagnostics must preserve the irreversible activation boundary"
    )
    assert state["failure"]["after_step"] == "ACTIVATION_STARTED", (
        "the diagnostic lost the persisted activation intent"
    )
    assert state["failure"]["error"] == (
        "BootstrapError: " + diagnostic_text(str(scenario.failure_error))
    ), "the post-activation cause must also be sanitized"
    assert not any(
        mode in {"fail-cluster", "rollback-cluster"}
        for mode, _cluster in scenario.driver_calls
    ), "failure reporting must not compensate after activation starts"
