from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional.acceptance_runner_common import (
    processor_queue_backlog,
    replica_vanished,
)
from scripts.e2e.regional.live_driver_guard import (
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence
from tools.run_regional_acceptance import build_local_environment, sanitize_value

CASE_ID = "GF-REGIONAL-TEST-999"
CONFIRMATION = "TEST999_EXECUTE"


@pytest.fixture(autouse=True)
def isolated_scope(monkeypatch: pytest.MonkeyPatch):
    previous_umask = os.umask(0o077)
    monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", "formal")
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE", raising=False)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SITE_PROFILE", raising=False)
    monkeypatch.setattr(live_driver_guard, "install_abort_signals", lambda: None)
    try:
        yield
    finally:
        os.umask(previous_umask)


@pytest.mark.parametrize(
    "queue",
    [
        None,
        {},
        {"depth": None},
        {"depth": -1},
        {"depth": True},
        {"depth": 0.5},
        {"depth": 3, "fault_backlog_depth": None},
        {"depth": 3, "fault_backlog_depth": -1},
    ],
)
def test_unknown_queue_state_never_proves_a_safe_idle_window(queue) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        processor_queue_backlog(queue)


@pytest.mark.parametrize(
    "message",
    [
        "/bin/sh: kubectl: not found",
        'Error from server (NotFound): configmaps "configuration" not found',
        'Error from server (Forbidden): cannot read pods; audit says "not found"',
        "remote Python module not found",
    ],
)
def test_unrelated_read_failures_are_not_vanished_replicas(message: str) -> None:
    assert not replica_vanished(RuntimeError(message)), (
        "an unrelated command or authorization failure is not evidence of a gone Pod"
    )


def predecessor_record(path: Path, **changes: object) -> None:
    path.write_text(
        json.dumps(
            {
                "case_id": CASE_ID,
                "verdict": "PASS",
                "status": "COMPLETED",
                "execution_scope": "formal",
                "formal_sequence_satisfied": True,
                "release_id": "release-test",
                "cluster_id": "cluster-test",
                **changes,
            }
        )
    )


@pytest.mark.parametrize("value", ["false", "true", 1, 0, [], {}])
def test_formal_sequence_flag_must_be_an_actual_boolean(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "predecessor.json"
    predecessor_record(path, formal_sequence_satisfied=value)
    result = predecessor_evidence(
        path, CASE_ID, release_id="release-test", cluster_id="cluster-test"
    )
    assert result["execution_allowed"] is False


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "ABORTED"])
def test_a_retained_pass_verdict_does_not_override_failed_execution(
    tmp_path: Path, status: str
) -> None:
    path = tmp_path / "predecessor.json"
    predecessor_record(path, status=status)
    result = predecessor_evidence(
        path, CASE_ID, release_id="release-test", cluster_id="cluster-test"
    )
    assert result["execution_allowed"] is False


def test_current_completed_predecessor_remains_usable(tmp_path: Path) -> None:
    path = tmp_path / "predecessor.json"
    predecessor_record(path)
    result = predecessor_evidence(
        path, CASE_ID, release_id="release-test", cluster_id="cluster-test"
    )
    assert result["execution_allowed"] is True


def test_an_untracked_source_change_invalidates_focused_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "isolated fixture",
        ],
        check=True,
    )
    source = tmp_path / "src" / "new_helper.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(live_driver_guard, "ROOT", tmp_path)
    before = live_driver_guard.source_digest()
    source.write_text("VALUE = 2\n")
    assert live_driver_guard.source_digest() != before, (
        "untracked code participates in the executed source identity"
    )


@pytest.mark.parametrize("origin", ["arguments", "environment"])
def test_same_path_kubeconfig_replacement_invalidates_approved_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    kubeconfig = tmp_path / "cluster.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\ncurrent-context: cluster-a\n")
    args = argparse.Namespace(
        run_dir=tmp_path,
        attempt=1,
        plan=True,
        execute=False,
        confirm="",
        maintenance_window_end="",
    )
    environment = {}
    if origin == "arguments":
        args.cpu_kubeconfig = str(kubeconfig)
    else:
        environment["CPU_KUBECONFIG"] = str(kubeconfig)
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "same-source")
    live_driver_guard.build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        details={},
        arguments=args,
        preflight_passed=True,
        environment=environment,
    )
    args.execute, args.plan, args.confirm = True, False, CONFIRMATION
    args.maintenance_window_end = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    live_driver_guard.authorize_execution(
        args, case_id=CASE_ID, confirmation=CONFIRMATION, environment=environment
    )
    kubeconfig.write_text("apiVersion: v1\ncurrent-context: cluster-b\n")
    with pytest.raises(RuntimeError, match="connections"):
        live_driver_guard.authorize_execution(
            args, case_id=CASE_ID, confirmation=CONFIRMATION, environment=environment
        )


@dataclass(frozen=True)
class Settings:
    node: str

    def environment(self) -> dict[str, str]:
        return {"SITE": "isolated-test"}


def fixture_case(calls: list[str], *, errors: list[str]) -> CaseRunner[Settings]:
    def parser() -> argparse.ArgumentParser:
        result = argparse.ArgumentParser()
        result.add_argument("--node", default="node-a")
        add_live_arguments(result, confirmation=CONFIRMATION)
        return result

    def execute(
        settings: Settings, _root: Path, _attempt: int, _deadline: datetime
    ) -> int:
        calls.append(settings.node)
        return 0

    return CaseRunner(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser=parser,
        configure=lambda arguments: Settings(arguments.node),
        read_only_preflight=lambda _settings, _directory: {"errors": errors},
        plan_details=lambda settings, preflight: {
            "node": settings.node,
            "preflight": preflight,
        },
        execute_case=execute,
    )


def execute_arguments(root: Path, *, node: str = "node-a") -> list[str]:
    deadline = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    return [
        "runner",
        "--run-dir",
        str(root),
        "--node",
        node,
        "--execute",
        "--confirm",
        CONFIRMATION,
        "--maintenance-window-end",
        deadline,
    ]


def test_failed_plan_cannot_authorize_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    case = fixture_case(calls, errors=["target is not ready"])
    monkeypatch.setattr("sys.argv", ["runner", "--run-dir", str(tmp_path), "--plan"])
    assert run_standard_case(case) == 1
    monkeypatch.setattr("sys.argv", execute_arguments(tmp_path))
    with pytest.raises(RuntimeError, match="preflight"):
        run_standard_case(case)
    assert calls == [], "failed preflight must block the mutating callback"


def test_an_execute_cannot_change_the_planned_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    case = fixture_case(calls, errors=[])
    monkeypatch.setattr("sys.argv", ["runner", "--run-dir", str(tmp_path), "--plan"])
    assert run_standard_case(case) == 0
    monkeypatch.setattr("sys.argv", execute_arguments(tmp_path, node="node-b"))
    with pytest.raises(RuntimeError, match="arguments_sha256"):
        run_standard_case(case)
    assert calls == [], "a confirmation for node-a must not authorize node-b"


def test_structured_reports_redact_values_by_credential_key() -> None:
    digest = "a" * 64
    result = sanitize_value(
        {
            "password": "test-password",
            "nested": {
                "cluster_token": "test-cluster-token",
                "lease_token_sha256": digest,
            },
        }
    )
    assert result == {
        "password": "<redacted>",
        "nested": {"cluster_token": "<redacted>", "lease_token_sha256": digest},
    }


def test_isolated_analysis_cannot_fall_back_to_ambient_cloud_credentials() -> None:
    environment = build_local_environment(
        {"HOME": "/test-home", "PATH": "/usr/bin", "AWS_PROFILE": "external"}
    )
    assert environment.get("AWS_EC2_METADATA_DISABLED") == "true"
    assert environment.get("AWS_CONFIG_FILE") == os.devnull
    assert environment.get("AWS_SHARED_CREDENTIALS_FILE") == os.devnull
    assert environment.get("KUBECONFIG") == os.devnull
