from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.live_driver_guard import (
    add_live_arguments,
    authorize_execution,
    build_plan,
    details_sha256,
    record_focused_tests,
    reusable_focused_tests,
    source_digest,
)

CASE_ID = "GF-REGIONAL-TEST-001"
CONFIRMATION = "TEST_LIVE_CONFIRMATION"
ENVIRONMENT = {
    "CPU_KUBECONFIG": "/secure/cpu.kubeconfig",
    "GPU_KUBECONFIG": "/secure/gpu.kubeconfig",
    "GPU_EKS_CONTEXT": "gpu-context",
}


def _arguments(
    run_dir: Path, *, confirm: str = CONFIRMATION, deadline: datetime | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        execute=True,
        confirm=confirm,
        maintenance_window_end=(
            deadline or datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        run_dir=run_dir,
        attempt=1,
    )


def test_live_driver_defaults_to_non_execute_mode(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)

    arguments = parser.parse_args(["--run-dir", str(tmp_path)])

    assert arguments.execute is False
    assert arguments.plan is False


def test_live_driver_authorizes_matching_plan(tmp_path: Path) -> None:
    plan = build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={"mutation": "test-only"},
    )

    assert plan["schema_version"] == 2, plan
    assert plan["execution_scope"] == "formal", plan
    assert plan["selection_reference"] is None, plan
    deadline = authorize_execution(
        _arguments(tmp_path),
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
    )

    assert deadline > datetime.now(timezone.utc)


def test_live_driver_rejects_confirmation_and_environment_drift(tmp_path: Path) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )

    with pytest.raises(RuntimeError, match="confirmation"):
        authorize_execution(
            _arguments(tmp_path, confirm="wrong"),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )
    with pytest.raises(RuntimeError, match="environment"):
        authorize_execution(
            _arguments(tmp_path),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment={**ENVIRONMENT, "GPU_EKS_CONTEXT": "drifted"},
        )


def test_selective_scope_requires_an_audit_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.delenv(SELECTION_REFERENCE_ENV, raising=False)

    with pytest.raises(RuntimeError, match=SELECTION_REFERENCE_ENV):
        build_plan(
            run_dir=tmp_path,
            case_id=CASE_ID,
            attempt=1,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
            details={},
        )


def test_selective_scope_is_bound_to_plan_and_case_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-SELECTIVE-001")
    plan = build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )

    assert plan["execution_scope"] == "selective", plan
    assert plan["selection_reference"] == "CHG-SELECTIVE-001", plan
    authorize_execution(
        _arguments(tmp_path),
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
    )
    report_path = tmp_path / "cases" / CASE_ID / f"{CASE_ID}.json"
    write_json_atomic(report_path, {"case_id": CASE_ID, "verdict": "PASS"})
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["execution_scope"] == "selective", report
    assert report["selection_reference"] == "CHG-SELECTIVE-001", report
    assert report["formal_sequence_satisfied"] is False, report


def test_execute_rejects_execution_scope_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-SELECTIVE-002")

    with pytest.raises(RuntimeError, match="execution_scope"):
        authorize_execution(
            _arguments(tmp_path),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )


def test_live_driver_rejects_expired_window(tmp_path: Path) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )

    with pytest.raises(RuntimeError, match="window has ended"):
        authorize_execution(
            _arguments(
                tmp_path, deadline=datetime.now(timezone.utc) - timedelta(seconds=1)
            ),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )


def test_plan_records_a_details_digest_and_refuses_edited_details(
    tmp_path: Path,
) -> None:
    """The target node and the drill marker live in ``details``; a plan whose
    details were edited after approval is not the approved plan."""

    plan = build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={"node": "node-a", "marker": "m-1"},
    )

    assert plan["details_sha256"] == details_sha256(plan["details"])
    path = tmp_path / "cases" / CASE_ID / "plan.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["details"]["node"] = "node-b"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="drifted at details_sha256"):
        authorize_execution(
            _arguments(tmp_path),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )


def test_authorize_compares_the_callers_current_details_when_given(
    tmp_path: Path,
) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={"node": "node-a", "marker": "m-1"},
    )

    # Same content in another key order is the same plan.
    authorize_execution(
        _arguments(tmp_path),
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={"marker": "m-1", "node": "node-a"},
    )
    with pytest.raises(RuntimeError, match="drifted at details"):
        authorize_execution(
            _arguments(tmp_path),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
            details={"node": "node-b", "marker": "m-1"},
        )


def test_a_plan_without_a_details_digest_is_refused(tmp_path: Path) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )
    path = tmp_path / "cases" / CASE_ID / "plan.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["details_sha256"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="drifted at details_sha256"):
        authorize_execution(
            _arguments(tmp_path),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )


def test_source_digest_is_a_stable_sha256_over_the_checked_trees() -> None:
    first = source_digest()

    assert len(first) == 64 and int(first, 16) >= 0
    assert source_digest() == first, "two reads of an unchanged tree must agree"


def test_focused_tests_are_reused_only_for_the_same_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "digest-a")
    details: dict[str, Any] = {"node": "node-a"}
    record_focused_tests(details, {"passed": True, "returncode": 0})
    assert details["focused_tests_source_digest"] == "digest-a", details
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details=details,
    )
    plan_path = tmp_path / "cases" / CASE_ID / "plan.json"

    assert reusable_focused_tests(plan_path) == {"passed": True, "returncode": 0}

    # Any edit to the drivers, their tests or the package changes the digest,
    # and the recorded pass no longer speaks for the code about to run.
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "digest-b")
    assert reusable_focused_tests(plan_path) is None


def test_a_failed_or_absent_focused_result_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "digest-a")
    details: dict[str, Any] = {}
    record_focused_tests(details, {"passed": False, "returncode": 1})
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details=details,
    )
    plan_path = tmp_path / "cases" / CASE_ID / "plan.json"

    assert reusable_focused_tests(plan_path) is None
    assert reusable_focused_tests(tmp_path / "absent.json") is None
