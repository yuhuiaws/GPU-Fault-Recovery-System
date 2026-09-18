"""Case-level ordering and recovery routing over the real execution journal."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_resume as recovery
from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.destr008_controller_lock import read_private_document
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_destr_warm import NOW, WarmHarness


def entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[WarmHarness, list[str]]:
    harness = WarmHarness(case, tmp_path, monkeypatch)
    harness.settings = replace(harness.settings, scenarios=case.SCENARIOS)
    monkeypatch.setattr(
        case,
        "require_admission_api",
        lambda _regional: harness.call("admission.discovery"),
    )
    harness.plan(tmp_path)
    executed: list[str] = []

    def run_scenario(_settings: Any, **kwargs: Any) -> dict[str, Any]:
        scenario = kwargs["scenario"]
        assert kwargs["journal"].record.scenarios[scenario].state == "STARTED", (
            "the real journal must record intent before invoking the scenario"
        )
        executed.append(scenario)
        harness.call("scenario.execute", scenario)
        return {"scenario": scenario, "verdict": "PASS", "cleanup_complete": True}

    monkeypatch.setattr(case, "run_scenario", run_scenario)
    return harness, executed


def journal(tmp_path: Path) -> dict[str, Any]:
    return read_private_document(
        tmp_path / "cases" / case.CASE_ID / "execution-owner.json"
    )


def test_complete_matrix_commits_only_after_all_scenarios_and_prewarm_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, executed = entry(tmp_path, monkeypatch)
    code, result = h.execute(tmp_path)
    assert code == 0 and result["verdict"] == "PASS", result
    assert executed == list(case.SCENARIOS), executed
    assert result["complete_matrix"] is True and not result["errors"], result
    record = journal(tmp_path)
    assert record["completed"] and record["prewarm_cleaned"], record
    assert all(row["state"] == "CLEANED" for row in record["scenarios"].values()), (
        record
    )


@pytest.mark.parametrize("cleanup", [False, None])
def test_full_matrix_does_not_pass_without_each_cleanup_receipt(
    cleanup: bool | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, _ = entry(tmp_path, monkeypatch)

    def run_scenario(_settings: Any, **kwargs: Any) -> dict[str, Any]:
        scenario = kwargs["scenario"]
        proof = cleanup if scenario == case.SCENARIOS[-1] else True
        return {"scenario": scenario, "verdict": "PASS", "cleanup_complete": proof}

    monkeypatch.setattr(case, "run_scenario", run_scenario)
    code, result = h.execute(tmp_path)
    assert code == 1 and result["verdict"] == "FAIL", (
        "a local PASS string cannot replace final resource-closure proof",
        result,
    )
    assert journal(tmp_path)["scenarios"][case.SCENARIOS[-1]]["state"] == "STARTED"


def test_existing_run_routes_only_to_cleanup_before_healthy_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, executed = entry(tmp_path, monkeypatch)
    assert h.execute(tmp_path)[0] == 0, "establish an actual completed local journal"
    case_dir = tmp_path / "cases" / case.CASE_ID
    original = (case_dir / f"{case.CASE_ID}.json").read_bytes()
    calls: list[int] = []

    def cleanup(_settings: Any, **kwargs: Any) -> dict[str, Any]:
        value = kwargs["journal"]
        assert value.resuming and value.record.completed, value.record
        calls.append(value.record.attempt)
        return {
            "case_id": case.CASE_ID,
            "verdict": "FAIL",
            "cleanup_only": True,
            "cleanup_complete": True,
            "errors": [],
        }

    monkeypatch.setattr(recovery, "resume_execution", cleanup)
    monkeypatch.setattr(
        case,
        "read_only_preflight",
        lambda *_a, **_k: pytest.fail("cleanup must not enter injection preflight"),
    )
    code = case.execute_case(h.settings, tmp_path, 99, NOW)
    assert code == 1 and calls == [2], calls
    assert executed == list(case.SCENARIOS), "recovery must not repeat any scenario"
    assert (case_dir / f"{case.CASE_ID}.json").read_bytes() == original, (
        "cleanup evidence must not overwrite original execution evidence"
    )
    report = json.loads((case_dir / "cleanup-attempt-99.json").read_text())
    assert report["verdict"] == "FAIL" and report["cleanup_only"], report


@pytest.mark.parametrize("legacy", ["scenarios", "prewarm-owner.json"])
def test_unowned_previous_artifacts_block_new_execution(
    legacy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, executed = entry(tmp_path, monkeypatch)
    path = tmp_path / "cases" / case.CASE_ID / legacy
    if legacy == "scenarios":
        path.mkdir(mode=0o700)
    else:
        path.write_text("{}")
    before = len(h.calls)
    with pytest.raises(RegionalFixtureError, match="no execution owner"):
        h.execute(tmp_path)
    assert not executed and len(h.calls) == before, h.calls
    assert path.exists(), "unrecognized recovery artifacts must be preserved"


def test_shared_administrator_directory_is_rejected_without_changing_its_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, executed = entry(tmp_path, monkeypatch)
    tmp_path.chmod(0o755)
    try:
        with pytest.raises(RegionalFixtureError, match="canonical private"):
            h.execute(tmp_path)
        assert not executed, "an unprivate site cannot authorize a case"
        assert tmp_path.stat().st_mode & 0o777 == 0o755
    finally:
        tmp_path.chmod(0o700)


def test_scenario_exception_keeps_original_intent_and_cleans_prewarm_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, executed = entry(tmp_path, monkeypatch)
    h.failures["scenario.execute"] = RuntimeError("controlled scenario failure")
    code, result = h.execute(tmp_path)
    assert code == 1 and result["verdict"] == "FAIL", result
    assert executed == [case.SCENARIOS[0]], executed
    record = journal(tmp_path)
    assert record["scenarios"][case.SCENARIOS[0]]["state"] == "STARTED", record
    assert record["prewarm_cleaned"] and not record["completed"], record


@pytest.mark.parametrize("failure", ["exception", "residual"])
def test_failed_prewarm_cleanup_prevents_a_full_matrix_pass(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, _ = entry(tmp_path, monkeypatch)
    if failure == "exception":
        h.failures["prewarm.cleanup"] = RuntimeError("controlled cleanup failure")
    else:
        h.prewarm_residuals["pods"] = True
    code, result = h.execute(tmp_path)
    assert code == 1 and result["verdict"] == "FAIL", result
    assert not journal(tmp_path)["completed"], "residual prewarm cannot close the run"


def test_expired_window_never_starts_prewarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, executed = entry(tmp_path, monkeypatch)
    code, result = h.execute(tmp_path, seconds=0)
    assert code == 1 and "before image prewarm" in result["error"], result
    assert executed == [] and not journal(tmp_path)["prewarm_started"], result


def test_cpu_blast_drift_prevents_pass_even_when_each_scenario_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, _ = entry(tmp_path, monkeypatch)
    original = h.regional.cpu_blast_snapshot
    reads = 0

    def snapshot() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return original() if reads == 1 else {"nodes": ["replaced-cpu"]}

    monkeypatch.setattr(h.regional, "cpu_blast_snapshot", snapshot)
    code, result = h.execute(tmp_path)
    assert (
        code == 1
        and "control-plane EKS state differs from baseline" in result["errors"]
    )
