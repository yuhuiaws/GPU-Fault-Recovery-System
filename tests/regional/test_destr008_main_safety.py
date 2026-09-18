from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.destr008_journal import (
    ExecutionJournal,
    ExecutionRecord,
    ScenarioState,
)
from tests.regional._destr008_main import MainHarness

BOUNDED = ["kubernetes-not-ready", "agent-unavailable", "active-gpu-pod"]


@pytest.mark.parametrize("scenario", BOUNDED)
def test_main_bounded_case_requires_real_safety_before_fixture_and_post(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, scenario)
    report = h.run()
    assert report["verdict"] == "PASS", report
    assert report["cleanup_complete"] is True, report
    receipt = report["independent_safety_cleanup"]["receipt"]
    assert receipt["workflow_ids"] == ["workflow-shortage"], receipt
    assert receipt["source_complete"] and receipt["producer_revoked"], receipt
    names = [name for name, _ in h.warm.calls]
    assert names.index("workload.running") < names.index("replacement.post"), names
    fixture_create = (
        "holder.create" if scenario == "active-gpu-pod" else "service.create"
    )
    assert names.index("workload.running") < names.index(fixture_create), names
    assert names.count("replacement.post") == 1, names
    assert not h.causal.gpu.objects and h.causal.cpu.journal()["closed"]


@pytest.mark.parametrize("scenario", BOUNDED)
def test_main_unknown_post_retains_independent_inhibition_and_never_restores(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, scenario)
    h.warm.failures["replacement.post"] = TimeoutError("lost controlled response")
    report = h.run()
    assert report["verdict"] == "FAIL" and report["cleanup_deferred"], report
    assert report["independent_safety_cleanup"]["quiescent"] is False, report
    assert h.causal.gpu.objects, "an unresolved POST must retain the fence"
    names = [name for name, _ in h.warm.calls]
    assert names.count("replacement.post") == 1, names
    assert not set(names) & {
        "workload.delete",
        "service.restore",
        "service.cleanup",
        "holder.cleanup",
        "spares.release",
        "restore.create",
    }, names


@pytest.mark.parametrize("scenario", BOUNDED)
def test_profile_drift_is_rejected_before_a_bounded_fixture_is_created(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, scenario)
    h.warm.observations[0]["runtime_profile_version"] = "stale-profile"
    report = h.run()
    assert report["verdict"] == "FAIL" and "profile differs" in report["error"], report
    names = {name for name, _ in h.warm.calls}
    assert not names & {"holder.create", "service.create", "replacement.post"}, names
    assert not h.causal.gpu.objects, "profile drift must stop before fence creation"


def test_bounded_main_persists_each_stage_before_its_real_safety_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, "active-gpu-pod")

    def binding() -> dict[str, Any]:
        return case.execution_binding(h.settings, h.causal.cpu.regional, tmp_path)

    journal = ExecutionJournal(
        tmp_path / "execution-owner.json",
        binding,
        initial=ExecutionRecord(
            schema_version=1,
            binding=binding(),
            attempt=1,
            maintenance_expires_at=int(h.at(3600).timestamp()),
            release_id="release-a",
            profile_version="profile-v1",
            fault_uid="uid-node-a",
            spare_uid="uid-node-b",
            scenarios={"active-gpu-pod": ScenarioState()},
        ),
    )
    journal.start_prewarm()
    journal.start_scenario("active-gpu-pod")
    report = h.run(journal=journal)
    assert report["verdict"] == "PASS" and report["cleanup_complete"], report
    record = journal.record.scenarios["active-gpu-pod"]
    assert all(
        getattr(record, field)
        for field in (
            "workload_started",
            "safety_started",
            "fixture_started",
            "post_started",
        )
    ), record


@pytest.mark.parametrize("change", ["inventory", "provider-event"])
def test_provider_drift_is_not_a_successful_shortage(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, "active-gpu-pod")
    if change == "inventory":
        monkeypatch.setattr(
            h.warm.warm, "provider_inventory", lambda: {"sha256": "changed"}
        )
    else:
        h.warm.events = [{"event_name": "BatchReplaceClusterNodes"}]
    report = h.run()
    assert report["verdict"] == "FAIL" and report["cleanup_complete"], report
    assert any("provider" in error.lower() for error in report["errors"]), report
