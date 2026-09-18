from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture
from tests.regional._cov95_destr_warm import Clock, WarmHarness, observation


@pytest.mark.parametrize("missing", ["site_file", "manifest"])
def test_missing_inputs_fail_before_any_preflight_io(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(case, tmp_path, monkeypatch)
    getattr(h.settings, missing).unlink()
    with pytest.raises(RegionalFixtureError, match="does not exist"):
        case.read_only_preflight(h.settings, tmp_path)
    assert h.calls == [], "missing local input must not trigger remote inspection"


def test_unknown_scenario_and_failed_capability_are_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(case, tmp_path, monkeypatch)
    h.settings = replace(h.settings, scenarios=("unknown",))
    h.failures["activation.capability"] = RuntimeError("controlled unavailable proof")
    report = case.read_only_preflight(h.settings, tmp_path)
    assert "unknown DESTR-008 scenario selected" in report["errors"], report
    assert any("safeguards unavailable" in error for error in report["errors"]), report


@pytest.mark.parametrize("problem", ["missing", "not-running", "wrong-gpu-count"])
def test_observation_wait_requires_running_and_the_complete_gpu_allocation(
    problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(case, "time", clock)
    value = observation()
    if problem == "not-running":
        value["workload_phase"] = "PENDING"
    elif problem == "wrong-gpu-count":
        value["containers"] = []
    first = [] if problem == "missing" else [value]
    snapshots = iter([{"observations": first}, {"observations": [observation()]}])
    warm = SimpleNamespace(store_snapshot=lambda **_kwargs: next(snapshots))
    result = case.wait_observation(
        cast(WarmSpareLiveFixture, warm), job_id="job", attempt_id="attempt"
    )
    assert result == observation() and clock.elapsed == 5, result


def test_observation_absence_eventually_expires_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(case, "time", clock)
    warm = SimpleNamespace(store_snapshot=lambda **_kwargs: {"observations": []})
    with pytest.raises(RegionalFixtureError, match="did not appear"):
        case.wait_observation(
            cast(WarmSpareLiveFixture, warm),
            job_id="job",
            attempt_id="attempt",
            timeout_seconds=5,
        )
    assert clock.elapsed == 5


def test_replacement_without_a_completion_timestamp_has_no_expiry_proof() -> None:
    errors = case.bound_errors(
        {"status": "FAILED"}, bound_at=Clock().now(), label="controlled fixture"
    )
    assert errors and "no timestamp" in errors[0], errors
