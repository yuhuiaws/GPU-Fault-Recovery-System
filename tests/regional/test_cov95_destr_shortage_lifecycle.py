from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from tests.regional._cov95_destr_warm import NOW, WarmHarness, settings_for
from tests.regional._destr008_scenario import bind_fixture


class Resource:
    def __init__(self, label: str, calls: list[Any], *, bound: bool) -> None:
        self.label = label
        self.calls = calls
        self.name = "fake-holder"
        self.deadline_at: Any = None
        self.failsafe_at: Any = None
        self.bound = bound
        self.failure = ""
        self.residual = False

    def create(self) -> None:
        self.calls.append(f"{self.label}.create")
        if self.label == "holder" and self.bound:
            self.deadline_at = NOW + timedelta(seconds=840)

    def stop(self, service: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("service.stop", service, kwargs))
        if self.bound:
            self.failsafe_at = NOW + timedelta(
                seconds=kwargs["restore_seconds"] + kwargs["delay_seconds"]
            )
        return {"scheduled": bool(kwargs["delay_seconds"])}

    def restore(self) -> dict[str, Any]:
        self.calls.append("service.restore")
        if self.failure == "restore":
            raise RuntimeError("fake service restore failure")
        return {"restored": True}

    def cleanup(self) -> Any:
        self.calls.append(f"{self.label}.cleanup")
        if self.failure == "cleanup":
            raise RuntimeError(f"fake {self.label} cleanup failure")
        return self.residual if self.label == "holder" else {"pod": self.residual}


def fixture(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bound: bool = True,
) -> tuple[Any, Resource, list[Any]]:
    calls: list[Any] = []
    label = "holder" if scenario == "active-gpu-pod" else "service"
    resource = Resource(label, calls, bound=bound)
    monkeypatch.setattr(case, "GpuHolderFixture", lambda *_a, **_k: resource)
    monkeypatch.setattr(case, "WarmSpareServiceFixture", lambda *_a, **_k: resource)
    warm = SimpleNamespace(
        wait_node_ready=lambda *args, **kwargs: calls.append(("node.ready", kwargs)),
        wait_fleet_readiness=lambda *args, **kwargs: calls.append(
            ("fleet.ready", kwargs)
        ),
    )
    result = case.ScenarioFixture(
        settings_for(case, tmp_path),
        warm,
        scenario=scenario,
        run_id="owned",
        state_directory=tmp_path / "probe-state",
    )
    bind_fixture(result)
    return result, resource, calls


@pytest.mark.parametrize(
    "scenario", ["active-gpu-pod", "kubernetes-not-ready", "agent-unavailable"]
)
@pytest.mark.parametrize("bound", [False, True])
def test_unsupported_shortage_fixture_lifecycle_records_bounds_without_admitting_case(
    scenario: str, bound: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert case.scenario_admission_errors([scenario]), scenario
    scenario_fixture, resource, calls = fixture(
        scenario, tmp_path, monkeypatch, bound=bound
    )
    early = scenario_fixture.apply()
    assert scenario_fixture.bound_at is None, early
    late = scenario_fixture.apply_late()
    assert (scenario_fixture.bound_at is not None) is bound, late
    restored = scenario_fixture.restore()
    assert restored["errors"] == [], restored
    assert f"{resource.label}.cleanup" in calls, calls
    if scenario == "kubernetes-not-ready":
        waits = [
            entry[1]["ready"]
            for entry in calls
            if isinstance(entry, tuple) and entry[0] == "node.ready"
        ]
        assert waits == [False, True], calls
        assert calls.index(
            ("node.ready", {"ready": True, "timeout_seconds": 600})
        ) < calls.index("service.restore"), calls
    elif scenario == "agent-unavailable":
        waits = [
            entry[1]["ready"]
            for entry in calls
            if isinstance(entry, tuple) and entry[0] == "fleet.ready"
        ]
        assert waits == [False, True], calls
    else:
        assert early["armed"] == "late" and late["holder_pod"] == resource.name, (
            early,
            late,
        )
    assert case.scenario_admission_errors([scenario]), (
        "unit fixture must not alter admission"
    )


@pytest.mark.parametrize(
    ("scenario", "failure", "residual", "expected"),
    [
        ("active-gpu-pod", "cleanup", False, "holder cleanup"),
        ("active-gpu-pod", "", True, "holder Pod remains"),
        ("agent-unavailable", "restore", False, "service restore"),
        ("agent-unavailable", "cleanup", False, "probe cleanup"),
        ("agent-unavailable", "", True, "probe resources remain"),
    ],
)
def test_shortage_cleanup_records_failures_and_retains_refusal(
    scenario: str,
    failure: str,
    residual: bool,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_fixture, resource, calls = fixture(scenario, tmp_path, monkeypatch)
    scenario_fixture.apply()
    scenario_fixture.apply_late()
    resource.failure = failure
    resource.residual = residual
    report = scenario_fixture.restore()
    assert any(expected in error for error in report["errors"]), report
    assert f"{resource.label}.cleanup" in calls, calls
    assert case.scenario_admission_errors([scenario]), scenario


def test_unknown_shortage_scenario_never_creates_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_fixture, _, calls = fixture("unknown", tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unknown scenario"):
        scenario_fixture.apply()
    assert calls == [], calls
    assert scenario_fixture.restore() == {"errors": []}, scenario_fixture.scenario


def test_shortage_fault_restore_failed_workflow_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(case, tmp_path, monkeypatch)
    h.restore_status = "FAILED"
    h.nodes["node-a"]["annotations"]["gpu-fault.io/incident-id"] = "owned"
    result = case.restore_fault_node(
        h.warm, settings=h.settings, incident_id="owned", profile_version="profile-v1"
    )
    assert "fault-node restore workflow failed" in result["errors"], result


@pytest.mark.parametrize("phase", ["shortage.apply", "workload.running"])
def test_safe_shortage_rechecks_window_before_next_mutation(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(case, tmp_path, monkeypatch)
    h.advance_at[phase] = 61
    result = case.run_scenario(
        h.settings,
        warm=h.warm,
        regional=h.regional,
        case_dir=tmp_path,
        run_dir=tmp_path,
        attempt=1,
        scenario="no-spare",
        maintenance_window_end=NOW + timedelta(seconds=60),
        provider_baseline=h.provider,
        profile_version="profile-v1",
    )
    assert result["verdict"] == "FAIL" and "window ended" in result["error"], result
    assert not h.posted, h.calls
