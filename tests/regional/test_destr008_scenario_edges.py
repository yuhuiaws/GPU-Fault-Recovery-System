from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture
from tests.regional._cov95_destr_warm import settings_for
from tests.regional._destr008_scenario import bind_fixture


class FixturePort:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.residual = False

    def create(self) -> None:
        pytest.fail("cleanup-only recovery must not create a fixture")

    def stop(self, *args: Any, **kwargs: Any) -> None:
        pytest.fail("cleanup-only recovery must not stop a service")

    def cleanup(self) -> bool:
        self.calls.append("holder.cleanup")
        return self.residual

    def resume_cleanup(self) -> dict[str, Any]:
        self.calls.append("service.resume")
        return {"phase": "CLOSED"}

    def close(self) -> None:
        self.calls.append("service.close")


def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> tuple[case.ScenarioFixture, FixturePort, list[tuple[str, Any]]]:
    port = FixturePort()
    waits: list[tuple[str, Any]] = []
    warm = SimpleNamespace(
        wait_node_ready=lambda *args, **kwargs: waits.append(("node", kwargs)),
        wait_fleet_readiness=lambda *args, **kwargs: waits.append(("agent", kwargs)),
    )
    monkeypatch.setattr(case, "GpuHolderFixture", lambda *_a, **_k: port)
    monkeypatch.setattr(case, "WarmSpareServiceFixture", lambda *_a, **_k: port)
    fixture = case.ScenarioFixture(
        settings_for(case, tmp_path),
        cast(WarmSpareLiveFixture, warm),
        scenario=scenario,
        run_id="scenario-run",
        state_directory=tmp_path / "state",
    )
    return fixture, port, waits


@pytest.mark.parametrize(
    "scenario", ["active-gpu-pod", "kubernetes-not-ready", "agent-unavailable"]
)
def test_scenario_resumption_uses_only_bound_original_cleanup(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, port, waits = setup(tmp_path, monkeypatch, scenario)
    bind_fixture(fixture)
    result = fixture.resume_cleanup()
    fixture.close()
    assert result["errors"] == [], result
    if scenario == "active-gpu-pod":
        assert port.calls == ["holder.cleanup"] and not waits, (port.calls, waits)
    else:
        assert port.calls == ["service.resume", "service.close"], port.calls
        assert waits[-1][0] == "agent" and waits[-1][1]["ready"], waits
        if scenario == "kubernetes-not-ready":
            assert waits[0][0] == "node" and waits[0][1]["ready"], waits


def test_holder_resumption_never_treats_a_remaining_pod_as_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, port, _ = setup(tmp_path, monkeypatch, "active-gpu-pod")
    bind_fixture(fixture)
    port.residual = True
    assert fixture.resume_cleanup()["errors"] == ["GPU holder remains"]


@pytest.mark.parametrize(
    "operation", ["bound_inputs", "service_fixture", "resume_cleanup"]
)
def test_missing_original_plan_cannot_authorize_fixture_recovery(
    operation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, port, _ = setup(tmp_path, monkeypatch, "agent-unavailable")
    with pytest.raises(RegionalFixtureError):
        getattr(fixture, operation)()
    fixture.close()
    assert port.calls == [], "missing authority must stop before fixture I/O"


def test_existing_safety_binding_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, port, _ = setup(tmp_path, monkeypatch, "agent-unavailable")
    bind_fixture(fixture)
    assert fixture.plan is not None
    with pytest.raises(RegionalFixtureError, match="binding differs"):
        fixture.bind_safety(fixture.plan, datetime.now(timezone.utc))
    assert port.calls == [], "a second binding cannot authorize recovery"


def test_metadata_scenario_without_original_mutation_custody_requires_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, port, _ = setup(tmp_path, monkeypatch, "no-spare")
    with pytest.raises(RegionalFixtureError, match="original mutation owner"):
        fixture.resume_cleanup()
    assert port.calls == [], "metadata must not be guessed from its current value"
