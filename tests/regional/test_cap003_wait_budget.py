from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from scripts.e2e.regional import capacity_acceptance_cases as cases
from scripts.e2e.regional.probes import cap_probe_app as probe
from tests.regional.test_cov95_cap_case_lifecycle import model as capacity_model_fixture
from tests.regional.test_cov95_cap_probe_app import HoldIO, app_with_io

model = capacity_model_fixture


@pytest.mark.parametrize(
    ("budget", "maximum", "effective"),
    [(15, 30, 12), (0, 30, 20), (30, 30, 20), (15, 8, 8), (2, 30, 0)],
)
def test_private_probe_reports_deployed_budget_and_server_cap(
    monkeypatch: pytest.MonkeyPatch, budget: float, maximum: float, effective: float
) -> None:
    monkeypatch.setenv("GPU_FAULT_REQUEST_BUDGET_SECONDS", str(budget))
    io = HoldIO()
    app = app_with_io(monkeypatch, io)
    app.state.remote_command_wakeups = SimpleNamespace(
        listener_alive=True,
        connected=True,
        disconnects_total=0,
        wakeups_total=0,
        max_wait_seconds=maximum,
    )

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://example",
        ) as client:
            response = await client.post(
                "/__cap__/claim-wakeups", json={"tag": "budget", "wait_seconds": 20}
            )
            assert response.status_code == 200
            report = response.json()
            assert report["requested_wait_seconds"] == 20
            assert report["request_budget_seconds"] == budget
            assert report["claim_wait_reserve_seconds"] == 3
            assert report["server_max_wait_seconds"] == maximum
            assert report["effective_wait_seconds"] == effective
            legacy = await client.post("/__cap__/claim-wakeups", json={"tag": "budget"})
            assert set(legacy.json()) == {
                "listener_alive",
                "listener_connected",
                "disconnects_total",
                "wakeups_total",
            }
            invalid = await client.post(
                "/__cap__/claim-wakeups", json={"tag": "budget", "wait_seconds": True}
            )
            assert invalid.status_code == 422

    asyncio.run(exercise())
    assert io.attempts == 0, "budget inspection must not occupy Store I/O"


@pytest.mark.parametrize(
    ("requested", "budget", "maximum"),
    [
        (0, 15, 30),
        (-1, 15, 30),
        ("20", 15, 30),
        (float("nan"), 15, 30),
        (20, float("inf"), 30),
        (20, -1, 30),
        (20, 15, 0),
        (20, 15, True),
    ],
)
def test_invalid_runtime_wait_parameters_are_not_reported_as_healthy(
    requested: Any, budget: float, maximum: Any
) -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        probe.claim_wait_configuration(
            requested, request_budget=budget, maximum=maximum
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("request_budget_seconds", None),
        ("request_budget_seconds", float("nan")),
        ("server_max_wait_seconds", 0),
        ("claim_wait_reserve_seconds", -1),
        ("effective_wait_seconds", 20),
        ("effective_wait_seconds", 0),
        ("effective_wait_seconds", True),
        ("requested_wait_seconds", 10),
    ],
)
def test_runner_refuses_missing_or_inconsistent_budget_evidence(
    key: str, value: Any
) -> None:
    report = probe.claim_wait_configuration(20, request_budget=15, maximum=30)
    report[key] = value
    with pytest.raises(cases.CapError, match="CAP003 effective wait"):
        cases.cap003_wait_configuration(report, 20)


@pytest.mark.parametrize(
    ("elapsed", "passed"), [(10.9, False), (12.2, True), (13.1, False)]
)
def test_clipped_wait_retains_early_return_and_overhead_guards(
    model: Any, tmp_path: Path, elapsed: float, passed: bool
) -> None:
    value = model()
    value.claim_elapsed_override = elapsed
    if passed:
        assert value.case_003()["status"] == "PASS"
    else:
        with pytest.raises(cases.CapError, match="CAP-003 failed"):
            value.case_003()
    rows = json.loads((tmp_path / "CAP-003/summary.json").read_text())[
        "production_long_poll"
    ]
    assert len(rows) == 4
    assert all(row["passed"] is passed for row in rows), (
        'test_clipped_wait_retains_early_return_and_overhead_guards: expected all(row["passed"] is passed for row in rows)'
    )
    assert all(row["wait_seconds"] == 20 for row in rows), (
        'test_clipped_wait_retains_early_return_and_overhead_guards: expected all(row["wait_seconds"] == 20 for row in rows)'
    )
    assert all(row["effective_wait_seconds"] == 12 for row in rows), (
        'test_clipped_wait_retains_early_return_and_overhead_guards: expected all(row["effective_wait_seconds"] == 12 for row in rows)'
    )
    assert all(
        abs(sample["overhead_ms"] - max(0, elapsed - 12) * 1000) < 5
        for row in rows
        for sample in row["samples"]
    ), (
        'test_clipped_wait_retains_early_return_and_overhead_guards: expected all( abs(sample["overhead_ms"] - max(0, elapsed - 12) * 1000) < 5 fo...'
    )
    assert value.events[-1] == ("cleanup", "cap-probe")


@pytest.mark.parametrize("budget", [0, 30])
def test_unclipped_runtime_configuration_still_requires_the_full_requested_hold(
    model: Any, budget: float
) -> None:
    value = model()
    value.claim_request_budget = budget
    result = value.case_003()
    assert result["status"] == "PASS"
    assert all(
        row["effective_wait_seconds"] == 20
        and row["wait_configuration"]["request_budget_seconds"] == budget
        for row in result["production_long_poll"]
    ), (
        'test_unclipped_runtime_configuration_still_requires_the_full_requested_hold: expected all( row["effective_wait_seconds"] == 20 and row["w...'
    )


def test_configuration_drift_during_a_scale_is_not_accepted(
    model: Any, tmp_path: Path
) -> None:
    value = model()
    original = value.probe_control
    reads = 0

    def changing(probe: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal reads
        if path == "/__cap__/claim-wakeups":
            reads += 1
            if reads == 2:
                value.claim_request_budget = 16
        return original(probe, path, payload)

    value.probe_control = changing
    with pytest.raises(cases.CapError, match="CAP-003 failed"):
        value.case_003()
    rows = json.loads((tmp_path / "CAP-003/summary.json").read_text())[
        "production_long_poll"
    ]
    assert rows[0]["passed"] is False
    assert value.events[-1] == ("cleanup", "cap-probe")
