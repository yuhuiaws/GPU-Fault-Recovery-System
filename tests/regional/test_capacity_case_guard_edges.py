"""CAP-002/CAP-004 guard edges on the capacity harness.

The AMP target reader with an exact label match, the CAP-002 preconditions
(previous scrape stop proven, plan-bound scrape source, enough window for the
alert hold), a scrape companion that cannot start or starts without evidence,
a companion stop interrupted by the operator, and a CAP-004 Store probe that
answers for another run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import capacity_acceptance_cases as cases
from scripts.e2e.regional.capacity_acceptance_base import CapError
from tests.regional import test_capacity_scrape_lifecycle as scrape_lifecycle
from tests.regional._cov95_capacity_support import capacity_fixture as capacity_fixture
from tests.regional.test_capacity_scrape_lifecycle import LifecycleHarness

lifecycle = scrape_lifecycle.lifecycle


def test_cap002_target_value_accepts_an_exact_label_match() -> None:
    now = time.time()
    result = [{"metric": {"pod": "probe", "capacity_run": "r1"}, "value": [now, "0.5"]}]
    assert cases.cap002_target_value(
        result, expected_labels={"pod": "probe", "capacity_run": "r1"}
    ) == pytest.approx(0.5)
    assert cases.cap002_target_value(result, expected_labels={"pod": "other"}) is None


def test_cap002_refuses_to_start_while_the_previous_scrape_stop_is_unproven(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.cap002_scrape_stopped = False
    with pytest.raises(CapError, match="previous scrape shutdown is unverified"):
        lifecycle.case_002_v2()
    assert lifecycle.events == [], "an unproven stop must not deploy or query anything"


def test_cap002_requires_the_plan_bound_scrape_source(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.scrape_source_binding = {}
    with pytest.raises(CapError, match="plan-bound scrape source"):
        lifecycle.case_002_v2()
    assert lifecycle.events == []


def test_cap002_alert_hold_needs_enough_maintenance_window(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.maintenance_deadline = datetime.fromtimestamp(
        lifecycle.clock.time() + cases.CAP002_ALERT_HOLD_SECONDS / 2, timezone.utc
    )
    with pytest.raises(CapError, match="insufficient for the alert hold"):
        lifecycle.cap002_alert(
            lifecycle.probe, lifecycle.run_dir, {"passed": True}, selector="pod"
        )
    assert not any(event[0] == "/__cap__/hold" for event in lifecycle.events), (
        "a short window must never place a saturation hold"
    )


def test_cap002_scrape_start_without_evidence_fails_and_still_stops_the_companion(
    lifecycle: LifecycleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion_calls: list[str] = []

    def companion(*_arguments: Any, **_keywords: Any) -> SimpleNamespace:
        return SimpleNamespace(
            selector="pod",
            start=lambda: companion_calls.append("start") or ["not", "a", "dict"],
            stop=lambda: companion_calls.append("stop")
            or {"cleanup_complete": True, "process_termination_proven": True},
        )

    monkeypatch.setattr(cases, "ScrapeCompanion", companion)
    with pytest.raises(CapError, match="scrape start evidence is invalid"):
        lifecycle.case_002_v2()
    assert companion_calls == ["start", "stop"]
    assert lifecycle.cap002_scrape_stopped is True
    summary = json.loads((lifecycle.run_dir / "CAP-002" / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert summary["scrape_ready"] is False
    assert summary["error_type"] == "CapError"


def test_cap002_companion_construction_failure_leaves_no_companion_to_stop(
    lifecycle: LifecycleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def companion(*_arguments: Any, **_keywords: Any) -> Any:
        raise OSError("scrape companion binary is unavailable")

    monkeypatch.setattr(cases, "ScrapeCompanion", companion)
    with pytest.raises(OSError, match="companion binary"):
        lifecycle.case_002_v2()
    assert not any(event[0] == "scrape-stop" for event in lifecycle.events), (
        "a companion that never existed cannot be stopped"
    )
    assert ("delete", "deployment") in lifecycle.events, (
        "the probe is still cleaned up when no scrape ever started"
    )
    summary = json.loads((lifecycle.run_dir / "CAP-002" / "summary.json").read_text())
    assert summary["cleanup_errors"] == []
    assert summary["scrape_stopped"] is True


def test_cap002_operator_interrupt_during_companion_stop_is_the_raised_error(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.resolve_values = [("1", 0), ("0", 91), ("0", 0), ("0", 0)]
    lifecycle.resolve_states = [[], [], ["pending"], []]
    lifecycle.stop_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as interrupted:
        lifecycle.case_002_v2()
    assert interrupted.value.__notes__ == ["scrape companion stop: KeyboardInterrupt"]
    summary = json.loads((lifecycle.run_dir / "CAP-002" / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert summary["scrape_stopped"] is False
    assert summary["cleanup_errors"] == ["scrape companion stop: KeyboardInterrupt"]


def test_cap004_store_probe_must_answer_for_this_run(
    lifecycle: LifecycleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lifecycle,
        "kubectl",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps({"run_id": "another-run"})),
    )
    with pytest.raises(CapError, match="invalid identity"):
        lifecycle.cap004_store(
            SimpleNamespace(database="gpu_fault_unit", pod="probe-pod"), "snapshot"
        )


def test_cap002_target_value_without_label_expectations_reads_the_sample() -> None:
    now = time.time()
    result = [{"metric": {"pod": "any"}, "value": [now, "0.25"]}]
    assert cases.cap002_target_value(result) == pytest.approx(0.25)


def test_cap004_store_probe_for_this_run_returns_its_rows(
    lifecycle: LifecycleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = {"run_id": lifecycle.run_id, "commands": [{"status": "SUCCEEDED"}]}
    issued: list[tuple[str, ...]] = []

    def kubectl(*arguments: str, **keywords: Any) -> SimpleNamespace:
        issued.append(arguments)
        return SimpleNamespace(stdout=json.dumps(rows))

    monkeypatch.setattr(lifecycle, "kubectl", kubectl)
    probe = SimpleNamespace(database="gpu_fault_unit", pod="probe-pod")
    assert lifecycle.cap004_store(probe, "cleanup") == rows
    assert issued[0][0] == "exec"
    assert {"probe-pod", "cleanup", lifecycle.run_id} <= set(issued[0])
